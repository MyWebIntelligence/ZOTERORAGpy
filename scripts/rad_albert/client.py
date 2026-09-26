"""Client HTTP de l'API Albert (DINUM) — httpx seul, sans SDK openai.

Règles (décision 4 du sprint) :

* la clé ``api_key`` est **obligatoire** et n'est jamais lue dans
  l'environnement ; elle ne voyage que dans l'en-tête ``Authorization`` ;
* en-têtes envoyés : ``Authorization`` et ``User-Agent`` seulement (plus
  ceux que le corps impose : ``Content-Type``, ``Content-Length``, ``Host``) ;
* ``follow_redirects=False`` : une redirection est une erreur ;
* jamais de champ ``user`` ; jamais de ``dimensions`` pour les embeddings ;
* une seule couche de réessai (``retry.call_with_retry``), limiteur proactif
  par rôle acquis hors sémaphore, sommeil hors sémaphore ; les créations non
  idempotentes (collection, document, envoi de chunks) ne sont renvoyées que
  si la requête n'a sûrement pas été traitée (connexion impossible, 429, 503),
  sinon ``AlbertUncertainWriteError`` ;
* erreurs classées sur le statut HTTP (``errors.classify_http_error``), corps
  masqué (clé retirée) avant toute journalisation ;
* ids de modèles épinglés : ``response.model`` recopie l'alias demandé (D5,
  D12), il est journalisé mais ne sert jamais à connaître l'id ; dès que la
  liste ``/v1/models`` est connue (preflight, ``models()``), c'est l'id résolu
  qui part sur le fil, jamais l'alias.

Le transport est injectable (``httpx.MockTransport``, voir
``tests/albert_fakes.FakeAlbert``).
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import logging
import math
import dataclasses
import threading
import time
from datetime import date
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import httpx

from . import catalog
from .config import AlbertConfig, normalise_base_url, root_url
from .errors import (
    AlbertAuthError,
    AlbertModelBusy,
    AlbertPermanentError,
    AlbertTransientError,
    AlbertTruncatedError,
    AlbertUncertainWriteError,
    classify_http_error,
    redact,
)
from .limiter import bucket_for_role, estimate_input_tokens, get_limiter
from .retry import RetryPolicy, call_with_retry, parse_retry_after
from .usage import UsageLedger, extract_usage

logger = logging.getLogger(__name__)

USER_AGENT = "RAGpy-Albert/1.0"
"""Valeur de l'en-tête ``User-Agent`` (aucune donnée personnelle)."""

MAX_EMBED_BATCH = 64
"""Textes au plus par requête d'embeddings (D12 : 65 → 413)."""

MAX_CHUNKS_PER_POST = 64
"""Chunks au plus par ``POST /v1/documents/{id}/chunks`` (D16 : 65 → 422)."""

PAGE_LIMIT = 100
"""Taille de page maximale des listes (D14 : 101 → 422)."""

SEARCH_METHODS = ("semantic", "lexical", "hybrid")
"""Méthodes de recherche acceptées (``exact`` n'existe pas, D17)."""

CHUNKS_PER_EMBED_REQUEST = 32
"""Chunks vectorisés par requête d'embeddings côté serveur (D19 : 64 chunks = 2 requêtes)."""

PUSH_LEDGER_ENDPOINT = "/v1/documents/{document_id}/chunks"
"""Libellé ``endpoint`` des envois de chunks dans le ledger (gabarit, sans id de document)."""

_MAX_PAGES = 10000
_FORBIDDEN_BODY_KEYS = frozenset({"user", "model", "messages", "dimensions", "max_completion_tokens"})
_NOT_SENT_ERRORS: Tuple[type, ...] = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ProxyError)
"""Erreurs httpx levées avant tout envoi de la requête : un renvoi est sans risque."""
_SAFE_RESEND_STATUSES = frozenset({429, 503})
"""Statuts d'une création refusée avant traitement (quota, surcharge) : renvoi sans risque."""
_SCOPE_SERIAL = itertools.count(1)

ENDPOINTS: Tuple[Tuple[str, str], ...] = (
    ("GET", "/health"),
    ("GET", "/health/models"),
    ("GET", "/v1/me"),
    ("GET", "/v1/models"),
    ("GET", "/v1/models/{model}"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/embeddings"),
    ("POST", "/v1/ocr"),
    ("GET", "/v1/collections"),
    ("POST", "/v1/collections"),
    ("GET", "/v1/collections/{collection_id}"),
    ("DELETE", "/v1/collections/{collection_id}"),
    ("GET", "/v1/documents"),
    ("POST", "/v1/documents"),
    ("GET", "/v1/documents/{document_id}"),
    ("DELETE", "/v1/documents/{document_id}"),
    ("GET", "/v1/documents/{document_id}/chunks"),
    ("POST", "/v1/documents/{document_id}/chunks"),
    ("POST", "/v1/search"),
)
"""Points d'accès appelés par ce client (méthode, gabarit relatif à la racine du service)."""

MISSING_KEY_MESSAGE = (
    "Clé API Albert (DINUM) requise. Configurez-la dans Paramètres > Mes Identifiants."
)


# ---------------------------------------------------------------------------
# Erreurs propres au client
# ---------------------------------------------------------------------------
class AlbertMissingKeyError(AlbertAuthError, ValueError):
    """Clé absente ou vide à la construction du client (jamais lue dans l'environnement)."""


class AlbertInputError(AlbertPermanentError, ValueError):
    """Requête refusée côté client, avant tout envoi (entrée vide, lot trop grand…)."""


class AlbertResponseError(AlbertPermanentError):
    """Réponse 2xx inexploitable (dimension inattendue, vecteur nul, index manquant…)."""


# ---------------------------------------------------------------------------
# Résultat de chat
# ---------------------------------------------------------------------------
def _rebuild_chat_result(content: str, state: Dict[str, Any]) -> "ChatResult":
    """Reconstruit un ``ChatResult`` (support de ``pickle``)."""
    obj = ChatResult(content)
    obj.__dict__.update(state)
    return obj


class ChatResult(str):
    """Contenu d'une réponse de chat (``message.content`` seulement), avec ses métadonnées.

    Sous-classe de ``str`` : le résultat se compare et s'utilise comme le texte
    renvoyé. Le champ ``reasoning`` des modèles à raisonnement n'est jamais
    repris.

    Attributes:
        finish_reason: ``stop``, ``length``, ``content_filter``…
        model: id épinglé envoyé (modèle servi, repli compris).
        response_model: champ ``model`` de la réponse (peut recopier un alias).
        requested_model: premier modèle de la chaîne essayée.
        fallback_from: modèle primaire quand un repli a servi la réponse.
        usage: usage extrait (tokens, coût, impacts).
        request_id: identifiant de la réponse.
        role: rôle applicatif.
    """

    def __new__(
        cls,
        content: Optional[str] = "",
        *,
        finish_reason: Optional[str] = None,
        model: Optional[str] = None,
        response_model: Optional[str] = None,
        requested_model: Optional[str] = None,
        fallback_from: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> "ChatResult":
        """Crée le résultat (``content`` ``None`` devient ``''``)."""
        obj = super().__new__(cls, content or "")
        obj.finish_reason = finish_reason
        obj.model = model
        obj.response_model = response_model
        obj.requested_model = requested_model if requested_model is not None else model
        obj.fallback_from = fallback_from
        obj.usage = usage or {}
        obj.request_id = request_id
        obj.role = role
        return obj

    @property
    def content(self) -> str:
        """Texte de la réponse (``str`` simple)."""
        return str.__str__(self)

    @property
    def served_model(self) -> Optional[str]:
        """Synonyme de ``model`` (modèle qui a servi la réponse)."""
        return self.model

    @property
    def truncated(self) -> bool:
        """Vrai si la génération s'est arrêtée sur ``length`` ou ``content_filter``."""
        return self.finish_reason in ("length", "content_filter")

    def __reduce__(self) -> Tuple[Any, ...]:
        """Sérialisation ``pickle`` (contenu et métadonnées)."""
        return (_rebuild_chat_result, (self.content, dict(self.__dict__)))


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def _sniff_mime(raw: bytes, default: str) -> str:
    """Type MIME d'après la signature des octets (PDF, PNG, JPEG, WEBP, TIFF, GIF)."""
    if raw.startswith(b"%PDF"):
        return "application/pdf"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return default


def _data_uri(data: Union[bytes, bytearray, str], mime: Optional[str], default: str) -> Tuple[str, str]:
    """(URI ``data:``, type MIME) pour des octets, du base64 ou une URI existante."""
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
        kind = mime or _sniff_mime(raw, default)
        return f"data:{kind};base64,{base64.b64encode(raw).decode('ascii')}", kind
    text = str(data).strip()
    if text.startswith("data:"):
        kind = text[5:].split(";", 1)[0].split(",", 1)[0] or (mime or default)
        return text, kind
    if text.startswith(("https://", "http://")):
        return text, mime or default
    kind = mime or default
    return f"data:{kind};base64,{text}", kind


def _message_list(messages: Any) -> List[Dict[str, Any]]:
    """Liste de messages (une chaîne devient un message ``user``)."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, Mapping):
        return [dict(messages)]
    if not isinstance(messages, (list, tuple)) or not messages:
        raise AlbertInputError("Chat Albert : liste de messages vide ou invalide.", reason="bad_request")
    return [dict(m) if isinstance(m, Mapping) else m for m in messages]


def _content_text(content: Any) -> Optional[str]:
    """Texte d'un ``message.content`` (chaîne, ou liste de parties ``text``)."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text") for p in content if isinstance(p, Mapping) and isinstance(p.get("text"), str)]
        return "".join(parts)
    return str(content)


def _resend_is_safe(error: AlbertTransientError) -> bool:
    """Vrai si une création a sûrement échoué avant tout traitement (renvoi sans doublon).

    Cas sûrs : 429 et 503 (requête refusée avant traitement), ou erreur httpx
    levée avant l'envoi (connexion, délai de connexion ou d'attente du pool,
    proxy). Tout le reste (délai de lecture ou d'écriture, protocole rompu,
    autre 5xx, 2xx illisible) laisse l'issue inconnue.
    """
    if error.status is not None:
        return error.status in _SAFE_RESEND_STATUSES
    return isinstance(error.__cause__, _NOT_SENT_ERRORS)


def _as_int(value: Any, what: str) -> int:
    """Identifiant entier strictement positif (collection, document)."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise AlbertInputError(f"Identifiant Albert invalide ({what}) : {value!r}.", reason="bad_request") from None
    if number < 0 or isinstance(value, bool):
        raise AlbertInputError(f"Identifiant Albert invalide ({what}) : {value!r}.", reason="bad_request")
    return number


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class AlbertClient:
    """Client synchrone et thread-safe de l'API Albert.

    Un nom de modèle est envoyé tel quel (préfixe ``albert/`` retiré), sauf
    s'il a été résolu en id par ``/v1/models`` (``run_preflight`` ou
    ``resolve_model_id``) : c'est alors l'id épinglé qui part.

    Temps : par défaut, les limiteurs sont les instances partagées du
    processus (``limiter.get_limiter``), quel que soit le ``sleep`` fourni
    (un sommeil réel enveloppé, pour journaliser ou annuler, garde le budget
    commun). Le temps n'est **virtuel** que sur demande (``virtual_time=True``)
    ou, par défaut, quand le transport injecté est un ``httpx.MockTransport``
    (aucun serveur réel) : les limiteurs, locaux, sont alors propres au client
    et suivent une horloge qui n'avance que par ses propres sommeils (une
    attente réelle n'est jamais comptée deux fois ; aucune attente réelle si
    le sommeil injecté n'attend pas).

    Attributes:
        cfg: configuration Albert (``AlbertConfig``).
        base_url: URL de base normalisée (``…/v1``).
        root: racine du service (``/health``).
        ledger: ledger d'usage (fourni ou créé).
    """

    def __init__(
        self,
        cfg: Optional[AlbertConfig],
        api_key: str,
        *,
        transport: Optional[httpx.BaseTransport] = None,
        ledger: Optional[UsageLedger] = None,
        sleep: Callable[[float], Any] = time.sleep,
        limiters: Optional[Union[Mapping[str, Any], Callable[[str], Any]]] = None,
        use_limiter: bool = True,
        policy: Optional[RetryPolicy] = None,
        today: Optional[date] = None,
        clock: Callable[[], float] = time.monotonic,
        rng: Any = None,
        user_agent: str = USER_AGENT,
        virtual_time: Optional[bool] = None,
    ) -> None:
        """Construit le client ; aucune requête n'est émise.

        Args:
            cfg: configuration (``None`` : ``AlbertConfig.from_env()``, qui ne lit
                jamais la clé).
            api_key: clé Bearer, **obligatoire** (jamais lue dans l'environnement).
            transport: transport httpx injecté (``httpx.MockTransport`` en test).
            ledger: ledger d'usage partagé (défaut : un ledger propre au client).
            sleep: sommeil des réessais et, en temps virtuel, des limiteurs
                (injectable ; ne rend pas à lui seul le temps virtuel).
            limiters: limiteurs par rôle ou par budget (dict ou fonction
                ``rôle → limiteur``) ; défaut : ``limiter.get_limiter(rôle, cfg)``.
            use_limiter: ``False`` désactive le limiteur proactif.
            policy: politique de réessai (défaut : dérivée de ``cfg``).
            today: date de référence des chaînes de repli (défaut : jour courant).
            clock: horloge monotone (latences du ledger).
            rng: générateur de la gigue des réessais.
            user_agent: valeur de ``User-Agent``.
            virtual_time: ``True`` : limiteurs propres au client sur une horloge
                virtuelle avancée par ``sleep`` ; ``False`` : limiteurs partagés
                du processus ; ``None`` (défaut) : virtuel seulement si
                ``transport`` est un ``httpx.MockTransport``.

        Raises:
            AlbertMissingKeyError: clé absente ou vide (``AlbertAuthError`` et
                ``ValueError``).
            ValueError: URL de base refusée.
        """
        if not isinstance(api_key, str) or not api_key.strip():
            raise AlbertMissingKeyError(MISSING_KEY_MESSAGE)
        self.cfg: AlbertConfig = cfg if cfg is not None else AlbertConfig.from_env()
        self.__api_key = api_key.strip()
        self.base_url = normalise_base_url(self.cfg.base_url, allow_custom_host=self.cfg.allow_custom_host)
        self.root = root_url(self.base_url)
        self.ledger: UsageLedger = ledger if ledger is not None else UsageLedger()
        if virtual_time is None:
            virtual_time = isinstance(transport, httpx.MockTransport)
        self._virtual_time = bool(virtual_time)
        self._virtual_base = time.monotonic()
        self._virtual_offset = 0.0
        self._time_lock = threading.Lock()
        self._injected_sleep = sleep
        self._sleep: Callable[[float], Any] = self._virtual_sleep if self._virtual_time else sleep
        self._own_limiters: Dict[str, Any] = {}
        self._limiters = limiters
        self._use_limiter = bool(use_limiter)
        self._policy = policy if policy is not None else RetryPolicy.from_config(self.cfg)
        self._today = today
        self._clock = clock
        self._rng = rng
        self._headers = {"Authorization": f"Bearer {self.__api_key}", "User-Agent": user_agent}
        self._http = httpx.Client(transport=transport, follow_redirects=False)
        self._scope = None if transport is None else f"transport-{next(_SCOPE_SERIAL)}"
        self._lock = threading.Lock()
        self._resolved: Dict[str, str] = {}
        self._listing: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------ temps virtuel
    def _virtual_sleep(self, seconds: float) -> None:
        """Sommeil en temps virtuel : avance l'horloge virtuelle puis délègue au sommeil fourni."""
        with self._time_lock:
            self._virtual_offset += max(0.0, float(seconds))
        self._injected_sleep(seconds)

    def _virtual_clock(self) -> float:
        """Horloge des limiteurs en temps virtuel : n'avance que par les sommeils du client.

        Le temps réel écoulé n'y entre pas : un sommeil injecté qui attend
        vraiment n'est compté qu'une fois, et le débit ne dépasse jamais le
        plafond configuré.
        """
        with self._time_lock:
            return self._virtual_base + self._virtual_offset

    # ------------------------------------------------------------------ cycle de vie
    def close(self) -> None:
        """Ferme les connexions HTTP."""
        self._http.close()

    def __enter__(self) -> "AlbertClient":
        """Contexte ``with AlbertClient(...) as client``."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Ferme le client en sortie de contexte."""
        self.close()

    def __repr__(self) -> str:
        """Représentation sans la clé."""
        return f"AlbertClient(base_url={self.base_url!r})"

    @property
    def key_fingerprint(self) -> str:
        """Empreinte non réversible de la clé (``sha256[:12]``), pour les caches."""
        return hashlib.sha256(self.__api_key.encode("utf-8")).hexdigest()[:12]

    @property
    def cache_scope(self) -> Tuple[str, str, Optional[str]]:
        """Portée des caches (preflight) : empreinte de clé, URL de base, transport injecté.

        Sans transport injecté, deux clients de même clé et même URL partagent la
        portée ; un transport injecté (tests) donne une portée propre au client.
        """
        return (self.key_fingerprint, self.base_url, self._scope)

    @property
    def policy(self) -> RetryPolicy:
        """Politique de réessai effective."""
        return self._policy

    # ------------------------------------------------------------------ modèles
    def remember_models(self, mapping: Mapping[str, str]) -> None:
        """Mémorise des résolutions nom → id épinglé (issues du preflight)."""
        with self._lock:
            for name, ident in mapping.items():
                if name and ident:
                    self._resolved[str(name)] = str(ident)

    def wire_model(self, model: Any) -> str:
        """Nom envoyé pour ``model`` : préfixe ``albert/`` retiré, puis id épinglé.

        L'id vient des résolutions mémorisées (preflight, ``resolve_model_id``)
        ou, à défaut, de la liste ``/v1/models`` déjà chargée par ``models()``
        (sans requête) ; il est alors mémorisé. Sans liste connue, le nom part
        tel quel. Un alias n'est jamais envoyé quand son id est connu.

        Raises:
            AlbertInputError: nom vide.
        """
        name = str(model).strip() if model is not None else ""
        if name[:7].lower() == "albert/":
            name = name[7:].strip()
        if not name:
            raise AlbertInputError("Modèle Albert vide.", reason="bad_request")
        with self._lock:
            resolved = self._resolved.get(name) or self._resolved.get(name.lower())
            listing = self._listing
        if resolved:
            return resolved
        if listing is not None:
            try:
                ident = catalog.resolve_model(name, listing)
            except (catalog.ModelNotFoundError, ValueError):
                return name
            self.remember_models({name: ident})
            return ident
        return name

    def resolve_model_id(self, model: Any, *, refresh: bool = False) -> str:
        """Résout un id ou un alias par ``/v1/models`` (liste mise en cache par le client).

        Raises:
            catalog.ModelNotFoundError: modèle absent de la liste du compte.
        """
        listing = self.models() if (refresh or self._listing is None) else self._listing
        name = str(model).strip() if model is not None else ""
        if name[:7].lower() == "albert/":
            name = name[7:].strip()
        ident = catalog.resolve_model(name, listing)
        self.remember_models({name: ident})
        return ident

    # ------------------------------------------------------------------ transport
    def _limiter_for(self, role: Optional[str]) -> Any:
        """Limiteur du rôle (``None`` si désactivé ou sans budget)."""
        if not self._use_limiter or not role:
            return None
        if callable(self._limiters):
            return self._limiters(role)
        bucket = bucket_for_role(role)
        if isinstance(self._limiters, Mapping):
            if role in self._limiters:
                return self._limiters[role]
            if bucket in self._limiters:
                return self._limiters[bucket]
        if not self._virtual_time:
            return get_limiter(role, self.cfg)
        if bucket is None:
            return None
        with self._lock:
            limiter = self._own_limiters.get(bucket)
            if limiter is None:
                local_cfg = self.cfg
                if getattr(local_cfg, "limiter_backend", "local") != "local" and dataclasses.is_dataclass(local_cfg):
                    # Temps virtuel : une fenêtre Redis partagée n'a pas de sens.
                    local_cfg = dataclasses.replace(local_cfg, limiter_backend="local")
                limiter = get_limiter(role, local_cfg, clock=self._virtual_clock, sleep=self._sleep)
                self._own_limiters[bucket] = limiter
            return limiter

    def _redact(self, text: Any, limit: Optional[int] = None) -> str:
        """Masque la clé (et les jetons connus) dans un texte."""
        return redact(text, secrets=(self.__api_key,), limit=limit, emails=False)

    def _send(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        json_body: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        files: Any = None,
    ) -> httpx.Response:
        """Un échange HTTP ; lève une ``AlbertError`` classée pour tout statut non 2xx."""
        url = self.root + path
        clean_params = {k: v for k, v in (params or {}).items() if v is not None} or None
        request = httpx.Request(
            method,
            url,
            params=clean_params,
            json=json_body,
            files=files,
            headers=self._headers,
            extensions={"timeout": httpx.Timeout(float(timeout)).as_dict()},
        )
        try:
            response = self._http.send(request, follow_redirects=False)
        except httpx.TimeoutException as exc:
            raise AlbertTransientError(
                endpoint=path, detail=f"délai dépassé ({type(exc).__name__})"
            ) from exc
        except (httpx.NetworkError, httpx.RemoteProtocolError, httpx.ProxyError) as exc:
            raise AlbertTransientError(
                endpoint=path, detail=f"{type(exc).__name__} : {self._redact(exc, 200)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AlbertPermanentError(
                reason="bad_request", endpoint=path, detail=f"{type(exc).__name__} : {self._redact(exc, 200)}"
            ) from exc
        logger.debug("Albert %s %s -> HTTP %s", method, path, response.status_code)
        if 200 <= response.status_code < 300:
            return response
        body = self._redact(response.text)
        retry_after = parse_retry_after(response.headers.get("retry-after"))
        raise classify_http_error(
            response.status_code,
            body,
            response.headers,
            endpoint=path,
            retry_after=retry_after,
            retry_after_max=self._policy.retry_after_max,
        )

    def _decode(self, response: httpx.Response, path: str) -> Any:
        """Corps JSON d'une réponse 2xx (``{}`` si vide)."""
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            raise AlbertTransientError(
                status=response.status_code, endpoint=path, detail="réponse non JSON"
            ) from None

    def _call(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        role: Optional[str] = None,
        json_body: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        files: Any = None,
        tokens: int = 0,
        requests: int = 1,
        semaphore: Any = None,
        single_attempt: bool = False,
        handler: Optional[Callable[[Any, float], Any]] = None,
        idempotent: bool = True,
    ) -> Any:
        """Requête avec réessais ; renvoie ``handler(corps, latence)`` ou le corps JSON.

        Avec ``idempotent=False`` (créations : ``POST /v1/collections``,
        ``POST /v1/documents``, ``POST /v1/documents/{id}/chunks``), un échec
        transitoire n'est retenté que si la requête n'a sûrement pas été
        traitée (connexion impossible, 429, 503) ; sinon (délai de lecture ou
        d'écriture, protocole rompu, autre 5xx, 2xx illisible) il devient
        ``AlbertUncertainWriteError``, relevé sans renvoi.
        """
        policy = self._policy.single() if single_attempt else self._policy

        def attempt() -> Any:
            """Un essai : envoi, décodage, traitement du corps."""
            started = self._clock()
            try:
                response = self._send(
                    method, path, timeout=timeout, json_body=json_body, params=params, files=files
                )
                data = self._decode(response, path)
                latency = self._clock() - started
                return handler(data, latency) if handler is not None else data
            except AlbertTransientError as exc:
                if idempotent or _resend_is_safe(exc):
                    raise
                raise AlbertUncertainWriteError(
                    status=exc.status, endpoint=path, detail=exc.detail or type(exc).__name__
                ) from exc

        return call_with_retry(
            attempt,
            policy=policy,
            semaphore=semaphore,
            limiter=self._limiter_for(role),
            tokens=tokens,
            requests=requests,
            sleep=self._sleep,
            rng=self._rng,
            label=path,
        )

    def _paginate(
        self,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        limit: int = PAGE_LIMIT,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Parcourt une liste paginée (``offset``/``limit``, ``limit`` <= 100) jusqu'à une page incomplète."""
        size = max(1, min(int(limit), PAGE_LIMIT))
        items: List[Dict[str, Any]] = []
        offset = 0
        for _ in range(_MAX_PAGES):
            query = dict(params or {})
            query.update({"offset": offset, "limit": size})
            page = self._call("GET", path, timeout=timeout or self.cfg.timeout_collections, params=query)
            data = page.get("data") if isinstance(page, Mapping) else page
            data = list(data or [])
            items.extend(data)
            if len(data) < size:
                break
            offset += len(data)
        return items

    # ------------------------------------------------------------------ compte et santé
    def me(self) -> Dict[str, Any]:
        """``GET /v1/me`` : compte, budget, limites, expiration."""
        return self._call("GET", "/v1/me", timeout=self.cfg.timeout_collections)

    def models(self) -> List[Dict[str, Any]]:
        """``GET /v1/models`` : liste des modèles du compte (mise en cache sur le client)."""
        data = self._call("GET", "/v1/models", timeout=self.cfg.timeout_collections)
        listing = list(catalog.listing_entries(data))
        with self._lock:
            self._listing = listing
        return listing

    def model_info(self, model: Any) -> Dict[str, Any]:
        """``GET /v1/models/{model}`` (id ou alias, préfixe ``albert/`` accepté).

        Le nom est d'abord ramené à son id (``wire_model`` ; pour un alias de
        type ``organisation/modèle``, le catalogue statique), puis encodé comme
        **un seul** segment de chemin (``/``, ``?``, ``#``, ``%`` encodés) : il ne
        peut ni changer de route ni ajouter de paramètre.

        Raises:
            AlbertInputError: nom vide ou réduit à des points (aucun envoi).
        """
        name = self.wire_model(model)
        if "/" in name:
            name = catalog.canonical_id(name)
        if not name.strip("."):
            raise AlbertInputError(f"Modèle Albert invalide : {name!r}.", reason="bad_request")
        return self._call(
            "GET", "/v1/models/" + quote(name, safe=""), timeout=self.cfg.timeout_collections
        )

    def health(self) -> Dict[str, Any]:
        """``GET /health`` à la racine du service (hors ``/v1``)."""
        return self._call("GET", "/health", timeout=self.cfg.timeout_collections)

    def health_models(self) -> Dict[str, Any]:
        """``GET /health/models`` à la racine du service."""
        return self._call("GET", "/health/models", timeout=self.cfg.timeout_collections)

    # ------------------------------------------------------------------ chat
    def _today_for(self, today: Optional[date]) -> date:
        """Date de référence des chaînes de repli."""
        if today is not None:
            return catalog.as_date(today)
        if self._today is not None:
            return catalog.as_date(self._today)
        return date.today()

    def chat_chain(
        self,
        role: str,
        model: Any = None,
        *,
        today: Optional[date] = None,
        single_attempt: bool = False,
        fallback: bool = True,
    ) -> List[str]:
        """Modèles essayés pour un appel de chat, dans l'ordre.

        Le modèle explicite (s'il est fourni) passe en tête ; la chaîne du rôle
        suit si ``fallback`` est vrai, sauf avec ``ALBERT_MODEL_FALLBACK=0``, en
        mode ``single_attempt`` ou pour un rôle sans repli.

        Args:
            role: rôle applicatif.
            model: modèle explicite (id, alias ou ``albert/<id>``).
            today: date de référence (défaut : date du client, sinon jour courant).
            single_attempt: un seul modèle.
            fallback: ajoute la chaîne du rôle après le modèle explicite.

        Returns:
            Les noms envoyés, dans l'ordre d'essai.

        Raises:
            AlbertPermanentError: aucun modèle disponible pour le rôle.
        """
        spec = catalog.role_spec(role)
        day = self._today_for(today)
        if spec.name == "ocr_chat":
            chain = [self.wire_model(self.cfg.ocr_chat_model)]
        else:
            chain = [self.wire_model(m) for m in catalog.fallback_chain(spec.name, today=day)]
        if model is not None and str(model).strip():
            primary = self.wire_model(model)
            key = catalog.canonical_id(primary)
            rest = [m for m in chain if catalog.canonical_id(m) != key] if fallback else []
            chain = [primary] + rest
        if not chain:
            raise AlbertPermanentError(
                f"Aucun modèle Albert disponible pour le rôle {spec.name} au {day.isoformat()}.",
                reason="not_found",
            )
        if single_attempt or not fallback or not self.cfg.model_fallback or spec.name in catalog.NO_FALLBACK_ROLES:
            return chain[:1]
        return chain

    def _chat_body(
        self,
        wire: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        response_format: Any,
        seed: Optional[int],
        stop: Any,
        reasoning_effort: Optional[str],
        extra: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Corps de ``/v1/chat/completions`` pour un modèle donné (marge de raisonnement comprise)."""
        body: Dict[str, Any] = {"model": wire, "messages": messages}
        reasoning = catalog.is_reasoning(wire)
        if max_tokens is not None:
            budget = int(max_tokens)
            if reasoning:
                budget += int(self.cfg.reasoning_headroom)
            body["max_tokens"] = budget
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if response_format is not None:
            body["response_format"] = response_format
        if seed is not None:
            body["seed"] = seed
        if stop is not None:
            body["stop"] = stop
        if reasoning:
            body["reasoning_effort"] = reasoning_effort or self.cfg.reasoning_effort
        for key, value in extra.items():
            if key in _FORBIDDEN_BODY_KEYS or value is None:
                continue
            if key == "reasoning_effort" and not reasoning:
                continue
            body.setdefault(key, value)
        return body

    def chat(
        self,
        messages: Any,
        model: Any = None,
        *,
        role: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Any = None,
        seed: Optional[int] = None,
        stop: Any = None,
        reasoning_effort: Optional[str] = None,
        timeout: Optional[float] = None,
        single_attempt: bool = False,
        semaphore: Any = None,
        today: Optional[date] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        **extra: Any,
    ) -> ChatResult:
        """``POST /v1/chat/completions`` ; repli de rôle sur 503 répétés si ``role`` est donné.

        ``max_tokens`` est envoyé tel quel (D8) ; pour un modèle à raisonnement
        (gpt-oss), la marge ``ALBERT_REASONING_HEADROOM`` s'y ajoute et
        ``reasoning_effort`` est envoyé (D9). Seul ``message.content`` est
        renvoyé, jamais le champ ``reasoning``. Sans ``role``, seul ``model``
        est essayé (budget du limiteur : ``notes`` pour un modèle à raisonnement,
        sinon ``recode``) ; avec ``role``, la chaîne du rôle suit ``model`` (ou
        la remplace si ``model`` est absent) après des 503 répétés. L'ordre
        historique ``chat(model, messages)`` est reconnu (nom de modèle en
        premier, liste de messages en second).

        Args:
            messages: liste de messages (ou une chaîne : un message ``user``).
            model: modèle (id, alias ou ``albert/<id>``) ; ``None`` avec ``role``
                = chaîne du rôle.
            role: rôle (chaîne de repli, limiteur, délai).
            max_tokens: budget de la réponse (avant marge de raisonnement) ;
                ``max_completion_tokens`` (dans ``extra``) n'est jamais envoyé
                et ne sert de budget qu'à défaut de ``max_tokens`` (D8).
            temperature: température (omise si ``None``).
            top_p: ``top_p`` (omis si ``None``).
            response_format: ``json_object`` / ``json_schema`` (D10).
            seed: graine (mode durci, D11).
            stop: séquences d'arrêt.
            reasoning_effort: effort de raisonnement (défaut ``ALBERT_REASONING_EFFORT``).
            timeout: délai HTTP (défaut selon le rôle).
            single_attempt: un seul envoi, sans réessai ni repli.
            semaphore: sémaphore(s) tenu(s) pendant l'envoi seulement.
            today: date de référence de la chaîne de repli.
            extra_body: paramètres supplémentaires (``user`` est toujours retiré).
            **extra: autres paramètres du corps (mêmes règles que ``extra_body``).

        Returns:
            ``ChatResult`` (le texte, plus ``finish_reason``, ``model``…).

        Raises:
            AlbertInputError: ni modèle ni rôle, ou messages invalides.
            AlbertTruncatedError: contenu vide avec ``finish_reason=length``.
            AlbertModelBusy: 503 persistant sur tous les modèles essayés.
            AlbertError: autres erreurs classées.
        """
        if isinstance(messages, str) and isinstance(model, (list, tuple, Mapping)):
            messages, model = model, messages
        if messages is None:
            raise AlbertInputError("Chat Albert : messages manquants.", reason="bad_request")
        has_model = model is not None and bool(str(model).strip())
        if role is None:
            if not has_model:
                raise AlbertInputError("Chat Albert : modèle ou rôle requis.", reason="bad_request")
            role_name = "notes" if catalog.is_reasoning(self.wire_model(model)) else "recode"
        else:
            role_name = role
        spec = catalog.role_spec(role_name)
        if spec.endpoint != "chat":
            raise AlbertInputError(f"Le rôle {spec.name} n'utilise pas le chat.", reason="bad_request")
        message_list = _message_list(messages)
        chain = self.chat_chain(
            spec.name, model, today=today, single_attempt=single_attempt, fallback=role is not None
        )
        for wire in chain:
            catalog.check_endpoint_type(wire, "chat")
        merged_extra: Dict[str, Any] = dict(extra_body or {})
        merged_extra.update(extra)
        # D8 : envoyé avec max_tokens, max_completion_tokens l'emporte et annulerait
        # la marge de raisonnement ; il n'est donc jamais envoyé et ne sert que
        # de budget quand max_tokens est absent.
        completion_budget = merged_extra.pop("max_completion_tokens", None)
        if max_tokens is None and completion_budget is not None:
            max_tokens = int(completion_budget)
        if timeout is None:
            if spec.name == "notes":
                timeout = self.cfg.timeout_notes
            elif spec.name == "ocr_chat":
                timeout = self.cfg.timeout_ocr_page
            else:
                timeout = self.cfg.timeout_chat
        tokens = estimate_input_tokens(message_list, estimator=self.cfg.token_estimator)
        last_busy: Optional[AlbertModelBusy] = None
        for index, wire in enumerate(chain):
            body = self._chat_body(
                wire,
                message_list,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
                seed=seed,
                stop=stop,
                reasoning_effort=reasoning_effort,
                extra=merged_extra,
            )
            fallback_from = chain[0] if index > 0 else None
            try:
                return self._chat_once(
                    body,
                    role=spec.name,
                    requested=chain[0],
                    fallback_from=fallback_from,
                    timeout=float(timeout),
                    tokens=tokens,
                    semaphore=semaphore,
                    single_attempt=single_attempt,
                )
            except AlbertModelBusy as exc:
                last_busy = exc
                if index + 1 < len(chain):
                    logger.warning(
                        "Albert : modèle %s surchargé (503), repli sur %s (rôle %s).",
                        wire,
                        chain[index + 1],
                        spec.name,
                    )
                    continue
                raise
        assert last_busy is not None  # chaîne non vide : la boucle renvoie ou relève
        raise last_busy

    def chat_role(self, role: str, messages: Any, *, model: Any = None, **kwargs: Any) -> ChatResult:
        """Chat par rôle : chaîne du rôle (``model`` éventuel en tête), repli sur 503 répétés.

        Args:
            role: rôle (``recode``, ``citation``, ``notes``, ``book_structure``,
                ``long_context``, ``ocr_chat``).
            messages: messages (ou une chaîne).
            model: modèle placé en tête de la chaîne (optionnel).
            **kwargs: paramètres de ``chat`` (``max_tokens``, ``today``…).

        Returns:
            ``ChatResult`` ; ``fallback_from`` est posé si un repli a servi.
        """
        return self.chat(messages, model, role=role, **kwargs)

    def _chat_once(
        self,
        body: Dict[str, Any],
        *,
        role: str,
        requested: str,
        fallback_from: Optional[str],
        timeout: float,
        tokens: int,
        semaphore: Any,
        single_attempt: bool,
    ) -> ChatResult:
        """Un modèle de la chaîne : envoi avec réessais, ledger, contrôle de troncature."""
        path = "/v1/chat/completions"
        wire = body["model"]

        def handle(data: Any, latency: float) -> ChatResult:
            """Extrait ``message.content`` et ``finish_reason`` ; enregistre l'usage."""
            choices = data.get("choices") if isinstance(data, Mapping) else None
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
                raise AlbertTransientError(status=200, endpoint=path, detail="réponse de chat sans choices")
            choice = choices[0]
            message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
            content = _content_text(message.get("content"))
            finish = choice.get("finish_reason")
            self.ledger.record(
                endpoint=path,
                model=wire,
                response_model=data.get("model"),
                usage=data,
                role=role,
                latency_s=latency,
                request_id=data.get("id"),
                status=200,
                fallback_from=fallback_from,
                finish_reason=finish,
            )
            if finish == "length" and (content is None or not content.strip()):
                raise AlbertTruncatedError(
                    status=200,
                    endpoint=path,
                    detail=(
                        f"finish_reason=length sans contenu (modèle {wire}, "
                        f"max_tokens={body.get('max_tokens')})"
                    ),
                )
            return ChatResult(
                content or "",
                finish_reason=finish,
                model=wire,
                response_model=data.get("model"),
                requested_model=requested,
                fallback_from=fallback_from,
                usage=extract_usage(data),
                request_id=data.get("id"),
                role=role,
            )

        try:
            return self._call(
                "POST",
                path,
                timeout=timeout,
                role=role,
                json_body=body,
                tokens=tokens,
                semaphore=semaphore,
                single_attempt=single_attempt,
                handler=handle,
            )
        except (AlbertTransientError, AlbertAuthError, AlbertPermanentError):
            self.ledger.record_error(endpoint=path, model=wire)
            raise

    # ------------------------------------------------------------------ OCR
    def ocr_image(
        self,
        image: Union[bytes, bytearray, str],
        *,
        model: Any = None,
        mime: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[float] = None,
        single_attempt: bool = False,
        semaphore: Any = None,
    ) -> ChatResult:
        """OCR d'une page image par le chat (LightOnOCR) : message **image seule** (§9.1).

        Args:
            image: octets de l'image, base64 ou URI ``data:``.
            model: modèle (défaut ``ALBERT_OCR_CHAT_MODEL``).
            mime: type MIME (défaut : détecté, sinon ``image/png``).
            max_tokens: défaut ``ALBERT_OCR_MAX_TOKENS``.
            temperature: défaut ``ALBERT_OCR_TEMPERATURE``.
            top_p: défaut ``ALBERT_OCR_TOP_P``.
            timeout: défaut ``ALBERT_TIMEOUT_OCR_PAGE``.
            single_attempt: un seul envoi.
            semaphore: sémaphore(s) tenu(s) pendant l'envoi.

        Returns:
            ``ChatResult`` ; ``finish_reason == "length"`` signale une page tronquée.

        Raises:
            AlbertTruncatedError: page sans contenu avec ``finish_reason=length``.
        """
        if image is None or (isinstance(image, (bytes, bytearray, str)) and len(image) == 0):
            raise AlbertInputError("OCR Albert : image vide.", reason="bad_request")
        uri, _kind = _data_uri(image, mime, "image/png")
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}]}]
        return self.chat(
            messages,
            model if model is not None else self.cfg.ocr_chat_model,
            role="ocr_chat",
            max_tokens=max_tokens if max_tokens is not None else self.cfg.ocr_max_tokens,
            temperature=temperature if temperature is not None else self.cfg.ocr_temperature,
            top_p=top_p if top_p is not None else self.cfg.ocr_top_p,
            timeout=timeout if timeout is not None else self.cfg.timeout_ocr_page,
            single_attempt=single_attempt,
            semaphore=semaphore,
        )

    def ocr_document(
        self,
        document: Union[bytes, bytearray, str, Mapping[str, Any]],
        *,
        pages: Any = None,
        model: Any = None,
        mime: Optional[str] = None,
        include_image_base64: bool = False,
        timeout: Optional[float] = None,
        single_attempt: bool = False,
        semaphore: Any = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """``POST /v1/ocr`` (Mistral Document AI, accès restreint) : document en URI ``data:``.

        Les ``pages`` sont transmises **telles quelles** (indices à partir de 0 :
        c'est à l'appelant de les fournir ; aucune conversion ici).

        Args:
            document: octets (PDF ou image), URI ``data:``/https, ou objet
                ``document`` déjà construit.
            pages: indices de pages (liste ou chaîne « 0,2,4-6 »), omis si ``None``.
            model: défaut ``ALBERT_OCR_DOC_MODEL``.
            mime: type MIME (défaut : détecté, sinon PDF).
            include_image_base64: images extraites en base64.
            timeout: défaut ``ALBERT_TIMEOUT_OCR_DOC``.
            single_attempt: un seul envoi.
            semaphore: sémaphore(s) tenu(s) pendant l'envoi.
            extra: autres paramètres (``table_format``…).

        Returns:
            Le corps JSON (``pages[]`` avec ``index`` et ``markdown``, ``usage_info``…).

        Raises:
            AlbertPermanentError: en particulier, sans accès au modèle restreint,
                le compte sondé répond **404** ``Model mistral-ocr-2512 not
                found.`` (D3), classé ``not_found`` et non ``no_access`` : la
                bascule sur LightOnOCR se décide avec
                ``errors.is_ocr_access_denied(erreur)`` (403 et 404 reconnus) ou
                ``PreflightResult.unavailable['ocr_doc']``.
        """
        if isinstance(document, Mapping):
            doc = dict(document)
        else:
            if document is None or len(document) == 0:
                raise AlbertInputError("OCR Albert : document vide.", reason="bad_request")
            uri, kind = _data_uri(document, mime, "application/pdf")
            if kind.startswith("image/"):
                doc = {"type": "image_url", "image_url": uri}
            else:
                doc = {"type": "document_url", "document_url": uri}
        wire = self.wire_model(model if model is not None else self.cfg.ocr_doc_model)
        catalog.check_endpoint_type(wire, "ocr")
        body: Dict[str, Any] = {"model": wire, "document": doc, "include_image_base64": bool(include_image_base64)}
        if pages is not None:
            body["pages"] = pages
        for key, value in (extra or {}).items():
            if key not in _FORBIDDEN_BODY_KEYS and value is not None:
                body.setdefault(key, value)
        path = "/v1/ocr"

        def handle(data: Any, latency: float) -> Dict[str, Any]:
            """Enregistre l'usage et renvoie le corps."""
            if not isinstance(data, Mapping) or not isinstance(data.get("pages"), list):
                raise AlbertTransientError(status=200, endpoint=path, detail="réponse OCR sans pages")
            self.ledger.record(
                endpoint=path,
                model=wire,
                response_model=data.get("model"),
                usage=data,
                role="ocr_doc",
                latency_s=latency,
                request_id=data.get("id"),
                status=200,
                items=len(data["pages"]),
            )
            return dict(data)

        try:
            return self._call(
                "POST",
                path,
                timeout=float(timeout if timeout is not None else self.cfg.timeout_ocr_doc),
                role="ocr_doc",
                json_body=body,
                semaphore=semaphore,
                single_attempt=single_attempt,
                handler=handle,
            )
        except (AlbertTransientError, AlbertAuthError, AlbertPermanentError):
            self.ledger.record_error(endpoint=path, model=wire)
            raise

    # ------------------------------------------------------------------ embeddings
    def embed(
        self,
        texts: Union[str, Sequence[str]],
        *,
        model: Any = None,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
        single_attempt: bool = False,
        semaphore: Any = None,
    ) -> List[List[float]]:
        """``POST /v1/embeddings`` par tranches de 64 au plus, vecteurs dans l'ordre des textes.

        Contrat (invariant 25) : tranches de 64 au plus, tri par ``index``,
        contrôle de la dimension, aucun texte vide envoyé, jamais de
        ``dimensions``, normalisation L2 si ``ALBERT_EMBED_L2_NORMALIZE=1`` (D12).
        Aucun repli de modèle.

        Args:
            texts: textes (une chaîne seule est acceptée).
            model: défaut ``ALBERT_EMBED_MODEL``.
            batch_size: taille de tranche (plafonnée à 64 et à ``ALBERT_EMBED_BATCH``).
            timeout: défaut ``ALBERT_TIMEOUT_EMBED``.
            single_attempt: un seul envoi par tranche.
            semaphore: sémaphore(s) tenu(s) pendant chaque envoi.

        Returns:
            Un vecteur par texte, dans l'ordre d'entrée.

        Raises:
            AlbertInputError: texte vide ou non textuel (aucune requête émise).
            AlbertResponseError: réponse incohérente (nombre, index, dimension, vecteur nul).
        """
        items = [texts] if isinstance(texts, str) else list(texts)
        for position, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise AlbertInputError(
                    f"Embeddings Albert : texte vide ou invalide à la position {position} "
                    "(une chaîne vide n'est jamais envoyée).",
                    reason="bad_request",
                )
        if not items:
            return []
        wire = self.wire_model(model if model is not None else self.cfg.embed_model)
        catalog.check_endpoint_type(wire, "embeddings")
        expected_dim = catalog.embedding_dim(wire)
        size = batch_size if batch_size is not None else self.cfg.embed_batch
        size = max(1, min(int(size), int(self.cfg.embed_batch) or MAX_EMBED_BATCH, MAX_EMBED_BATCH))
        vectors: List[List[float]] = []
        for start in range(0, len(items), size):
            chunk = items[start:start + size]
            vectors.extend(
                self._embed_slice(
                    wire,
                    chunk,
                    expected_dim=expected_dim,
                    timeout=float(timeout if timeout is not None else self.cfg.timeout_embed),
                    single_attempt=single_attempt,
                    semaphore=semaphore,
                )
            )
            if expected_dim is None and vectors:
                expected_dim = len(vectors[0])
        return vectors

    def _embed_slice(
        self,
        wire: str,
        texts: List[str],
        *,
        expected_dim: Optional[int],
        timeout: float,
        single_attempt: bool,
        semaphore: Any,
    ) -> List[List[float]]:
        """Une tranche d'embeddings : envoi, tri par index, contrôles, normalisation."""
        path = "/v1/embeddings"
        body = {"model": wire, "input": list(texts), "encoding_format": "float"}

        def handle(data: Any, latency: float) -> List[List[float]]:
            """Vérifie et ordonne les vecteurs ; enregistre l'usage."""
            rows = data.get("data") if isinstance(data, Mapping) else None
            if not isinstance(rows, list):
                raise AlbertTransientError(status=200, endpoint=path, detail="réponse d'embeddings sans data")
            self.ledger.record(
                endpoint=path,
                model=wire,
                response_model=data.get("model"),
                usage=data,
                role="embed",
                latency_s=latency,
                request_id=data.get("id"),
                status=200,
                items=len(texts),
            )
            return self._check_vectors(rows, len(texts), expected_dim, path)

        try:
            return self._call(
                "POST",
                path,
                timeout=timeout,
                role="embed",
                json_body=body,
                tokens=estimate_input_tokens(texts, estimator=self.cfg.token_estimator),
                semaphore=semaphore,
                single_attempt=single_attempt,
                handler=handle,
            )
        except (AlbertTransientError, AlbertAuthError, AlbertPermanentError):
            self.ledger.record_error(endpoint=path, model=wire)
            raise

    def _check_vectors(
        self, rows: List[Any], count: int, expected_dim: Optional[int], path: str
    ) -> List[List[float]]:
        """Trie par ``index`` et valide nombre, indices, dimension et norme ; normalise en L2."""
        if len(rows) != count:
            raise AlbertResponseError(
                reason="validation", status=200, endpoint=path,
                detail=f"{len(rows)} vecteur(s) reçu(s) pour {count} texte(s)",
            )
        try:
            ordered = sorted(rows, key=lambda row: int(row["index"]))
        except (KeyError, TypeError, ValueError):
            raise AlbertResponseError(
                reason="validation", status=200, endpoint=path, detail="index d'embedding manquant"
            ) from None
        if [int(row["index"]) for row in ordered] != list(range(count)):
            raise AlbertResponseError(
                reason="validation", status=200, endpoint=path, detail="index d'embedding incohérents"
            )
        out: List[List[float]] = []
        dim = expected_dim
        for row in ordered:
            vector = row.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise AlbertResponseError(
                    reason="validation", status=200, endpoint=path, detail="vecteur absent (encoding_format=float attendu)"
                )
            values = [float(v) for v in vector]
            if dim is None:
                dim = len(values)
            if len(values) != dim:
                raise AlbertResponseError(
                    reason="validation", status=200, endpoint=path,
                    detail=f"dimension {len(values)} au lieu de {dim}",
                )
            norm = math.sqrt(sum(v * v for v in values))
            if not math.isfinite(norm) or norm == 0.0:
                raise AlbertResponseError(
                    reason="validation", status=200, endpoint=path, detail="vecteur nul ou non fini"
                )
            if self.cfg.embed_l2_normalize:
                values = [v / norm for v in values]
            out.append(values)
        return out

    # ------------------------------------------------------------------ collections
    def list_collections(
        self, *, name: Optional[str] = None, visibility: Optional[str] = None, page_size: int = PAGE_LIMIT
    ) -> List[Dict[str, Any]]:
        """``GET /v1/collections`` paginé ; ``name`` filtré exactement (serveur et client, D14)."""
        items = self._paginate("/v1/collections", {"name": name, "visibility": visibility}, limit=page_size)
        if name is not None:
            items = [c for c in items if isinstance(c, Mapping) and c.get("name") == name]
        return items

    def get_collection(self, collection_id: Any) -> Dict[str, Any]:
        """``GET /v1/collections/{id}``."""
        cid = _as_int(collection_id, "collection")
        return self._call("GET", f"/v1/collections/{cid}", timeout=self.cfg.timeout_collections)

    def create_collection(
        self, name: str, *, description: Optional[str] = None, visibility: str = "private"
    ) -> int:
        """``POST /v1/collections`` : collection **privée** (toute autre visibilité est refusée).

        Returns:
            L'id de la collection créée.

        Raises:
            AlbertInputError: nom vide ou visibilité autre que ``private``.
            AlbertUncertainWriteError: issue inconnue (délai de lecture, 5xx
                autre que 503…) : jamais renvoyée, les doublons de nom étant
                acceptés (D14) ; rapprocher par ``list_collections(name=…)``.
        """
        if not isinstance(name, str) or not name.strip():
            raise AlbertInputError("Collection Albert : nom vide.", reason="bad_request")
        if visibility is not None and str(visibility).strip().lower() != "private":
            raise AlbertInputError(
                "Collection Albert : seule la visibilité « private » est autorisée par RAGpy.",
                reason="bad_request",
            )
        body: Dict[str, Any] = {"name": name, "visibility": "private"}
        if description is not None:
            body["description"] = description
        data = self._call(
            "POST", "/v1/collections", timeout=self.cfg.timeout_collections, json_body=body, idempotent=False
        )
        return _as_int(data.get("id") if isinstance(data, Mapping) else None, "collection créée")

    def delete_collection(self, collection_id: Any) -> None:
        """``DELETE /v1/collections/{id}`` (documents et chunks compris)."""
        cid = _as_int(collection_id, "collection")
        self._call("DELETE", f"/v1/collections/{cid}", timeout=self.cfg.timeout_collections)

    # ------------------------------------------------------------------ documents et chunks
    def list_documents(
        self, collection_id: Any = None, *, name: Optional[str] = None, page_size: int = PAGE_LIMIT
    ) -> List[Dict[str, Any]]:
        """``GET /v1/documents`` paginé (filtres ``collection_id`` et ``name`` exact)."""
        cid = _as_int(collection_id, "collection") if collection_id is not None else None
        items = self._paginate("/v1/documents", {"collection_id": cid, "name": name}, limit=page_size)
        if name is not None:
            items = [d for d in items if isinstance(d, Mapping) and d.get("name") == name]
        return items

    def get_document(self, document_id: Any) -> Dict[str, Any]:
        """``GET /v1/documents/{id}``."""
        did = _as_int(document_id, "document")
        return self._call("GET", f"/v1/documents/{did}", timeout=self.cfg.timeout_collections)

    def create_document(self, collection_id: Any, name: Any, *, metadata: Optional[Mapping[str, Any]] = None) -> int:
        """``POST /v1/documents`` en multipart **sans fichier** (D15), à remplir par ``add_chunks``.

        Corps : ``files={'name': (None, nom), 'collection_id': (None, str(id))}``
        (plus ``metadata`` sérialisée en JSON si fournie).

        Returns:
            L'id du document créé.

        Raises:
            AlbertUncertainWriteError: issue inconnue : jamais renvoyée ;
                rapprocher par ``list_documents(collection_id, name=…)``.
        """
        if isinstance(collection_id, str) and not collection_id.strip().isdigit() and isinstance(name, int):
            collection_id, name = name, collection_id
        cid = _as_int(collection_id, "collection")
        if not isinstance(name, str) or not name.strip():
            raise AlbertInputError("Document Albert : nom vide.", reason="bad_request")
        files: Dict[str, Tuple[None, str]] = {"name": (None, name), "collection_id": (None, str(cid))}
        if metadata:
            files["metadata"] = (None, json.dumps(dict(metadata), ensure_ascii=False))
        data = self._call(
            "POST", "/v1/documents", timeout=self.cfg.timeout_collections, files=files, idempotent=False
        )
        return _as_int(data.get("id") if isinstance(data, Mapping) else None, "document créé")

    def delete_document(self, document_id: Any) -> None:
        """``DELETE /v1/documents/{id}``."""
        did = _as_int(document_id, "document")
        self._call("DELETE", f"/v1/documents/{did}", timeout=self.cfg.timeout_collections)

    def list_chunks(self, document_id: Any, *, page_size: int = PAGE_LIMIT) -> List[Dict[str, Any]]:
        """``GET /v1/documents/{id}/chunks``, toutes les pages (``limit`` 100)."""
        did = _as_int(document_id, "document")
        return self._paginate(f"/v1/documents/{did}/chunks", limit=page_size)

    def add_chunks(
        self,
        document_id: Any,
        chunks: Sequence[Union[str, Mapping[str, Any]]],
        *,
        single_attempt: bool = False,
        semaphore: Any = None,
    ) -> List[int]:
        """``POST /v1/documents/{id}/chunks`` : 1 à 64 chunks en un envoi (atomique, D16).

        La vectorisation côté serveur consomme le quota bge-m3 (D19) : l'envoi
        passe par le limiteur du rôle ``embed`` et chaque envoi réussi est
        inscrit au ledger (rôle ``push``, ``items`` = nombre de chunks, sans
        modèle : le client n'en envoie pas) ; un envoi en échec y est compté
        par ``record_error``. Le découpage en tranches et l'assainissement des
        métadonnées reviennent à l'appelant.

        Args:
            document_id: document cible.
            chunks: ``{"content": str, "metadata": dict | None}`` ou chaînes.
            single_attempt: un seul envoi.
            semaphore: sémaphore(s) tenu(s) pendant l'envoi.

        Returns:
            Les ids de chunks attribués (D16), ou ``[]`` si le serveur n'en renvoie pas.

        Raises:
            AlbertInputError: 0 ou plus de 64 chunks, contenu vide (aucun envoi).
            AlbertUncertainWriteError: issue inconnue (l'envoi atomique a pu
                réussir, sans clé d'idempotence, D16) : jamais renvoyé ;
                rapprocher par ``list_chunks`` (``content_id``) ou supprimer le
                document (rollback D16).
        """
        did = _as_int(document_id, "document")
        items = list(chunks or [])
        if not 1 <= len(items) <= MAX_CHUNKS_PER_POST:
            raise AlbertInputError(
                f"Chunks Albert : {len(items)} par envoi (de 1 à {MAX_CHUNKS_PER_POST}).",
                reason="bad_request",
            )
        payload: List[Dict[str, Any]] = []
        for position, item in enumerate(items):
            if isinstance(item, str):
                entry: Dict[str, Any] = {"content": item}
            elif isinstance(item, Mapping):
                entry = {"content": item.get("content")}
                if item.get("metadata") is not None:
                    entry["metadata"] = dict(item["metadata"])
            else:
                entry = {"content": None}
            if not isinstance(entry["content"], str) or not entry["content"].strip():
                raise AlbertInputError(
                    f"Chunks Albert : contenu vide à la position {position}.", reason="bad_request"
                )
            payload.append(entry)
        path = f"/v1/documents/{did}/chunks"

        def handle(data: Any, latency: float) -> Any:
            """Inscrit l'envoi au ledger (quota bge-m3 consommé côté serveur, D19) ; renvoie le corps."""
            self.ledger.record(
                endpoint=PUSH_LEDGER_ENDPOINT,
                usage=data,
                role="push",
                latency_s=latency,
                items=len(payload),
            )
            return data

        try:
            data = self._call(
                "POST",
                path,
                timeout=self.cfg.timeout_collections,
                role="push",
                json_body={"chunks": payload},
                requests=max(1, math.ceil(len(payload) / CHUNKS_PER_EMBED_REQUEST)),
                semaphore=semaphore,
                single_attempt=single_attempt,
                handler=handle,
                idempotent=False,
            )
        except (AlbertTransientError, AlbertAuthError, AlbertPermanentError):
            self.ledger.record_error(endpoint=PUSH_LEDGER_ENDPOINT)
            raise
        ids = data.get("ids") if isinstance(data, Mapping) else None
        return [int(i) for i in ids] if isinstance(ids, list) else []

    # ------------------------------------------------------------------ recherche
    def search(
        self,
        query: str,
        *,
        collection_ids: Optional[Iterable[Any]] = None,
        document_ids: Optional[Iterable[Any]] = None,
        method: str = "semantic",
        limit: int = 10,
        offset: int = 0,
        rff_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """``POST /v1/search`` ; ``method`` toujours envoyée, ``query`` obligatoire (D17).

        Args:
            query: requête (non vide).
            collection_ids: collections (100 au plus).
            document_ids: documents (100 au plus).
            method: ``semantic``, ``lexical`` ou ``hybrid``.
            limit: 1 à 100.
            offset: décalage.
            rff_k: constante RRF (nom ``rff_k``), envoyée si fournie.
            score_threshold: seuil, accepté **seulement** en ``semantic``.
            metadata_filters: filtre de comparaison ou composé (2 à 4 filtres).
            timeout: défaut ``ALBERT_TIMEOUT_COLLECTIONS``.

        Returns:
            Les résultats (``{"method", "score", "chunk"}``).

        Raises:
            AlbertInputError: requête vide, méthode inconnue, seuil hors
                ``semantic``, bornes dépassées (aucun envoi).
        """
        if not isinstance(query, str) or not query.strip():
            raise AlbertInputError("Recherche Albert : query obligatoire (D17).", reason="bad_request")
        method_name = str(method or "").strip().lower()
        if method_name not in SEARCH_METHODS:
            raise AlbertInputError(
                f"Recherche Albert : méthode {method!r} inconnue (semantic, lexical ou hybrid).",
                reason="bad_request",
            )
        if score_threshold is not None and method_name != "semantic":
            raise AlbertInputError(
                "Recherche Albert : score_threshold n'est accepté qu'avec method='semantic'.",
                reason="bad_request",
            )
        if not 1 <= int(limit) <= PAGE_LIMIT:
            raise AlbertInputError("Recherche Albert : limit doit être compris entre 1 et 100.", reason="bad_request")
        body: Dict[str, Any] = {"query": query, "method": method_name, "limit": int(limit), "offset": int(offset)}
        for field, values in (("collection_ids", collection_ids), ("document_ids", document_ids)):
            if values is not None:
                ids = [_as_int(v, field) for v in values]
                if len(ids) > PAGE_LIMIT:
                    raise AlbertInputError(f"Recherche Albert : {field} limité à 100 ids.", reason="bad_request")
                body[field] = ids
        if rff_k is not None:
            body["rff_k"] = int(rff_k)
        if score_threshold is not None:
            body["score_threshold"] = float(score_threshold)
        if metadata_filters is not None:
            body["metadata_filters"] = dict(metadata_filters)
        path = "/v1/search"

        def handle(data: Any, latency: float) -> List[Dict[str, Any]]:
            """Enregistre l'usage et renvoie les résultats."""
            rows = data.get("data") if isinstance(data, Mapping) else None
            self.ledger.record(
                endpoint=path, usage=data, role="search", latency_s=latency, status=200,
                items=len(rows) if isinstance(rows, list) else 0,
            )
            return list(rows or [])

        return self._call(
            "POST",
            path,
            timeout=float(timeout if timeout is not None else self.cfg.timeout_collections),
            json_body=body,
            handler=handle,
        )


__all__ = [
    "CHUNKS_PER_EMBED_REQUEST",
    "ENDPOINTS",
    "MAX_CHUNKS_PER_POST",
    "MAX_EMBED_BATCH",
    "PAGE_LIMIT",
    "PUSH_LEDGER_ENDPOINT",
    "SEARCH_METHODS",
    "USER_AGENT",
    "AlbertClient",
    "AlbertInputError",
    "AlbertMissingKeyError",
    "AlbertResponseError",
    "ChatResult",
]
