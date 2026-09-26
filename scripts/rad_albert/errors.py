"""Taxonomie des erreurs de l'API Albert (DINUM) — stdlib seulement.

La classification repose sur le **statut HTTP** (décision D6 : le type
d'erreur ne se lit pas dans le corps), avec deux exceptions textuelles
documentées : ``InsufficientBudget`` dans un 400 (budget épuisé) et
``account has expired`` dans un 403 (compte expiré).

Hiérarchie ::

    AlbertError
    ├── AlbertAuthError        (invalid_key | account_expired | budget_exhausted)
    │   └── AlbertQuotaExhausted   (429 persistant ou Retry-After > plafond)
    ├── AlbertPermanentError   (bad_request | no_access | not_found | conflict
    │                           | payload_too_large | validation)
    ├── AlbertTransientError   (408, 5xx, réseau, 429 court ; ``retry_after``)
    │   ├── AlbertModelBusy        (503 « Model is too busy »)
    │   └── AlbertUncertainWriteError  (création dont l'issue est inconnue :
    │                                   jamais renvoyée, ``retryable=False``)
    └── AlbertTruncatedError   (``finish_reason=length`` sans contenu)

    AlbertDisabledError(ValueError)   (``albert/…`` demandé avec Albert OFF)

Chaque ``AlbertError`` porte ``status`` (int ou None), ``endpoint`` (str ou
None) et ``detail`` (extrait du corps, 300 caractères au plus, secrets
masqués). Les erreurs de compte (``AlbertAuthError`` et sa sous-classe
``AlbertQuotaExhausted``) ne sont jamais retentées et arrêtent le job entier
(décision 22).

Accès à ``/v1/ocr`` (D3) : sur le compte sondé, l'absence d'accès au modèle
restreint ``mistral-ocr-2512`` répond **404** ``{"detail": "Model
mistral-ocr-2512 not found."}``, classé ``AlbertPermanentError(not_found)``
comme tout 404 (le 403 ``no_access`` reste possible). Pour décider de la
bascule sur LightOnOCR, utiliser ``is_ocr_access_denied(erreur)`` (qui
reconnaît les deux formes) ou ``PreflightResult.unavailable['ocr_doc']``,
jamais le seul motif ``no_access``.

``parse_retry_after`` (secondes et date HTTP, plafond optionnel) est l'unique
analyseur de ``Retry-After`` du chemin Albert ; ``retry.py`` le réexporte.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Mapping, Optional

DETAIL_MAX_CHARS = 300
"""Longueur maximale de ``AlbertError.detail`` (après masquage)."""

CREDENTIAL_KEY = "albert_api_key"
"""Identifiant utilisateur à renseigner (champ ``credential_required``)."""

AUTH_REASONS = ("invalid_key", "account_expired", "budget_exhausted")
"""Motifs acceptés par ``AlbertAuthError``."""

QUOTA_REASON = "quota_exhausted"
"""Motif unique de ``AlbertQuotaExhausted`` (sous-classe d'``AlbertAuthError``)."""

PERMANENT_REASONS = (
    "bad_request",
    "no_access",
    "not_found",
    "conflict",
    "payload_too_large",
    "validation",
)
"""Motifs acceptés par ``AlbertPermanentError``."""

MESSAGES_FR = {
    # Génériques (clé = ``message_key`` de la classe)
    "error": "Erreur de l'API Albert.",
    "transient": "Erreur temporaire de l'API Albert ; une nouvelle tentative est possible.",
    "model_busy": (
        "Modèle Albert surchargé (« Model is too busy ») : nouvelle tentative, "
        "puis bascule sur le modèle de repli du rôle."
    ),
    "write_uncertain": (
        "Création Albert à l'issue incertaine : la requête a pu être traitée par "
        "le serveur, elle n'est donc pas renvoyée. Rapprocher avant tout nouvel "
        "envoi (collection par nom exact, document par nom, chunks par content_id "
        "via list_chunks) ; en cas de doute, supprimer le document."
    ),
    "truncated": (
        "Réponse Albert tronquée (finish_reason=length) sans contenu : elle n'est "
        "jamais traitée comme une réponse vide valide."
    ),
    "disabled": (
        "Albert est désactivé sur ce serveur (ALBERT_ENABLED=0) : les modèles "
        "« albert/… » et les capacités Albert ne sont pas disponibles."
    ),
    "redirect": (
        "Redirection refusée par le client Albert : vérifier ALBERT_BASE_URL "
        "(les redirections ne sont jamais suivies)."
    ),
    # Erreurs de compte (arrêt du job entier)
    "invalid_key": (
        "Clé API Albert invalide ou révoquée. Configurez-la dans Paramètres > "
        "Mes Identifiants."
    ),
    "account_expired": (
        "Compte Albert expiré : écrire à albert.api@numerique.gouv.fr pour le "
        "renouveler."
    ),
    "budget_exhausted": (
        "Budget Albert épuisé : le traitement est arrêté (contacter les "
        "administrateurs d'Albert pour relever le budget)."
    ),
    "quota_exhausted": (
        "Quota journalier Albert atteint : le traitement est arrêté ; réessayer "
        "plus tard ou réduire le volume."
    ),
    # Erreurs permanentes (jamais retentées)
    "bad_request": "Requête refusée par l'API Albert (requête invalide).",
    "no_access": (
        "Accès refusé par l'API Albert : droit manquant sur ce modèle ou ce "
        "service (par exemple droit d'écriture des collections)."
    ),
    "not_found": (
        "Ressource introuvable sur l'API Albert : vérifier l'identifiant du "
        "modèle via /v1/models (ou celui de la collection ou du document). Un "
        "modèle d'accès restreint absent du compte (mistral-ocr-2512 sur "
        "/v1/ocr) répond aussi 404."
    ),
    "conflict": "Conflit signalé par l'API Albert (ressource déjà existante).",
    "payload_too_large": (
        "Requête trop volumineuse pour l'API Albert (lot ou document à découper)."
    ),
    "validation": (
        "Requête rejetée par la validation de l'API Albert (paramètre invalide "
        "ou mauvais type de modèle) : erreur de configuration."
    ),
}
"""Messages français, indexés par motif (``reason``) ou par ``message_key``."""


# ---------------------------------------------------------------------------
# Masquage et extraction du détail
# ---------------------------------------------------------------------------

_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_PREFIXED_TOKEN_RE = re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{6,}")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{4,}")
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password"
    r"|authorization)\b(\s*[=:]\s*)(\"?)(?!bearer\s|\*\*\*)(?:basic\s+)?[^\s\"'&,;}]+"
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_MASK = "***"


def redact(
    text: Any,
    secrets: Iterable[str] = (),
    *,
    limit: Optional[int] = DETAIL_MAX_CHARS,
    emails: bool = True,
) -> str:
    """Masque les secrets d'un texte et le tronque.

    Retire les ``secrets`` fournis (valeurs littérales de 4 caractères au
    moins), les jetons ``Bearer …``, les jetons préfixés (``sk-…``), les JWT,
    les affectations ``api_key=…`` / ``token: …`` et, si ``emails`` est vrai,
    les adresses e-mail. Les blancs sont réduits à un espace.

    Args:
        text: Texte à nettoyer (``None`` donne ``''``).
        secrets: Valeurs exactes à masquer (par exemple la clé du client).
        limit: Longueur maximale du résultat (``None`` = pas de troncature).
        emails: Masque aussi les adresses e-mail.

    Returns:
        Le texte masqué, d'au plus ``limit`` caractères (suffixe ``…`` si
        tronqué).
    """
    if text is None:
        return ""
    out = str(text)
    for secret in secrets or ():
        if secret and len(str(secret)) >= 4:
            out = out.replace(str(secret), _MASK)
    out = _BEARER_RE.sub("Bearer " + _MASK, out)
    out = _JWT_RE.sub(_MASK, out)
    out = _PREFIXED_TOKEN_RE.sub(_MASK, out)
    out = _ASSIGNMENT_RE.sub(lambda m: m.group(1) + m.group(2) + m.group(3) + _MASK, out)
    if emails:
        out = _EMAIL_RE.sub(_MASK, out)
    out = " ".join(out.split())
    if limit is not None and limit >= 1 and len(out) > limit:
        out = out[: limit - 1] + "…"
    return out


def _validation_item_text(item: Any) -> str:
    """Rend lisible un élément d'erreur de validation FastAPI (``loc`` + ``msg``).

    Le champ ``input`` (qui peut recopier le contenu envoyé, textes de chunks
    compris) n'est jamais repris.
    """
    if not isinstance(item, Mapping):
        return str(item)
    loc = item.get("loc")
    msg = item.get("msg") or item.get("message") or item.get("type") or ""
    if isinstance(loc, (list, tuple)) and loc:
        return ".".join(str(part) for part in loc) + ": " + str(msg)
    return str(msg)


def detail_from_body(body_text: Any) -> str:
    """Extrait un message lisible d'un corps d'erreur Albert.

    Formes reconnues : ``{"detail": "…"}`` ; ``{"detail": [{"loc", "msg", …}]}``
    (validation 422, sans le champ ``input``) ; ``{"error": {"message": …}}``
    et ``{"message": …}`` (erreurs amont relayées). Un corps non JSON est
    renvoyé tel quel. Aucun masquage ni troncature ici (voir ``redact``).

    Args:
        body_text: Corps brut de la réponse (str, bytes ou None).

    Returns:
        Le texte extrait, blancs réduits (``''`` si vide).
    """
    if body_text is None:
        return ""
    if isinstance(body_text, (bytes, bytearray)):
        body_text = bytes(body_text).decode("utf-8", errors="replace")
    raw = str(body_text).strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return " ".join(raw.split())
    text: str
    if isinstance(payload, Mapping):
        detail = payload.get("detail")
        error = payload.get("error")
        if isinstance(detail, str):
            text = detail
        elif isinstance(detail, (list, tuple)):
            text = "; ".join(_validation_item_text(item) for item in detail)
        elif isinstance(detail, Mapping):
            text = str(detail.get("message") or detail.get("msg") or json.dumps(detail, ensure_ascii=False))
        elif isinstance(error, Mapping):
            text = str(error.get("message") or error.get("type") or json.dumps(error, ensure_ascii=False))
        elif isinstance(error, str):
            text = error
        elif isinstance(payload.get("message"), str):
            text = payload["message"]
        else:
            text = json.dumps(payload, ensure_ascii=False)
    elif isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, ensure_ascii=False)
    return " ".join(text.split())


def _letters_only(text: str) -> str:
    """Réduit un texte à ses lettres minuscules (comparaison robuste aux espaces)."""
    return re.sub(r"[^a-z]", "", text.lower())


# ---------------------------------------------------------------------------
# Hiérarchie
# ---------------------------------------------------------------------------


class AlbertError(Exception):
    """Erreur de l'API Albert, base de toute la taxonomie.

    Attributes:
        status: Statut HTTP (``None`` pour une erreur réseau ou locale).
        endpoint: Chemin appelé (par exemple ``/v1/chat/completions``).
        detail: Extrait du corps, 300 caractères au plus, secrets masqués.
    """

    message_key = "error"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        status: Optional[int] = None,
        endpoint: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """Construit l'erreur ; le message par défaut vient de ``MESSAGES_FR``.

        Args:
            message: Message complet ; à défaut, message français de la
                classe (ou du motif), suivi du statut et du détail.
            status: Statut HTTP.
            endpoint: Chemin appelé (masqué et tronqué à 200 caractères).
            detail: Extrait du corps (masqué et tronqué à 300 caractères).
        """
        self.status: Optional[int] = int(status) if status is not None else None
        self.endpoint: Optional[str] = (redact(endpoint, limit=200) or None) if endpoint else None
        self.detail: Optional[str] = (redact(detail) or None) if detail else None
        if message is None:
            message = self._default_message()
        else:
            message = redact(message, limit=None, emails=False)
        super().__init__(message)

    def _message_base(self) -> str:
        """Renvoie le message français de base de l'instance."""
        return MESSAGES_FR.get(self.message_key, MESSAGES_FR["error"])

    def _default_message(self) -> str:
        """Compose le message par défaut : base française, statut, détail."""
        parts = [self._message_base()]
        if self.status is not None:
            where = f" {self.endpoint}" if self.endpoint else ""
            parts.append(f"[HTTP {self.status}{where}]")
        if self.detail:
            parts.append(f"Détail : {self.detail}")
        return " ".join(parts)


class AlbertAuthError(AlbertError):
    """Erreur de compte : clé invalide, compte expiré ou budget épuisé.

    Jamais retentée ; arrête le job entier et renvoie l'utilisateur vers
    l'identifiant ``albert_api_key`` (attribut ``credential_required``).

    Attributes:
        reason: ``invalid_key``, ``account_expired`` ou ``budget_exhausted``.
        credential_required: Toujours ``albert_api_key``.
    """

    _REASONS = AUTH_REASONS
    _DEFAULT_REASON = "invalid_key"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        reason: Optional[str] = None,
        status: Optional[int] = None,
        endpoint: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """Construit l'erreur de compte.

        Args:
            message: Message complet (défaut : ``MESSAGES_FR[reason]``).
            reason: Motif, parmi ``_REASONS`` de la classe.
            status: Statut HTTP.
            endpoint: Chemin appelé.
            detail: Extrait du corps.

        Raises:
            ValueError: Motif inconnu pour cette classe.
        """
        reason = self._DEFAULT_REASON if reason is None else reason
        if reason not in self._REASONS:
            raise ValueError(
                f"Motif {reason!r} invalide pour {type(self).__name__} "
                f"(attendu : {', '.join(self._REASONS)})."
            )
        self.reason: str = reason
        self.credential_required: str = CREDENTIAL_KEY
        super().__init__(message, status=status, endpoint=endpoint, detail=detail)

    def _message_base(self) -> str:
        """Renvoie le message français associé au motif."""
        return MESSAGES_FR[self.reason]


class AlbertQuotaExhausted(AlbertAuthError):
    """Quota épuisé : 429 persistant ou ``Retry-After`` au-delà du plafond.

    Le backoff ne résout pas un plafond journalier : abandon et arrêt du job.

    Attributes:
        reason: Toujours ``quota_exhausted``.
        retry_after: Délai annoncé par le serveur, en secondes (ou None).
    """

    _REASONS = (QUOTA_REASON,)
    _DEFAULT_REASON = QUOTA_REASON

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        reason: Optional[str] = None,
        status: Optional[int] = None,
        endpoint: Optional[str] = None,
        detail: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        """Construit l'erreur de quota.

        Args:
            message: Message complet (défaut : ``MESSAGES_FR['quota_exhausted']``).
            reason: Ignoré sauf ``quota_exhausted`` (seul motif accepté).
            status: Statut HTTP (429 en pratique).
            endpoint: Chemin appelé.
            detail: Extrait du corps.
            retry_after: Délai annoncé (secondes), conservé pour le journal.
        """
        self.retry_after: Optional[float] = _clean_retry_after(retry_after)
        super().__init__(
            message, reason=reason, status=status, endpoint=endpoint, detail=detail
        )


class AlbertPermanentError(AlbertError):
    """Erreur permanente (configuration ou requête) : jamais retentée.

    Attributes:
        reason: ``bad_request``, ``no_access``, ``not_found``, ``conflict``,
            ``payload_too_large`` ou ``validation``.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        reason: str = "bad_request",
        status: Optional[int] = None,
        endpoint: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """Construit l'erreur permanente.

        Args:
            message: Message complet (défaut : ``MESSAGES_FR[reason]``).
            reason: Motif, parmi ``PERMANENT_REASONS``.
            status: Statut HTTP.
            endpoint: Chemin appelé.
            detail: Extrait du corps.

        Raises:
            ValueError: Motif inconnu.
        """
        if reason not in PERMANENT_REASONS:
            raise ValueError(
                f"Motif {reason!r} invalide pour AlbertPermanentError "
                f"(attendu : {', '.join(PERMANENT_REASONS)})."
            )
        self.reason: str = reason
        super().__init__(message, status=status, endpoint=endpoint, detail=detail)

    def _message_base(self) -> str:
        """Renvoie le message français associé au motif."""
        return MESSAGES_FR[self.reason]


class AlbertTransientError(AlbertError):
    """Erreur transitoire (408, 5xx, réseau, 429 court) : réessai possible.

    Attributes:
        retry_after: Délai imposé par le serveur (secondes), ou None ; quand
            il est présent, il est respecté sans gigue.
        retryable: Vrai si la couche de réessai peut renvoyer la requête
            (faux pour ``AlbertUncertainWriteError``).
    """

    message_key = "transient"
    retryable = True

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        status: Optional[int] = None,
        endpoint: Optional[str] = None,
        detail: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        """Construit l'erreur transitoire.

        Args:
            message: Message complet (défaut : message français de la classe).
            status: Statut HTTP (None pour une erreur réseau).
            endpoint: Chemin appelé.
            detail: Extrait du corps ou description de l'erreur réseau.
            retry_after: Délai annoncé (secondes) ; négatif ramené à 0.
        """
        self.retry_after: Optional[float] = _clean_retry_after(retry_after)
        super().__init__(message, status=status, endpoint=endpoint, detail=detail)


class AlbertModelBusy(AlbertTransientError):
    """503 « Model is too busy » : essais courts, puis repli de rôle."""

    message_key = "model_busy"


class AlbertUncertainWriteError(AlbertTransientError):
    """Échec d'une création non idempotente dont l'issue est inconnue : jamais renvoyée.

    Levée pour ``POST /v1/collections``, ``POST /v1/documents`` et
    ``POST /v1/documents/{id}/chunks`` quand la requête a pu atteindre le
    traitement (délai de lecture ou d'écriture, protocole rompu, 5xx autre
    que 503, réponse 2xx illisible). Les doublons sont acceptés par l'API
    (D14) et un envoi de chunks n'a pas de clé d'idempotence (D16) : l'appelant
    rapproche (nom exact, ``list_chunks`` par ``content_id``) ou supprime le
    document avant tout nouvel envoi. ``retryable`` est faux : la couche de
    réessai la relève au premier essai.
    """

    message_key = "write_uncertain"
    retryable = False


class AlbertTruncatedError(AlbertError):
    """``finish_reason=length`` avec un contenu vide : jamais une réponse valide."""

    message_key = "truncated"


class AlbertDisabledError(ValueError):
    """Albert demandé (``albert/…``, ``embedding_provider=albert``…) alors qu'il est OFF.

    Sous-classe de ``ValueError`` (et non d'``AlbertError``) : les routes la
    traduisent en 400 ou en événement SSE d'erreur, sans sous-processus.
    """

    def __init__(self, message: Optional[str] = None, *, model: Optional[str] = None) -> None:
        """Construit l'erreur.

        Args:
            message: Message complet (défaut : ``MESSAGES_FR['disabled']``).
            model: Modèle ou fournisseur demandé, cité dans le message par défaut.
        """
        self.model: Optional[str] = model
        if message is None:
            message = MESSAGES_FR["disabled"]
            if model:
                message = f"{message} (demande : {redact(model, limit=120)})"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _clean_retry_after(value: Any) -> Optional[float]:
    """Normalise un délai ``Retry-After`` : None si absent ou non fini, sinon >= 0."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _header(headers: Optional[Mapping[str, Any]], name: str) -> Optional[str]:
    """Lit un en-tête sans tenir compte de la casse (dict ou ``httpx.Headers``)."""
    if not headers:
        return None
    try:
        value = headers.get(name)
        if value is None:
            value = headers.get(name.title())
    except Exception:
        value = None
    if value is None:
        try:
            for key, item in headers.items():
                if str(key).lower() == name.lower():
                    value = item
                    break
        except Exception:
            return None
    return None if value is None else str(value)


def _now_utc(now: Any) -> datetime:
    """Instant de référence en UTC (``now`` : datetime, timestamp ou ``None``)."""
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, datetime):
        return now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(float(now), tz=timezone.utc)


def parse_retry_after(
    value: Any,
    cap: Optional[float] = None,
    *,
    now: Any = None,
    max_seconds: Optional[float] = None,
) -> Optional[float]:
    """Analyse un en-tête ``Retry-After`` en secondes (portage de ``rad_dataframe``).

    Unique analyseur du chemin Albert (``retry.parse_retry_after`` est le même
    objet). Sans plafond, le résultat est identique à ``_parse_retry_after`` :
    délai en secondes renvoyé tel quel (négatif → ``None``) ; date HTTP
    convertie en secondes à partir de maintenant (date passée → 0) ; valeur
    absente ou illisible → ``None``.

    Args:
        value: valeur brute de l'en-tête (ou ``None``).
        cap: plafond optionnel (secondes) appliqué au résultat.
        now: instant de référence pour une date HTTP (datetime ou timestamp ;
            défaut : maintenant).
        max_seconds: synonyme de ``cap``.

    Returns:
        Secondes (>= 0), ou ``None``.
    """
    if not value:
        return None
    text = str(value).strip()
    result: Optional[float]
    try:
        seconds = float(text)
        result = seconds if seconds >= 0 else None
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if target is None:
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        result = max(0.0, (target - _now_utc(now)).total_seconds())
    limit = cap if cap is not None else max_seconds
    if result is not None and limit is not None:
        result = min(result, float(limit))
    return result


def classify_http_error(
    status: int,
    body_text: str,
    headers: Mapping,
    *,
    endpoint: Optional[str] = None,
    retry_after: Optional[float] = None,
    retry_after_max: float = 120.0,
) -> AlbertError:
    """Classe une réponse HTTP d'échec selon son statut (taxonomie du sprint).

    ====================================  ==================================
    Statut / condition                    Erreur renvoyée
    ====================================  ==================================
    400 + ``InsufficientBudget``          ``AlbertAuthError(budget_exhausted)``
    400 autre                             ``AlbertPermanentError(bad_request)``
    401                                   ``AlbertAuthError(invalid_key)``
    403 + ``account has expired``         ``AlbertAuthError(account_expired)``
    403 autre                             ``AlbertPermanentError(no_access)``
    404                                   ``AlbertPermanentError(not_found)``
    409 / 413 / 422                       ``AlbertPermanentError`` (conflict /
                                          payload_too_large / validation)
    429, Retry-After <= plafond ou absent ``AlbertTransientError(retry_after)``
    429, Retry-After > plafond            ``AlbertQuotaExhausted``
    503                                   ``AlbertModelBusy``
    408 et autres 5xx                     ``AlbertTransientError``
    3xx (redirection jamais suivie)       ``AlbertPermanentError(bad_request)``
    autre statut                          ``AlbertPermanentError(bad_request)``
    ====================================  ==================================

    Args:
        status: Statut HTTP de la réponse.
        body_text: Corps brut (le détail extrait est masqué et tronqué ; le
            client masque en plus sa clé avant l'appel).
        headers: En-têtes de la réponse (``Retry-After`` lu par
            ``parse_retry_after`` si ``retry_after`` n'est pas fourni).
        endpoint: Chemin appelé, recopié dans l'erreur.
        retry_after: Délai déjà analysé par l'appelant (secondes, **non
            plafonné**) ; prioritaire sur l'en-tête.
        retry_after_max: Plafond (``ALBERT_RETRY_AFTER_MAX``) au-delà duquel un
            429 signifie un quota épuisé.

    Returns:
        L'instance d'``AlbertError`` correspondante (jamais levée ici).
    """
    code = int(status)
    full_detail = detail_from_body(body_text)
    letters = _letters_only(full_detail)
    common = {"status": code, "endpoint": endpoint, "detail": full_detail}

    if 300 <= code < 400:
        return AlbertPermanentError(
            f"{MESSAGES_FR['redirect']} [HTTP {code}]", reason="bad_request", **common
        )
    if code == 400:
        if "insufficientbudget" in letters:
            return AlbertAuthError(reason="budget_exhausted", **common)
        return AlbertPermanentError(reason="bad_request", **common)
    if code == 401:
        return AlbertAuthError(reason="invalid_key", **common)
    if code == 403:
        if "accounthasexpired" in letters:
            return AlbertAuthError(reason="account_expired", **common)
        return AlbertPermanentError(reason="no_access", **common)
    if code == 404:
        return AlbertPermanentError(reason="not_found", **common)
    if code == 409:
        return AlbertPermanentError(reason="conflict", **common)
    if code == 413:
        return AlbertPermanentError(reason="payload_too_large", **common)
    if code == 422:
        return AlbertPermanentError(reason="validation", **common)

    announced = _clean_retry_after(retry_after)
    if announced is None:
        announced = _clean_retry_after(parse_retry_after(_header(headers, "retry-after")))

    if code == 429:
        if announced is not None and announced > float(retry_after_max):
            return AlbertQuotaExhausted(retry_after=announced, **common)
        return AlbertTransientError(retry_after=announced, **common)
    if code == 503:
        return AlbertModelBusy(retry_after=announced, **common)
    if code == 408 or 500 <= code < 600:
        return AlbertTransientError(retry_after=announced, **common)
    return AlbertPermanentError(reason="bad_request", **common)


def is_ocr_access_denied(error: BaseException) -> bool:
    """Vrai si ``error`` signale l'absence d'accès au modèle de ``/v1/ocr`` (D3).

    Deux formes sont reconnues sur le point d'accès ``/v1/ocr`` :

    * ``AlbertPermanentError(no_access)`` (403, droit manquant) ;
    * ``AlbertPermanentError(not_found)`` en 404 dont le détail signale un
      modèle introuvable (``Model mistral-ocr-2512 not found.``), forme
      mesurée sur le compte sondé (D3) ; un id mal saisi donne la même
      réponse, et ``/v1/ocr`` reste inutilisable dans les deux cas.

    Un 404 générique (``Not Found``, route absente) n'est pas un refus d'accès.

    Args:
        error: exception levée par le client (toute autre classe donne ``False``).

    Returns:
        ``True`` si la bascule sur LightOnOCR doit être mémorisée.
    """
    if not isinstance(error, AlbertPermanentError):
        return False
    endpoint = (error.endpoint or "").split("?", 1)[0].rstrip("/")
    if not endpoint.endswith("/ocr"):
        return False
    if error.reason == "no_access":
        return True
    if error.reason != "not_found" or error.status != 404:
        return False
    letters = _letters_only(error.detail or "")
    return "model" in letters and "notfound" in letters
