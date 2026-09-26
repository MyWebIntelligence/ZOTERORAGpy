"""Doubles de test hors ligne de l'API Albert (DINUM) et de Redis.

* ``FAKE_ALBERT_KEY`` : l'unique clé Albert factice de la suite.
* ``FakeAlbert`` : API Albert simulée derrière un ``httpx.MockTransport`` (aucun
  réseau ; la garde réseau du ``conftest.py`` racine ne voit pas ce transport).
  Elle couvre tous les points d'accès de ``scripts/rad_albert/client.py``
  (``ROUTES``), avec des réponses calquées sur les fixtures réelles
  ``tests/fixtures/albert/P*.json`` : catalogue ``/v1/models``, ``/v1/me``,
  chat (``usage.cost`` / ``impacts``, champ ``reasoning`` de gpt-oss, piège de
  troncature ``finish_reason=length`` avec contenu ``null``), OCR par image,
  embeddings de 1024 dimensions normalisés et triés par index (413 au-delà de
  64 textes, 400 sur chaîne vide), ``/v1/ocr`` en 404 par défaut
  (``mistral-ocr-2512`` sans accès, D3), collections / documents (multipart) /
  chunks en mémoire avec pagination, recherche avec ``metadata_filters``,
  ``/health`` à la racine. Des statuts, en-têtes ``Retry-After``, exceptions
  réseau ou réponses arbitraires s'injectent par route (``inject``). Chaque
  requête est journalisée dans ``calls``.
* ``FakeRedis`` : sous-ensemble de redis-py (``get``, ``set``, ``incr``,
  ``expire``, ``ttl``…) avec expirations et horloge injectable.

Fichier gelé après la vague W1 : toute évolution passe par le gardien.
"""

from __future__ import annotations

import base64
import copy
import fnmatch
import functools
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qsl

import httpx

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FAKE_ALBERT_KEY = "fake-albert-key-0001"
FAKE_ROOT_URL = "https://albert.api.etalab.gouv.fr"
FAKE_BASE_URL = FAKE_ROOT_URL + "/v1"
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "albert")

EMBED_DIM = 1024
MAX_EMBED_BATCH = 64
MAX_CHUNKS_PER_POST = 64
MAX_METADATA_PROPS = 10
MAX_METADATA_STR = 255
MAX_METADATA_NUMBER = 1e16
DEFAULT_PAGE_LIMIT = 10
MAX_PAGE_LIMIT = 100
MAX_COMPOUND_FILTERS = 4
MIN_COMPOUND_FILTERS = 2
DEFAULT_RFF_K = 60
FIRST_COLLECTION_ID = 302360
FIRST_DOCUMENT_ID = 4934607
FIXED_CREATED = 1790442000

OCR_DOC_MODEL = "mistral-ocr-2512"
BUSY_DETAIL = "Model is too busy, please try again later"
CHAT_MODEL_TYPES = ("text-generation", "image-text-to-text")
EMBED_MODEL_TYPE = "text-embeddings-inference"
OCR_MODEL_TYPES = ("image-to-text", "image-text-to-text")
SEARCH_METHODS = ("hybrid", "semantic", "lexical")
FILTER_TYPES = ("eq", "sw", "ew", "co")

# Points d'accès servis (méthode, gabarit de chemin). ``/health*`` est à la racine.
ROUTES: Tuple[Tuple[str, str], ...] = (
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
    ("PATCH", "/v1/collections/{collection_id}"),
    ("DELETE", "/v1/collections/{collection_id}"),
    ("GET", "/v1/documents"),
    ("POST", "/v1/documents"),
    ("GET", "/v1/documents/{document_id}"),
    ("DELETE", "/v1/documents/{document_id}"),
    ("GET", "/v1/documents/{document_id}/chunks"),
    ("POST", "/v1/documents/{document_id}/chunks"),
    ("GET", "/v1/documents/{document_id}/chunks/{chunk_id}"),
    ("DELETE", "/v1/documents/{document_id}/chunks/{chunk_id}"),
    ("POST", "/v1/search"),
)
PUBLIC_ROUTES = frozenset({("GET", "/health")})
# Routes à état (collections, documents, chunks, recherche) : traitées sous verrou.
# Les autres (chat, embeddings, OCR…) appellent les réponses configurées hors
# verrou, pour que des requêtes concurrentes restent réellement concurrentes.
_STATEFUL_PREFIXES = ("/v1/collections", "/v1/documents", "/v1/search")
_NO_REPLY = object()

# Corps d'erreur nommés (statut, corps), textes repris des sondes P6/P12/P14 et
# de la référence API (§3.4) ; ``budget_exhausted`` contient ``InsufficientBudget``.
NAMED_ERRORS: Dict[str, Tuple[int, Any]] = {
    "bad_request": (400, {"detail": "Bad request."}),
    "budget_exhausted": (400, {"detail": "InsufficientBudgetException: Insufficient budget."}),
    "invalid_key": (401, {"detail": "Invalid API key."}),
    "account_expired": (403, {"detail": "Your account has expired. Please contact support to renew your account."}),
    "no_access": (403, {"detail": "Insufficient rights."}),
    "not_found": (404, {"detail": "Model not found."}),
    "request_timeout": (408, {"detail": "Request Timeout"}),
    "conflict": (409, {"detail": "Conflict."}),
    "payload_too_large": (413, {"detail": "batch size 65 > maximum allowed batch size 64"}),
    "wrong_model_type": (422, {"detail": "Model has wrong type. Expected: text-embeddings-inference. Actual: text-generation."}),
    "rate_limited": (429, {"detail": "Too many requests."}),
    "server_error": (500, {"detail": "Internal Server Error"}),
    "bad_gateway": (502, {"detail": "Bad Gateway"}),
    "model_busy": (503, {"detail": BUSY_DETAIL}),
    "gateway_timeout": (504, {"detail": "Gateway Timeout"}),
}
_ERROR_NAME_BY_STATUS = {
    400: "bad_request", 401: "invalid_key", 403: "no_access", 404: "not_found", 408: "request_timeout",
    409: "conflict", 413: "payload_too_large", 422: "wrong_model_type", 429: "rate_limited",
    500: "server_error", 502: "bad_gateway", 503: "model_busy", 504: "gateway_timeout",
}

# En-têtes de limitation observés (P12/P13) sur les réponses 200 de chat et d'embeddings.
RATELIMIT_HEADERS = {
    "x-ratelimit-limit-request": "500",
    "x-ratelimit-remaining-request": "499",
    "x-ratelimit-reset-requests": "0s",
    "x-ratelimit-limit-token": "None",
    "x-ratelimit-remaining-token": "0",
    "x-ratelimit-reset-token": "0s",
}

_IMPACTS = {"kWh": 5.095698496904152e-07, "kgCO2eq": 2.6710792561089838e-08}


# ---------------------------------------------------------------------------
# Fixtures JSON et vecteurs déterministes
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _read_fixture(fixtures_dir: str, name: str) -> str:
    """Texte brut d'une fixture (mis en cache)."""
    with open(os.path.join(fixtures_dir, name), encoding="utf-8") as fh:
        return fh.read()


def load_fixture(name: str, fixtures_dir: Optional[str] = None) -> Dict[str, Any]:
    """Fixture ``tests/fixtures/albert/<name>`` décodée (copie indépendante)."""
    return json.loads(_read_fixture(fixtures_dir or FIXTURES_DIR, name))


@functools.lru_cache(maxsize=8192)
def _embedding_tuple(text: str, dim: int) -> Tuple[float, ...]:
    """Vecteur déterministe de norme L2 = 1 dérivé du sha256 du texte."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return tuple(v / norm for v in values)


def fake_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    """Vecteur d'embedding que ``FakeAlbert`` renvoie pour ``text`` (norme 1, déterministe)."""
    return list(_embedding_tuple(text, dim))


def _json_response(status: int, body: Any = None, headers: Optional[Mapping[str, str]] = None) -> httpx.Response:
    """Réponse JSON (ou vide pour ``body is None``) avec en-têtes optionnels."""
    hdrs = dict(headers or {})
    if body is None:
        return httpx.Response(status, headers=hdrs)
    if isinstance(body, bytes):
        return httpx.Response(status, content=body, headers=hdrs)
    if isinstance(body, str):
        hdrs.setdefault("content-type", "text/plain; charset=utf-8")
        return httpx.Response(status, text=body, headers=hdrs)
    return httpx.Response(status, json=body, headers=hdrs)


def _validation_error(loc: Sequence[Any], msg: str, input_value: Any = None, err_type: str = "value_error",
                      ctx: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Entrée de détail 422 au format FastAPI/pydantic."""
    item: Dict[str, Any] = {"type": err_type, "loc": list(loc), "msg": msg, "input": input_value}
    if ctx:
        item["ctx"] = dict(ctx)
    return item


def _unprocessable(errors: List[Dict[str, Any]]) -> httpx.Response:
    """Réponse 422 ``{"detail": [...]}``."""
    return _json_response(422, {"detail": errors})


def _retry_after_value(value: Any) -> str:
    """Valeur d'en-tête ``Retry-After`` (secondes ou date HTTP transmise telle quelle)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Multipart minimal (POST /v1/documents)
# ---------------------------------------------------------------------------
_DISPOSITION_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_multipart(content_type: str, body: bytes) -> Tuple[Dict[str, str], Dict[str, Tuple[str, bytes]]]:
    """Décode un corps ``multipart/form-data`` en (champs texte, fichiers ``(nom, octets)``)."""
    match = re.search(r'boundary="?([^";]+)"?', content_type or "")
    fields: Dict[str, str] = {}
    files: Dict[str, Tuple[str, bytes]] = {}
    if not match:
        return fields, files
    delimiter = b"--" + match.group(1).encode("latin-1")
    for part in body.split(delimiter):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        head, _, payload = part.partition(b"\r\n\r\n")
        headers = head.decode("utf-8", "replace")
        disposition = next((line for line in headers.split("\r\n") if line.lower().startswith("content-disposition")), "")
        params = dict(_DISPOSITION_RE.findall(disposition))
        name = params.get("name")
        if name is None:
            continue
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        if "filename" in params:
            files[name] = (params["filename"], payload)
        else:
            fields[name] = payload.decode("utf-8", "replace")
    return fields, files


# ---------------------------------------------------------------------------
# Journal des appels
# ---------------------------------------------------------------------------
@dataclass
class FakeCall:
    """Requête reçue par ``FakeAlbert`` (journal ``calls``).

    Attributes:
        method: verbe HTTP.
        path: chemin normalisé (``/v1/...`` ou ``/health...``).
        url: URL complète reçue.
        params: paramètres de requête (dernière valeur par nom).
        headers: en-têtes (noms en minuscules).
        content: corps brut.
        json: corps JSON décodé, sinon ``None``.
        form: champs de formulaire (multipart ou urlencoded).
        files: fichiers multipart ``{champ: (nom, octets)}``.
        status: statut renvoyé (``None`` si une exception a été injectée).
        injected: ``True`` si la réponse venait d'une injection.
    """

    method: str
    path: str
    url: str
    params: Dict[str, str]
    headers: Dict[str, str]
    content: bytes
    json: Any = None
    form: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, Tuple[str, bytes]] = field(default_factory=dict)
    status: Optional[int] = None
    injected: bool = False


InjectionEntry = Union[int, str, tuple, Mapping[str, Any], httpx.Response, BaseException, type, Callable[..., Any]]


def _compile_route(template: str) -> "re.Pattern[str]":
    """Expression régulière d'un gabarit ``/v1/documents/{document_id}``."""
    pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", template)
    return re.compile("^" + pattern + "/?$")


_COMPILED_ROUTES = [(method, template, _compile_route(template)) for method, template in ROUTES]


# ---------------------------------------------------------------------------
# API Albert simulée
# ---------------------------------------------------------------------------
class FakeAlbert:
    """API Albert simulée, servie par ``httpx.MockTransport`` (sync et async).

    Usage typique::

        fake = FakeAlbert()
        client = AlbertClient(cfg, FAKE_ALBERT_KEY, transport=fake.transport)
        fake.inject("POST", "/v1/chat/completions", 429, retry_after=2)
        ...
        assert [c.path for c in fake.calls] == [...]

    Réglages publics (modifiables après construction) : ``chat_reply``,
    ``ocr_chat_reply``, ``ocr_doc_reply``, ``ocr_access``, ``reasoning_models``,
    ``reasoning_tokens``, ``embed_shuffle``, ``me``, ``models``, ``health_models``.
    """

    def __init__(
        self,
        fixtures_dir: Optional[str] = None,
        *,
        api_key: Optional[str] = FAKE_ALBERT_KEY,
        ocr_access: bool = False,
        models: Optional[Sequence[Mapping[str, Any]]] = None,
        me_overrides: Optional[Mapping[str, Any]] = None,
        embed_dim: int = EMBED_DIM,
        reasoning_models: Iterable[str] = ("gpt-oss-120b",),
        reasoning_tokens: int = 256,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        """Construit l'API simulée.

        Args:
            fixtures_dir: dossier des fixtures ``P*.json`` (défaut ``tests/fixtures/albert``).
            api_key: clé Bearer attendue (``None`` = toute requête acceptée).
            ocr_access: ``True`` pour servir ``/v1/ocr`` (``mistral-ocr-2512``) ;
                par défaut 404 comme le compte réel (D3).
            models: catalogue ``/v1/models`` (défaut : fixture P2).
            me_overrides: champs remplacés dans la réponse ``/v1/me`` (ex. ``expires``).
            embed_dim: dimension des embeddings de ``bge-m3``.
            reasoning_models: ids des modèles à raisonnement (champ ``reasoning``).
            reasoning_tokens: tokens de raisonnement simulés ; un ``max_tokens``
                inférieur produit le piège ``content=None`` / ``finish_reason=length`` (P9).
            clock: horloge des champs ``created`` (défaut : valeur fixe).
        """
        self.fixtures_dir = fixtures_dir or FIXTURES_DIR
        self.api_key = api_key
        self.ocr_access = ocr_access
        self.embed_dim = embed_dim
        self.reasoning_models = set(reasoning_models)
        self.reasoning_tokens = reasoning_tokens
        self._clock = clock or (lambda: float(FIXED_CREATED))
        self._lock = threading.RLock()

        self.calls: List[FakeCall] = []
        self.injections: Dict[Tuple[str, str], Deque[Tuple[InjectionEntry, Any]]] = {}

        # Réponses configurables.
        self.chat_reply: Union[None, str, Mapping[str, Any], Callable[[Dict[str, Any]], Any]] = None
        self.chat_replies: Deque[Any] = deque()
        self.ocr_chat_reply: Union[None, str, Callable[[str, int], str]] = None
        self.ocr_doc_reply: Union[None, str, Callable[[int], str]] = None
        self.embed_shuffle = False

        self.models: List[Dict[str, Any]] = (
            [dict(m) for m in models] if models is not None else load_fixture("P2_models.json", self.fixtures_dir)["body"]["data"]
        )
        me = load_fixture("P1_me.json", self.fixtures_dir)["body"]
        me.setdefault("expires", None)
        me.update(me_overrides or {})
        self.me: Dict[str, Any] = me
        self.health_models: Dict[str, Any] = load_fixture("P20_health_models.json", self.fixtures_dir)["body"]
        self._fixture_chat = load_fixture("P7_chat_usage.json", self.fixtures_dir)["body"]
        self._fixture_reasoning = load_fixture("P9_reasoning_low_2048.json", self.fixtures_dir)["body"]
        self._fixture_trap = load_fixture("P9_reasoning_trap_64.json", self.fixtures_dir)["body"]
        self._fixture_ocr_chat = load_fixture("P5_ocr_chat_id.json", self.fixtures_dir)["body"]
        self._fixture_ocr_doc = load_fixture("P3_ocr_page0_synthetic.json", self.fixtures_dir)["body"]

        # Stockage en mémoire.
        self.collections: Dict[int, Dict[str, Any]] = {}
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.chunks: Dict[int, List[Dict[str, Any]]] = {}
        self._next_collection_id = FIRST_COLLECTION_ID
        self._next_document_id = FIRST_DOCUMENT_ID
        self._next_chunk_id: Dict[int, int] = {}
        self._counter = 0
        self._ocr_chat_index = 0

        self.transport = httpx.MockTransport(self.handler)

    # ------------------------------------------------------------------ clients
    def client(self, base_url: str = FAKE_BASE_URL, *, api_key: Optional[str] = FAKE_ALBERT_KEY,
               **kwargs: Any) -> httpx.Client:
        """``httpx.Client`` branché sur ce faux (en-tête ``Authorization`` si ``api_key``)."""
        headers = dict(kwargs.pop("headers", {}) or {})
        if api_key is not None:
            headers.setdefault("Authorization", f"Bearer {api_key}")
        return httpx.Client(transport=self.transport, base_url=base_url, headers=headers, **kwargs)

    def async_client(self, base_url: str = FAKE_BASE_URL, *, api_key: Optional[str] = FAKE_ALBERT_KEY,
                     **kwargs: Any) -> httpx.AsyncClient:
        """``httpx.AsyncClient`` branché sur ce faux (en-tête ``Authorization`` si ``api_key``)."""
        headers = dict(kwargs.pop("headers", {}) or {})
        if api_key is not None:
            headers.setdefault("Authorization", f"Bearer {api_key}")
        return httpx.AsyncClient(transport=self.transport, base_url=base_url, headers=headers, **kwargs)

    # ---------------------------------------------------------------- injection
    def inject(self, method: str, path: str, *entries: InjectionEntry, retry_after: Any = None) -> "FakeAlbert":
        """Programme des réponses pour les prochains appels d'une route, dans l'ordre.

        Chaque entrée est consommée par un appel correspondant, avant le
        traitement normal (et avant le contrôle de la clé). Formes admises :

        * ``int`` : statut avec le corps d'erreur par défaut (``NAMED_ERRORS``) ;
        * ``str`` : nom de ``NAMED_ERRORS`` (ex. ``"model_busy"``, ``"account_expired"``) ;
        * ``(statut, corps)`` ou ``(statut, corps, en-têtes)`` ;
        * ``{"status": …, "body": …, "headers": …, "retry_after": …}`` ;
        * ``httpx.Response`` ; une exception (instance, ou classe ``httpx.RequestError``
          instanciée avec la requête) ; un appelable ``f(request)`` renvoyant l'une
          de ces formes.

        Args:
            method: verbe HTTP (``"*"`` = tous).
            path: chemin exact ou motif ``fnmatch`` (ex. ``"/v1/documents/*/chunks"``).
            entries: réponses successives.
            retry_after: en-tête ``Retry-After`` ajouté à chaque entrée (secondes ou date HTTP).

        Returns:
            ``self`` (chaînable).
        """
        key = (method.upper(), path)
        with self._lock:
            queue = self.injections.setdefault(key, deque())
            for entry in entries:
                queue.append((entry, retry_after))
        return self

    def clear_injections(self) -> None:
        """Supprime toutes les injections restantes."""
        with self._lock:
            self.injections.clear()

    def pending_injections(self) -> int:
        """Nombre d'injections non encore consommées."""
        with self._lock:
            return sum(len(q) for q in self.injections.values())

    def queue_chat_replies(self, *replies: Any) -> "FakeAlbert":
        """Réponses de chat consommées dans l'ordre (``str`` ou ``dict``), avant ``chat_reply``."""
        with self._lock:
            self.chat_replies.extend(replies)
        return self

    # ------------------------------------------------------------------ journal
    def calls_to(self, method: Optional[str] = None, path: Optional[str] = None) -> List[FakeCall]:
        """Appels filtrés par verbe et chemin (motif ``fnmatch`` accepté)."""
        with self._lock:
            calls = list(self.calls)
        return [
            c for c in calls
            if (method is None or c.method == method.upper())
            and (path is None or c.path == path or fnmatch.fnmatchcase(c.path, path))
        ]

    def reset_calls(self) -> None:
        """Vide le journal des appels."""
        with self._lock:
            self.calls.clear()

    # ------------------------------------------------------------------ handler
    @staticmethod
    def normalise_path(raw_path: str) -> str:
        """Chemin relatif à la racine de l'API (préfixe éventuel avant ``/v1/`` retiré)."""
        path = raw_path or "/"
        if not path.startswith("/v1/") and "/v1/" in path:
            path = path[path.index("/v1/"):]
        return path

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Point d'entrée du ``MockTransport`` : journal, injection, puis routage."""
        call = self._record(request)
        with self._lock:
            injected = self._pop_injection(call.method, call.path)
        if injected is not None:
            call.injected = True
            entry, retry_after = injected
            response = self._materialise(entry, request, retry_after)
        else:
            response = self._dispatch(request, call)
        call.status = response.status_code
        return response

    def _record(self, request: httpx.Request) -> FakeCall:
        """Ajoute la requête au journal et décode son corps."""
        try:
            content = request.content
        except httpx.RequestNotRead:
            content = request.read()
        headers = {k.lower(): v for k, v in request.headers.items()}
        ctype = headers.get("content-type", "")
        parsed_json: Any = None
        form: Dict[str, str] = {}
        files: Dict[str, Tuple[str, bytes]] = {}
        if content:
            if "multipart/form-data" in ctype:
                form, files = parse_multipart(ctype, content)
            elif "application/x-www-form-urlencoded" in ctype:
                form = dict(parse_qsl(content.decode("utf-8", "replace"), keep_blank_values=True))
            else:
                try:
                    parsed_json = json.loads(content)
                except ValueError:
                    parsed_json = None
        call = FakeCall(
            method=request.method.upper(),
            path=self.normalise_path(request.url.path),
            url=str(request.url),
            params=dict(request.url.params),
            headers=headers,
            content=content,
            json=parsed_json,
            form=form,
            files=files,
        )
        with self._lock:
            self.calls.append(call)
        return call

    def _pop_injection(self, method: str, path: str) -> Optional[Tuple[InjectionEntry, Any]]:
        """Première injection correspondant à la requête (ordre d'enregistrement des routes)."""
        for (m, pattern), queue in self.injections.items():
            if not queue or (m != "*" and m != method):
                continue
            if pattern == path or fnmatch.fnmatchcase(path, pattern):
                return queue.popleft()
        return None

    def _materialise(self, entry: InjectionEntry, request: httpx.Request, retry_after: Any) -> httpx.Response:
        """Convertit une entrée injectée en réponse (ou lève l'exception injectée)."""
        if isinstance(entry, type) and issubclass(entry, BaseException):
            if issubclass(entry, httpx.RequestError):
                raise entry("erreur réseau simulée", request=request)
            raise entry("erreur simulée")
        if isinstance(entry, BaseException):
            raise entry
        if isinstance(entry, httpx.Response):
            response = entry
        elif callable(entry):
            return self._materialise(entry(request), request, retry_after)
        else:
            status, body, headers, entry_retry = self._entry_parts(entry)
            if entry_retry is not None:
                retry_after = entry_retry
            headers = dict(headers)
            if retry_after is not None and not any(k.lower() == "retry-after" for k in headers):
                headers["Retry-After"] = _retry_after_value(retry_after)
            return _json_response(status, body, headers)
        if retry_after is not None and "retry-after" not in response.headers:
            response.headers["Retry-After"] = _retry_after_value(retry_after)
        return response

    @staticmethod
    def _entry_parts(entry: Any) -> Tuple[int, Any, Mapping[str, str], Any]:
        """(statut, corps, en-têtes, retry_after) d'une entrée non appelable."""
        if isinstance(entry, bool):
            raise TypeError("entrée d'injection invalide : booléen")
        if isinstance(entry, int):
            name = _ERROR_NAME_BY_STATUS.get(entry)
            body = copy.deepcopy(NAMED_ERRORS[name][1]) if name else (
                None if entry == 204 else {"detail": httpx.codes.get_reason_phrase(entry) or "Error"}
            )
            return entry, body, {}, None
        if isinstance(entry, str):
            status, body = NAMED_ERRORS[entry]
            return status, copy.deepcopy(body), {}, None
        if isinstance(entry, tuple):
            status = entry[0]
            body = entry[1] if len(entry) > 1 else None
            headers = entry[2] if len(entry) > 2 else {}
            return status, body, headers or {}, None
        if isinstance(entry, Mapping):
            body = entry.get("body", entry.get("json"))
            return int(entry["status"]), body, entry.get("headers") or {}, entry.get("retry_after")
        raise TypeError(f"entrée d'injection non prise en charge : {type(entry).__name__}")

    def _dispatch(self, request: httpx.Request, call: FakeCall) -> httpx.Response:
        """Route la requête vers le gestionnaire du point d'accès."""
        path_matched = False
        for method, template, regex in _COMPILED_ROUTES:
            match = regex.match(call.path)
            if not match:
                continue
            path_matched = True
            if method != call.method:
                continue
            if (method, template) not in PUBLIC_ROUTES and not self._authorised(call):
                return _json_response(*NAMED_ERRORS["invalid_key"])
            handler = getattr(self, "_route_" + self._handler_name(method, template))
            if template.startswith(_STATEFUL_PREFIXES):
                with self._lock:
                    return handler(call, **match.groupdict())
            return handler(call, **match.groupdict())
        if path_matched:
            return _json_response(405, {"detail": "Method Not Allowed"})
        return _json_response(404, {"detail": "Not Found"})

    @staticmethod
    def _handler_name(method: str, template: str) -> str:
        """Nom de méthode ``_route_<verbe>_<chemin>`` d'une route."""
        slug = re.sub(r"[^a-z0-9]+", "_", re.sub(r"\{\w+\}", "id", template.lower())).strip("_")
        return f"{method.lower()}_{slug}"

    def _authorised(self, call: FakeCall) -> bool:
        """Vrai si l'en-tête ``Authorization`` porte la clé attendue."""
        if self.api_key is None:
            return True
        return call.headers.get("authorization") == f"Bearer {self.api_key}"

    def _now(self) -> int:
        """Horodatage Unix des objets créés."""
        return int(self._clock())

    def _next_id(self, prefix: str) -> str:
        """Identifiant de réponse séquentiel (``chatcmpl-fake-000001``…)."""
        with self._lock:
            self._counter += 1
            return f"{prefix}-fake-{self._counter:06d}"

    # ------------------------------------------------------------------ modèles
    def find_model(self, name: Any) -> Optional[Dict[str, Any]]:
        """Modèle du catalogue désigné par son id ou un alias (``None`` si absent)."""
        if not isinstance(name, str) or not name:
            return None
        for model in self._catalog():
            if model.get("id") == name or name in (model.get("aliases") or []):
                return model
        return None

    def _catalog(self) -> List[Dict[str, Any]]:
        """Catalogue servi : fixture P2, plus ``mistral-ocr-2512`` si l'accès OCR est ouvert."""
        models = list(self.models)
        if self.ocr_access and not any(m.get("id") == OCR_DOC_MODEL for m in models):
            models.append({
                "object": "model", "id": OCR_DOC_MODEL, "type": "image-to-text", "aliases": [],
                "created": FIXED_CREATED, "owned_by": "Direction interministerielle du numerique (DINUM)",
                "max_context_length": 16384, "costs": {"prompt_tokens": 0.0, "completion_tokens": 0.0},
            })
        return models

    @staticmethod
    def _wrong_type(expected: str, actual: str) -> httpx.Response:
        """422 « Model has wrong type » (P6)."""
        return _json_response(422, {"detail": f"Model has wrong type. Expected: {expected}. Actual: {actual}."})

    def _usage(self, prompt_tokens: int, completion_tokens: int, *, carbon: bool = True) -> Dict[str, Any]:
        """Objet ``usage`` (tokens, ``cost``, ``impacts``, ``requests``)."""
        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": 0.0,
        }
        if carbon:
            usage["carbon"] = {k: {"min": v, "max": v} for k, v in _IMPACTS.items()}
        usage["impacts"] = dict(_IMPACTS)
        usage["requests"] = 1
        return usage

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimation grossière de tokens (4 caractères par token)."""
        return max(1, len(text) // 4) if text else 0

    # ------------------------------------------------------------------ santé / compte
    def _route_get_health(self, call: FakeCall) -> httpx.Response:
        """``GET /health`` (racine, sans authentification)."""
        return _json_response(200, {"status": "ok"})

    def _route_get_health_models(self, call: FakeCall) -> httpx.Response:
        """``GET /health/models`` (racine)."""
        return _json_response(200, copy.deepcopy(self.health_models))

    def _route_get_v1_me(self, call: FakeCall) -> httpx.Response:
        """``GET /v1/me`` (fixture P1 masquée + ``me_overrides``)."""
        return _json_response(200, copy.deepcopy(self.me))

    def _route_get_v1_models(self, call: FakeCall) -> httpx.Response:
        """``GET /v1/models`` (fixture P2)."""
        return _json_response(200, {"object": "list", "data": copy.deepcopy(self._catalog())})

    def _route_get_v1_models_id(self, call: FakeCall, model: str) -> httpx.Response:
        """``GET /v1/models/{model}`` (id ou alias)."""
        found = self.find_model(model)
        if found is None:
            return _json_response(404, {"detail": "Model not found."})
        return _json_response(200, copy.deepcopy(found))

    # ------------------------------------------------------------------ chat
    def _route_post_v1_chat_completions(self, call: FakeCall) -> httpx.Response:
        """``POST /v1/chat/completions`` : texte, image (OCR LightOnOCR) et raisonnement."""
        body = call.json
        if not isinstance(body, dict):
            return _unprocessable([_validation_error(["body"], "Input should be a valid dictionary", body, "dict_type")])
        model_name = body.get("model")
        messages = body.get("messages")
        if not isinstance(model_name, str) or not model_name:
            return _unprocessable([_validation_error(["body", "model"], "Field required", None, "missing")])
        if not isinstance(messages, list) or not messages:
            return _unprocessable([_validation_error(["body", "messages"], "Field required", messages, "missing")])
        model = self.find_model(model_name)
        if model is None:
            return _json_response(404, {"detail": "Model not found."})
        if model.get("type") not in CHAT_MODEL_TYPES:
            return self._wrong_type("text-generation", str(model.get("type")))
        prompt_text, image_urls = self._message_parts(messages)
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        reasoning: Optional[str] = None
        if image_urls:
            content: Optional[str] = self._ocr_chat_text(image_urls[0])
            finish = "stop"
        else:
            reply = self._chat_reply_for(body)
            content, finish, reasoning = reply["content"], reply["finish_reason"], reply.get("reasoning")
        if model.get("id") in self.reasoning_models:
            if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens < self.reasoning_tokens:
                content, finish = None, "length"
                reasoning = self._fixture_trap["choices"][0]["message"]["reasoning"]
            elif reasoning is None:
                reasoning = self._fixture_reasoning["choices"][0]["message"]["reasoning"]
        message = copy.deepcopy(self._fixture_chat["choices"][0]["message"])
        message["content"] = content
        message["reasoning"] = reasoning
        response = copy.deepcopy(self._fixture_chat)
        response.update({
            "id": self._next_id("chatcmpl"),
            "created": self._now(),
            "model": model_name,
            "choices": [{
                "index": 0, "message": message, "logprobs": None, "finish_reason": finish,
                "stop_reason": None, "token_ids": None,
            }],
            "usage": self._usage(self._estimate_tokens(prompt_text), self._estimate_tokens(content or "")),
        })
        return _json_response(200, response, RATELIMIT_HEADERS)

    @staticmethod
    def _message_parts(messages: List[Any]) -> Tuple[str, List[str]]:
        """(texte concaténé, URLs d'images) des messages."""
        texts: List[str] = []
        images: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        texts.append(part["text"])
                    elif part.get("type") == "image_url":
                        image = part.get("image_url")
                        url = image.get("url") if isinstance(image, dict) else image
                        images.append(str(url or ""))
        return "\n".join(texts), images

    def _chat_reply_for(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Réponse de chat : file ``chat_replies``, puis ``chat_reply``, puis « OK. » (P7)."""
        with self._lock:
            queued = self.chat_replies.popleft() if self.chat_replies else _NO_REPLY
        if queued is not _NO_REPLY:
            reply = queued
        elif callable(self.chat_reply):
            reply = self.chat_reply(body)
        else:
            reply = self.chat_reply
        if reply is None:
            reply = self._fixture_chat["choices"][0]["message"]["content"]
        if isinstance(reply, Mapping):
            return {
                "content": reply.get("content"),
                "finish_reason": reply.get("finish_reason", "stop"),
                "reasoning": reply.get("reasoning"),
            }
        return {"content": str(reply), "finish_reason": "stop", "reasoning": None}

    def _ocr_chat_text(self, image_url: str) -> str:
        """Transcription d'une page image (``ocr_chat_reply`` ou texte de la fixture P5)."""
        with self._lock:
            index = self._ocr_chat_index
            self._ocr_chat_index += 1
        if callable(self.ocr_chat_reply):
            return str(self.ocr_chat_reply(image_url, index))
        if self.ocr_chat_reply is not None:
            return str(self.ocr_chat_reply)
        return self._fixture_ocr_chat["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------ embeddings
    def _route_post_v1_embeddings(self, call: FakeCall) -> httpx.Response:
        """``POST /v1/embeddings`` : 64 textes au plus (413), pas de chaîne vide (400)."""
        body = call.json
        if not isinstance(body, dict):
            return _unprocessable([_validation_error(["body"], "Input should be a valid dictionary", body, "dict_type")])
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not inputs or not all(isinstance(t, str) for t in inputs):
            return _unprocessable([_validation_error(["body", "input"], "Input should be a valid list of strings", inputs, "list_type")])
        if len(inputs) > MAX_EMBED_BATCH:
            return _json_response(413, {"detail": f"batch size {len(inputs)} > maximum allowed batch size {MAX_EMBED_BATCH}"})
        if any(t == "" for t in inputs):
            return _json_response(400, {"detail": "Input validation error: `inputs` cannot be empty"})
        model_name = body.get("model")
        model = self.find_model(model_name)
        if model is None:
            return _json_response(404, {"detail": "Model not found."})
        if model.get("type") != EMBED_MODEL_TYPE:
            return self._wrong_type(EMBED_MODEL_TYPE, str(model.get("type")))
        dim = 4096 if model.get("id") == "qwen3-vl-embedding-8b" else self.embed_dim
        data = [{"embedding": fake_embedding(text, dim), "index": i, "object": "embedding"} for i, text in enumerate(inputs)]
        if self.embed_shuffle:
            data.reverse()
        prompt_tokens = sum(self._estimate_tokens(t) for t in inputs)
        response = {
            "data": data,
            "model": model_name,
            "object": "list",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens,
                      "cost": 0.0, "impacts": {"kWh": 0.0, "kgCO2eq": 0.0}},
            "id": self._next_id("request"),
        }
        return _json_response(200, response, RATELIMIT_HEADERS)

    # ------------------------------------------------------------------ OCR document
    def _route_post_v1_ocr(self, call: FakeCall) -> httpx.Response:
        """``POST /v1/ocr`` : 404 sans accès (D3) ; sinon pages indexées à partir de 0."""
        body = call.json if isinstance(call.json, dict) else {}
        model_name = body.get("model") or OCR_DOC_MODEL
        model = self.find_model(model_name)
        if model is None:
            return _json_response(404, {"detail": f"Model {model_name} not found."})
        if model.get("type") not in OCR_MODEL_TYPES:
            return self._wrong_type("image-to-text", str(model.get("type")))
        document = body.get("document")
        if not isinstance(document, dict) or document.get("type") not in ("document_url", "image_url"):
            return _unprocessable([_validation_error(["body", "document"], "Field required", document, "missing")])
        url = str(document.get("document_url") or document.get("image_url") or "")
        raw = self._data_uri_bytes(url)
        total = self._pdf_page_count(raw) if document.get("type") == "document_url" else 1
        try:
            indices = self._parse_pages(body.get("pages"), total)
        except ValueError as exc:
            return _unprocessable([_validation_error(["body", "pages"], str(exc), body.get("pages"))])
        template = self._fixture_ocr_doc
        pages = []
        for index in indices:
            page = copy.deepcopy(template["pages"][0])
            page["index"] = index
            page["markdown"] = self._ocr_doc_text(index)
            pages.append(page)
        response = {
            "id": self._next_id("ocr"),
            "model": model_name,
            "pages": pages,
            "usage_info": {"pages_processed": len(pages), "doc_size_bytes": len(raw)},
            "usage": self._usage(0, 0, carbon=False),
        }
        return _json_response(200, response)

    def _ocr_doc_text(self, index: int) -> str:
        """Markdown d'une page ``/v1/ocr`` (``ocr_doc_reply`` ou fixture P3 pour l'index 0)."""
        if callable(self.ocr_doc_reply):
            return str(self.ocr_doc_reply(index))
        if self.ocr_doc_reply is not None:
            return str(self.ocr_doc_reply)
        if index == 0:
            return self._fixture_ocr_doc["pages"][0]["markdown"]
        return f"PAGE_{index + 1}_7Q"

    @staticmethod
    def _data_uri_bytes(url: str) -> bytes:
        """Octets d'une URI ``data:…;base64,…`` (vide sinon)."""
        if not url.startswith("data:") or "," not in url:
            return b""
        try:
            return base64.b64decode(url.split(",", 1)[1], validate=False)
        except (ValueError, TypeError):
            return b""

    @staticmethod
    def _pdf_page_count(raw: bytes) -> int:
        """Nombre de pages d'un PDF (objets ``/Type /Page``), 1 par défaut."""
        count = len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", raw))
        return count or 1

    @staticmethod
    def _parse_pages(pages: Any, total: int) -> List[int]:
        """Indices de pages demandés (liste ou chaîne « 0,2,4-6 »), toutes par défaut."""
        if pages is None:
            return list(range(total))
        if isinstance(pages, list):
            if not all(isinstance(p, int) and not isinstance(p, bool) and p >= 0 for p in pages):
                raise ValueError("pages must be non-negative integers")
            return list(pages)
        if isinstance(pages, str):
            indices: List[int] = []
            for part in pages.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start, end = (int(x) for x in part.split("-", 1))
                    indices.extend(range(start, end + 1))
                else:
                    indices.append(int(part))
            return indices
        raise ValueError("pages must be a list or a string")

    # ------------------------------------------------------------------ pagination
    @staticmethod
    def _page_params(call: FakeCall) -> Tuple[Optional[httpx.Response], int, int]:
        """(erreur 422 éventuelle, offset, limit) des paramètres de pagination."""
        try:
            offset = int(call.params.get("offset", 0))
            limit = int(call.params.get("limit", DEFAULT_PAGE_LIMIT))
        except ValueError:
            return _unprocessable([_validation_error(["query", "limit"], "Input should be a valid integer",
                                                     call.params.get("limit"), "int_parsing")]), 0, 0
        if limit > MAX_PAGE_LIMIT:
            return _unprocessable([_validation_error(
                ["query", "limit"], f"Input should be less than or equal to {MAX_PAGE_LIMIT}", str(limit),
                "less_than_equal", {"le": MAX_PAGE_LIMIT})]), 0, 0
        if limit < 1 or offset < 0:
            return _unprocessable([_validation_error(["query", "limit"], "Input should be greater than or equal to 1",
                                                     str(limit), "greater_than_equal", {"ge": 1})]), 0, 0
        return None, offset, limit

    @staticmethod
    def _ordered(items: List[Dict[str, Any]], call: FakeCall) -> List[Dict[str, Any]]:
        """Tri par ``order_by`` (``id`` par défaut) et ``order_direction``."""
        key = call.params.get("order_by", "id")
        reverse = call.params.get("order_direction", "asc").lower() == "desc"
        return sorted(items, key=lambda item: (item.get(key) is None, item.get(key)), reverse=reverse)

    # ------------------------------------------------------------------ collections
    def _collection_view(self, collection: Dict[str, Any]) -> Dict[str, Any]:
        """Objet ``collection`` avec compteurs recalculés."""
        view = dict(collection)
        docs = [d for d in self.documents.values() if d["collection_id"] == collection["id"]]
        view["documents"] = len(docs)
        view["size"] = sum(self._document_size(d["id"]) for d in docs)
        return view

    def _route_get_v1_collections(self, call: FakeCall) -> httpx.Response:
        """``GET /v1/collections`` : filtres ``name`` (exact, D14) et ``visibility``, pagination."""
        error, offset, limit = self._page_params(call)
        if error is not None:
            return error
        items = [self._collection_view(c) for c in self.collections.values()]
        if "name" in call.params:
            items = [c for c in items if c["name"] == call.params["name"]]
        if "visibility" in call.params:
            items = [c for c in items if c["visibility"] == call.params["visibility"]]
        items = self._ordered(items, call)[offset:offset + limit]
        return _json_response(200, {"object": "list", "data": items})

    def _route_post_v1_collections(self, call: FakeCall) -> httpx.Response:
        """``POST /v1/collections`` : 201 ``{"id": …}`` (visibilité ``private`` par défaut)."""
        body = call.json if isinstance(call.json, dict) else None
        name = body.get("name") if body else None
        if not isinstance(name, str) or not name:
            return _unprocessable([_validation_error(["body", "name"], "Field required", name, "missing")])
        visibility = body.get("visibility") or "private"
        if visibility not in ("private", "public"):
            return _unprocessable([_validation_error(["body", "visibility"], "Input should be 'private' or 'public'",
                                                     visibility, "enum")])
        cid = self._next_collection_id
        self._next_collection_id += 1
        now = self._now()
        self.collections[cid] = {
            "object": "collection", "id": cid, "name": name, "owner": "<masqué>",
            "description": body.get("description"), "visibility": visibility,
            "created": now, "updated": now, "documents": 0, "size": 0,
        }
        return _json_response(201, {"id": cid})

    def _collection_or_404(self, collection_id: Any) -> Tuple[Optional[Dict[str, Any]], Optional[httpx.Response]]:
        """(collection, None) ou (None, réponse 404)."""
        try:
            cid = int(collection_id)
        except (TypeError, ValueError):
            cid = None
        collection = self.collections.get(cid) if cid is not None else None
        if collection is None:
            return None, _json_response(404, {"detail": "Collection not found."})
        return collection, None

    def _route_get_v1_collections_id(self, call: FakeCall, collection_id: str) -> httpx.Response:
        """``GET /v1/collections/{id}``."""
        collection, error = self._collection_or_404(collection_id)
        return error or _json_response(200, self._collection_view(collection))

    def _route_patch_v1_collections_id(self, call: FakeCall, collection_id: str) -> httpx.Response:
        """``PATCH /v1/collections/{id}`` : 204."""
        collection, error = self._collection_or_404(collection_id)
        if error is not None:
            return error
        body = call.json if isinstance(call.json, dict) else {}
        for key in ("name", "description", "visibility"):
            if key in body and body[key] is not None:
                collection[key] = body[key]
        collection["updated"] = self._now()
        return _json_response(204)

    def _route_delete_v1_collections_id(self, call: FakeCall, collection_id: str) -> httpx.Response:
        """``DELETE /v1/collections/{id}`` : 204, documents et chunks supprimés."""
        collection, error = self._collection_or_404(collection_id)
        if error is not None:
            return error
        for doc_id in [d["id"] for d in self.documents.values() if d["collection_id"] == collection["id"]]:
            self._drop_document(doc_id)
        del self.collections[collection["id"]]
        return _json_response(204)

    # ------------------------------------------------------------------ documents
    def _document_size(self, document_id: int) -> int:
        """Taille (octets UTF-8) des chunks d'un document."""
        return sum(len(c["content"].encode("utf-8")) for c in self.chunks.get(document_id, []))

    def _document_view(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Objet ``document`` public (sans métadonnées internes)."""
        return {
            "object": "document", "id": document["id"], "name": document["name"],
            "collection_id": document["collection_id"], "created": document["created"],
            "chunks": len(self.chunks.get(document["id"], [])), "size": self._document_size(document["id"]),
        }

    def _drop_document(self, document_id: int) -> None:
        """Supprime un document et ses chunks."""
        self.documents.pop(document_id, None)
        self.chunks.pop(document_id, None)
        self._next_chunk_id.pop(document_id, None)

    def _document_or_404(self, document_id: Any) -> Tuple[Optional[Dict[str, Any]], Optional[httpx.Response]]:
        """(document, None) ou (None, réponse 404)."""
        try:
            did = int(document_id)
        except (TypeError, ValueError):
            did = None
        document = self.documents.get(did) if did is not None else None
        if document is None:
            return None, _json_response(404, {"detail": "Document not found."})
        return document, None

    def _route_get_v1_documents(self, call: FakeCall) -> httpx.Response:
        """``GET /v1/documents`` : filtres ``collection_id`` et ``name``, pagination."""
        error, offset, limit = self._page_params(call)
        if error is not None:
            return error
        items = [self._document_view(d) for d in self.documents.values()]
        if "collection_id" in call.params:
            try:
                cid = int(call.params["collection_id"])
            except ValueError:
                return _unprocessable([_validation_error(["query", "collection_id"], "Input should be a valid integer",
                                                         call.params["collection_id"], "int_parsing")])
            items = [d for d in items if d["collection_id"] == cid]
        if "name" in call.params:
            items = [d for d in items if d["name"] == call.params["name"]]
        items = self._ordered(items, call)[offset:offset + limit]
        return _json_response(200, {"object": "list", "data": items})

    def _route_post_v1_documents(self, call: FakeCall) -> httpx.Response:
        """``POST /v1/documents`` (multipart ou urlencoded, D15) : 201 ``{"id": …}``."""
        form = call.form
        raw_cid = form.get("collection_id")
        try:
            cid = int(raw_cid) if raw_cid is not None else None
        except ValueError:
            cid = None
        if cid is None or cid <= 0:
            return _unprocessable([_validation_error(["body", "collection_id"], "Input should be a valid integer",
                                                     raw_cid, "int_parsing")])
        collection, error = self._collection_or_404(cid)
        if error is not None:
            return error
        metadata: Dict[str, Any] = {}
        if form.get("metadata"):
            try:
                metadata = json.loads(form["metadata"])
            except ValueError:
                return _unprocessable([_validation_error(["body", "metadata"], "Invalid JSON", form["metadata"], "json_invalid")])
        upload = call.files.get("file")
        name = form.get("name") or (upload[0] if upload else None)
        if not name:
            return _unprocessable([_validation_error(["body", "name"], "Field required", None, "missing")])
        did = self._next_document_id
        self._next_document_id += 1
        self.documents[did] = {
            "id": did, "name": name, "collection_id": collection["id"], "created": self._now(), "metadata": metadata,
        }
        self.chunks[did] = []
        self._next_chunk_id[did] = 0
        if upload:
            text = upload[1].decode("utf-8", "replace")
            try:
                size = max(1, int(form.get("chunk_size", 2048)))
            except ValueError:
                size = 2048
            for start in range(0, len(text), size):
                piece = text[start:start + size]
                if piece.strip():
                    self._store_chunk(did, piece, dict(metadata))
        return _json_response(201, {"id": did})

    def _route_get_v1_documents_id(self, call: FakeCall, document_id: str) -> httpx.Response:
        """``GET /v1/documents/{id}``."""
        document, error = self._document_or_404(document_id)
        return error or _json_response(200, self._document_view(document))

    def _route_delete_v1_documents_id(self, call: FakeCall, document_id: str) -> httpx.Response:
        """``DELETE /v1/documents/{id}`` : 204."""
        document, error = self._document_or_404(document_id)
        if error is not None:
            return error
        self._drop_document(document["id"])
        return _json_response(204)

    # ------------------------------------------------------------------ chunks
    def _store_chunk(self, document_id: int, content: str, metadata: Optional[Dict[str, Any]]) -> int:
        """Enregistre un chunk et renvoie son id (séquentiel par document, à partir de 0)."""
        chunk_id = self._next_chunk_id.get(document_id, 0)
        self._next_chunk_id[document_id] = chunk_id + 1
        self.chunks.setdefault(document_id, []).append({
            "object": "chunk", "id": chunk_id, "collection_id": self.documents[document_id]["collection_id"],
            "document_id": document_id, "content": content, "metadata": dict(metadata or {}),
            "created": self._now(),
        })
        return chunk_id

    @staticmethod
    def metadata_errors(metadata: Any, loc: Sequence[Any]) -> List[Dict[str, Any]]:
        """Erreurs 422 d'un dictionnaire de métadonnées de chunk (P16).

        Règles : 10 propriétés au plus ; clés de 1 à 255 caractères ; valeurs
        chaîne (1 à 255 caractères), entier ou flottant (|v| ≤ 1e16), booléen.
        """
        if metadata is None:
            return []
        if not isinstance(metadata, dict):
            return [_validation_error(list(loc), "Input should be a valid dictionary", metadata, "dict_type")]
        errors: List[Dict[str, Any]] = []
        if len(metadata) > MAX_METADATA_PROPS:
            errors.append(_validation_error(
                list(loc), f"Dictionary should have at most {MAX_METADATA_PROPS} items after validation, not {len(metadata)}",
                metadata, "too_long", {"field_type": "Dictionary", "max_length": MAX_METADATA_PROPS,
                                       "actual_length": len(metadata)}))
        for key, value in metadata.items():
            where = list(loc) + [key]
            if not isinstance(key, str) or not 1 <= len(key) <= MAX_METADATA_STR:
                errors.append(_validation_error(where, f"String should have at most {MAX_METADATA_STR} characters",
                                                key, "string_too_long", {"max_length": MAX_METADATA_STR}))
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if isinstance(value, float) and math.isnan(value) or abs(value) > MAX_METADATA_NUMBER:
                    errors.append(_validation_error(where, "Input should be less than or equal to 1e16", value, "less_than_equal"))
                continue
            if isinstance(value, str):
                if not value:
                    errors.append(_validation_error(where, "String should have at least 1 character", value,
                                                    "string_too_short", {"min_length": 1}))
                elif len(value) > MAX_METADATA_STR:
                    errors.append(_validation_error(where, f"String should have at most {MAX_METADATA_STR} characters",
                                                    value, "string_too_long", {"max_length": MAX_METADATA_STR}))
                continue
            errors.append(_validation_error(where, "Input should be a valid string", value, "string_type"))
        return errors

    def _route_get_v1_documents_id_chunks(self, call: FakeCall, document_id: str) -> httpx.Response:
        """``GET /v1/documents/{id}/chunks`` : pagination ``offset`` / ``limit`` (100 au plus)."""
        document, error = self._document_or_404(document_id)
        if error is not None:
            return error
        error, offset, limit = self._page_params(call)
        if error is not None:
            return error
        items = copy.deepcopy(self.chunks.get(document["id"], [])[offset:offset + limit])
        return _json_response(200, {"object": "list", "data": items})

    def _route_post_v1_documents_id_chunks(self, call: FakeCall, document_id: str) -> httpx.Response:
        """``POST /v1/documents/{id}/chunks`` : 1 à 64 chunks, envoi atomique, ids renvoyés (P16)."""
        document, error = self._document_or_404(document_id)
        if error is not None:
            return error
        body = call.json if isinstance(call.json, dict) else {}
        chunks = body.get("chunks")
        if not isinstance(chunks, list):
            return _unprocessable([_validation_error(["body", "chunks"], "Field required", chunks, "missing")])
        if len(chunks) > MAX_CHUNKS_PER_POST:
            return _unprocessable([_validation_error(
                ["body", "chunks"], f"List should have at most {MAX_CHUNKS_PER_POST} items after validation, not {len(chunks)}",
                chunks, "too_long", {"field_type": "List", "max_length": MAX_CHUNKS_PER_POST, "actual_length": len(chunks)})])
        if not chunks:
            return _unprocessable([_validation_error(["body", "chunks"], "List should have at least 1 item after validation, not 0",
                                                     chunks, "too_short", {"field_type": "List", "min_length": 1})])
        errors: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                errors.append(_validation_error(["body", "chunks", i], "Input should be a valid dictionary", chunk, "dict_type"))
                continue
            content = chunk.get("content")
            if not isinstance(content, str) or not content:
                errors.append(_validation_error(["body", "chunks", i, "content"], "String should have at least 1 character",
                                                content, "string_too_short", {"min_length": 1}))
            errors.extend(self.metadata_errors(chunk.get("metadata"), ["body", "chunks", i, "metadata"]))
        if errors:
            return _unprocessable(errors)
        ids = [self._store_chunk(document["id"], c["content"], c.get("metadata")) for c in chunks]
        return _json_response(201, {"document_id": document["id"], "ids": ids})

    def _chunk_or_404(self, document_id: str, chunk_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[httpx.Response]]:
        """(chunk, None) ou (None, réponse 404)."""
        document, error = self._document_or_404(document_id)
        if error is not None:
            return None, error
        for chunk in self.chunks.get(document["id"], []):
            if str(chunk["id"]) == str(chunk_id):
                return chunk, None
        return None, _json_response(404, {"detail": "Chunk not found."})

    def _route_get_v1_documents_id_chunks_id(self, call: FakeCall, document_id: str, chunk_id: str) -> httpx.Response:
        """``GET /v1/documents/{id}/chunks/{chunk_id}``."""
        chunk, error = self._chunk_or_404(document_id, chunk_id)
        return error or _json_response(200, copy.deepcopy(chunk))

    def _route_delete_v1_documents_id_chunks_id(self, call: FakeCall, document_id: str, chunk_id: str) -> httpx.Response:
        """``DELETE /v1/documents/{id}/chunks/{chunk_id}`` : 204."""
        chunk, error = self._chunk_or_404(document_id, chunk_id)
        if error is not None:
            return error
        self.chunks[chunk["document_id"]].remove(chunk)
        return _json_response(204)

    # ------------------------------------------------------------------ recherche
    @classmethod
    def _filter_errors(cls, flt: Any, loc: List[Any], *, nested: bool = False) -> List[Dict[str, Any]]:
        """Erreurs 422 d'un ``metadata_filters`` (comparaison ou composé de 2 à 4, sans imbrication)."""
        if flt is None:
            return []
        if not isinstance(flt, dict):
            return [_validation_error(loc, "Input should be a valid dictionary", flt, "dict_type")]
        if "operator" in flt or "filters" in flt:
            if nested:
                return [_validation_error(loc, "Nested compound filters are not allowed", flt, "value_error")]
            filters = flt.get("filters")
            if flt.get("operator") not in ("and", "or") or not isinstance(filters, list):
                return [_validation_error(loc + ["operator"], "Input should be 'and' or 'or'", flt.get("operator"), "enum")]
            if not MIN_COMPOUND_FILTERS <= len(filters) <= MAX_COMPOUND_FILTERS:
                return [_validation_error(loc + ["filters"], "Field required", flt, "missing")]
            errors: List[Dict[str, Any]] = []
            for i, sub in enumerate(filters):
                errors.extend(cls._filter_errors(sub, loc + ["filters", i], nested=True))
            return errors
        if not isinstance(flt.get("key"), str) or flt.get("type") not in FILTER_TYPES or "value" not in flt:
            return [_validation_error(loc, "Field required", flt, "missing")]
        return []

    @classmethod
    def _filter_matches(cls, metadata: Mapping[str, Any], flt: Optional[Mapping[str, Any]]) -> bool:
        """Vrai si les métadonnées satisfont le filtre (``eq``, ``sw``, ``ew``, ``co``)."""
        if not flt:
            return True
        if "operator" in flt:
            results = [cls._filter_matches(metadata, sub) for sub in flt["filters"]]
            return all(results) if flt["operator"] == "and" else any(results)
        key, kind, value = flt["key"], flt["type"], flt["value"]
        if key not in metadata:
            return False
        actual = metadata[key]
        if kind == "eq":
            return actual == value and type(actual) is type(value) or (
                isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and isinstance(value, (int, float)) and not isinstance(value, bool) and actual == value)
        if not isinstance(actual, str):
            return False
        text = str(value)
        if kind == "sw":
            return actual.startswith(text)
        if kind == "ew":
            return actual.endswith(text)
        return text in actual

    def _route_post_v1_search(self, call: FakeCall) -> httpx.Response:
        """``POST /v1/search`` : ``semantic`` / ``lexical`` / ``hybrid`` (RRF), filtres de métadonnées."""
        body = call.json if isinstance(call.json, dict) else {}
        query = body.get("query")
        if not isinstance(query, str) or not query:
            return _unprocessable([_validation_error(["body", "query"], "Input should be a valid string", query, "string_type")])
        method = body.get("method", "semantic")
        if method not in SEARCH_METHODS:
            return _unprocessable([_validation_error(
                ["body", "method"], "Input should be 'hybrid', 'semantic' or 'lexical'", method, "enum",
                {"expected": "'hybrid', 'semantic' or 'lexical'"})])
        limit = body.get("limit", 10)
        offset = body.get("offset", 0)
        if not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_LIMIT:
            return _unprocessable([_validation_error(["body", "limit"], "Input should be less than or equal to 100", limit,
                                                     "less_than_equal", {"le": MAX_PAGE_LIMIT})])
        threshold = body.get("score_threshold")
        if threshold and method != "semantic":
            return _json_response(400, {"detail": "Score threshold is only available for semantic search method"})
        errors = self._filter_errors(body.get("metadata_filters"), ["body", "metadata_filters"])
        if errors:
            return _unprocessable(errors)
        collection_ids = body.get("collection_ids") or []
        document_ids = body.get("document_ids") or []
        for cid in collection_ids:
            if cid not in self.collections:
                return _json_response(404, {"detail": "Collection not found."})
        candidates = [
            c for doc_id, chunks in self.chunks.items() for c in chunks
            if (not collection_ids or c["collection_id"] in collection_ids)
            and (not document_ids or doc_id in document_ids)
            and self._filter_matches(c["metadata"], body.get("metadata_filters"))
        ]
        scored = self._score(query, candidates, method, int(body.get("rff_k", DEFAULT_RFF_K) or DEFAULT_RFF_K))
        if method == "semantic" and threshold:
            scored = [(s, c) for s, c in scored if s >= float(threshold)]
        page = scored[offset:offset + limit]
        data = [{"method": method, "score": score, "chunk": copy.deepcopy(chunk)} for score, chunk in page]
        tokens = self._estimate_tokens(query)
        return _json_response(200, {
            "object": "list", "data": data,
            "usage": {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens, "cost": 0.0,
                      "impacts": {"kWh": 0.0, "kgCO2eq": 0.0}},
        })

    def _score(self, query: str, chunks: List[Dict[str, Any]], method: str, rff_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """Scores triés : cosinus ramené à [0, 1], recouvrement lexical, ou fusion RRF."""
        qvec = _embedding_tuple(query, self.embed_dim)
        terms = set(re.findall(r"\w+", query.lower()))

        def semantic(chunk: Dict[str, Any]) -> float:
            """Similarité cosinus (vecteurs de norme 1) ramenée à [0, 1]."""
            cvec = _embedding_tuple(chunk["content"], self.embed_dim)
            return (1.0 + sum(a * b for a, b in zip(qvec, cvec))) / 2.0

        def lexical(chunk: Dict[str, Any]) -> float:
            """Nombre d'occurrences des termes de la requête."""
            words = re.findall(r"\w+", chunk["content"].lower())
            return float(sum(1 for w in words if w in terms))

        if method == "semantic":
            ranked = [(semantic(c), c) for c in chunks]
        elif method == "lexical":
            ranked = [(lexical(c), c) for c in chunks]
            ranked = [(s, c) for s, c in ranked if s > 0]
        else:
            by_sem = sorted(chunks, key=semantic, reverse=True)
            by_lex = sorted(chunks, key=lexical, reverse=True)
            fused: Dict[int, float] = {}
            for ranking in (by_sem, by_lex):
                for rank, chunk in enumerate(ranking):
                    fused[id(chunk)] = fused.get(id(chunk), 0.0) + 1.0 / (rff_k + rank + 1)
            ranked = [(fused[id(c)], c) for c in chunks]
        ranked.sort(key=lambda item: (-item[0], item[1]["document_id"], item[1]["id"]))
        return ranked


# ---------------------------------------------------------------------------
# Redis simulé
# ---------------------------------------------------------------------------
class FakeRedisError(Exception):
    """Erreur de commande (équivalent de ``redis.ResponseError``)."""


class FakeRedis:
    """Sous-ensemble en mémoire de ``redis.Redis`` avec expirations et horloge injectable.

    Les valeurs sont stockées en octets et renvoyées en ``bytes`` (ou en ``str``
    avec ``decode_responses=True``), comme redis-py. Chaque commande est
    journalisée dans ``calls`` sous la forme ``(commande, arguments)``.
    """

    def __init__(self, clock: Optional[Callable[[], float]] = None, *, decode_responses: bool = False) -> None:
        """Crée une base vide.

        Args:
            clock: horloge en secondes (défaut ``time.monotonic``) ; une horloge
                factice permet de simuler l'écoulement des fenêtres.
            decode_responses: renvoie des ``str`` au lieu de ``bytes``.
        """
        self._clock = clock or time.monotonic
        self.decode_responses = decode_responses
        self._data: Dict[str, bytes] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []
        self.url: Optional[str] = None

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "FakeRedis":
        """Équivalent de ``redis.Redis.from_url`` (l'URL est seulement mémorisée)."""
        instance = cls(clock=kwargs.get("clock"), decode_responses=bool(kwargs.get("decode_responses", False)))
        instance.url = url
        return instance

    # --- utilitaires --------------------------------------------------------
    @staticmethod
    def _key(name: Any) -> str:
        """Nom de clé normalisé en ``str``."""
        return name.decode("utf-8") if isinstance(name, bytes) else str(name)

    @staticmethod
    def _encode(value: Any) -> bytes:
        """Valeur encodée comme redis-py (``bytes``)."""
        if isinstance(value, bytes):
            return value
        if isinstance(value, bool):
            raise FakeRedisError("Invalid input of type: 'bool'")
        return str(value).encode("utf-8")

    def _out(self, value: Optional[bytes]) -> Any:
        """Valeur renvoyée selon ``decode_responses``."""
        if value is None or not self.decode_responses:
            return value
        return value.decode("utf-8")

    @staticmethod
    def _seconds(value: Any) -> float:
        """Durée en secondes (nombre ou ``timedelta``)."""
        if isinstance(value, timedelta):
            return value.total_seconds()
        return float(value)

    def _purge(self, key: str) -> None:
        """Supprime la clé si elle a expiré."""
        deadline = self._expiry.get(key)
        if deadline is not None and self._clock() >= deadline:
            self._data.pop(key, None)
            self._expiry.pop(key, None)

    def _log(self, command: str, *args: Any) -> None:
        """Journalise une commande."""
        self.calls.append((command, args))

    # --- commandes ----------------------------------------------------------
    def ping(self) -> bool:
        """``PING``."""
        self._log("ping")
        return True

    def time(self) -> Tuple[int, int]:
        """``TIME`` : (secondes, microsecondes) de l'horloge injectée."""
        now = self._clock()
        return int(now), int(round((now - int(now)) * 1_000_000))

    def get(self, name: Any) -> Any:
        """``GET``."""
        key = self._key(name)
        with self._lock:
            self._log("get", key)
            self._purge(key)
            return self._out(self._data.get(key))

    def set(self, name: Any, value: Any, ex: Any = None, px: Any = None, nx: bool = False, xx: bool = False,
            keepttl: bool = False) -> Optional[bool]:
        """``SET`` avec ``EX`` / ``PX`` / ``NX`` / ``XX`` / ``KEEPTTL``."""
        key = self._key(name)
        with self._lock:
            self._log("set", key, value, ex, px, nx, xx)
            self._purge(key)
            exists = key in self._data
            if (nx and exists) or (xx and not exists):
                return None
            self._data[key] = self._encode(value)
            if ex is not None:
                self._expiry[key] = self._clock() + self._seconds(ex)
            elif px is not None:
                self._expiry[key] = self._clock() + self._seconds(px) / 1000.0
            elif not keepttl:
                self._expiry.pop(key, None)
            return True

    def incrby(self, name: Any, amount: int = 1) -> int:
        """``INCRBY`` (crée la clé à 0 si absente, conserve le TTL)."""
        key = self._key(name)
        with self._lock:
            self._log("incrby", key, amount)
            self._purge(key)
            try:
                value = int(self._data.get(key, b"0")) + int(amount)
            except ValueError:
                raise FakeRedisError("value is not an integer or out of range") from None
            self._data[key] = str(value).encode("utf-8")
            return value

    def incr(self, name: Any, amount: int = 1) -> int:
        """``INCR`` / ``INCRBY``."""
        return self.incrby(name, amount)

    def decr(self, name: Any, amount: int = 1) -> int:
        """``DECRBY``."""
        return self.incrby(name, -int(amount))

    def expire(self, name: Any, time: Any, nx: bool = False, xx: bool = False, gt: bool = False,
               lt: bool = False) -> bool:
        """``EXPIRE`` (secondes ou ``timedelta``) ; faux si la clé n'existe pas."""
        key = self._key(name)
        with self._lock:
            self._log("expire", key, time)
            self._purge(key)
            if key not in self._data:
                return False
            current = self._expiry.get(key)
            deadline = self._clock() + self._seconds(time)
            if (nx and current is not None) or (xx and current is None):
                return False
            if gt and (current is None or deadline <= current):
                return False
            if lt and current is not None and deadline >= current:
                return False
            self._expiry[key] = deadline
            return True

    def pexpire(self, name: Any, time: Any) -> bool:
        """``PEXPIRE`` (millisecondes)."""
        milliseconds = time.total_seconds() * 1000.0 if isinstance(time, timedelta) else float(time)
        return self.expire(name, milliseconds / 1000.0)

    def pttl(self, name: Any) -> int:
        """``PTTL`` : -2 si absente, -1 sans expiration, sinon millisecondes restantes."""
        key = self._key(name)
        with self._lock:
            self._log("pttl", key)
            self._purge(key)
            if key not in self._data:
                return -2
            deadline = self._expiry.get(key)
            if deadline is None:
                return -1
            return max(0, int(round((deadline - self._clock()) * 1000.0)))

    def ttl(self, name: Any) -> int:
        """``TTL`` : -2 si absente, -1 sans expiration, sinon secondes restantes (arrondies)."""
        remaining = self.pttl(name)
        if remaining < 0:
            return remaining
        return int((remaining + 500) // 1000)

    def delete(self, *names: Any) -> int:
        """``DEL`` : nombre de clés supprimées."""
        removed = 0
        with self._lock:
            for name in names:
                key = self._key(name)
                self._log("delete", key)
                self._purge(key)
                if key in self._data:
                    removed += 1
                    self._data.pop(key, None)
                    self._expiry.pop(key, None)
        return removed

    def exists(self, *names: Any) -> int:
        """``EXISTS`` : nombre de clés présentes."""
        with self._lock:
            count = 0
            for name in names:
                key = self._key(name)
                self._purge(key)
                count += key in self._data
            return count

    def keys(self, pattern: str = "*") -> List[Any]:
        """``KEYS`` (motif ``fnmatch``)."""
        with self._lock:
            for key in list(self._data):
                self._purge(key)
            return [self._out(k.encode("utf-8")) for k in sorted(self._data) if fnmatch.fnmatchcase(k, pattern)]

    def flushall(self) -> bool:
        """``FLUSHALL``."""
        with self._lock:
            self._data.clear()
            self._expiry.clear()
        return True

    flushdb = flushall

    def pipeline(self, transaction: bool = True) -> "FakeRedisPipeline":
        """Pipeline : commandes mises en file puis exécutées par ``execute()``."""
        return FakeRedisPipeline(self)

    def close(self) -> None:
        """Sans effet (compatibilité redis-py)."""


class FakeRedisPipeline:
    """Pipeline de ``FakeRedis`` : les commandes sont exécutées dans l'ordre à ``execute()``."""

    _COMMANDS = frozenset({
        "get", "set", "incr", "incrby", "decr", "expire", "pexpire", "ttl", "pttl", "delete", "exists", "time", "ping",
    })

    def __init__(self, redis: FakeRedis) -> None:
        """Associe le pipeline à ``redis``."""
        self._redis = redis
        self._queue: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Callable[..., "FakeRedisPipeline"]:
        """Met en file une commande connue et renvoie le pipeline (chaînable)."""
        if name not in self._COMMANDS:
            raise AttributeError(name)

        def queue_command(*args: Any, **kwargs: Any) -> "FakeRedisPipeline":
            """Ajoute la commande à la file."""
            self._queue.append((name, args, kwargs))
            return self

        return queue_command

    def execute(self) -> List[Any]:
        """Exécute les commandes en file (sous le verrou de la base) et renvoie leurs résultats."""
        with self._redis._lock:
            results = [getattr(self._redis, name)(*args, **kwargs) for name, args, kwargs in self._queue]
        self._queue.clear()
        return results

    def __enter__(self) -> "FakeRedisPipeline":
        """Contexte ``with r.pipeline() as pipe``."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Vide la file en sortie de contexte."""
        self._queue.clear()
