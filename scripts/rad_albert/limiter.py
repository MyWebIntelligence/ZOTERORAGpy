"""Limiteur proactif de débit Albert, par rôle — stdlib (redis importé paresseusement).

Deux budgets par rôle, multipliés par ``ALBERT_PROCESS_SHARE`` :

* requêtes par minute (``ALBERT_RECODE_RPM``, ``_NOTES_RPM``, ``_OCR_RPM``,
  ``_EMBED_RPM``) ;
* tokens d'entrée par minute pour les rôles de chat (``ALBERT_CHAT_TPM``),
  estimés à ``len/3`` (``estimate_input_tokens``).

Backends :

* ``TokenBucket`` (``local``, défaut) : seau à jetons thread-safe, horloge et
  sommeil injectables ; chaque appelant réserve sa part sous verrou puis dort
  **hors** verrou. ``pause(retry_after)`` suspend le débit après un 429.
* ``RedisWindowLimiter`` (``redis``) : fenêtres fixes d'une minute partagées
  entre processus (Celery, plusieurs sessions) ; URL ``ALBERT_LIMITER_REDIS_URL``,
  sinon ``CELERY_BROKER_URL``, sinon ``redis://localhost:6379/0``. Redis
  indisponible : repli sur un seau local, avec un seul WARNING.

``get_limiter(role, cfg)`` renvoie l'instance partagée du processus pour le
budget du rôle. En asynchrone, appeler ``await limiter.aacquire(...)`` (ou
``asyncio.to_thread(limiter.acquire, ...)``) : la boucle n'est jamais bloquée.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
"""URL Redis de dernier recours."""

REDIS_PREFIX = "ragpy:albert:limiter"
"""Préfixe des clés Redis du limiteur."""

CHARS_PER_TOKEN = 3.0
"""Estimation ``chars`` : un token pour trois caractères (décision 7)."""

WINDOW_SECONDS = 60.0

# Rôle → budget du limiteur (même budget = même instance partagée).
ROLE_BUCKETS: Mapping[str, str] = {
    "recode": "recode",
    "citation": "recode",
    "book_structure": "recode",
    "long_context": "recode",
    "chat": "recode",
    "notes": "notes",
    "ocr": "ocr",
    "ocr_chat": "ocr",
    "ocr_doc": "ocr",
    "embed": "embed",
    "embeddings": "embed",
    "push": "embed",
}


# ---------------------------------------------------------------------------
# Estimation des tokens d'entrée
# ---------------------------------------------------------------------------
def _collect_text(payload: Any, out: list) -> None:
    """Ajoute à ``out`` les textes d'une charge (messages, parties, listes, chaînes)."""
    if payload is None:
        return
    if isinstance(payload, str):
        out.append(payload)
        return
    if isinstance(payload, (bytes, bytearray)):
        return
    if isinstance(payload, Mapping):
        if "messages" in payload:
            _collect_text(payload.get("messages"), out)
            return
        if "input" in payload:
            _collect_text(payload.get("input"), out)
            return
        if payload.get("type") == "text":
            _collect_text(payload.get("text"), out)
            return
        if "content" in payload:
            _collect_text(payload.get("content"), out)
        return
    if isinstance(payload, (list, tuple)):
        for item in payload:
            _collect_text(item, out)


_TIKTOKEN_ENCODING: Any = None


def _tiktoken_count(text: str) -> Optional[int]:
    """Compte les tokens ``o200k_base`` (``None`` si tiktoken est indisponible)."""
    global _TIKTOKEN_ENCODING
    try:
        if _TIKTOKEN_ENCODING is None:
            import tiktoken  # import paresseux : seulement pour ALBERT_TOKEN_ESTIMATOR=tiktoken

            _TIKTOKEN_ENCODING = tiktoken.get_encoding("o200k_base")
        return len(_TIKTOKEN_ENCODING.encode(text, disallowed_special=()))
    except Exception:  # encodage absent ou non téléchargeable : estimation par caractères
        return None


def estimate_input_tokens(payload: Any, *, estimator: str = "chars") -> int:
    """Estime les tokens d'entrée d'une requête (budget TPM du limiteur).

    Args:
        payload: texte, liste de messages (contenu texte ou parties
            ``{"type": "text"}``), liste de textes, ou corps de requête
            (``{"messages": ...}`` / ``{"input": ...}``). Les images ne sont
            pas comptées.
        estimator: ``chars`` (longueur / 3, arrondi supérieur) ou ``tiktoken``
            (``o200k_base`` si disponible, sinon ``chars``).

    Returns:
        Un entier >= 0.
    """
    texts: list = []
    _collect_text(payload, texts)
    text = "\n".join(texts)
    if not text:
        return 0
    if estimator == "tiktoken":
        counted = _tiktoken_count(text)
        if counted is not None:
            return counted
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Seau à jetons local
# ---------------------------------------------------------------------------
def _positive(value: Any) -> Optional[float]:
    """``float(value)`` s'il est strictement positif et fini, sinon ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _share(share: Any, process_share: Any) -> float:
    """Part du quota (``process_share`` prioritaire), bornée à ]0, 1]."""
    value = process_share if process_share is not None else share
    number = _positive(value)
    if number is None:
        return 1.0
    return min(number, 1.0)


class TokenBucket:
    """Seau à jetons thread-safe : requêtes par minute et tokens d'entrée par minute.

    Chaque budget a une capacité d'une minute de débit (``rpm × part``,
    ``tpm × part``) et se remplit en continu. ``acquire`` réserve sous verrou
    (le niveau peut devenir négatif, ce qui ordonne les appelants) puis dort
    hors verrou le temps nécessaire. Une requête plus grosse que la capacité
    TPM attend un seau plein puis passe (dette remboursée par les suivantes).

    Attributes:
        rpm: requêtes par minute **effectives** (``rpm × part`` ; ``None`` = illimité).
        tpm: tokens d'entrée par minute effectifs (``tpm × part`` ; ``None`` = illimité).
        base_rpm: requêtes par minute configurées, avant la part.
        base_tpm: tokens par minute configurés, avant la part.
        share: part du quota attribuée au processus.
        name: nom du budget (journaux).
    """

    def __init__(
        self,
        rpm: Optional[float] = None,
        tpm: Optional[float] = None,
        *,
        share: float = 1.0,
        process_share: Optional[float] = None,
        name: str = "",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = time.sleep,
    ) -> None:
        """Crée un seau plein.

        Args:
            rpm: requêtes par minute (``None`` ou <= 0 = illimité).
            tpm: tokens d'entrée par minute (``None`` ou <= 0 = illimité).
            share: part du quota (0 < part <= 1).
            process_share: synonyme de ``share`` (prioritaire s'il est fourni).
            name: nom du budget.
            clock: horloge monotone en secondes (injectable).
            sleep: fonction de sommeil (injectable).
        """
        self.base_rpm = _positive(rpm)
        self.base_tpm = _positive(tpm)
        self.share = _share(share, process_share)
        self.rpm = self.base_rpm * self.share if self.base_rpm is not None else None
        self.tpm = self.base_tpm * self.share if self.base_tpm is not None else None
        self.name = name
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        now = clock()
        self._last = now
        self._req_level = self.request_capacity or 0.0
        self._tok_level = self.token_capacity or 0.0
        self._paused_until = now

    # --- capacités ---------------------------------------------------------
    @property
    def effective_rpm(self) -> Optional[float]:
        """Requêtes par minute effectives (synonyme de ``rpm``)."""
        return self.rpm

    @property
    def effective_tpm(self) -> Optional[float]:
        """Tokens par minute effectifs (synonyme de ``tpm``)."""
        return self.tpm

    @property
    def request_capacity(self) -> Optional[float]:
        """Capacité du budget de requêtes (au moins 1)."""
        rate = self.effective_rpm
        return max(1.0, rate) if rate is not None else None

    @property
    def token_capacity(self) -> Optional[float]:
        """Capacité du budget de tokens (au moins 1)."""
        rate = self.effective_tpm
        return max(1.0, rate) if rate is not None else None

    # --- état ----------------------------------------------------------------
    def _refill(self, now: float) -> None:
        """Remplit les deux budgets pour le temps écoulé (sous verrou)."""
        elapsed = max(0.0, now - self._last)
        self._last = max(self._last, now)
        if self.effective_rpm is not None:
            self._req_level = min(self.request_capacity, self._req_level + elapsed * self.effective_rpm / WINDOW_SECONDS)
        if self.effective_tpm is not None:
            self._tok_level = min(self.token_capacity, self._tok_level + elapsed * self.effective_tpm / WINDOW_SECONDS)

    @property
    def request_level(self) -> Optional[float]:
        """Jetons de requête disponibles maintenant (négatif = réservations en attente)."""
        with self._lock:
            self._refill(self._clock())
            return self._req_level if self.effective_rpm is not None else None

    @property
    def token_level(self) -> Optional[float]:
        """Jetons de tokens disponibles maintenant (négatif = réservations en attente)."""
        with self._lock:
            self._refill(self._clock())
            return self._tok_level if self.effective_tpm is not None else None

    def reserve(self, tokens: int = 0, *, requests: int = 1) -> float:
        """Réserve une requête (et ses tokens) et renvoie l'attente nécessaire, sans dormir.

        Args:
            tokens: tokens d'entrée estimés.
            requests: unités de requête consommées.

        Returns:
            Secondes à attendre avant l'envoi (0 si immédiat).
        """
        with self._lock:
            now = self._clock()
            self._refill(now)
            waits = [max(0.0, self._paused_until - now)]
            rate_r = self.effective_rpm
            if rate_r is not None and requests > 0:
                need = min(float(requests), self.request_capacity)
                self._req_level -= float(requests)
                deficit = (need - float(requests)) - self._req_level
                waits.append(max(0.0, deficit) * WINDOW_SECONDS / rate_r)
            rate_t = self.effective_tpm
            if rate_t is not None and tokens and tokens > 0:
                need = min(float(tokens), self.token_capacity)
                self._tok_level -= float(tokens)
                deficit = (need - float(tokens)) - self._tok_level
                waits.append(max(0.0, deficit) * WINDOW_SECONDS / rate_t)
            return max(waits)

    def acquire(self, tokens: int = 0, *, requests: int = 1) -> float:
        """Attend (hors verrou) que le débit autorise une requête de ``tokens`` tokens.

        Args:
            tokens: tokens d'entrée estimés.
            requests: unités de requête consommées.

        Returns:
            Secondes attendues.
        """
        wait = self.reserve(tokens, requests=requests)
        if wait > 0:
            self._sleep(wait)
        return wait

    async def aacquire(self, tokens: int = 0, *, requests: int = 1) -> float:
        """``acquire`` exécuté dans un thread (``asyncio.to_thread``)."""
        return await asyncio.to_thread(self.acquire, tokens, requests=requests)

    def pause(self, retry_after: Optional[float] = None) -> float:
        """Suspend le débit après un 429.

        Args:
            retry_after: durée (secondes) ; ``None`` ou <= 0 = le temps d'une
                requête au débit courant (1 s au moins).

        Returns:
            La durée de pause appliquée.
        """
        seconds = _positive(retry_after)
        if seconds is None:
            rate = self.effective_rpm
            seconds = max(1.0, WINDOW_SECONDS / rate) if rate else 1.0
        with self._lock:
            now = self._clock()
            self._paused_until = max(self._paused_until, now + seconds)
        logger.warning("Limiteur Albert %s : pause de %.1f s après un 429.", self.name or "-", seconds)
        return seconds

    def __repr__(self) -> str:
        """Représentation lisible (débits effectifs)."""
        return f"TokenBucket(name={self.name!r}, rpm={self.effective_rpm}, tpm={self.effective_tpm})"


class NullLimiter:
    """Limiteur sans effet (rôle sans budget)."""

    name = "none"

    def acquire(self, tokens: int = 0, *, requests: int = 1) -> float:
        """N'attend jamais."""
        return 0.0

    async def aacquire(self, tokens: int = 0, *, requests: int = 1) -> float:
        """N'attend jamais."""
        return 0.0

    def pause(self, retry_after: Optional[float] = None) -> float:
        """Sans effet."""
        return 0.0


# ---------------------------------------------------------------------------
# Fenêtres Redis partagées
# ---------------------------------------------------------------------------
def resolve_redis_url(cfg: Any = None, env: Optional[Mapping[str, str]] = None) -> str:
    """URL Redis du limiteur : ``ALBERT_LIMITER_REDIS_URL``, puis ``CELERY_BROKER_URL``, puis défaut.

    Args:
        cfg: ``AlbertConfig`` (champ ``limiter_redis_url``) ou ``None``.
        env: environnement à lire pour ``CELERY_BROKER_URL`` (défaut ``os.environ``).

    Returns:
        L'URL retenue.
    """
    configured = getattr(cfg, "limiter_redis_url", None) if cfg is not None else None
    if configured and str(configured).strip():
        return str(configured).strip()
    source = os.environ if env is None else env
    broker = source.get("CELERY_BROKER_URL")
    if broker and str(broker).strip():
        return str(broker).strip()
    return DEFAULT_REDIS_URL


class RedisWindowLimiter:
    """Limiteur à fenêtres fixes d'une minute partagé par Redis (tous processus).

    Les compteurs ``INCRBY`` + ``EXPIRE`` sont indexés par minute ; une
    requête qui dépasserait le budget annule sa réservation et attend la
    fenêtre suivante. ``pause`` pose une échéance commune à tous les
    processus. Le premier appel de la fenêtre passe toujours (une requête
    plus grosse que le budget TPM ne bloque pas indéfiniment).

    Attributes:
        name: nom du budget (partie des clés Redis ; même nom = fenêtre partagée).
        url: URL Redis (le client n'est créé qu'au premier usage).
        rpm, tpm, base_rpm, base_tpm, share: comme ``TokenBucket``.
    """

    def __init__(
        self,
        rpm: Optional[float] = None,
        tpm: Optional[float] = None,
        *,
        name: str = "default",
        share: float = 1.0,
        process_share: Optional[float] = None,
        url: Optional[str] = None,
        client: Any = None,
        redis_client: Any = None,
        prefix: str = REDIS_PREFIX,
        window: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Any] = time.sleep,
    ) -> None:
        """Prépare le limiteur (aucune connexion avant le premier ``acquire``).

        Args:
            rpm: requêtes par minute (``None`` = illimité).
            tpm: tokens d'entrée par minute (``None`` = illimité).
            name: nom du budget (partie des clés).
            share: part du quota (0 < part <= 1).
            process_share: synonyme de ``share``.
            url: URL Redis (défaut : ``resolve_redis_url()``).
            client: client Redis déjà construit (tests : ``FakeRedis``).
            redis_client: synonyme de ``client``.
            prefix: préfixe des clés.
            window: durée d'une fenêtre, en secondes.
            clock: horloge murale partagée (``time.time``).
            sleep: fonction de sommeil.
        """
        self.base_rpm = _positive(rpm)
        self.base_tpm = _positive(tpm)
        self.share = _share(share, process_share)
        self.rpm = self.base_rpm * self.share if self.base_rpm is not None else None
        self.tpm = self.base_tpm * self.share if self.base_tpm is not None else None
        self.name = str(name)
        self.prefix = prefix
        self.window = float(window) if _positive(window) else WINDOW_SECONDS
        self.url = url or resolve_redis_url()
        self._client = client if client is not None else redis_client
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._fallback: Optional[TokenBucket] = None

    @property
    def request_limit(self) -> Optional[int]:
        """Requêtes autorisées par fenêtre (``None`` = illimité)."""
        return max(1, int(self.rpm)) if self.rpm is not None else None

    @property
    def token_limit(self) -> Optional[int]:
        """Tokens autorisés par fenêtre (``None`` = illimité)."""
        return max(1, int(self.tpm)) if self.tpm is not None else None

    def _redis(self) -> Any:
        """Client Redis (import paresseux de ``redis`` au premier usage)."""
        with self._lock:
            if self._client is None:
                import redis  # import paresseux : backend optionnel

                self._client = redis.from_url(self.url)
            return self._client

    def _key(self, kind: str, index: Optional[int] = None) -> str:
        """Clé Redis d'un compteur (``req``/``tok``) ou de la pause."""
        base = f"{self.prefix}:{self.name}:{kind}"
        return base if index is None else f"{base}:{index}"

    def _degrade(self, exc: Exception) -> TokenBucket:
        """Bascule sur un seau local (Redis injoignable), avec un seul WARNING."""
        with self._lock:
            if self._fallback is None:
                logger.warning(
                    "Limiteur Albert %s : Redis indisponible (%s) ; repli sur un limiteur local.",
                    self.name,
                    type(exc).__name__,
                )
                self._fallback = TokenBucket(
                    self.base_rpm, self.base_tpm, share=self.share, name=self.name, sleep=self._sleep
                )
            return self._fallback

    def _paused_until(self, client: Any) -> float:
        """Échéance de pause partagée (0 si aucune)."""
        raw = client.get(self._key("pause"))
        if raw is None:
            return 0.0
        try:
            return float(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except (TypeError, ValueError):
            return 0.0

    def _count(self, client: Any, key: str, amount: int, ttl: int) -> int:
        """``INCRBY`` + ``EXPIRE`` dans un pipeline ; renvoie le nouveau total."""
        pipe = client.pipeline()
        pipe.incrby(key, amount)
        pipe.expire(key, ttl)
        result = pipe.execute()
        return int(result[0])

    def acquire(self, tokens: int = 0, *, requests: int = 1) -> float:
        """Attend qu'une requête de ``tokens`` tokens tienne dans la fenêtre courante.

        Args:
            tokens: tokens d'entrée estimés.
            requests: unités de requête consommées.

        Returns:
            Secondes attendues.
        """
        if self._fallback is not None:
            return self._fallback.acquire(tokens, requests=requests)
        waited = 0.0
        ttl = int(math.ceil(self.window * 2))
        while True:
            try:
                client = self._redis()
                now = self._clock()
                pause_left = self._paused_until(client) - now
                if pause_left > 0:
                    self._sleep(pause_left)
                    waited += pause_left
                    continue
                index = int(now // self.window)
                counted = []
                allowed = True
                if self.request_limit is not None and requests > 0:
                    key = self._key("req", index)
                    total = self._count(client, key, int(requests), ttl)
                    counted.append((key, int(requests)))
                    if total > self.request_limit and total != int(requests):
                        allowed = False
                if allowed and self.token_limit is not None and tokens and tokens > 0:
                    key = self._key("tok", index)
                    total = self._count(client, key, int(tokens), ttl)
                    counted.append((key, int(tokens)))
                    if total > self.token_limit and total != int(tokens):
                        allowed = False
                if allowed:
                    return waited
                for key, amount in counted:
                    client.decr(key, amount)
                wait = max(0.01, (index + 1) * self.window - now + 0.01)
            except ImportError as exc:
                return waited + self._degrade(exc).acquire(tokens, requests=requests)
            except Exception as exc:  # erreurs de connexion Redis : repli local, jamais d'échec du job
                if type(exc).__module__.split(".")[0] != "redis" and not isinstance(exc, (OSError, ConnectionError)):
                    raise
                return waited + self._degrade(exc).acquire(tokens, requests=requests)
            self._sleep(wait)
            waited += wait

    async def aacquire(self, tokens: int = 0, *, requests: int = 1) -> float:
        """``acquire`` exécuté dans un thread (``asyncio.to_thread``)."""
        return await asyncio.to_thread(self.acquire, tokens, requests=requests)

    def pause(self, retry_after: Optional[float] = None) -> float:
        """Pose une pause commune à tous les processus après un 429.

        Args:
            retry_after: durée (secondes) ; ``None`` = une seconde.

        Returns:
            La durée de pause appliquée.
        """
        seconds = _positive(retry_after) or 1.0
        if self._fallback is not None:
            return self._fallback.pause(seconds)
        try:
            client = self._redis()
            until = self._clock() + seconds
            if until > self._paused_until(client):
                client.set(self._key("pause"), repr(until), ex=int(math.ceil(seconds)) + 1)
        except Exception as exc:  # Redis injoignable : pause locale
            return self._degrade(exc).pause(seconds)
        logger.warning("Limiteur Albert %s : pause partagée de %.1f s après un 429.", self.name, seconds)
        return seconds

    def __repr__(self) -> str:
        """Représentation lisible (sans l'URL, qui peut contenir un mot de passe)."""
        return f"RedisWindowLimiter(name={self.name!r}, rpm={self.request_limit}, tpm={self.token_limit})"


# ---------------------------------------------------------------------------
# Instances partagées par rôle
# ---------------------------------------------------------------------------
_REGISTRY: Dict[Tuple[Any, ...], Any] = {}
_REGISTRY_LOCK = threading.Lock()


def bucket_for_role(role: Any) -> Optional[str]:
    """Budget du limiteur d'un rôle (``None`` : rôle sans limiteur)."""
    key = str(role or "").strip().lower()
    return ROLE_BUCKETS.get(key)


def role_rates(role: Any, cfg: Any) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """(budget, rpm, tpm) configurés pour un rôle d'après ``cfg``."""
    bucket = bucket_for_role(role)
    if bucket == "recode":
        return bucket, getattr(cfg, "recode_rpm", None), getattr(cfg, "chat_tpm", None)
    if bucket == "notes":
        return bucket, getattr(cfg, "notes_rpm", None), getattr(cfg, "chat_tpm", None)
    if bucket == "ocr":
        return bucket, getattr(cfg, "ocr_rpm", None), None
    if bucket == "embed":
        return bucket, getattr(cfg, "embed_rpm", None), None
    return None, None, None


def get_limiter(
    role: Any,
    cfg: Any = None,
    *,
    clock: Optional[Callable[[], float]] = None,
    sleep: Optional[Callable[[float], Any]] = None,
    redis_client: Any = None,
    env: Optional[Mapping[str, str]] = None,
) -> Any:
    """Limiteur du rôle ; l'instance est partagée par tout le processus.

    Args:
        role: rôle (``recode``, ``citation``, ``notes``, ``book_structure``,
            ``long_context``, ``ocr_chat``, ``ocr_doc``, ``embed``, ``push``…).
        cfg: ``AlbertConfig`` (défaut : ``AlbertConfig.from_env()``).
        clock: horloge injectée (instance dédiée, non partagée).
        sleep: sommeil injecté (instance dédiée, non partagée).
        redis_client: client Redis injecté (instance dédiée, non partagée).
        env: environnement pour ``CELERY_BROKER_URL`` (backend redis).

    Returns:
        ``TokenBucket``, ``RedisWindowLimiter`` ou ``NullLimiter`` (rôle sans budget).
    """
    if cfg is None:
        from .config import AlbertConfig

        cfg = AlbertConfig.from_env()
    bucket, rpm, tpm = role_rates(role, cfg)
    if bucket is None:
        return NullLimiter()
    share = getattr(cfg, "process_share", 1.0)
    backend = getattr(cfg, "limiter_backend", "local")
    url = resolve_redis_url(cfg, env=env) if backend == "redis" else None
    dedicated = clock is not None or sleep is not None or redis_client is not None
    key = (backend, bucket, rpm, tpm, share, url)
    if not dedicated:
        with _REGISTRY_LOCK:
            existing = _REGISTRY.get(key)
            if existing is not None:
                return existing
    if backend == "redis":
        kwargs: Dict[str, Any] = {"share": share, "url": url, "client": redis_client}
        if clock is not None:
            kwargs["clock"] = clock
        if sleep is not None:
            kwargs["sleep"] = sleep
        limiter: Any = RedisWindowLimiter(rpm, tpm, name=bucket, **kwargs)
    else:
        kwargs = {"share": share, "name": bucket}
        if clock is not None:
            kwargs["clock"] = clock
        if sleep is not None:
            kwargs["sleep"] = sleep
        limiter = TokenBucket(rpm, tpm, **kwargs)
    if dedicated:
        return limiter
    with _REGISTRY_LOCK:
        return _REGISTRY.setdefault(key, limiter)


def reset_limiters() -> None:
    """Oublie les limiteurs partagés du processus (tests, changement de configuration)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


__all__ = [
    "DEFAULT_REDIS_URL",
    "ROLE_BUCKETS",
    "NullLimiter",
    "RedisWindowLimiter",
    "TokenBucket",
    "bucket_for_role",
    "estimate_input_tokens",
    "get_limiter",
    "reset_limiters",
    "resolve_redis_url",
    "role_rates",
]
