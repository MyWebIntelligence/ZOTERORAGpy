"""Limiteur proactif de débit Albert, par rôle — stdlib (redis importé paresseusement).

Deux budgets par rôle, multipliés par ``ALBERT_PROCESS_SHARE`` :

* requêtes par minute (``ALBERT_RECODE_RPM``, ``_NOTES_RPM``, ``_OCR_RPM``,
  ``_EMBED_RPM``) ;
* tokens d'entrée par minute pour les rôles de chat (``ALBERT_CHAT_TPM``),
  estimés à ``len/3`` (``estimate_input_tokens``).

Backends :

* ``TokenBucket`` (``local``, défaut) : seau à jetons thread-safe, horloge et
  sommeil injectables ; chaque appelant réserve sa part sous verrou puis dort
  **hors** verrou. ``pause(retry_after)`` suspend le débit après un 429. Le
  niveau accumulé ne dépasse jamais la rafale (``burst`` requêtes ;
  ``token_burst`` tokens, la même fraction de minute), même après une
  inactivité : au plus ``rpm × part + burst − 1`` requêtes et
  ``tpm × part + token_burst − 1`` tokens sur toute minute glissante. La
  rafale de départ est prise sur le débit de la première minute : jamais plus
  de ``rpm × part`` requêtes ni ``tpm × part`` tokens sur la première minute
  d'un processus. Une requête plus grosse que ``token_burst`` part dès que le
  seau contient cette rafale : les bornes de tokens ne sont alors dépassées
  que de son propre excédent.
* ``RedisWindowLimiter`` (``redis``) : fenêtres fixes d'une minute partagées
  entre processus (Celery, plusieurs sessions) ; URL ``ALBERT_LIMITER_REDIS_URL``,
  sinon ``CELERY_BROKER_URL``, sinon ``redis://localhost:6379/0``. Redis
  indisponible : repli sur un seau local, avec un seul WARNING par panne ;
  Redis est retenté après ``REDIS_RETRY_SECONDS`` (jamais de repli définitif).

``get_limiter(role, cfg, model=...)`` renvoie l'instance partagée du processus
pour le budget de l'appel : celui du rôle, sauf pour un modèle à raisonnement
(gpt-oss) sous un rôle du budget ``recode``, qui consomme le budget ``notes``
(``limiter_role``). Le seau démarre presque vide : sa rafale vaut la
concurrence configurée du budget (``bucket_burst``), au plus une minute de débit,
et plafonne aussi le niveau accumulé pendant une inactivité.

En asynchrone, passer par ``retry.acall_with_retry`` : un seau local y est
réservé (``reserve``) puis attendu par ``asyncio.sleep`` ; un limiteur Redis,
et la pause posée après un 429, passent par l'exécuteur dédié
``albert-limiter``. La boucle d'événements n'est jamais bloquée et aucun fil de
l'exécuteur par défaut n'est occupé par une attente de débit.
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

REDIS_RETRY_SECONDS = 30.0
"""Délai (secondes) avant de retenter Redis après une panne ; seau local entre-temps."""

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
    "answer": "recode",
    "vision": "recode",
    "audio": "audio",
}

# Budget → champ de ``AlbertConfig`` donnant sa rafale de départ (concurrence du budget).
BUCKET_CONCURRENCY: Mapping[str, str] = {
    "recode": "recode_concurrency",
    "notes": "notes_concurrency",
    "ocr": "ocr_concurrency",
    "embed": "embed_concurrency",
    "audio": "audio_concurrency",
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


def _repay(level: float, advance: float, gain: float, ceiling: float) -> Tuple[float, float]:
    """Applique un remplissage ``gain`` : l'avance de départ est remboursée d'abord.

    Args:
        level: niveau courant du budget.
        advance: avance restant à rembourser (rafale de départ prise sur la première minute).
        gain: jetons apportés par le temps écoulé.
        ceiling: plafond du niveau accumulé (la rafale du budget).

    Returns:
        ``(niveau, avance)`` après remplissage (niveau plafonné à ``ceiling``).
    """
    paid = min(advance, gain)
    return min(ceiling, level + (gain - paid)), advance - paid


def _wait_for(deficit: float, advance: float, rate: float) -> float:
    """Attente (secondes) d'une réservation : 0 si le niveau la couvre, sinon avance puis déficit.

    Args:
        deficit: jetons manquants après la réservation (<= 0 : aucun).
        advance: avance de départ restant à rembourser avant tout nouveau jeton.
        rate: débit du budget, en jetons par minute.

    Returns:
        Secondes d'attente.
    """
    if deficit <= 0:
        return 0.0
    return (deficit + advance) * WINDOW_SECONDS / rate


class TokenBucket:
    """Seau à jetons thread-safe : requêtes par minute et tokens d'entrée par minute.

    Chaque budget se remplit en continu au débit d'une minute (``rpm × part``,
    ``tpm × part``) ; son niveau accumulé est plafonné à sa **rafale** :
    ``burst`` requêtes (au plus une minute de débit ; défaut : une minute) et
    ``token_burst`` tokens, la même fraction de minute
    (``burst × tpm / rpm`` ; une minute si le débit de requêtes est illimité).
    ``acquire`` réserve sous verrou (le niveau peut devenir négatif, ce qui
    ordonne les appelants) puis dort hors verrou le temps nécessaire. Une
    requête plus grosse que la rafale de tokens attend que le seau contienne
    cette rafale puis passe (dette remboursée par les suivantes).

    Borne : même après une inactivité, le seau ne contient jamais plus que sa
    rafale ; toute fenêtre glissante de 60 s admet donc au plus
    ``rpm × part + burst − 1`` requêtes et ``tpm × part + token_burst − 1``
    tokens (une requête plus grosse que ``token_burst`` ne dépasse les bornes
    de tokens, ici et au départ, que de son propre excédent).

    Départ : la rafale part sans attente, mais elle est une **avance** sur la
    première minute : le remplissage ne reprend qu'une fois l'avance
    remboursée (au-delà d'une requête pour le budget de requêtes). Sur débit
    saturé dès le départ, la première minute n'admet donc jamais plus de
    ``rpm × part`` requêtes ni ``tpm × part`` tokens ; ensuite, un jeton toutes
    les ``60 / (rpm × part)`` secondes.

    Attributes:
        rpm: requêtes par minute **effectives** (``rpm × part`` ; ``None`` = illimité).
        tpm: tokens d'entrée par minute effectifs (``tpm × part`` ; ``None`` = illimité).
        base_rpm: requêtes par minute configurées, avant la part.
        base_tpm: tokens par minute configurés, avant la part.
        share: part du quota attribuée au processus.
        burst: rafale de requêtes : admises d'emblée au départ et plafond du
            niveau accumulé (``None`` si le débit est illimité).
        token_burst: rafale de tokens, plafond du niveau de tokens accumulé
            (``None`` si le débit de tokens est illimité).
        name: nom du budget (journaux).
    """

    def __init__(
        self,
        rpm: Optional[float] = None,
        tpm: Optional[float] = None,
        *,
        share: float = 1.0,
        process_share: Optional[float] = None,
        burst: Optional[float] = None,
        name: str = "",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = time.sleep,
    ) -> None:
        """Crée un seau plafonné à sa rafale, rafale de départ prise sur la première minute.

        Args:
            rpm: requêtes par minute (``None`` ou <= 0 = illimité).
            tpm: tokens d'entrée par minute (``None`` ou <= 0 = illimité).
            share: part du quota (0 < part <= 1).
            process_share: synonyme de ``share`` (prioritaire s'il est fourni).
            burst: requêtes admises sans attente au départ et plafond du niveau
                accumulé, entre 1 et la capacité (``None`` ou <= 0 : une minute
                de débit) ; la rafale de tokens en est la même fraction de minute.
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
        capacity = self.request_capacity
        wanted = _positive(burst)
        if capacity is None:
            self.burst: Optional[float] = None
        else:
            self.burst = capacity if wanted is None else min(capacity, max(1.0, wanted))
        self.token_burst = self._token_burst()
        # Rafale de départ = avance sur la première minute (au-delà d'une requête ; tous
        # les tokens) : remboursée par le remplissage avant tout nouveau jeton.
        self._req_level = self.burst or 0.0
        self._req_advance = max(0.0, self._req_level - 1.0)
        self._tok_level = self.token_burst or 0.0
        self._tok_advance = self._tok_level
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

    def _token_burst(self) -> Optional[float]:
        """Rafale de tokens : la même fraction de minute que la rafale de requêtes.

        Returns:
            ``token_capacity × burst / request_capacity``, entre 1 et la
            capacité de tokens (une minute de tokens si le débit de requêtes
            est illimité) ; ``None`` si le débit de tokens est illimité.
        """
        capacity = self.token_capacity
        requests = self.request_capacity
        if capacity is None:
            return None
        if requests is None or self.burst is None:
            return capacity
        return min(capacity, max(1.0, capacity * self.burst / requests))

    # --- état ----------------------------------------------------------------
    def _refill(self, now: float) -> None:
        """Remplit les deux budgets pour le temps écoulé, avance de départ remboursée d'abord (sous verrou).

        Le niveau accumulé est plafonné à la rafale du budget (``burst``,
        ``token_burst``), jamais à une minute entière : une inactivité ne
        laisse pas repartir plus que la rafale d'un coup.
        """
        elapsed = max(0.0, now - self._last)
        self._last = max(self._last, now)
        if self.effective_rpm is not None:
            self._req_level, self._req_advance = _repay(
                self._req_level, self._req_advance, elapsed * self.effective_rpm / WINDOW_SECONDS,
                self.burst,
            )
        if self.effective_tpm is not None:
            self._tok_level, self._tok_advance = _repay(
                self._tok_level, self._tok_advance, elapsed * self.effective_tpm / WINDOW_SECONDS,
                self.token_burst,
            )

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
                need = min(float(requests), self.burst)
                self._req_level -= float(requests)
                deficit = (need - float(requests)) - self._req_level
                waits.append(_wait_for(deficit, self._req_advance, rate_r))
            rate_t = self.effective_tpm
            if rate_t is not None and tokens and tokens > 0:
                need = min(float(tokens), self.token_burst)
                self._tok_level -= float(tokens)
                deficit = (need - float(tokens)) - self._tok_level
                waits.append(_wait_for(deficit, self._tok_advance, rate_t))
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
        """``acquire`` exécuté dans un thread (``asyncio.to_thread``).

        Le fil de l'exécuteur par défaut reste occupé pendant toute l'attente :
        les appels Albert passent plutôt par ``retry.acall_with_retry``.
        """
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

    Redis injoignable : repli sur un seau local (un seul WARNING par panne),
    puis nouvelle tentative Redis toutes les ``retry_interval`` secondes ;
    dès que Redis répond, la fenêtre partagée reprend et le seau local est
    abandonné (un INFO le signale). Avec ``require_redis``
    (``ALBERT_LIMITER_REQUIRE_REDIS=1``), aucun seau local : la panne lève
    ``AlbertQuotaExhausted`` (arrêt explicite du job, aucun appel envoyé), car
    un seau par processus ne garantit pas le plafond du compte.

    Attributes:
        name: nom du budget (partie des clés Redis ; même nom = fenêtre partagée).
        url: URL Redis (le client n'est créé qu'au premier usage).
        retry_interval: délai (secondes) avant de retenter Redis après un échec.
        burst: rafale de départ du seau local de repli (``None`` = une minute de débit).
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
        retry_interval: float = REDIS_RETRY_SECONDS,
        burst: Optional[float] = None,
        require_redis: bool = False,
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
            retry_interval: délai (secondes) avant de retenter Redis après un
                échec (défaut ``REDIS_RETRY_SECONDS``).
            burst: rafale de départ du seau local de repli (voir ``TokenBucket``).
            require_redis: refuse tout repli local (panne Redis = arrêt explicite).
        """
        self.require_redis = bool(require_redis)
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
        self.retry_interval = _positive(retry_interval) or REDIS_RETRY_SECONDS
        self._retry_at = 0.0
        self.burst = _positive(burst)

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
        """Bascule sur un seau local (Redis injoignable) jusqu'à la prochaine tentative Redis.

        Un seul WARNING par panne : un nouvel échec de la tentative suivante
        garde le même seau local (son état est conservé) et n'est journalisé
        qu'en DEBUG.

        Raises:
            AlbertQuotaExhausted: ``require_redis`` est vrai (aucun repli local).
        """
        if self.require_redis:
            from .errors import AlbertQuotaExhausted  # import paresseux : module stdlib

            logger.error(
                "Limiteur Albert %s : Redis indisponible (%s) et ALBERT_LIMITER_REQUIRE_REDIS=1 : "
                "aucun nouvel appel Albert.", self.name, type(exc).__name__,
            )
            raise AlbertQuotaExhausted(
                "Limiteur partagé Albert indisponible (Redis injoignable) et "
                "ALBERT_LIMITER_REQUIRE_REDIS=1 : aucun nouvel appel n'est envoyé ; "
                "rétablir Redis puis relancer le traitement.",
                endpoint=None,
            ) from exc
        with self._lock:
            if self._fallback is None:
                logger.warning(
                    "Limiteur Albert %s : Redis indisponible (%s) ; repli sur un limiteur local, "
                    "nouvelle tentative dans %.0f s.",
                    self.name,
                    type(exc).__name__,
                    self.retry_interval,
                )
                self._fallback = TokenBucket(
                    self.base_rpm, self.base_tpm, share=self.share, burst=self.burst, name=self.name,
                    clock=self._clock, sleep=self._sleep,
                )
            else:
                logger.debug(
                    "Limiteur Albert %s : Redis toujours indisponible (%s).", self.name, type(exc).__name__
                )
            self._retry_at = self._clock() + self.retry_interval
            return self._fallback

    def _local_bucket(self) -> Optional[TokenBucket]:
        """Seau local à utiliser maintenant (``None`` : Redis joignable ou à retenter)."""
        with self._lock:
            if self._fallback is None or self._clock() >= self._retry_at:
                return None
            return self._fallback

    def _recovered(self) -> None:
        """Redis a répondu : abandonne le seau local de repli (un INFO si une panne était en cours)."""
        with self._lock:
            if self._fallback is None:
                return
            self._fallback = None
            self._retry_at = 0.0
        logger.info("Limiteur Albert %s : Redis de nouveau joignable ; fin du repli local.", self.name)

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
        local = self._local_bucket()
        if local is not None:
            return local.acquire(tokens, requests=requests)
        waited = 0.0
        ttl = int(math.ceil(self.window * 2))
        while True:
            try:
                client = self._redis()
                now = self._clock()
                pause_left = self._paused_until(client) - now
                self._recovered()
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
        """``acquire`` exécuté dans un thread (``asyncio.to_thread``).

        Le fil de l'exécuteur par défaut reste occupé pendant toute l'attente :
        les appels Albert passent plutôt par ``retry.acall_with_retry``.
        """
        return await asyncio.to_thread(self.acquire, tokens, requests=requests)

    def pause(self, retry_after: Optional[float] = None) -> float:
        """Pose une pause commune à tous les processus après un 429.

        Args:
            retry_after: durée (secondes) ; ``None`` = une seconde.

        Returns:
            La durée de pause appliquée.
        """
        seconds = _positive(retry_after) or 1.0
        local = self._local_bucket()
        if local is not None:
            return local.pause(seconds)
        try:
            client = self._redis()
            until = self._clock() + seconds
            if until > self._paused_until(client):
                client.set(self._key("pause"), repr(until), ex=int(math.ceil(seconds)) + 1)
            self._recovered()
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


def limiter_role(role: Any, model: Any = None) -> Any:
    """Rôle dont le budget s'applique à un appel de ``model`` sous ``role``.

    Le budget suit le modèle envoyé : un modèle à raisonnement (gpt-oss, 10 RPM
    mesurés, D1) consomme le budget ``notes`` même sous un rôle du budget
    ``recode`` (``recode``, ``citation``, ``book_structure``, ``long_context``,
    ``chat``, ``answer``, ``vision``). Tous les autres cas gardent le rôle donné : un repli ministral
    du rôle ``notes`` reste sur ``notes`` ; OCR et embeddings ne changent jamais.

    Args:
        role: rôle applicatif.
        model: modèle envoyé (id, alias ou ``albert/<id>``) ; ``None`` = rôle seul.

    Returns:
        ``"notes"`` pour un modèle à raisonnement sous un rôle du budget
        ``recode``, sinon ``role`` inchangé.
    """
    name = str(model).strip() if model is not None else ""
    if name[:7].lower() == "albert/":
        name = name[7:].strip()
    if not name or bucket_for_role(role) != "recode":
        return role
    from . import catalog  # import paresseux : le limiteur reste importable seul

    return "notes" if catalog.is_reasoning(name) else role


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
    if bucket == "audio":
        return bucket, getattr(cfg, "audio_rpm", None), None
    return None, None, None


def bucket_burst(bucket: Any, cfg: Any) -> float:
    """Rafale de départ d'un budget : sa concurrence configurée (1 si inconnue).

    Les envois parallèles d'un processus partent ensemble au démarrage, sans
    que la première minute dépasse le débit réglé (``TokenBucket`` plafonne la
    rafale à une minute de débit et la prend sur la première minute). La
    rafale plafonne aussi le niveau accumulé pendant une inactivité : au plus
    ``rpm × part + rafale − 1`` requêtes sur toute minute glissante.

    Args:
        bucket: budget (``recode``, ``notes``, ``ocr``, ``embed``).
        cfg: ``AlbertConfig`` (champs ``*_concurrency``) ou objet équivalent.

    Returns:
        Le nombre de requêtes admises d'emblée (au moins 1).
    """
    field = BUCKET_CONCURRENCY.get(str(bucket or ""))
    value = _positive(getattr(cfg, field, None)) if field else None
    return value if value is not None else 1.0


def get_limiter(
    role: Any,
    cfg: Any = None,
    *,
    clock: Optional[Callable[[], float]] = None,
    sleep: Optional[Callable[[float], Any]] = None,
    redis_client: Any = None,
    env: Optional[Mapping[str, str]] = None,
    model: Any = None,
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
        model: modèle envoyé ; un modèle à raisonnement sous un rôle du budget
            ``recode`` prend le budget ``notes`` (``limiter_role``).

    Le seau démarre presque vide : sa rafale vaut la concurrence du budget
    (``bucket_burst``), prise sur le débit de la première minute, et plafonne
    le niveau accumulé pendant une inactivité (seau de repli d'un limiteur
    Redis compris).

    Returns:
        ``TokenBucket``, ``RedisWindowLimiter`` ou ``NullLimiter`` (rôle sans budget).
    """
    role = limiter_role(role, model)
    if cfg is None:
        from .config import AlbertConfig

        cfg = AlbertConfig.from_env()
    bucket, rpm, tpm = role_rates(role, cfg)
    if bucket is None:
        return NullLimiter()
    share = getattr(cfg, "process_share", 1.0)
    burst = bucket_burst(bucket, cfg)
    backend = getattr(cfg, "limiter_backend", "local")
    url = resolve_redis_url(cfg, env=env) if backend == "redis" else None
    dedicated = clock is not None or sleep is not None or redis_client is not None
    require_redis = bool(getattr(cfg, "limiter_require_redis", False)) and backend == "redis"
    key = (backend, bucket, rpm, tpm, share, url, burst, require_redis)
    if not dedicated:
        with _REGISTRY_LOCK:
            existing = _REGISTRY.get(key)
            if existing is not None:
                return existing
    if backend == "redis":
        kwargs: Dict[str, Any] = {
            "share": share, "url": url, "client": redis_client, "burst": burst, "require_redis": require_redis,
        }
        if clock is not None:
            kwargs["clock"] = clock
        if sleep is not None:
            kwargs["sleep"] = sleep
        limiter: Any = RedisWindowLimiter(rpm, tpm, name=bucket, **kwargs)
    else:
        kwargs = {"share": share, "name": bucket, "burst": burst}
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
    "BUCKET_CONCURRENCY",
    "DEFAULT_REDIS_URL",
    "REDIS_RETRY_SECONDS",
    "ROLE_BUCKETS",
    "NullLimiter",
    "RedisWindowLimiter",
    "TokenBucket",
    "bucket_burst",
    "bucket_for_role",
    "estimate_input_tokens",
    "get_limiter",
    "limiter_role",
    "reset_limiters",
    "resolve_redis_url",
    "role_rates",
]
