"""Couche unique de réessai du chemin Albert — stdlib seulement.

* ``parse_retry_after`` : portage de ``_parse_retry_after`` (``rad_dataframe``) ;
  lit ``Retry-After`` en secondes **et** en date HTTP, avec un plafond optionnel.
  Défini une seule fois dans ``errors.py`` (qui s'en sert pour classer les
  réponses) et réexporté ici.
* ``backoff_seconds`` : portage paramétré de ``_mistral_retry_wait`` ; un
  ``Retry-After`` est respecté sans gigue, sinon backoff exponentiel plafonné
  (gigue optionnelle, uniquement sur ce chemin, comme le chemin Mistral).
* ``RetryPolicy`` : paramètres figés (``ALBERT_MAX_RETRIES``, backoff, plafond
  de ``Retry-After``, essais courts sur 503, mode ``single_attempt``).
* ``call_with_retry`` / ``acall_with_retry`` : boucle de réessai. Ordre de
  chaque essai : acquisition du limiteur **hors** sémaphore, sémaphore(s)
  tenu(s) **pendant l'envoi seulement**, sommeil **après** libération. En
  asynchrone, la boucle d'événements n'est jamais bloquée et l'exécuteur par
  défaut n'est jamais occupé par une attente de débit : un seau local
  (``TokenBucket``) est réservé sans attendre puis attendu par
  ``asyncio.sleep`` ; les autres limiteurs (Redis) sont acquis, et toute pause
  après un 429 posée, dans un exécuteur dédié (``albert-limiter``).

Classes d'erreur (``errors.py``) : seules ``AlbertTransientError`` et
``AlbertModelBusy`` sont retentées, et seulement si leur attribut ``retryable``
est vrai (``AlbertUncertainWriteError`` : création à l'issue inconnue, jamais
renvoyée). Les erreurs de compte (``AlbertAuthError``, ``AlbertQuotaExhausted``),
permanentes et de troncature remontent au premier essai. Un 429 qui persiste
au-delà des réessais devient ``AlbertQuotaExhausted`` ; sans ``Retry-After``
(en-tête facultatif chez Albert), seulement une fois qu'une fenêtre de débit
complète (``RATE_WINDOW_SECONDS``) s'est écoulée depuis le premier 429, pour
qu'un dépassement par minute ne soit jamais pris pour un quota journalier.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Sequence, TypeVar, Union

from .errors import AlbertModelBusy, AlbertQuotaExhausted, AlbertTransientError, parse_retry_after
from .limiter import NullLimiter, TokenBucket

logger = logging.getLogger(__name__)

T = TypeVar("T")

BUSY_MAX_WAIT = 10.0
"""Plafond (secondes) des attentes courtes entre deux 503 « Model is too busy »."""

JITTER_FRACTION = 0.25
JITTER_MAX = 5.0

RATE_WINDOW_SECONDS = 60.0
"""Fenêtre des compteurs de débit d'Albert (RPM/TPM : fenêtre fixe d'une minute).

Un 429 sans ``Retry-After`` n'est déclaré quota épuisé qu'après que les
attentes cumulées depuis le premier 429 ont couvert cette fenêtre (plus
``RATE_WINDOW_MARGIN``) : un dépassement par minute s'est alors résorbé."""

RATE_WINDOW_MARGIN = 1.0
"""Marge (secondes) ajoutée à la fenêtre pour que l'essai suive sa remise à zéro."""

_REAL_SLEEP = time.sleep
"""Sommeil réel : un seau local qui l'utilise peut être attendu par ``asyncio.sleep``."""


# ---------------------------------------------------------------------------
# Retry-After et backoff
# ---------------------------------------------------------------------------
def backoff_seconds(
    attempt: int,
    retry_after: Optional[float] = None,
    *,
    base: float = 2.0,
    max_backoff: float = 60.0,
    retry_after_cap: Optional[float] = None,
    jitter: bool = False,
    rng: Any = None,
    policy: Optional["RetryPolicy"] = None,
) -> float:
    """Délai (secondes) avant le réessai qui suit l'essai ``attempt`` (0 = premier).

    Portage paramétré de ``_mistral_retry_wait`` (résultat identique sans
    gigue) : un ``retry_after`` > 0 est respecté, plafonné à
    ``retry_after_cap`` (défaut : ``max_backoff``), sans gigue ; sinon
    ``base * 2**attempt`` plafonné à ``max_backoff``, plus, si ``jitter`` est
    vrai, la gigue du chemin Mistral ``uniform(0, min(25 %, 5 s))`` (attente
    bornée par ``max_backoff + min(0,25 × max_backoff, 5)``).

    Args:
        attempt: numéro (à partir de 0) de l'essai qui vient d'échouer.
        retry_after: délai imposé par le serveur, en secondes.
        base: base du backoff exponentiel (``ALBERT_RETRY_BACKOFF``).
        max_backoff: plafond du backoff (``ALBERT_RETRY_MAX_BACKOFF``).
        retry_after_cap: plafond d'un ``Retry-After`` (``ALBERT_RETRY_AFTER_MAX``).
        jitter: ajoute la gigue sur le chemin exponentiel.
        rng: générateur (``random.Random``) pour une gigue déterministe.
        policy: politique dont ``backoff``, ``max_backoff`` et
            ``retry_after_max`` remplacent les trois paramètres précédents.

    Returns:
        Le délai, >= 0.
    """
    if policy is not None:
        base = policy.backoff
        max_backoff = policy.max_backoff
        if retry_after_cap is None:
            retry_after_cap = policy.retry_after_max
    if retry_after is not None and retry_after > 0:
        limit = retry_after_cap if retry_after_cap is not None else max_backoff
        return max(0.0, min(float(retry_after), float(limit)))
    exponent = min(max(int(attempt), 0), 60)
    wait = max(0.0, min(float(base) * (2 ** exponent), float(max_backoff)))
    if jitter and wait > 0:
        generator = rng if rng is not None else random
        wait += generator.uniform(0, min(wait * JITTER_FRACTION, JITTER_MAX))
    return wait


# ---------------------------------------------------------------------------
# Politique
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetryPolicy:
    """Paramètres de réessai du chemin Albert.

    Attributes:
        max_retries: réessais au plus sur erreur transitoire (``ALBERT_MAX_RETRIES``).
        backoff: base du backoff exponentiel, en secondes.
        max_backoff: plafond du backoff exponentiel.
        retry_after_max: ``Retry-After`` au-delà duquel un 429 signifie un quota
            épuisé (classification) ; plafond des attentes imposées.
        busy_retries: nombre de réponses 503 tolérées sur un même modèle avant
            de le quitter (repli de rôle géré par l'appelant).
        busy_max_wait: plafond des attentes courtes entre deux 503.
        jitter: gigue sur le chemin exponentiel.
        single_attempt: un seul envoi, aucune attente ; l'appelant gère le
            réessai (web asynchrone, décision 15).
    """

    max_retries: int = 4
    backoff: float = 2.0
    max_backoff: float = 60.0
    retry_after_max: float = 120.0
    busy_retries: int = 2
    busy_max_wait: float = BUSY_MAX_WAIT
    jitter: bool = True
    single_attempt: bool = False

    @classmethod
    def from_config(cls, cfg: Any, **overrides: Any) -> "RetryPolicy":
        """Construit la politique depuis un ``AlbertConfig``.

        Args:
            cfg: configuration Albert (champs ``max_retries``, ``retry_backoff``,
                ``retry_max_backoff``, ``retry_after_max``, ``busy_retries``).
            **overrides: champs de la politique à remplacer.

        Returns:
            La politique figée.
        """
        policy = cls(
            max_retries=int(getattr(cfg, "max_retries", cls.max_retries)),
            backoff=float(getattr(cfg, "retry_backoff", cls.backoff)),
            max_backoff=float(getattr(cfg, "retry_max_backoff", cls.max_backoff)),
            retry_after_max=float(getattr(cfg, "retry_after_max", cls.retry_after_max)),
            busy_retries=int(getattr(cfg, "busy_retries", cls.busy_retries)),
        )
        return replace(policy, **overrides) if overrides else policy

    def single(self) -> "RetryPolicy":
        """Copie en mode ``single_attempt`` (un seul envoi)."""
        return replace(self, single_attempt=True)

    def transient_wait(self, attempt: int, retry_after: Optional[float] = None, *, rng: Any = None) -> float:
        """Attente avant le réessai ``attempt + 1`` d'une erreur transitoire."""
        return backoff_seconds(
            attempt,
            retry_after,
            base=self.backoff,
            max_backoff=self.max_backoff,
            retry_after_cap=self.retry_after_max,
            jitter=self.jitter,
            rng=rng,
        )

    def busy_wait(self, busy_index: int, retry_after: Optional[float] = None, *, rng: Any = None) -> float:
        """Attente courte avant un nouvel essai après un 503 « Model is too busy »."""
        return backoff_seconds(
            busy_index,
            retry_after,
            base=self.backoff,
            max_backoff=min(self.max_backoff, self.busy_max_wait),
            retry_after_cap=min(self.retry_after_max, self.busy_max_wait),
            jitter=self.jitter,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# Sémaphores et limiteur
# ---------------------------------------------------------------------------
SemaphoreArg = Union[None, Any, Sequence[Any]]


def _as_semaphores(semaphore: SemaphoreArg) -> List[Any]:
    """Liste ordonnée des sémaphores à tenir (``None``, un seul ou une séquence)."""
    if semaphore is None:
        return []
    if isinstance(semaphore, (list, tuple)):
        return [sem for sem in semaphore if sem is not None]
    return [semaphore]


@contextlib.contextmanager
def _held(semaphores: Sequence[Any]):
    """Tient les sémaphores dans l'ordre donné et les libère en ordre inverse."""
    acquired: List[Any] = []
    try:
        for sem in semaphores:
            if hasattr(sem, "acquire") and hasattr(sem, "release"):
                sem.acquire()
            else:
                sem.__enter__()
            acquired.append(sem)
        yield
    finally:
        for sem in reversed(acquired):
            if hasattr(sem, "acquire") and hasattr(sem, "release"):
                sem.release()
            else:
                sem.__exit__(None, None, None)


@contextlib.asynccontextmanager
async def _aheld(semaphores: Sequence[Any]):
    """Version asynchrone de ``_held`` (sémaphores asyncio ou, via un thread, threading)."""
    acquired: List[Any] = []
    try:
        for sem in semaphores:
            if hasattr(sem, "__aenter__"):
                await sem.__aenter__()
            else:
                result = sem.acquire()
                if inspect.isawaitable(result):
                    await result
            acquired.append(sem)
        yield
    finally:
        for sem in reversed(acquired):
            if hasattr(sem, "__aexit__"):
                await sem.__aexit__(None, None, None)
            else:
                sem.release()


def _acquire_limiter(limiter: Any, tokens: int, requests: int) -> Any:
    """Appelle ``limiter.acquire`` (le paramètre ``requests`` n'est passé que s'il diffère de 1)."""
    if requests != 1:
        return limiter.acquire(tokens, requests=requests)
    return limiter.acquire(tokens)


def _pause_limiter(limiter: Any, seconds: float) -> None:
    """Met le limiteur en pause après un 429 (si le limiteur le permet)."""
    pause = getattr(limiter, "pause", None)
    if callable(pause):
        pause(seconds)


_LIMITER_EXECUTOR: Optional[ThreadPoolExecutor] = None
_LIMITER_EXECUTOR_LOCK = threading.Lock()


def _limiter_executor() -> ThreadPoolExecutor:
    """Exécuteur dédié aux attentes et entrées-sorties bloquantes des limiteurs (créé au premier usage).

    Distinct de l'exécuteur par défaut de la boucle : un limiteur qui dort
    (pause après un 429, fenêtre Redis pleine) n'y retient aucun fil utile au
    reste de l'application (arrêt d'une session, appels OpenAI, Zotero).
    """
    global _LIMITER_EXECUTOR
    with _LIMITER_EXECUTOR_LOCK:
        if _LIMITER_EXECUTOR is None:
            _LIMITER_EXECUTOR = ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 1) + 4),
                thread_name_prefix="albert-limiter",
            )
        return _LIMITER_EXECUTOR


def _is_local_bucket(limiter: Any) -> bool:
    """Vrai pour un ``TokenBucket`` en temps réel (réservation en mémoire, attente par ``asyncio.sleep``).

    Un seau dont le sommeil est injecté (horloge virtuelle) garde son propre
    ``acquire`` : son attente n'est pas du temps réel.
    """
    return isinstance(limiter, TokenBucket) and getattr(limiter, "_sleep", None) is _REAL_SLEEP


async def _run_in_limiter_executor(func: Callable[..., T], *args: Any) -> T:
    """Exécute ``func(*args)`` dans l'exécuteur dédié aux limiteurs, sans bloquer la boucle."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_limiter_executor(), functools.partial(func, *args))


async def _aacquire_limiter(limiter: Any, tokens: int, requests: int) -> Any:
    """Acquisition asynchrone du limiteur, sans occuper de fil de l'exécuteur par défaut.

    Un seau local (``TokenBucket``) est réservé sous son verrou (quelques
    microsecondes, aucune attente) puis attendu par ``asyncio.sleep`` ; tout
    autre limiteur (Redis, espions de test) est acquis dans l'exécuteur dédié.

    Args:
        limiter: limiteur proactif (``acquire``, éventuellement ``reserve``).
        tokens: tokens d'entrée estimés de la requête.
        requests: unités de requête consommées.

    Returns:
        Ce que renvoie l'acquisition (secondes attendues pour les limiteurs du module).
    """
    if isinstance(limiter, NullLimiter):
        return 0.0
    if _is_local_bucket(limiter):
        wait = limiter.reserve(tokens, requests=requests)
        if wait > 0:
            await asyncio.sleep(wait)
        return wait
    return await _run_in_limiter_executor(_acquire_limiter, limiter, tokens, requests)


async def _apause_limiter(limiter: Any, seconds: float) -> None:
    """Pause asynchrone après un 429 : en mémoire pour un seau local, sinon hors boucle.

    La pause d'un ``RedisWindowLimiter`` lit et écrit Redis (client sans délai
    d'expiration) : elle passe par l'exécuteur dédié pour que la boucle
    d'événements ne soit jamais bloquée par un lien Redis lent.
    """
    if isinstance(limiter, (TokenBucket, NullLimiter)):
        _pause_limiter(limiter, seconds)
        return
    await _run_in_limiter_executor(_pause_limiter, limiter, seconds)


def quota_exhausted_from(error: AlbertTransientError) -> AlbertQuotaExhausted:
    """Convertit un 429 persistant en ``AlbertQuotaExhausted`` (arrêt du job)."""
    return AlbertQuotaExhausted(
        status=error.status,
        endpoint=error.endpoint,
        detail=error.detail,
        retry_after=getattr(error, "retry_after", None),
    )


class _RetryState:
    """Compteurs d'une boucle de réessai et décision après un échec."""

    def __init__(self, policy: RetryPolicy, rng: Any, *, defer_pause: bool = False) -> None:
        """Initialise les compteurs (réessais transitoires, réponses 503, attente cumulée sur 429).

        Args:
            policy: politique de réessai.
            rng: générateur de la gigue.
            defer_pause: ne pose pas la pause du limiteur dans ``next_wait`` ;
                elle est laissée dans ``pending_pause`` pour l'appelant
                asynchrone, qui la pose hors de la boucle d'événements.
        """
        self.policy = policy
        self.rng = rng
        self.attempt = 0
        self.busy = 0
        self.rate_waited = 0.0
        self.defer_pause = defer_pause
        self.pending_pause: Optional[float] = None

    def _pause(self, limiter: Any, seconds: float) -> None:
        """Pose (ou, en mode différé, retient) la pause du limiteur après un 429."""
        if limiter is None:
            return
        if self.defer_pause:
            self.pending_pause = seconds
        else:
            _pause_limiter(limiter, seconds)

    def _window_completion(self, exc: AlbertTransientError) -> Optional[float]:
        """Attente qui complète une fenêtre de débit avant de conclure à un quota, ou ``None``.

        Seulement pour un 429 sans ``Retry-After`` quand des réessais sont
        permis : les attentes cumulées depuis le premier 429 doivent couvrir
        ``RATE_WINDOW_SECONDS`` (plus la marge) ; au-delà, le 429 persistant
        est bien un quota épuisé. Un seul essai supplémentaire au plus.
        """
        if exc.retry_after is not None or self.policy.max_retries <= 0:
            return None
        remaining = RATE_WINDOW_SECONDS + RATE_WINDOW_MARGIN - self.rate_waited
        return remaining if remaining > 0 else None

    def next_wait(self, exc: AlbertTransientError, limiter: Any) -> float:
        """Attente avant le prochain essai, ou relève l'erreur si le réessai est exclu.

        Raises:
            AlbertModelBusy: 503 au-delà de ``busy_retries`` (repli de rôle).
            AlbertTransientError: réessais épuisés, mode ``single_attempt`` ou
                erreur marquée ``retryable = False`` (relevée telle quelle).
            AlbertQuotaExhausted: 429 persistant (sans ``Retry-After`` : après
                une fenêtre de débit complète).
        """
        policy = self.policy
        if not getattr(exc, "retryable", True):
            raise exc
        if isinstance(exc, AlbertModelBusy):
            self.busy += 1
            if policy.single_attempt or self.busy >= max(1, policy.busy_retries):
                raise exc
            return policy.busy_wait(self.busy - 1, exc.retry_after, rng=self.rng)
        if policy.single_attempt:
            if exc.status == 429 and limiter is not None:
                self._pause(limiter, policy.transient_wait(0, exc.retry_after, rng=self.rng))
            raise exc
        if self.attempt >= policy.max_retries:
            if exc.status == 429:
                extra = self._window_completion(exc)
                if extra is None:
                    raise quota_exhausted_from(exc) from exc
                self.rate_waited += extra
                self._pause(limiter, extra)
                return extra
            raise exc
        wait = policy.transient_wait(self.attempt, exc.retry_after, rng=self.rng)
        self.attempt += 1
        if exc.status == 429:
            self.rate_waited += wait
            self._pause(limiter, wait)
        return wait

    async def apply_pending_pause(self, limiter: Any) -> None:
        """Pose la pause retenue par ``next_wait`` (mode différé), sans bloquer la boucle."""
        seconds, self.pending_pause = self.pending_pause, None
        if seconds is not None and limiter is not None:
            await _apause_limiter(limiter, seconds)

    def log(self, exc: AlbertTransientError, wait: float, label: Optional[str]) -> None:
        """Journalise un réessai (WARNING ; le message d'erreur est déjà masqué)."""
        logger.warning(
            "Albert%s : erreur transitoire (%s), nouvel essai dans %.1f s.",
            f" [{label}]" if label else "",
            exc,
            wait,
        )


# ---------------------------------------------------------------------------
# Boucles de réessai
# ---------------------------------------------------------------------------
def call_with_retry(
    fn: Callable[[], T],
    *,
    policy: Optional[RetryPolicy] = None,
    semaphore: SemaphoreArg = None,
    limiter: Any = None,
    tokens: int = 0,
    sleep: Callable[[float], Any] = time.sleep,
    requests: int = 1,
    rng: Any = None,
    on_retry: Optional[Callable[[AlbertTransientError, float], Any]] = None,
    label: Optional[str] = None,
) -> T:
    """Exécute ``fn`` avec réessais, limiteur hors sémaphore et sommeil hors sémaphore.

    Pour chaque essai : ``limiter.acquire(tokens)`` (hors sémaphore), puis
    ``fn()`` sémaphore(s) tenu(s), puis, en cas d'erreur transitoire, sommeil
    **après** libération. Un 429 met le limiteur en pause (``pause``).

    Args:
        fn: envoi d'une requête (sans argument) ; lève une ``AlbertError``.
        policy: politique de réessai (défaut ``RetryPolicy()``).
        semaphore: sémaphore ou séquence de sémaphores (acquis dans l'ordre,
            libérés en ordre inverse), tenus pendant l'envoi seulement.
        limiter: limiteur proactif (``acquire``/``pause``), ou ``None``.
        tokens: tokens d'entrée estimés de la requête (budget TPM).
        sleep: fonction de sommeil (injectable).
        requests: unités de requête consommées (1 par défaut).
        rng: générateur de la gigue.
        on_retry: rappel ``(erreur, attente)`` avant chaque sommeil.
        label: libellé ajouté aux journaux.

    Returns:
        La valeur renvoyée par ``fn``.

    Raises:
        AlbertError: erreur non retentée, ou dernière erreur après épuisement.
    """
    state = _RetryState(policy or RetryPolicy(), rng)
    semaphores = _as_semaphores(semaphore)
    while True:
        if limiter is not None:
            _acquire_limiter(limiter, tokens, requests)
        try:
            with _held(semaphores):
                return fn()
        except AlbertTransientError as exc:
            wait = state.next_wait(exc, limiter)
            state.log(exc, wait, label)
            if on_retry is not None:
                on_retry(exc, wait)
        sleep(wait)


async def acall_with_retry(
    fn: Callable[[], Union[Awaitable[T], T]],
    *,
    policy: Optional[RetryPolicy] = None,
    semaphore: SemaphoreArg = None,
    limiter: Any = None,
    tokens: int = 0,
    sleep: Optional[Callable[[float], Any]] = None,
    requests: int = 1,
    rng: Any = None,
    on_retry: Optional[Callable[[AlbertTransientError, float], Any]] = None,
    label: Optional[str] = None,
    run_sync_in_thread: bool = True,
) -> T:
    """Version asynchrone de ``call_with_retry`` ; la boucle d'événements n'est jamais bloquée.

    L'acquisition du limiteur se fait hors sémaphore sans occuper l'exécuteur
    par défaut (``_aacquire_limiter`` : seau local réservé puis attendu par
    ``asyncio.sleep``, autre limiteur dans l'exécuteur dédié) ; la pause
    après un 429 est posée hors de la boucle (``_apause_limiter``) ;
    l'attente passe par ``asyncio.sleep`` (hors sémaphore). ``fn`` peut être
    une fonction coroutine ou une fonction synchrone (exécutée dans un
    thread si ``run_sync_in_thread``).

    Args:
        fn: envoi d'une requête (coroutine ou appelable synchrone).
        policy: politique de réessai.
        semaphore: sémaphore(s) asyncio (ou threading, acquis sans bloquer la boucle).
        limiter: limiteur proactif.
        tokens: tokens d'entrée estimés.
        sleep: fonction de sommeil (défaut ``asyncio.sleep`` ; une fonction
            synchrone est aussi acceptée).
        requests: unités de requête consommées.
        rng: générateur de la gigue.
        on_retry: rappel ``(erreur, attente)`` avant chaque sommeil.
        label: libellé ajouté aux journaux.
        run_sync_in_thread: exécute un ``fn`` synchrone dans un thread.

    Returns:
        La valeur renvoyée par ``fn``.

    Raises:
        AlbertError: erreur non retentée, ou dernière erreur après épuisement.
    """
    state = _RetryState(policy or RetryPolicy(), rng, defer_pause=True)
    semaphores = _as_semaphores(semaphore)
    sleeper = sleep if sleep is not None else asyncio.sleep
    while True:
        if limiter is not None:
            await _aacquire_limiter(limiter, tokens, requests)
        try:
            async with _aheld(_threading_to_async(semaphores)):
                if inspect.iscoroutinefunction(fn) or not run_sync_in_thread:
                    result = fn()
                else:
                    result = await asyncio.to_thread(fn)
                if inspect.isawaitable(result):
                    result = await result
                return result
        except AlbertTransientError as exc:
            try:
                wait = state.next_wait(exc, limiter)
            finally:
                await state.apply_pending_pause(limiter)
            state.log(exc, wait, label)
            if on_retry is not None:
                outcome = on_retry(exc, wait)
                if inspect.isawaitable(outcome):
                    await outcome
        slept = sleeper(wait)
        if inspect.isawaitable(slept):
            await slept


class _ThreadSemaphoreAdapter:
    """Adapte un sémaphore ``threading`` à ``async with`` sans bloquer la boucle."""

    def __init__(self, semaphore: Any) -> None:
        """Enveloppe ``semaphore`` (doté de ``acquire``/``release`` synchrones)."""
        self._semaphore = semaphore

    async def __aenter__(self) -> None:
        """Acquiert le sémaphore dans un thread."""
        await asyncio.to_thread(self._semaphore.acquire)

    async def __aexit__(self, *exc: Any) -> None:
        """Libère le sémaphore."""
        self._semaphore.release()


def _threading_to_async(semaphores: Iterable[Any]) -> List[Any]:
    """Enveloppe les sémaphores synchrones (sans ``__aenter__``) pour ``async with``."""
    out: List[Any] = []
    for sem in semaphores:
        acquire = getattr(sem, "acquire", None)
        if hasattr(sem, "__aenter__") or (acquire is not None and inspect.iscoroutinefunction(acquire)):
            out.append(sem)
        else:
            out.append(_ThreadSemaphoreAdapter(sem))
    return out


__all__ = [
    "BUSY_MAX_WAIT",
    "RATE_WINDOW_SECONDS",
    "RetryPolicy",
    "acall_with_retry",
    "backoff_seconds",
    "call_with_retry",
    "parse_retry_after",
    "quota_exhausted_from",
]
