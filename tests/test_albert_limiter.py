"""Tests du limiteur proactif Albert (``scripts/rad_albert/limiter.py``).

Horloge et sommeil factices (aucune attente réelle, sauf le test de la boucle
asyncio, borné à moins d'une seconde). Le backend Redis est simulé par
``tests/albert_fakes.FakeRedis`` : aucun serveur Redis, aucun réseau.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time

import pytest

from scripts.rad_albert import limiter as albert_limiter
from scripts.rad_albert import retry as albert_retry
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_albert.errors import AlbertTransientError
from tests.albert_fakes import FakeRedis


@pytest.fixture(autouse=True)
def _fresh_limiters():
    """Oublie les limiteurs partagés du processus avant et après chaque test."""
    albert_limiter.reset_limiters()
    yield
    albert_limiter.reset_limiters()


class FakeClock:
    """Horloge factice : ``sleep`` avance le temps au lieu d'attendre."""

    def __init__(self, start=1000.0):
        """Démarre l'horloge à ``start`` secondes."""
        self.now = float(start)
        self.sleeps = []

    def __call__(self):
        """Heure courante (secondes)."""
        return self.now

    def sleep(self, seconds):
        """Avance l'horloge de ``seconds`` secondes (et note la demande)."""
        seconds = max(0.0, float(seconds))
        self.sleeps.append(seconds)
        self.now += seconds


def _cfg(**overrides):
    """Configuration Albert ON (défauts du sprint) avec surcharges typées."""
    return dataclasses.replace(AlbertConfig(enabled=True), **overrides)


def _bucket(clock, *, rpm, tpm=None, share=1.0):
    """Seau à jetons local piloté par l'horloge factice."""
    return albert_limiter.TokenBucket(rpm, tpm, share=share, clock=clock, sleep=clock.sleep)


def _drive(limiter, clock, count, tokens=0):
    """Acquiert ``count`` fois et renvoie les instants (horloge factice) de chaque acquisition."""
    stamps = []
    for _ in range(count):
        limiter.acquire(tokens)
        stamps.append(clock.now)
    return stamps


def _max_in_window(stamps, window=60.0):
    """Plus grand nombre d'acquisitions dans une fenêtre glissante de ``window`` secondes."""
    best = 0
    for i, start in enumerate(stamps):
        best = max(best, sum(1 for t in stamps[i:] if t < start + window))
    return best


# ---------------------------------------------------------------------------
# Seau local
# ---------------------------------------------------------------------------
def test_bucket_rpm_tpm_fake_clock():
    # RPM : la rafale initiale (une minute de débit) passe sans attendre ; ensuite,
    # un jeton toutes les 60/rpm secondes, jamais plus de rpm par minute glissante.
    clock = FakeClock()
    bucket = _bucket(clock, rpm=6)
    stamps = _drive(bucket, clock, 6)
    assert stamps == [1000.0] * 6
    assert clock.sleeps == []
    stamps += _drive(bucket, clock, 18)
    assert clock.sleeps == pytest.approx([10.0] * 18)
    assert stamps[-1] - stamps[0] == pytest.approx(180.0)
    assert _max_in_window(stamps[6:]) <= 6

    # Le débit se reconstitue avec le temps (horloge qui avance sans sommeil).
    clock.now += 60.0
    before = len(clock.sleeps)
    _drive(bucket, clock, 6)
    assert len(clock.sleeps) == before

    # TPM : les tokens d'entrée estimés consomment le second budget.
    clock = FakeClock()
    bucket = _bucket(clock, rpm=1000, tpm=600)  # 10 tokens par seconde
    bucket.acquire(600)
    assert clock.sleeps == []
    start = clock.now
    bucket.acquire(300)
    assert clock.now - start == pytest.approx(30.0)

    # Une requête plus grosse que tout le budget TPM n'attend pas indéfiniment.
    clock = FakeClock()
    bucket = _bucket(clock, rpm=1000, tpm=100)
    bucket.acquire(10_000)
    assert clock.now - 1000.0 <= 61.0


def test_bucket_is_thread_safe():
    # Horloge figée : chaque réservation renvoie son attente ; sous verrou, les attentes
    # sont exactement 0 (rafale de 60) puis 1, 2, …, 20 s (une par réservation en file).
    frozen = FakeClock()
    waits = []
    lock = threading.Lock()
    bucket = albert_limiter.TokenBucket(60, None, clock=frozen, sleep=lambda s: None)

    def worker():
        """Réserve 10 fois."""
        for _ in range(10):
            wait = bucket.acquire(0)
            with lock:
                waits.append(wait)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(waits) == 80
    assert sorted(round(w, 6) for w in waits) == [0.0] * 60 + [float(i) for i in range(1, 21)]


def test_pause_on_429():
    clock = FakeClock()
    bucket = _bucket(clock, rpm=600)
    bucket.acquire(0)
    bucket.pause(12.0)
    start = clock.now
    bucket.acquire(0)
    assert clock.now - start == pytest.approx(12.0)
    # Une pause plus courte n'avance pas l'échéance d'une pause plus longue.
    bucket.pause(20.0)
    bucket.pause(5.0)
    start = clock.now
    bucket.acquire(0)
    assert clock.now - start == pytest.approx(20.0)

    # ``call_with_retry`` met le limiteur en pause sur un 429 (durée = Retry-After).
    paused = []

    class SpyLimiter:
        """Limiteur espion."""

        def acquire(self, tokens=0, *, requests=1):
            """Acquisition sans effet."""
            return 0.0

        def pause(self, seconds=None):
            """Note la pause demandée."""
            paused.append(seconds)

    attempts = []

    def send():
        """429 avec Retry-After de 4 s, puis 503 transitoire hors 429, puis succès."""
        attempts.append(1)
        if len(attempts) == 1:
            raise AlbertTransientError(status=429, retry_after=4.0)
        if len(attempts) == 2:
            raise AlbertTransientError(status=502)
        return "ok"

    policy = albert_retry.RetryPolicy.from_config(_cfg())
    result = albert_retry.call_with_retry(send, policy=policy, semaphore=None, limiter=SpyLimiter(),
                                          tokens=0, sleep=lambda s: None)
    assert result == "ok"
    assert paused == [4.0]  # seul le 429 met le limiteur en pause


def test_process_share():
    cfg = _cfg(process_share=0.5)
    expected = {
        "recode": (cfg.recode_rpm, cfg.chat_tpm),
        "citation": (cfg.recode_rpm, cfg.chat_tpm),
        "notes": (cfg.notes_rpm, cfg.chat_tpm),
        "ocr_chat": (cfg.ocr_rpm, None),
        "embed": (cfg.embed_rpm, None),
        "push": (cfg.embed_rpm, None),
    }
    for role, (rpm, tpm) in expected.items():
        limiter = albert_limiter.get_limiter(role, cfg)
        assert limiter.effective_rpm == pytest.approx(rpm * 0.5), role
        if tpm is None:
            assert limiter.effective_tpm is None, role
        else:
            assert limiter.effective_tpm == pytest.approx(tpm * 0.5), role
    full = albert_limiter.get_limiter("recode", _cfg(process_share=1.0))
    assert full.effective_rpm == pytest.approx(_cfg().recode_rpm)

    # Un même budget est partagé par tout le processus (recode et citation).
    assert albert_limiter.get_limiter("recode", cfg) is albert_limiter.get_limiter("citation", cfg)
    assert albert_limiter.get_limiter("recode", cfg) is not albert_limiter.get_limiter("notes", cfg)

    # Effet mesuré : à part 0,5, deux fois moins d'acquisitions par minute.
    clock = FakeClock()
    bucket = _bucket(clock, rpm=60, share=0.5)
    stamps = _drive(bucket, clock, 120)
    assert stamps[:30] == [1000.0] * 30
    assert _max_in_window(stamps[30:]) <= 30
    assert stamps[-1] - stamps[0] == pytest.approx(90 * 2.0)


def test_estimate_input_tokens():
    assert albert_limiter.estimate_input_tokens("a" * 300) == 100
    assert albert_limiter.estimate_input_tokens("a" * 301) == 101
    messages = [{"role": "system", "content": "a" * 30}, {"role": "user", "content": "b" * 60}]
    assert albert_limiter.estimate_input_tokens(messages) == pytest.approx(30, abs=1)
    parts = [{"role": "user", "content": [{"type": "text", "text": "c" * 90},
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    assert albert_limiter.estimate_input_tokens(parts) == 30
    assert albert_limiter.estimate_input_tokens(["a" * 30, "b" * 30]) == pytest.approx(20, abs=1)
    assert albert_limiter.estimate_input_tokens("") == 0
    assert albert_limiter.estimate_input_tokens(None) == 0


# ---------------------------------------------------------------------------
# Backend Redis
# ---------------------------------------------------------------------------
def _redis_limiter(redis, clock, name="recode", rpm=3, tpm=None):
    """Limiteur Redis à fenêtres fixes branché sur ``FakeRedis`` et l'horloge factice."""
    return albert_limiter.RedisWindowLimiter(rpm, tpm, client=redis, name=name, clock=clock, sleep=clock.sleep)


def test_redis_window_fake_redis():
    clock = FakeClock()
    redis = FakeRedis(clock=clock)
    first = _redis_limiter(redis, clock)
    second = _redis_limiter(redis, clock)
    # Deux processus partagent la même fenêtre : 3 requêtes par minute au total.
    first.acquire(0)
    second.acquire(0)
    first.acquire(0)
    assert clock.sleeps == []
    start = clock.now
    second.acquire(0)
    waited = clock.now - start
    assert 0.0 < waited <= 60.0 + 0.1
    assert int(clock.now // 60) == int(start // 60) + 1  # fenêtre suivante
    # Les compteurs expirent (aucune clé éternelle).
    counters = [k.decode() if isinstance(k, bytes) else k for k in redis.keys("*")]
    assert counters
    assert all(redis.ttl(k) > 0 for k in counters)
    assert {cmd for cmd, _ in redis.calls} >= {"incrby", "expire"}

    # Un autre budget a sa propre fenêtre.
    other = _redis_limiter(redis, clock, name="embed")
    before = len(clock.sleeps)
    for _ in range(3):
        other.acquire(0)
    assert len(clock.sleeps) == before

    # Pause partagée après un 429 : elle s'impose à tous les processus.
    first.pause(30.0)
    start = clock.now
    second.acquire(0)
    assert clock.now - start >= 30.0 - 1e-6

    # Budget de tokens par fenêtre.
    clock = FakeClock(start=6000.0)
    redis = FakeRedis(clock=clock)
    tokens = _redis_limiter(redis, clock, name="notes", rpm=100, tpm=1000)
    tokens.acquire(600)
    tokens.acquire(300)
    assert clock.sleeps == []
    tokens.acquire(300)
    assert len(clock.sleeps) == 1

    # get_limiter en backend redis avec un client injecté.
    limiter = albert_limiter.get_limiter("recode", _cfg(limiter_backend="redis"), redis_client=FakeRedis(clock=clock),
                                         clock=clock, sleep=clock.sleep)
    assert isinstance(limiter, albert_limiter.RedisWindowLimiter)
    limiter.acquire(10)


def test_redis_url_resolution():
    resolve = albert_limiter.resolve_redis_url
    assert resolve(_cfg(limiter_redis_url="redis://limiter:6379/3"),
                   env={"CELERY_BROKER_URL": "redis://broker:6379/0"}) == "redis://limiter:6379/3"
    assert resolve(_cfg(), env={"CELERY_BROKER_URL": "redis://broker:6379/0"}) == "redis://broker:6379/0"
    assert resolve(_cfg(), env={"CELERY_BROKER_URL": ""}) == "redis://localhost:6379/0"
    assert resolve(_cfg(), env={}) == "redis://localhost:6379/0"
    assert resolve(AlbertConfig.from_env({"ALBERT_LIMITER_REDIS_URL": "redis://env-limiter:6379/1"}),
                   env={"CELERY_BROKER_URL": "redis://broker:6379/0"}) == "redis://env-limiter:6379/1"
    # Aucune connexion n'est ouverte à la construction (import paresseux de redis).
    limiter = albert_limiter.RedisWindowLimiter(10, None, url="redis://unused:6379/0")
    assert "unused" not in repr(limiter)


def test_get_limiter_local_by_default():
    limiter = albert_limiter.get_limiter("recode", _cfg())
    assert isinstance(limiter, albert_limiter.TokenBucket)
    assert limiter.effective_rpm == pytest.approx(45.0)
    assert limiter.effective_tpm == pytest.approx(115000.0)


# ---------------------------------------------------------------------------
# Boucle asyncio
# ---------------------------------------------------------------------------
def test_limiter_does_not_block_event_loop():
    class SlowLimiter:
        """Limiteur dont l'acquisition dort réellement (bloquante) 0,3 s."""

        def __init__(self):
            """Aucune acquisition au départ."""
            self.acquired = 0

        def acquire(self, tokens=0, *, requests=1):
            """Acquisition bloquante."""
            time.sleep(0.3)
            self.acquired += 1
            return 0.3

        def pause(self, seconds=None):
            """Pause sans effet."""

    async def scenario():
        """Compte les tics de la boucle pendant un appel limité puis réessayé."""
        ticks = 0
        stop = asyncio.Event()

        async def ticker():
            """Incrémente le compteur toutes les 5 ms jusqu'à l'arrêt."""
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.005)

        calls = []

        async def send():
            """Échec transitoire (Retry-After 0,2 s), puis succès."""
            calls.append(1)
            if len(calls) == 1:
                raise AlbertTransientError(status=429, retry_after=0.2)
            return "ok"

        limiter = SlowLimiter()
        task = asyncio.ensure_future(ticker())
        await asyncio.sleep(0)
        started = time.monotonic()
        result = await albert_retry.acall_with_retry(
            send, policy=albert_retry.RetryPolicy.from_config(_cfg()), semaphore=asyncio.Semaphore(1),
            limiter=limiter, tokens=10,
        )
        elapsed = time.monotonic() - started
        stop.set()
        await task
        return result, ticks, elapsed, limiter.acquired

    result, ticks, elapsed, acquired = asyncio.run(scenario())
    assert result == "ok"
    assert acquired == 2
    assert elapsed >= 0.75
    # Pendant ~0,8 s d'attente (2 × 0,3 s de limiteur + 0,2 s de Retry-After), la boucle tourne.
    assert ticks >= 40

    # ``aacquire`` du seau local passe aussi par un thread.
    async def via_aacquire():
        """Acquisition asynchrone sur un seau qui doit attendre 0,3 s réelles."""
        bucket = albert_limiter.TokenBucket(200, None)  # 200 RPM : 0,3 s par jeton après la rafale
        for _ in range(200):
            bucket.reserve(0)
        ticks = 0
        stop = asyncio.Event()

        async def ticker():
            """Incrémente le compteur toutes les 5 ms jusqu'à l'arrêt."""
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.005)

        task = asyncio.ensure_future(ticker())
        await asyncio.sleep(0)
        waited = await bucket.aacquire(0)
        stop.set()
        await task
        return waited, ticks

    waited, ticks = asyncio.run(via_aacquire())
    assert waited == pytest.approx(0.3, abs=0.05)
    assert ticks >= 15
