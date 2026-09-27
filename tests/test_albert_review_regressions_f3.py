"""Tests de non-régression de la revue du lot 9 (client, limiteur, formulaire d'identifiants).

Tout passe par ``tests/albert_fakes`` (``FakeAlbert`` sur ``httpx.MockTransport``,
``FakeRedis``) : aucun réseau, clé factice ``FAKE_ALBERT_KEY`` seulement, horloges
et sommeils factices (aucune attente réelle).

* budget du limiteur choisi d'après le modèle envoyé : gpt-oss (10 RPM mesurés,
  D1) consomme le budget ``notes`` même sous un rôle du budget ``recode`` ;
* chaîne de repli filtrée par ``/v1/models`` quand la liste du compte est connue :
  un modèle de repli absent n'est jamais envoyé (plus de 404 qui arrête le job
  après un simple 503 du modèle configuré) ;
* seau local : jamais plus de ``rpm`` requêtes (ni ``tpm`` tokens) sur la
  première minute, rafale initiale comprise : la rafale de départ, bornée par la
  concurrence du rôle, est prise sur le débit de la première minute ; elle
  plafonne aussi le niveau accumulé pendant une inactivité (au plus
  ``rpm × part + rafale − 1`` requêtes et ``tpm × part + rafale de tokens − 1``
  tokens sur toute minute glissante) ;
* limiteur Redis : après une panne, Redis est retenté au bout d'un délai au lieu
  d'un repli local définitif ;
* ``/save_credentials`` Albert OFF : seuls les séparateurs de ligne (au sens de
  ``str.splitlines``) et NUL sont refusés (400, rien d'écrit) ; une tabulation ou
  tout autre caractère est accepté comme sur ``main`` ;
* code mort retiré (``ChatResult.served_model``, ``AlbertClient.chat_role``,
  ``TokenBucket.request_level`` / ``token_level``) et docstring du module
  limiteur alignée sur ``retry.acall_with_retry``.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import date, datetime, timezone

import pytest

from app.routes import settings as settings_routes
from scripts.rad_albert import limiter as albert_limiter
from scripts.rad_albert import preflight as albert_preflight
from scripts.rad_albert.client import AlbertClient, ChatResult
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_albert.errors import AlbertModelBusy
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert, FakeRedis

# Fixtures du recodage rad_chunk (routage de tout AlbertClient vers le faux) et du
# formulaire admin (base SQLite temporaire, admin injecté), réutilisées telles quelles.
from tests.test_albert_recode import _clean_env, albert_on, chunk_env  # noqa: F401
from tests.test_albert_recode import _abort_error, _chat_calls, _raw_chunks, rc
from tests.test_credentials_ui_sync import admin_user, client, db_session  # noqa: F401
from app.core.credentials import get_user_credentials

FIXED_TODAY = date(2026, 9, 26)
FIXED_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc).timestamp()
PRIMARY_RECODE = "ministral-3-8b-instruct-2512"
FALLBACK_RECODE = "mistral-small-3-2-24b-instruct-2506"
LATE_FALLBACK_RECODE = "gemma-4-31b-it"
REASONING_MODEL = "gpt-oss-120b"
MESSAGES = [{"role": "user", "content": "Réponds seulement : OK."}]
RECODE_BUDGET_ROLES = ("recode", "citation", "book_structure", "long_context")


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _fresh_albert_state():
    """Oublie les limiteurs partagés et le cache du preflight avant et après chaque test."""
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    yield
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


class SleepRecorder:
    """Sommeil factice : enregistre les durées demandées sans attendre."""

    def __init__(self):
        """Crée un enregistreur vide."""
        self.calls = []

    def __call__(self, seconds):
        """Enregistre une demande de sommeil de ``seconds`` secondes."""
        self.calls.append(float(seconds))


class SpyLimiter:
    """Limiteur espion : acquisitions et pauses sans attente."""

    def acquire(self, tokens=0, *, requests=1):
        """Acquisition immédiate."""
        return 0.0

    def pause(self, seconds=None):
        """Pause sans effet."""
        return 0.0


class FakeClock:
    """Horloge factice : ``sleep`` avance le temps au lieu d'attendre."""

    def __init__(self, start=6000.0):
        """Démarre l'horloge à ``start`` secondes."""
        self.now = float(start)

    def __call__(self):
        """Heure courante (secondes)."""
        return self.now

    def sleep(self, seconds):
        """Avance l'horloge de ``seconds`` secondes."""
        self.now += max(0.0, float(seconds))


class FlakyRedis(FakeRedis):
    """``FakeRedis`` dont les ``failures`` premiers ``GET`` échouent (connexion refusée)."""

    def __init__(self, *args, failures=1, **kwargs):
        """Prépare le faux avec ``failures`` échecs de connexion à venir."""
        super().__init__(*args, **kwargs)
        self.failures = failures

    def get(self, name):
        """``GET`` : ``ConnectionError`` tant qu'il reste des échecs, sinon réponse normale."""
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("Redis injoignable (simulé)")
        return super().get(name)


def _cfg(**overrides):
    """Configuration Albert ON (défauts du sprint) avec surcharges typées."""
    return dataclasses.replace(AlbertConfig(enabled=True), **overrides)


def _models_without(*ids):
    """Catalogue ``/v1/models`` de la fixture P2 privé des modèles ``ids``."""
    return [m for m in FakeAlbert().models if m.get("id") not in ids]


def _preflight(client):
    """Preflight du rôle ``recode`` (charge la liste ``/v1/models`` du compte dans le client)."""
    return albert_preflight.run_preflight(client, roles=("recode",), today=FIXED_TODAY, clock=lambda: FIXED_NOW)


def _sent_models(fake):
    """Modèles reçus par ``/v1/chat/completions``, dans l'ordre."""
    return [call.json["model"] for call in fake.calls_to("POST", "/v1/chat/completions")]


def _redis_keys(redis, pattern):
    """Clés ``FakeRedis`` correspondant à ``pattern`` (chaînes)."""
    return [k.decode() if isinstance(k, bytes) else k for k in redis.keys(pattern)]


# ---------------------------------------------------------------------------
# Constat 1 (majeur) : le budget du limiteur suit le modèle, pas seulement le rôle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", RECODE_BUDGET_ROLES)
def test_explicit_reasoning_model_uses_notes_budget(role):
    asked = []

    def limiters(name):
        """Note le rôle demandé au limiteur et renvoie un espion."""
        asked.append(name)
        return SpyLimiter()

    fake = FakeAlbert()
    client = AlbertClient(_cfg(), FAKE_ALBERT_KEY, transport=fake.transport, sleep=SleepRecorder(),
                          limiters=limiters)
    result = client.chat(MESSAGES, REASONING_MODEL, role=role, max_tokens=16, today=FIXED_TODAY)
    assert _sent_models(fake) == [REASONING_MODEL]
    assert asked and {albert_limiter.bucket_for_role(name) for name in asked} == {"notes"}
    # Le rôle applicatif (ledger, résultat) reste celui de l'appelant.
    assert result.role == role
    assert client.ledger.records[-1]["role"] == role

    # Témoin : ministral sous le même rôle garde le budget recode.
    asked.clear()
    client.chat(MESSAGES, PRIMARY_RECODE, role=role, max_tokens=16, today=FIXED_TODAY)
    assert {albert_limiter.bucket_for_role(name) for name in asked} == {"recode"}


def test_reasoning_model_paced_at_notes_rpm_with_default_limiter():
    # Limiteur par défaut (temps virtuel) : recode très large, notes à 1 RPM.
    fake = FakeAlbert()
    sleep = SleepRecorder()
    client = AlbertClient(_cfg(recode_rpm=1000, notes_rpm=1), FAKE_ALBERT_KEY, transport=fake.transport,
                          sleep=sleep)
    for _ in range(2):
        client.chat(MESSAGES, REASONING_MODEL, role="recode", max_tokens=16, today=FIXED_TODAY)
    assert sleep.calls == pytest.approx([60.0], abs=0.5)

    # Témoin : ministral sous le rôle recode n'attend pas (budget recode à 1000 RPM).
    fake = FakeAlbert()
    sleep = SleepRecorder()
    client = AlbertClient(_cfg(recode_rpm=1000, notes_rpm=1), FAKE_ALBERT_KEY, transport=fake.transport,
                          sleep=sleep)
    for _ in range(2):
        client.chat(MESSAGES, PRIMARY_RECODE, role="recode", max_tokens=16, today=FIXED_TODAY)
    assert sleep.calls == []


def test_get_limiter_budget_follows_model():
    cfg = _cfg()
    notes = albert_limiter.get_limiter("notes", cfg)
    recode = albert_limiter.get_limiter("recode", cfg)
    for role in RECODE_BUDGET_ROLES:
        assert albert_limiter.get_limiter(role, cfg, model="albert/" + REASONING_MODEL) is notes, role
        assert albert_limiter.get_limiter(role, cfg, model="openweight-large") is notes, role
        assert albert_limiter.get_limiter(role, cfg, model=PRIMARY_RECODE) is recode, role
        assert albert_limiter.get_limiter(role, cfg, model=None) is recode, role
    # Les budgets OCR et embeddings ne changent jamais d'après le modèle.
    assert albert_limiter.limiter_role("ocr_chat", REASONING_MODEL) == "ocr_chat"
    assert albert_limiter.limiter_role("embed", REASONING_MODEL) == "embed"
    assert albert_limiter.limiter_role("notes", PRIMARY_RECODE) == "notes"


# ---------------------------------------------------------------------------
# Constat 2 : la chaîne de repli ignore un modèle absent de /v1/models
# ---------------------------------------------------------------------------
def test_chain_skips_fallback_absent_from_models():
    cfg = _cfg(busy_retries=2, model_fallback=True, max_retries=4)
    fake = FakeAlbert(models=_models_without(FALLBACK_RECODE))
    client = AlbertClient(cfg, FAKE_ALBERT_KEY, transport=fake.transport, sleep=SleepRecorder())
    result = _preflight(client)
    assert result.chains["recode"] == [PRIMARY_RECODE]
    assert any(FALLBACK_RECODE in w for w in result.warnings)
    fake.inject("POST", "/v1/chat/completions", "model_busy", "model_busy")
    # Surcharge du seul modèle disponible : erreur transitoire, jamais un 404 « not_found ».
    with pytest.raises(AlbertModelBusy):
        client.chat(MESSAGES, role="recode", max_tokens=16, today=FIXED_TODAY)
    assert FALLBACK_RECODE not in _sent_models(fake)
    assert _sent_models(fake) == [PRIMARY_RECODE, PRIMARY_RECODE]
    assert client.chat_chain("recode", today=FIXED_TODAY) == [PRIMARY_RECODE]

    # Même chose à partir du 2026-12-01, quand gemma (expérimental) entre dans la chaîne.
    fake = FakeAlbert(models=_models_without(LATE_FALLBACK_RECODE))
    client = AlbertClient(cfg, FAKE_ALBERT_KEY, transport=fake.transport, sleep=SleepRecorder())
    client.models()
    fake.inject("POST", "/v1/chat/completions", "model_busy", "model_busy")
    with pytest.raises(AlbertModelBusy):
        client.chat(MESSAGES, role="recode", max_tokens=16, today=date(2026, 12, 1))
    assert _sent_models(fake) == [PRIMARY_RECODE, PRIMARY_RECODE]


def test_chain_filter_keeps_listed_fallback_and_explicit_primary():
    cfg = _cfg(busy_retries=2, model_fallback=True, max_retries=4)
    # Repli présent dans la liste : il sert toujours après deux 503 du primaire.
    fake = FakeAlbert(models=_models_without(LATE_FALLBACK_RECODE))
    client = AlbertClient(cfg, FAKE_ALBERT_KEY, transport=fake.transport, sleep=SleepRecorder())
    _preflight(client)
    fake.inject("POST", "/v1/chat/completions", "model_busy", "model_busy")
    result = client.chat(MESSAGES, role="recode", max_tokens=16, today=FIXED_TODAY)
    assert result.model == FALLBACK_RECODE
    assert result.fallback_from == PRIMARY_RECODE
    assert _sent_models(fake) == [PRIMARY_RECODE, PRIMARY_RECODE, FALLBACK_RECODE]

    # Un modèle explicite absent reste en tête : son 404 est l'erreur explicite (invariant 30).
    fake = FakeAlbert(models=_models_without(FALLBACK_RECODE))
    client = AlbertClient(cfg, FAKE_ALBERT_KEY, transport=fake.transport, sleep=SleepRecorder())
    client.models()
    assert client.chat_chain("recode", "modele-disparu-2026", today=FIXED_TODAY) == [
        "modele-disparu-2026", PRIMARY_RECODE,
    ]

    # Liste du compte inconnue (ni preflight ni models()) : chaîne du catalogue inchangée.
    client = AlbertClient(cfg, FAKE_ALBERT_KEY, transport=fake.transport, sleep=SleepRecorder())
    assert client.chat_chain("recode", today=FIXED_TODAY) == [PRIMARY_RECODE, FALLBACK_RECODE]


def test_rad_chunk_busy_primary_with_unlisted_fallback_does_not_abort(albert_on):
    fake = albert_on.fake
    fake.models = _models_without(FALLBACK_RECODE, LATE_FALLBACK_RECODE)
    fake.inject("POST", "/v1/chat/completions", "model_busy", "model_busy")
    raws = _raw_chunks(2)
    _texts, statuses = rc.gpt_recode_batch(raws, rc._RECODE_INSTRUCTIONS, model="albert/" + PRIMARY_RECODE)
    models = {call.json["model"] for call in _chat_calls(fake)}
    assert models == {PRIMARY_RECODE}
    # Une surcharge passagère ne coupe pas le recodage du job.
    assert _abort_error() is None
    assert statuses[-1] == "recoded"


# ---------------------------------------------------------------------------
# Constat 3 : jamais plus de rpm requêtes sur la première minute, rafale initiale
# comprise ; ensuite, au plus rpm × part + rafale − 1 sur toute minute glissante
# ---------------------------------------------------------------------------
def _max_in_window(stamps, window=60.0, tolerance=1e-6):
    """Plus grand nombre d'acquisitions dans une fenêtre glissante ``[t, t + window)``.

    ``tolerance`` absorbe l'arrondi des sommeils cumulés de l'horloge factice
    (45 × 60/45 s donne 59,999… au lieu de 60 s).
    """
    best = 0
    for i, start in enumerate(stamps):
        best = max(best, sum(1 for t in stamps[i:] if t < start + window - tolerance))
    return best


def test_bucket_first_minute_never_exceeds_rpm():
    now = [0.0]
    bucket = albert_limiter.get_limiter("recode", _cfg(), clock=lambda: now[0], sleep=lambda s: None)
    admitted = 0
    while bucket.reserve(0) < 60.0:
        admitted += 1
        assert admitted < 1000
    assert admitted <= _cfg().recode_rpm


def test_bucket_first_minute_never_exceeds_tpm():
    now = [0.0]
    cfg = _cfg(notes_rpm=1000, chat_tpm=600)
    bucket = albert_limiter.get_limiter("notes", cfg, clock=lambda: now[0], sleep=lambda s: None)
    admitted = 0
    while bucket.reserve(100, requests=0) < 60.0:
        admitted += 100
        assert admitted < 100_000
    assert 0 < admitted <= cfg.chat_tpm


@pytest.mark.parametrize("burst", [None, 1, 2, 45])
def test_bucket_saturated_from_start_never_exceeds_rpm_in_any_minute(burst):
    # Débit saturé dès le départ, sur plus de deux minutes : aucune fenêtre glissante
    # de 60 s ne contient plus de rpm acquisitions, quelle que soit la rafale de départ
    # (None : une minute de débit). La rafale part sans attente, puis le débit régulier.
    clock = FakeClock(start=0.0)
    bucket = albert_limiter.TokenBucket(45, None, burst=burst, clock=clock, sleep=clock.sleep)
    stamps = []
    for _ in range(100):
        bucket.acquire(0)
        stamps.append(clock.now)
    assert _max_in_window(stamps) <= 45
    immediate = 45 if burst is None else burst
    assert stamps[:immediate] == [0.0] * immediate
    assert stamps[immediate] > 0.0
    assert stamps[-1] - stamps[-2] == pytest.approx(60.0 / 45)


@pytest.mark.parametrize(
    "role, field",
    [
        ("recode", "recode_concurrency"),
        ("citation", "recode_concurrency"),
        ("notes", "notes_concurrency"),
        ("ocr_chat", "ocr_concurrency"),
        ("embed", "embed_concurrency"),
        ("push", "embed_concurrency"),
    ],
)
def test_get_limiter_start_burst_is_role_concurrency(role, field):
    # Le seau d'un rôle démarre presque vide : la rafale de départ vaut la
    # concurrence configurée du budget (les envois parallèles partent ensemble).
    now = [0.0]
    limiter = albert_limiter.get_limiter(role, _cfg(**{field: 3}), clock=lambda: now[0], sleep=lambda s: None)
    waits = [limiter.reserve(0) for _ in range(4)]
    assert waits[:3] == [0.0] * 3
    assert waits[3] > 0.0


def test_get_limiter_start_burst_capped_by_one_minute_of_rate():
    now = [0.0]
    cfg = _cfg(recode_rpm=4, recode_concurrency=10)
    limiter = albert_limiter.get_limiter("recode", cfg, clock=lambda: now[0], sleep=lambda s: None)
    waits = [limiter.reserve(0) for _ in range(5)]
    assert waits[:4] == [0.0] * 4
    assert waits[4] >= 60.0


def test_redis_fallback_bucket_keeps_role_burst():
    # Redis injoignable : le seau local de repli démarre, lui aussi, avec la rafale du rôle.
    clock = FakeClock()
    redis = FlakyRedis(clock=clock, failures=10**6)
    cfg = _cfg(limiter_backend="redis", recode_concurrency=2)
    limiter = albert_limiter.get_limiter("recode", cfg, redis_client=redis, clock=clock, sleep=clock.sleep)
    start = clock.now
    limiter.acquire(0)
    limiter.acquire(0)
    assert clock.now == start
    limiter.acquire(0)
    assert clock.now - start == pytest.approx(2 * 60.0 / cfg.recode_rpm)


def _saturate(limiter, clock, count, tokens=0):
    """Enchaîne ``count`` acquisitions sans pause ; renvoie les couples (instant, tokens)."""
    events = []
    for _ in range(count):
        limiter.acquire(tokens)
        events.append((clock.now, tokens))
    return events


def _max_tokens_in_window(events, window=60.0, tolerance=1e-6):
    """Plus grand total de tokens acquis dans une fenêtre glissante ``[t, t + window)``."""
    best = 0
    for i, (start, _tokens) in enumerate(events):
        best = max(best, sum(n for t, n in events[i:] if t < start + window - tolerance))
    return best


@pytest.mark.parametrize("rpm, share, burst", [(45, 1.0, 2), (90, 0.5, 2), (9, 1.0, 2), (6, 1.0, 3), (45, 1.0, None)])
def test_bucket_after_idle_never_exceeds_rate_plus_burst_in_any_minute(rpm, share, burst):
    # Saturé, deux minutes d'inactivité, saturé de nouveau : le niveau accumulé ne
    # dépasse jamais la rafale, donc aucune fenêtre glissante de 60 s ne contient plus
    # de rpm × part + rafale − 1 requêtes (borne atteinte juste après l'inactivité).
    # Sans plafond, l'inactivité remplissait une minute entière : 89 requêtes en une
    # minute à 45 par minute. burst=None : rafale d'une minute (seau construit
    # directement), borne 2 × rpm − 1 inchangée.
    clock = FakeClock(start=0.0)
    bucket = albert_limiter.TokenBucket(rpm, None, share=share, burst=burst, clock=clock, sleep=clock.sleep)
    effective = rpm * share
    rafale = effective if burst is None else burst
    stamps = [t for t, _ in _saturate(bucket, clock, int(3 * effective))]
    clock.now += 120.0
    after_idle = [t for t, _ in _saturate(bucket, clock, int(3 * effective))]
    assert _max_in_window(stamps + after_idle) <= effective + rafale - 1
    assert _max_in_window(after_idle) == effective + rafale - 1
    assert after_idle[:int(rafale)] == [after_idle[0]] * int(rafale)
    assert bucket.burst == rafale


def test_get_limiter_after_idle_bound_uses_role_concurrency():
    # Seau d'un rôle (get_limiter) : la rafale vaut la concurrence du budget, part comprise.
    clock = FakeClock(start=0.0)
    cfg = _cfg(recode_rpm=90, process_share=0.5, recode_concurrency=3)
    limiter = albert_limiter.get_limiter("recode", cfg, clock=clock, sleep=clock.sleep)
    stamps = [t for t, _ in _saturate(limiter, clock, 135)]
    clock.now += 120.0
    stamps += [t for t, _ in _saturate(limiter, clock, 135)]
    assert _max_in_window(stamps) <= 45 + 3 - 1


def test_bucket_tokens_after_idle_never_exceed_tpm_plus_token_burst():
    # Même loi pour les tokens : la rafale de tokens est la même fraction de minute que
    # la rafale de requêtes (rafale × TPM / RPM) et le niveau de tokens accumulé ne la
    # dépasse jamais. Débit limité par les tokens (300 tokens par requête, 6 000 TPM
    # effectifs, 30 RPM effectifs) : jamais plus de TPM × part + rafale de tokens − 1
    # tokens sur une fenêtre glissante, avant comme après deux minutes d'inactivité.
    clock = FakeClock(start=0.0)
    bucket = albert_limiter.TokenBucket(60, 12000, share=0.5, burst=2, clock=clock, sleep=clock.sleep)
    token_burst = 2 * 6000 / 30
    events = _saturate(bucket, clock, 60, tokens=300)
    clock.now += 120.0
    after_idle = _saturate(bucket, clock, 60, tokens=300)
    assert _max_tokens_in_window(events + after_idle) <= 6000 + token_burst - 1
    assert bucket.token_burst == pytest.approx(token_burst)

    # Une requête isolée plus grosse que la rafale de tokens (ici une demi-minute de
    # débit) part sans attendre après une inactivité : elle attend seulement la rafale.
    clock.now += 120.0
    assert bucket.acquire(3000) == 0.0


# ---------------------------------------------------------------------------
# Constat 4 : le limiteur Redis retente Redis après une panne passagère
# ---------------------------------------------------------------------------
def test_redis_limiter_recovers_after_transient_outage(caplog):
    clock = FakeClock()
    redis = FlakyRedis(clock=clock, failures=1)
    limiter = albert_limiter.RedisWindowLimiter(45, None, name="recode", client=redis, clock=clock,
                                                sleep=clock.sleep)
    with caplog.at_level(logging.WARNING, logger=albert_limiter.__name__):
        limiter.acquire(0)  # GET refusé : repli local
    assert limiter._fallback is not None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    # Pendant le délai de nouvelle tentative, Redis n'est pas sollicité à chaque appel.
    before = len(redis.calls)
    for _ in range(3):
        limiter.acquire(0)
    assert len(redis.calls) == before

    # Délai écoulé et Redis revenu : la fenêtre partagée reprend.
    clock.now += 120.0
    for _ in range(5):
        limiter.acquire(0)
    assert _redis_keys(redis, "ragpy:albert:limiter:recode:req:*")
    assert limiter._fallback is None


def test_redis_limiter_persistent_outage_never_fails_and_warns_once(caplog):
    clock = FakeClock()
    redis = FlakyRedis(clock=clock, failures=10**6)
    limiter = albert_limiter.RedisWindowLimiter(45, None, name="recode", client=redis, clock=clock,
                                                sleep=clock.sleep)
    with caplog.at_level(logging.DEBUG, logger=albert_limiter.__name__):
        for _ in range(4):
            limiter.acquire(0)
            clock.now += 120.0
        limiter.pause(5.0)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "Redis" in r.getMessage()]
    assert len(warnings) == 1
    assert limiter._fallback is not None
    assert _redis_keys(redis, "ragpy:albert:limiter:recode:*") == []


# ---------------------------------------------------------------------------
# Constat 5 : /save_credentials Albert OFF, seuls les séparateurs de ligne et NUL refusés
# ---------------------------------------------------------------------------
STR_LINE_SEPARATORS = tuple(
    chr(code) for code in range(0x110000) if len(("a" + chr(code) + "b").splitlines()) > 1
)
"""Séparateurs de ligne au sens de ``str.splitlines`` (``\\n``, ``\\r``, NEL, U+2028…)."""


def test_forbidden_chars_are_line_separators_and_nul_only():
    assert {"\n", "\r", "\x85", " ", " "} <= set(STR_LINE_SEPARATORS)
    for char in STR_LINE_SEPARATORS + ("\x00", "\r\n"):
        assert settings_routes._has_forbidden_chars(f"a{char}b") is True, repr(char)
    # Tabulation, autres contrôles et espaces : acceptés comme sur main.
    for char in ("\t", "\x01", "\x1b", "\x1f", "\x7f", "\x9f", "\xa0", " ", "é", "​"):
        assert settings_routes._has_forbidden_chars(f"a{char}b") is False, repr(char)
    assert settings_routes._has_forbidden_chars("") is False


def test_off_save_credentials_accepts_tab_refuses_line_injection(client, admin_user, db_session, tmp_path,
                                                                  monkeypatch):
    monkeypatch.delenv("ALBERT_ENABLED", raising=False)
    monkeypatch.setattr("app.routes.settings.RAGPY_DIR", str(tmp_path))
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\n", encoding="utf-8")

    # Séparateur de ligne ou NUL interne : l'injection d'une variable est refusée, rien n'est écrit.
    for char in ("\n", "\r", "\r\n", "\x85", " ", " ", "\x00"):
        resp = client.post(
            "/save_credentials",
            json={"PINECONE_ENV": "us-east-1", "ZOTERO_USER_ID": f"12{char}OPENAI_API_KEY=INJECTED"},
        )
        assert resp.status_code == 400, repr(char)
        assert resp.json()["invalid_keys"] == ["ZOTERO_USER_ID"]
        assert "INJECTED" not in resp.text
        assert env_file.read_text(encoding="utf-8") == "OTHER=1\n"
    db_session.refresh(admin_user)
    assert get_user_credentials(admin_user) == {}

    # Tabulation interne : acceptée comme sur main (.env et base personnelle mis à jour).
    resp = client.post("/save_credentials", json={"PINECONE_ENV": "us-east-1", "ZOTERO_USER_ID": "12\t34"})
    assert resp.status_code == 200
    assert env_file.read_text(encoding="utf-8") == "OTHER=1\nPINECONE_ENV=us-east-1\nZOTERO_USER_ID=12\t34\n"
    db_session.refresh(admin_user)
    assert get_user_credentials(admin_user) == {"pinecone_env": "us-east-1", "zotero_user_id": "12\t34"}


# ---------------------------------------------------------------------------
# Constat 6 : code mort retiré, docstring du limiteur alignée sur le chemin asynchrone
# ---------------------------------------------------------------------------
def test_dead_code_removed():
    assert not hasattr(ChatResult, "served_model")
    assert not hasattr(AlbertClient, "chat_role")
    assert not hasattr(albert_limiter.TokenBucket, "request_level")
    assert not hasattr(albert_limiter.TokenBucket, "token_level")


def test_limiter_module_doc_points_to_acall_with_retry():
    doc = albert_limiter.__doc__
    assert "acall_with_retry" in doc
    assert "asyncio.sleep" in doc and "albert-limiter" in doc
    assert "to_thread" not in doc
    assert "aacquire" not in doc
