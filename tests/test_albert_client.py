"""Tests du client HTTP Albert (``scripts/rad_albert/client.py``), de sa couche de retry et du preflight.

Tout passe par ``tests/albert_fakes.FakeAlbert`` (``httpx.MockTransport``) : aucun
appel réseau, clé factice ``FAKE_ALBERT_KEY`` seulement. Le sommeil est injecté
(enregistreur) : aucun test n'attend réellement. Les assertions portent sur le
journal des requêtes reçues par le faux (forme sur le fil) et sur la taxonomie
d'erreurs du sprint (classification sur le statut HTTP, erreurs de compte
jamais retentées, ``Retry-After`` respecté jusqu'à ``ALBERT_RETRY_AFTER_MAX``).
"""

from __future__ import annotations

import asyncio
import base64
import copy
import dataclasses
import json
import logging
import math
import random
import re
import threading
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from scripts.rad_albert import catalog as albert_catalog
from scripts.rad_albert import client as albert_client_mod
from scripts.rad_albert import limiter as albert_limiter
from scripts.rad_albert import preflight as albert_preflight
from scripts.rad_albert import retry as albert_retry
from scripts.rad_albert import usage as albert_usage
from scripts.rad_albert.client import AlbertClient
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_albert.errors import (
    AlbertAuthError,
    AlbertError,
    AlbertModelBusy,
    AlbertPermanentError,
    AlbertQuotaExhausted,
    AlbertTransientError,
    AlbertTruncatedError,
)
from tests.albert_fakes import (
    FAKE_ALBERT_KEY,
    FAKE_ROOT_URL,
    NAMED_ERRORS,
    ROUTES,
    FakeAlbert,
    fake_embedding,
    load_fixture,
)

FIXED_TODAY = date(2026, 9, 26)
FIXED_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc).timestamp()
ALBERT_HOST = "albert.api.etalab.gouv.fr"
PRIMARY_RECODE = "ministral-3-8b-instruct-2512"
FALLBACK_RECODE = "mistral-small-3-2-24b-instruct-2506"
LATE_FALLBACK_RECODE = "gemma-4-31b-it"
REASONING_MODEL = "gpt-oss-120b"
EMBED_MODEL = "bge-m3"
OCR_DOC_MODEL = "mistral-ocr-2512"
OCR_CHAT_MODEL = "lightonocr-2-1b"
# En-têtes admis sur le fil : ceux du protocole (hôte, corps) plus les deux seuls
# que le client pose, ``Authorization`` et ``User-Agent`` (décision 4). Les
# en-têtes par défaut d'httpx (``accept``, ``accept-encoding``, ``connection``)
# sont exclus : un client qui les laisserait passer échouerait.
STANDARD_HEADERS = frozenset({
    "host", "content-length", "content-type", "authorization", "user-agent",
})
MESSAGES = [{"role": "user", "content": "Réponds seulement : OK."}]
MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >> endobj\n"
    b"3 0 obj << /Type /Page /Parent 2 0 R >> endobj\n"
    b"4 0 obj << /Type /Page /Parent 2 0 R >> endobj\n"
    b"5 0 obj << /Type /Page /Parent 2 0 R >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"
)
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _fresh_process_state():
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
    """Limiteur espion : note les acquisitions (tokens, requêtes) et les pauses, sans attendre."""

    def __init__(self):
        """Crée l'espion sans aucune acquisition."""
        self.acquired = []
        self.paused = []

    def acquire(self, tokens=0, *, requests=1):
        """Note l'acquisition."""
        self.acquired.append((tokens, requests))
        return 0.0

    def pause(self, seconds=None):
        """Note la pause demandée après un 429."""
        self.paused.append(seconds)
        return seconds


def _cfg(**overrides):
    """Configuration Albert ON (défauts du sprint) avec surcharges typées."""
    return dataclasses.replace(AlbertConfig(enabled=True), **overrides)


def _client(fake, cfg=None, *, sleep=None, ledger=None, api_key=FAKE_ALBERT_KEY, limiters=None):
    """Client Albert branché sur ``fake`` (sommeil factice ; limiteur désactivé sauf espion fourni)."""
    return AlbertClient(
        cfg or _cfg(),
        api_key,
        transport=fake.transport,
        ledger=ledger,
        sleep=sleep if sleep is not None else SleepRecorder(),
        limiters=limiters,
        use_limiter=limiters is not None,
    )


def _chat(client, model=PRIMARY_RECODE, messages=None, **kwargs):
    """Appel de chat du client avec un modèle explicite (rôle ``recode`` par défaut)."""
    return client.chat(messages or MESSAGES, model=model, **kwargs)


def _held(semaphore):
    """Vrai si le sémaphore (threading) est actuellement tenu."""
    if semaphore.acquire(blocking=False):
        semaphore.release()
        return False
    return True


def _policy(**overrides):
    """Politique de retry de ``retry.py`` construite depuis une configuration."""
    return albert_retry.RetryPolicy.from_config(_cfg(**overrides))


def _http_date(offset_seconds):
    """Date HTTP (RFC 7231, GMT) décalée de ``offset_seconds`` par rapport à maintenant."""
    return format_datetime(datetime.now(timezone.utc) + timedelta(seconds=offset_seconds), usegmt=True)


def _template_regex(template):
    """Expression régulière d'un gabarit de chemin (``/v1/documents/{document_id}``)."""
    return re.compile("^" + re.sub(r"\{\w+\}", r"[^/]+", template) + "/?$")


def _route_matches(method, path):
    """Vrai si (méthode, chemin) correspond à une route servie par ``FakeAlbert``."""
    return any(m == method and _template_regex(t).match(path) for m, t in ROUTES)


def _make_document(client, *, name="ragpy-test"):
    """Crée une collection et un document ; renvoie (collection_id, document_id)."""
    cid = client.create_collection(name)
    did = client.create_document(cid, name + ".md")
    return int(cid), int(did)


def _max_jittered(max_backoff):
    """Borne haute d'une attente exponentielle : plafond plus gigue (25 %, 5 s au plus)."""
    return max_backoff + min(0.25 * max_backoff, 5.0)


# ---------------------------------------------------------------------------
# Classification des erreurs
# ---------------------------------------------------------------------------
_P6_CASES = [
    ("P6_invalid_key", AlbertAuthError, "invalid_key"),
    ("P6_unknown_model", AlbertPermanentError, "not_found"),
    ("P6_wrong_model_type", AlbertPermanentError, "validation"),
]

_NAMED_CASES = [
    ("bad_request", AlbertPermanentError, "bad_request"),
    ("budget_exhausted", AlbertAuthError, "budget_exhausted"),
    ("invalid_key", AlbertAuthError, "invalid_key"),
    ("account_expired", AlbertAuthError, "account_expired"),
    ("no_access", AlbertPermanentError, "no_access"),
    ("not_found", AlbertPermanentError, "not_found"),
    ("conflict", AlbertPermanentError, "conflict"),
    ("payload_too_large", AlbertPermanentError, "payload_too_large"),
    ("wrong_model_type", AlbertPermanentError, "validation"),
    ("request_timeout", AlbertTransientError, None),
    ("server_error", AlbertTransientError, None),
    ("bad_gateway", AlbertTransientError, None),
    ("gateway_timeout", AlbertTransientError, None),
    ("model_busy", AlbertModelBusy, None),
]


def _call_for_path(client, method, path):
    """Appelle la méthode du client qui produit la requête (méthode, chemin) donnée."""
    if (method, path) == ("GET", "/v1/models"):
        return client.models()
    if (method, path) == ("POST", "/v1/chat/completions"):
        return _chat(client, max_tokens=16)
    if (method, path) == ("POST", "/v1/embeddings"):
        return client.embed(["x"])
    raise AssertionError(f"route non prévue : {method} {path}")


@pytest.mark.parametrize("fixture_name, cls, reason", _P6_CASES)
def test_classification_matrix(fixture_name, cls, reason):
    fx = load_fixture(fixture_name + ".json")
    method, path = fx["request"]["method"], fx["request"]["path"]
    fake = FakeAlbert()
    fake.inject(method, path, (fx["status"], fx["body"], fx.get("headers") or {}))
    client = _client(fake, _cfg(max_retries=0, busy_retries=0, model_fallback=False))
    with pytest.raises(AlbertError) as excinfo:
        _call_for_path(client, method, path)
    err = excinfo.value
    assert type(err) is cls
    assert err.reason == reason
    assert err.status == fx["status"]
    assert err.endpoint == path
    assert err.detail == fx["body"]["detail"]
    assert len(fake.calls) == 1


@pytest.mark.parametrize("name, cls, reason", _NAMED_CASES)
def test_classification_matrix_named_statuses(name, cls, reason):
    status = NAMED_ERRORS[name][0]
    fake = FakeAlbert()
    fake.inject("GET", "/v1/models", name)
    client = _client(fake, _cfg(max_retries=0, busy_retries=0, model_fallback=False))
    with pytest.raises(AlbertError) as excinfo:
        client.models()
    err = excinfo.value
    assert type(err) is cls
    if reason is not None:
        assert err.reason == reason
    assert err.status == status
    assert err.endpoint == "/v1/models"
    if isinstance(err, AlbertAuthError):
        assert err.credential_required == "albert_api_key"
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "exc_class",
    [httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError],
)
def test_classification_network_errors_are_transient(exc_class):
    fake = FakeAlbert()
    fake.inject("GET", "/v1/models", exc_class, exc_class)
    sleep = SleepRecorder()
    client = _client(fake, _cfg(max_retries=1), sleep=sleep)
    with pytest.raises(AlbertTransientError) as excinfo:
        client.models()
    assert type(excinfo.value) is AlbertTransientError
    assert excinfo.value.status is None
    assert excinfo.value.endpoint == "/v1/models"
    assert len(fake.calls) == 2
    assert len(sleep.calls) == 1


@pytest.mark.parametrize(
    "entry",
    ["bad_request", "budget_exhausted", "invalid_key", "account_expired", "no_access", "not_found",
     "conflict", "payload_too_large", "wrong_model_type"],
)
def test_permanent_single_attempt(entry):
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", entry, entry, entry)
    sleep = SleepRecorder()
    client = _client(fake, _cfg(max_retries=4, model_fallback=True), sleep=sleep)
    with pytest.raises((AlbertPermanentError, AlbertAuthError)):
        _chat(client, max_tokens=16)
    # Un seul envoi, aucun sommeil, et jamais de repli de rôle sur une erreur permanente.
    assert len(fake.calls_to("POST", "/v1/chat/completions")) == 1
    assert sleep.calls == []


# ---------------------------------------------------------------------------
# Réessais, Retry-After, quota
# ---------------------------------------------------------------------------
def test_transient_retried_then_success_backoff_capped():
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", 500, 502, httpx.ConnectError)
    sleep = SleepRecorder()
    client = _client(fake, _cfg(max_retries=4, retry_backoff=2.0, retry_max_backoff=5.0), sleep=sleep)
    assert _chat(client, max_tokens=16) == "OK."
    assert len(fake.calls_to("POST", "/v1/chat/completions")) == 4
    assert len(sleep.calls) == 3
    assert all(0.0 < s <= _max_jittered(5.0) for s in sleep.calls)
    assert sleep.calls[0] >= 2.0


def test_transient_exhausts_max_retries():
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", *([504] * 10))
    sleep = SleepRecorder()
    client = _client(fake, _cfg(max_retries=2), sleep=sleep)
    with pytest.raises(AlbertTransientError) as excinfo:
        _chat(client, max_tokens=16)
    assert excinfo.value.status == 504
    assert len(fake.calls_to("POST", "/v1/chat/completions")) == 3
    assert len(sleep.calls) == 2


def test_retry_after_seconds_and_http_date_capped():
    # Secondes : délai respecté exactement, sans gigue, même au-delà du plafond de backoff (60 s).
    for header, expected in (("7", 7.0), ("90", 90.0), ("120", 120.0)):
        fake = FakeAlbert()
        fake.inject("POST", "/v1/chat/completions", 429, retry_after=header)
        sleep = SleepRecorder()
        client = _client(fake, _cfg(retry_max_backoff=60.0, retry_after_max=120.0), sleep=sleep)
        assert _chat(client, max_tokens=16) == "OK."
        assert sleep.calls == [expected]
        assert len(fake.calls_to("POST", "/v1/chat/completions")) == 2

    # Date HTTP : convertie en secondes à partir de maintenant.
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", 429, retry_after=_http_date(30))
    sleep = SleepRecorder()
    client = _client(fake, _cfg(retry_after_max=120.0), sleep=sleep)
    assert _chat(client, max_tokens=16) == "OK."
    assert len(sleep.calls) == 1
    assert 27.0 <= sleep.calls[0] <= 30.5

    # Au-delà du plafond (secondes ou date) : quota épuisé, aucun réessai, aucun sommeil.
    for header in ("121", _http_date(600)):
        fake = FakeAlbert()
        fake.inject("POST", "/v1/chat/completions", 429, 429, retry_after=header)
        sleep = SleepRecorder()
        client = _client(fake, _cfg(retry_after_max=120.0), sleep=sleep)
        with pytest.raises(AlbertQuotaExhausted):
            _chat(client, max_tokens=16)
        assert len(fake.calls_to("POST", "/v1/chat/completions")) == 1
        assert sleep.calls == []

    # Plafond configurable (ALBERT_RETRY_AFTER_MAX).
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", 429, retry_after="45")
    client = _client(fake, _cfg(retry_after_max=30.0))
    with pytest.raises(AlbertQuotaExhausted):
        _chat(client, max_tokens=16)

    # Retry-After illisible ou négatif : backoff exponentiel plafonné (avec gigue).
    for header in ("-3", "demain"):
        fake = FakeAlbert()
        fake.inject("POST", "/v1/chat/completions", 429, retry_after=header)
        sleep = SleepRecorder()
        client = _client(fake, _cfg(retry_backoff=2.0, retry_max_backoff=4.0), sleep=sleep)
        assert _chat(client, max_tokens=16) == "OK."
        assert len(sleep.calls) == 1
        assert 2.0 <= sleep.calls[0] <= _max_jittered(4.0)


def test_quota_exhausted_abort():
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", 429, 429, 429, retry_after="3600")
    sleep = SleepRecorder()
    client = _client(fake, _cfg(max_retries=4, retry_after_max=120.0, model_fallback=True), sleep=sleep)
    with pytest.raises(AlbertQuotaExhausted) as excinfo:
        _chat(client, max_tokens=16)
    err = excinfo.value
    assert isinstance(err, AlbertAuthError)
    assert err.reason == "quota_exhausted"
    assert err.credential_required == "albert_api_key"
    assert err.status == 429
    assert err.retry_after == pytest.approx(3600.0)
    assert len(fake.calls) == 1
    assert sleep.calls == []

    # 429 persistant (Retry-After court) : abandon en quota épuisé après les réessais.
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", *([429] * 10), retry_after="1")
    sleep = SleepRecorder()
    client = _client(fake, _cfg(max_retries=2, retry_after_max=120.0), sleep=sleep)
    with pytest.raises(AlbertQuotaExhausted):
        _chat(client, max_tokens=16)
    assert len(fake.calls_to("POST", "/v1/chat/completions")) == 3
    assert sleep.calls == [1.0, 1.0]


def test_client_limiter_acquired_per_attempt_and_paused_on_429():
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", 429, retry_after="5")
    spy = SpyLimiter()
    embed_spy = SpyLimiter()
    client = _client(fake, limiters={"recode": spy, "embed": embed_spy})
    long_prompt = [{"role": "user", "content": "x" * 300}]
    assert _chat(client, messages=long_prompt, max_tokens=16) == "OK."
    # Une acquisition par essai, avec les tokens d'entrée estimés (len / 3).
    assert [tokens for tokens, _ in spy.acquired] == [100, 100]
    assert spy.paused == [5.0]
    client.embed(["abc", "def"])
    assert len(embed_spy.acquired) == 1
    assert spy.paused == [5.0]


def test_client_default_limiter_uses_role_rates_and_share():
    # Limiteur par défaut (sommeil injecté = temps virtuel) : débit du rôle × part du processus.
    fake = FakeAlbert()
    sleep = SleepRecorder()
    client = AlbertClient(_cfg(recode_rpm=2, embed_rpm=1000), FAKE_ALBERT_KEY, transport=fake.transport,
                          sleep=sleep)
    for _ in range(4):
        client.chat(MESSAGES, role="recode", max_tokens=16, today=FIXED_TODAY)
    assert sleep.calls == pytest.approx([30.0, 30.0], abs=0.5)
    client.embed(["a", "b"])  # budget embed distinct : aucune attente
    assert len(sleep.calls) == 2

    fake = FakeAlbert()
    sleep = SleepRecorder()
    client = AlbertClient(_cfg(recode_rpm=2, process_share=0.5), FAKE_ALBERT_KEY, transport=fake.transport,
                          sleep=sleep)
    for _ in range(3):
        client.chat(MESSAGES, role="recode", max_tokens=16, today=FIXED_TODAY)
    assert sleep.calls == pytest.approx([60.0, 60.0], abs=0.5)


def test_busy_then_role_fallback():
    cfg = _cfg(busy_retries=2, model_fallback=True, max_retries=4)
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", "model_busy", "model_busy")
    ledger = albert_usage.UsageLedger()
    sleep = SleepRecorder()
    client = _client(fake, cfg, ledger=ledger, sleep=sleep)
    result = client.chat(MESSAGES, role="recode", max_tokens=16, today=FIXED_TODAY)
    assert result == "OK."
    models = [c.json["model"] for c in fake.calls_to("POST", "/v1/chat/completions")]
    assert models == [PRIMARY_RECODE, PRIMARY_RECODE, FALLBACK_RECODE]
    # Le modèle servi est connu et journalisé (il entrera dans la clé de cache).
    assert result.model == FALLBACK_RECODE
    assert result.fallback_from == PRIMARY_RECODE
    assert ledger.records[-1]["model"] == FALLBACK_RECODE
    assert ledger.records[-1]["fallback_from"] == PRIMARY_RECODE
    assert len(sleep.calls) == 1  # un seul essai court entre les deux 503 du primaire

    # À partir du 2026-12-01, mistral-small quitte la chaîne : repli sur gemma.
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", "model_busy", "model_busy")
    client = _client(fake, cfg)
    client.chat(MESSAGES, role="recode", max_tokens=16, today=date(2026, 12, 1))
    models = [c.json["model"] for c in fake.calls_to("POST", "/v1/chat/completions")]
    assert models == [PRIMARY_RECODE, PRIMARY_RECODE, LATE_FALLBACK_RECODE]

    # Repli désactivé : l'erreur « Model is too busy » remonte, primaire seul.
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", *(["model_busy"] * 10))
    client = _client(fake, _cfg(busy_retries=2, model_fallback=False, max_retries=4))
    with pytest.raises(AlbertModelBusy):
        client.chat(MESSAGES, role="recode", max_tokens=16, today=FIXED_TODAY)
    assert [c.json["model"] for c in fake.calls_to("POST", "/v1/chat/completions")] == [PRIMARY_RECODE] * 2

    # Toute la chaîne surchargée : AlbertModelBusy après le dernier modèle.
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", *(["model_busy"] * 10))
    client = _client(fake, cfg)
    with pytest.raises(AlbertModelBusy):
        client.chat(MESSAGES, role="recode", max_tokens=16, today=FIXED_TODAY)
    models = [c.json["model"] for c in fake.calls_to("POST", "/v1/chat/completions")]
    assert models == [PRIMARY_RECODE] * 2 + [FALLBACK_RECODE] * 2

    # Jamais de repli sur 404.
    fake = FakeAlbert()
    fake.inject("POST", "/v1/chat/completions", "not_found")
    client = _client(fake, cfg)
    with pytest.raises(AlbertPermanentError):
        client.chat(MESSAGES, role="recode", max_tokens=16, today=FIXED_TODAY)
    assert len(fake.calls) == 1

    # Jamais de repli pour les embeddings.
    fake = FakeAlbert()
    fake.inject("POST", "/v1/embeddings", *(["model_busy"] * 10))
    client = _client(fake, _cfg(busy_retries=2, model_fallback=True, max_retries=2))
    with pytest.raises(AlbertModelBusy):
        client.embed(["x"])
    assert {c.json["model"] for c in fake.calls_to("POST", "/v1/embeddings")} == {EMBED_MODEL}


def test_sleep_outside_semaphore():
    semaphore = threading.BoundedSemaphore(1)
    events = []

    class EventLimiter:
        """Limiteur espion : note si le sémaphore est tenu à l'acquisition et à la pause."""

        def acquire(self, tokens=0, *, requests=1):
            """Acquisition (jamais sous le sémaphore)."""
            events.append(("acquire", _held(semaphore)))

        def pause(self, seconds=None):
            """Pause demandée par un 429 (jamais sous le sémaphore)."""
            events.append(("pause", _held(semaphore)))

    attempts = []

    def send():
        """Envoi : 502, puis 429 avec Retry-After, puis succès."""
        attempts.append(_held(semaphore))
        events.append(("send", _held(semaphore)))
        if len(attempts) == 1:
            raise AlbertTransientError(status=502)
        if len(attempts) == 2:
            raise AlbertTransientError(status=429, retry_after=3.0)
        return "ok"

    def sleep(seconds):
        """Sommeil factice (jamais sous le sémaphore)."""
        events.append(("sleep", _held(semaphore)))

    result = albert_retry.call_with_retry(
        send, policy=_policy(max_retries=4), semaphore=semaphore, limiter=EventLimiter(), tokens=12, sleep=sleep,
    )
    assert result == "ok"
    assert attempts == [True, True, True]
    kinds = [kind for kind, _ in events]
    assert kinds == ["acquire", "send", "sleep", "acquire", "send", "pause", "sleep", "acquire", "send"]
    for kind, held in events:
        assert held is (kind == "send"), (kind, held)

    # Même garantie à travers le client : envoi sous sémaphore, sommeil hors sémaphore.
    fake = FakeAlbert()
    seen_during_send = []

    def failing_send(request):
        """Première réponse 502, en notant l'état du sémaphore pendant l'envoi."""
        seen_during_send.append(_held(semaphore))
        return 502

    fake.inject("POST", "/v1/chat/completions", failing_send)
    held_during_sleep = []
    client = _client(fake, sleep=lambda s: held_during_sleep.append(_held(semaphore)))
    assert _chat(client, max_tokens=16, semaphore=semaphore) == "OK."
    assert seen_during_send == [True]
    assert held_during_sleep == [False]
    assert _held(semaphore) is False

    # Variante asynchrone : sommeil asyncio hors sémaphore, envoi sous sémaphore.
    async def scenario():
        """Exécute ``acall_with_retry`` avec un sémaphore asyncio et un sommeil espion."""
        sem = asyncio.Semaphore(1)
        seen = []

        async def asend():
            """Envoi asynchrone : 500 puis succès."""
            seen.append(("send", sem.locked()))
            if sum(1 for kind, _ in seen if kind == "send") == 1:
                raise AlbertTransientError(status=500)
            return "ok"

        async def asleep(seconds):
            """Sommeil asynchrone factice."""
            seen.append(("sleep", sem.locked()))

        out = await albert_retry.acall_with_retry(
            asend, policy=_policy(max_retries=2), semaphore=sem, limiter=None, tokens=0, sleep=asleep,
        )
        return out, seen

    out, seen = asyncio.run(scenario())
    assert out == "ok"
    assert seen == [("send", True), ("sleep", False), ("send", True)]


# Table partagée : même sémantique que ``rad_dataframe._parse_retry_after``.
# Les entrées ``+Ns`` deviennent une date HTTP à maintenant + N secondes.
_RETRY_AFTER_TABLE = [
    None, "", "   ", "0", "30", "1.5", " 12 ", "300", "-5", "abc", "demain",
    "Wed, 21 Oct 2015 07:28:00 GMT", "Wed, 21 Oct 2015 07:28:00", "Mon, 99 Foo 2020",
    "+45s", "+90s", "+600s",
]


def _resolve_retry_after_input(value):
    """Remplace une entrée relative (``+Ns``) par une date HTTP à maintenant + N secondes."""
    if isinstance(value, str) and value.startswith("+") and value.endswith("s"):
        return _http_date(int(value[1:-1]))
    return value


@pytest.mark.parametrize("attempt", range(7))
@pytest.mark.parametrize("retry_after", [None, 0, 5, 90])
def test_backoff_seconds_parity_with_mistral_retry_wait(monkeypatch, attempt, retry_after):
    # Décision 5 : même sémantique que le chemin Mistral (base 3 s, plafond 60 s,
    # Retry-After > 0 respecté et plafonné, 0 ou absent = exponentiel).
    from scripts import rad_dataframe

    monkeypatch.setattr(rad_dataframe, "MISTRAL_OCR_RETRY_BACKOFF", 3.0)
    monkeypatch.setattr(rad_dataframe, "MISTRAL_OCR_RETRY_MAX_BACKOFF", 60.0)
    expected = rad_dataframe._mistral_retry_wait(attempt, retry_after)
    assert albert_retry.backoff_seconds(attempt, retry_after, base=3.0, max_backoff=60.0) == expected

    # Gigue : même formule que le site d'appel Mistral, uniquement sur le chemin exponentiel.
    assert (albert_retry.JITTER_FRACTION, albert_retry.JITTER_MAX) == (0.25, 5.0)
    got = albert_retry.backoff_seconds(
        attempt, retry_after, base=3.0, max_backoff=60.0, jitter=True, rng=random.Random(attempt),
    )
    if retry_after:
        assert got == expected
    else:
        assert got == expected + random.Random(attempt).uniform(0, min(expected * 0.25, 5.0))


@pytest.mark.parametrize("raw", _RETRY_AFTER_TABLE)
def test_parse_retry_after_parity_with_rad_dataframe(raw):
    from scripts import rad_dataframe

    value = _resolve_retry_after_input(raw)
    expected = rad_dataframe._parse_retry_after(value)
    got = albert_retry.parse_retry_after(value)
    if expected is None:
        assert got is None
    else:
        assert got is not None
        assert got == pytest.approx(expected, abs=2.0)
        assert got >= 0.0


def test_parse_retry_after_cap():
    assert albert_retry.parse_retry_after("300", 120.0) == 120.0
    assert albert_retry.parse_retry_after("30", 120.0) == 30.0
    assert albert_retry.parse_retry_after(_http_date(600), 120.0) == 120.0
    assert albert_retry.parse_retry_after(None, 120.0) is None
    assert albert_retry.parse_retry_after("-1", 120.0) is None


# ---------------------------------------------------------------------------
# Clé, en-têtes, redirections
# ---------------------------------------------------------------------------
def test_constructor_requires_key_never_reads_environ(monkeypatch):
    monkeypatch.setenv("ALBERT_API_KEY", "fake-albert-env-key-9999")
    cfg = _cfg()
    for missing in (None, "", "   "):
        with pytest.raises(ValueError):
            AlbertClient(cfg, missing, transport=FakeAlbert().transport)
    with pytest.raises(TypeError):
        AlbertClient(cfg)  # type: ignore[call-arg]

    fake = FakeAlbert()
    client = _client(fake, cfg)
    client.models()
    assert fake.calls[0].headers["authorization"] == f"Bearer {FAKE_ALBERT_KEY}"
    assert all("fake-albert-env-key-9999" not in json.dumps(c.headers) for c in fake.calls)
    assert FAKE_ALBERT_KEY not in repr(client)

    # Contrôle statique (même motif que la gate) : ni ``os.environ`` ni ``os.getenv``.
    for module in (albert_client_mod, albert_preflight):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert not re.search(r"os\.(environ|getenv)", source), module.__name__


def _exercise_client(client):
    """Appelle une fois chaque point d'accès du client (compte, inférence, collections)."""
    client.me()
    client.models()
    client.model_info(PRIMARY_RECODE)
    client.health()
    client.health_models()
    _chat(client, max_tokens=16)
    client.embed(["alpha", "beta"])
    client.ocr_image(TINY_PNG, mime="image/png")
    cid, did = _make_document(client)
    client.get_collection(cid)
    client.list_collections()
    client.list_documents(collection_id=cid)
    client.get_document(did)
    client.add_chunks(did, [{"content": "texte du chunk", "metadata": {"content_id": "abc_0"}}])
    client.list_chunks(did)
    client.search("texte", collection_ids=[cid], method="semantic")
    client.delete_document(did)
    client.delete_collection(cid)


def test_key_only_in_authorization_header():
    fake = FakeAlbert(ocr_access=True)
    client = _client(fake)
    _exercise_client(client)
    client.ocr_document(MINIMAL_PDF, pages=[0])
    assert len(fake.calls) >= 19
    for call in fake.calls:
        assert FAKE_ALBERT_KEY not in call.url
        assert all(FAKE_ALBERT_KEY not in str(v) for v in call.params.values())
        assert FAKE_ALBERT_KEY.encode() not in call.content
        extra = set(call.headers) - STANDARD_HEADERS
        assert extra == set(), (call.method, call.path, extra)
        for name, value in call.headers.items():
            if name == "authorization":
                assert value == f"Bearer {FAKE_ALBERT_KEY}"
            else:
                assert FAKE_ALBERT_KEY not in value
        assert call.headers.get("user-agent")


def test_errors_and_logs_redact_key(caplog):
    caplog.set_level(logging.DEBUG)
    echo = {"detail": f"clé refusée : {FAKE_ALBERT_KEY} (Bearer {FAKE_ALBERT_KEY})"}
    cases = [
        (400, AlbertPermanentError),
        (401, AlbertAuthError),
        (422, AlbertPermanentError),
        (500, AlbertTransientError),
        (429, AlbertQuotaExhausted),
    ]
    for status, cls in cases:
        fake = FakeAlbert()
        fake.inject("POST", "/v1/chat/completions", *([(status, echo)] * 3))
        client = _client(fake, _cfg(max_retries=1))
        with pytest.raises(cls) as excinfo:
            _chat(client, max_tokens=16)
        err = excinfo.value
        for item in (err, err.__cause__, err.__context__):
            if item is None:
                continue
            for text in (str(item), repr(item), repr(getattr(item, "args", ())),
                         getattr(item, "detail", None) or "", getattr(item, "endpoint", None) or ""):
                assert FAKE_ALBERT_KEY not in text
        assert err.detail and "clé refusée" in err.detail
    assert FAKE_ALBERT_KEY not in caplog.text
    for record in caplog.records:
        assert FAKE_ALBERT_KEY not in record.getMessage()


def test_no_user_field():
    fake = FakeAlbert()
    client = _client(fake)
    _chat(client, max_tokens=16, user="ragpy-user")
    _chat(client, model=REASONING_MODEL, max_tokens=16, extra_body={"user": "ragpy-user"})
    client.embed(["alpha"])
    client.ocr_image(TINY_PNG, mime="image/png")
    bodies = [c.json for c in fake.calls if c.method == "POST"]
    assert len(bodies) == 4
    for body in bodies:
        assert "user" not in body
    assert b"ragpy-user" not in b"".join(c.content for c in fake.calls)


def test_no_redirect_follow():
    fake = FakeAlbert()
    redirect = httpx.Response(307, headers={"Location": "https://evil.example.com/v1/models"})
    fake.inject("GET", "/v1/models", redirect)
    client = _client(fake, _cfg(max_retries=2))
    with pytest.raises(AlbertPermanentError) as excinfo:
        client.models()
    assert excinfo.value.status == 307
    assert len(fake.calls) == 1
    assert all(httpx.URL(c.url).host == ALBERT_HOST for c in fake.calls)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def test_chat_reasoning_headroom():
    fake = FakeAlbert()
    client = _client(fake, _cfg(reasoning_headroom=2048, reasoning_effort="medium"))
    answer = _chat(client, model=REASONING_MODEL, max_tokens=500)
    body = fake.calls_to("POST", "/v1/chat/completions")[-1].json
    assert body["model"] == REASONING_MODEL
    assert body["max_tokens"] == 500 + 2048
    assert body["reasoning_effort"] == "medium"
    assert "max_completion_tokens" not in body
    # Le contenu est renvoyé, jamais le raisonnement.
    assert answer == "OK."
    reasoning = load_fixture("P9_reasoning_low_2048.json")["body"]["choices"][0]["message"]["reasoning"]
    assert reasoning not in answer

    # Effort explicite (le filtre de citations force « low »).
    _chat(client, model=REASONING_MODEL, max_tokens=100, reasoning_effort="low")
    body = fake.calls_to("POST", "/v1/chat/completions")[-1].json
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 100 + 2048

    # Modèle sans raisonnement : ni marge ni reasoning_effort, même demandé.
    _chat(client, model=PRIMARY_RECODE, max_tokens=500, reasoning_effort="low")
    body = fake.calls_to("POST", "/v1/chat/completions")[-1].json
    assert body["max_tokens"] == 500
    assert "reasoning_effort" not in body
    assert "max_completion_tokens" not in body

    # Marge configurable (ALBERT_REASONING_HEADROOM).
    client = _client(fake, _cfg(reasoning_headroom=1024))
    _chat(client, model=REASONING_MODEL, max_tokens=10)
    assert fake.calls_to("POST", "/v1/chat/completions")[-1].json["max_tokens"] == 1034


def test_truncated_empty_raises():
    fake = FakeAlbert()
    fake.queue_chat_replies(
        {"content": None, "finish_reason": "length"},
        {"content": "", "finish_reason": "length"},
        {"content": "   ", "finish_reason": "length"},
    )
    sleep = SleepRecorder()
    client = _client(fake, sleep=sleep)
    for _ in range(3):
        with pytest.raises(AlbertTruncatedError):
            _chat(client, max_tokens=16)
    # Troncature ni retentée ni repliée par le client (la politique appartient à l'appelant).
    assert len(fake.calls) == 3
    assert sleep.calls == []

    # Piège gpt-oss : raisonnement présent, contenu nul, finish_reason=length.
    fake = FakeAlbert(reasoning_tokens=10**6)
    client = _client(fake)
    with pytest.raises(AlbertTruncatedError) as excinfo:
        _chat(client, model=REASONING_MODEL, max_tokens=64)
    trap = load_fixture("P9_reasoning_trap_64.json")["body"]["choices"][0]["message"]["reasoning"]
    assert trap[:40] not in str(excinfo.value)

    # Contenu nul avec finish_reason=stop : jamais le raisonnement à la place.
    fake = FakeAlbert()
    fake.queue_chat_replies({"content": None, "finish_reason": "stop", "reasoning": "pensée interne"})
    client = _client(fake)
    out = _chat(client, model=REASONING_MODEL, max_tokens=16)
    assert out == ""
    assert "pensée interne" not in out


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def test_embed_slices_64_sorts_checks_dim_rejects_empty():
    texts = [f"texte numéro {i}" for i in range(130)]
    fake = FakeAlbert()
    fake.embed_shuffle = True
    client = _client(fake)
    vectors = client.embed(texts)
    posts = fake.calls_to("POST", "/v1/embeddings")
    assert [len(c.json["input"]) for c in posts] == [64, 64, 2]
    assert [t for c in posts for t in c.json["input"]] == texts
    for call in posts:
        assert call.json["model"] == EMBED_MODEL
        assert "dimensions" not in call.json
        assert "user" not in call.json
    assert len(vectors) == len(texts)
    for text, vector in zip(texts, vectors):
        assert len(vector) == 1024
        assert list(vector) == pytest.approx(fake_embedding(text), abs=1e-6)

    # Chaîne vide (ou blanche) refusée avant tout envoi.
    for bad in (["a", "", "b"], ["a", "   "]):
        fake = FakeAlbert()
        client = _client(fake)
        with pytest.raises(ValueError):
            client.embed(bad)
        assert fake.calls == []

    # Dimension inattendue : erreur explicite, jamais de vecteur tronqué ou complété.
    fake = FakeAlbert(embed_dim=512)
    client = _client(fake)
    with pytest.raises(AlbertError):
        client.embed(["a", "b"])

    # Normalisation L2 (D12) : un vecteur non normé est ramené à la norme 1.
    raw = [3.0, 4.0] + [0.0] * 1022

    def unnormalised(request):
        """Réponse d'embeddings non normalisée (un seul texte)."""
        return httpx.Response(200, json={
            "object": "list", "model": EMBED_MODEL,
            "data": [{"object": "embedding", "index": 0, "embedding": raw}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1, "cost": 0.0},
        })

    fake = FakeAlbert()
    fake.inject("POST", "/v1/embeddings", unnormalised, unnormalised)
    normed = _client(fake, _cfg(embed_l2_normalize=True)).embed(["a"])[0]
    assert math.sqrt(sum(v * v for v in normed)) == pytest.approx(1.0)
    assert normed[:2] == pytest.approx([0.6, 0.8])
    kept = _client(fake, _cfg(embed_l2_normalize=False)).embed(["a"])[0]
    assert kept[:2] == pytest.approx([3.0, 4.0])


# ---------------------------------------------------------------------------
# Collections, documents, chunks, recherche
# ---------------------------------------------------------------------------
def test_add_chunks_1_to_64():
    fake = FakeAlbert()
    client = _client(fake)
    _, did = _make_document(client)
    for count in (1, 64):
        chunks = [{"content": f"chunk {i}", "metadata": {"chunk_index": i}} for i in range(count)]
        fake.reset_calls()
        ids = client.add_chunks(did, chunks)
        posts = fake.calls_to("POST", "/v1/documents/*/chunks")
        assert len(posts) == 1
        assert posts[0].status == 201
        assert len(posts[0].json["chunks"]) == count
        assert len(ids) == count
    for bad in ([], [{"content": f"c{i}"} for i in range(65)]):
        fake.reset_calls()
        with pytest.raises(ValueError):
            client.add_chunks(did, bad)
        assert fake.calls == []


def test_create_document_multipart():
    fake = FakeAlbert()
    client = _client(fake)
    cid = int(client.create_collection("ragpy-test"))
    fake.reset_calls()
    did = client.create_document(cid, "doc.md")
    (call,) = fake.calls_to("POST", "/v1/documents")
    assert call.headers["content-type"].startswith("multipart/form-data")
    assert call.form == {"name": "doc.md", "collection_id": str(cid)}
    assert call.files == {}
    assert call.status == 201
    assert int(did) in fake.documents


def test_create_collection_private_forced():
    fake = FakeAlbert()
    client = _client(fake)
    cid = client.create_collection("ragpy-test", description="essai")
    (call,) = fake.calls_to("POST", "/v1/collections")
    assert call.json["visibility"] == "private"
    assert call.json["name"] == "ragpy-test"
    assert fake.collections[int(cid)]["visibility"] == "private"
    fake.reset_calls()
    with pytest.raises(ValueError):
        client.create_collection("ragpy-public", visibility="public")
    assert fake.calls == []


def test_search_rejects_threshold_non_semantic():
    fake = FakeAlbert()
    client = _client(fake)
    cid, did = _make_document(client)
    client.add_chunks(did, [{"content": "le chat dort", "metadata": {"chunk_index": 0}}])
    for method in ("hybrid", "lexical"):
        fake.reset_calls()
        with pytest.raises(ValueError):
            client.search("chat", collection_ids=[cid], method=method, score_threshold=0.5)
        assert fake.calls_to("POST", "/v1/search") == []

    fake.reset_calls()
    results = client.search("chat", collection_ids=[cid], method="semantic", score_threshold=0.1)
    body = fake.calls_to("POST", "/v1/search")[-1].json
    assert body["method"] == "semantic"
    assert body["score_threshold"] == pytest.approx(0.1)
    assert results and results[0]["chunk"]["content"] == "le chat dort"

    fake.reset_calls()
    client.search("chat", collection_ids=[cid], method="hybrid", rff_k=30)
    body = fake.calls_to("POST", "/v1/search")[-1].json
    assert body["method"] == "hybrid"
    assert body["rff_k"] == 30
    assert "rrf_k" not in body
    assert "score_threshold" not in body

    # La méthode est toujours explicite, même par défaut.
    fake.reset_calls()
    client.search("chat", collection_ids=[cid])
    assert fake.calls_to("POST", "/v1/search")[-1].json["method"] == "semantic"


def test_search_requires_query():
    # D17 : query obligatoire ; une requête vide ou blanche est refusée avant tout envoi.
    fake = FakeAlbert()
    client = _client(fake)
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            client.search(bad, collection_ids=[1])
    assert fake.calls == []


def test_paginate_caps_page_size_at_100():
    # D14 : limit=101 donne 422 ; une taille de page plus grande est ramenée à 100.
    fake = FakeAlbert()
    client = _client(fake)
    cid, did = _make_document(client)
    fake.reset_calls()
    client.list_collections(page_size=500)
    client.list_documents(collection_id=cid, page_size=101)
    client.list_chunks(did, page_size=1000)
    gets = [c for c in fake.calls if c.method == "GET"]
    assert [c.path for c in gets] == ["/v1/collections", "/v1/documents", f"/v1/documents/{did}/chunks"]
    assert [c.params.get("limit") for c in gets] == ["100", "100", "100"]
    assert all(c.status == 200 for c in gets)


def test_list_chunks_paginates_by_100():
    fake = FakeAlbert()
    client = _client(fake)
    _, did = _make_document(client)
    for start in range(0, 250, 50):
        client.add_chunks(did, [{"content": f"chunk {i}"} for i in range(start, start + 50)])
    fake.reset_calls()
    chunks = client.list_chunks(did)
    assert [c["content"] for c in chunks] == [f"chunk {i}" for i in range(250)]
    gets = fake.calls_to("GET", "/v1/documents/*/chunks")
    assert [(c.params.get("offset"), c.params.get("limit")) for c in gets] == [
        ("0", "100"), ("100", "100"), ("200", "100"),
    ]


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
def test_ocr_document_pages_passthrough():
    fake = FakeAlbert(ocr_access=True)
    fake.ocr_doc_reply = lambda index: f"markdown de la page index {index}"
    client = _client(fake)
    result = client.ocr_document(MINIMAL_PDF, pages=[0, 2])
    (call,) = fake.calls_to("POST", "/v1/ocr")
    assert call.json["model"] == OCR_DOC_MODEL
    assert call.json["pages"] == [0, 2]
    document = call.json["document"]
    assert document["type"] == "document_url"
    assert document["document_url"].startswith("data:application/pdf;base64,")
    assert base64.b64decode(document["document_url"].split(",", 1)[1]) == MINIMAL_PDF
    pages = result["pages"]
    assert [p["index"] for p in pages] == [0, 2]
    assert [p["markdown"] for p in pages] == ["markdown de la page index 0", "markdown de la page index 2"]

    # Sans « pages », le champ n'est pas envoyé (toutes les pages).
    fake.reset_calls()
    result = client.ocr_document(MINIMAL_PDF)
    assert "pages" not in fake.calls_to("POST", "/v1/ocr")[-1].json
    assert [p["index"] for p in result["pages"]] == [0, 1, 2]


def test_ocr_image_message_is_image_only():
    fake = FakeAlbert()
    fake.ocr_chat_reply = "texte transcrit"
    client = _client(fake)
    assert client.ocr_image(TINY_PNG, mime="image/png") == "texte transcrit"
    (call,) = fake.calls_to("POST", "/v1/chat/completions")
    body = call.json
    assert body["model"] == OCR_CHAT_MODEL
    assert body["max_tokens"] == 4096
    assert body["temperature"] == pytest.approx(0.2)
    assert body["top_p"] == pytest.approx(0.9)
    (message,) = body["messages"]
    parts = message["content"]
    assert [p["type"] for p in parts] == ["image_url"]
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Usage, preflight, santé, couverture du faux
# ---------------------------------------------------------------------------
def test_ledger_records_response_model_cost(tmp_path):
    fake = FakeAlbert()
    ledger = albert_usage.UsageLedger()
    client = _client(fake, ledger=ledger)
    _chat(client, model="openweight-small", max_tokens=16)
    client.embed(["alpha", "beta"])
    assert len(ledger.records) == 2
    chat_record, embed_record = ledger.records
    # response.model recopie le nom envoyé (D5/D12) : il est enregistré tel quel.
    sent = fake.calls_to("POST", "/v1/chat/completions")[0].json["model"]
    assert chat_record["model"] == sent
    assert chat_record["response_model"] == sent
    assert chat_record["cost"] == 0.0
    assert chat_record["impacts"] == pytest.approx(
        {"kWh": 5.095698496904152e-07, "kgCO2eq": 2.6710792561089838e-08}
    )
    assert chat_record["prompt_tokens"] > 0
    assert chat_record["endpoint"] == "/v1/chat/completions"
    assert embed_record["model"] == EMBED_MODEL
    assert embed_record["response_model"] == EMBED_MODEL
    assert embed_record["cost"] == 0.0

    line = ledger.summary_line()
    assert isinstance(line, str) and line
    assert FAKE_ALBERT_KEY not in line

    out = tmp_path / "albert_usage.jsonl"
    assert ledger.write_jsonl(str(out)) == str(out)
    text = out.read_text(encoding="utf-8")
    rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    assert [r["response_model"] for r in rows] == [sent, EMBED_MODEL]
    assert all("cost" in r and "impacts" in r for r in rows)
    assert FAKE_ALBERT_KEY not in text

    # Rien d'autre à écrire : aucun ajout ; un ledger vide ne crée aucun fichier.
    assert ledger.write_jsonl(str(out)) is None
    empty = albert_usage.UsageLedger()
    missing = tmp_path / "vide.jsonl"
    assert empty.write_jsonl(str(missing)) is None
    assert not missing.exists()

    usage = albert_usage.extract_usage(load_fixture("P7_chat_usage.json")["body"])
    assert usage["cost"] == 0.0
    assert set(usage["impacts"]) == {"kWh", "kgCO2eq"}


def test_ledger_model_is_sent_id_response_model_is_echo():
    # Après résolution (alias → id épinglé), l'id part sur le fil et entre dans
    # ``model`` ; ``response_model`` garde ce que la réponse recopie (ici l'alias).
    fake = FakeAlbert()
    ledger = albert_usage.UsageLedger()
    client = _client(fake, ledger=ledger)
    assert client.resolve_model_id("openweight-small") == PRIMARY_RECODE

    def alias_echo(request):
        """Réponse de chat (P7) dont le champ ``model`` vaut l'alias, quel que soit l'id reçu."""
        body = copy.deepcopy(load_fixture("P7_chat_usage.json")["body"])
        body["model"] = "openweight-small"
        return 200, body

    for requested in ("openweight-small", "albert/openweight-small"):
        fake.reset_calls()
        before = len(ledger.records)
        fake.inject("POST", "/v1/chat/completions", alias_echo)
        result = _chat(client, model=requested, max_tokens=16)
        (call,) = fake.calls_to("POST", "/v1/chat/completions")
        assert call.json["model"] == PRIMARY_RECODE
        assert result == "OK."
        assert result.model == PRIMARY_RECODE
        assert result.response_model == "openweight-small"
        chat_records = [r for r in ledger.records[before:] if r["endpoint"] == "/v1/chat/completions"]
        assert len(chat_records) == 1
        assert chat_records[0]["model"] == PRIMARY_RECODE
        assert chat_records[0]["response_model"] == "openweight-small"


def test_ledger_records_chunk_push_not_management_calls(tmp_path):
    # Un envoi de chunks consomme le quota bge-m3 côté serveur (D19) : il est
    # inscrit au ledger (rôle ``push``) ; les appels de gestion ne le sont pas.
    fake = FakeAlbert()
    ledger = albert_usage.UsageLedger()
    client = _client(fake, ledger=ledger)
    cid, did = _make_document(client)
    assert not ledger.called

    ids = client.add_chunks(did, [{"content": f"chunk {i}", "metadata": {"chunk_index": i}} for i in range(3)])
    assert len(ids) == 3
    (record,) = ledger.records
    assert record["endpoint"] == albert_client_mod.PUSH_LEDGER_ENDPOINT
    assert str(did) not in record["endpoint"]
    assert record["role"] == "push"
    assert record["items"] == 3
    assert record["model"] is None and record["response_model"] is None
    assert record["prompt_tokens"] is None and record["cost"] is None
    assert record["latency_s"] is not None

    client.list_chunks(did)
    client.list_documents(cid)
    client.get_document(did)
    client.list_collections()
    assert len(ledger.records) == 1 and ledger.errors == 0

    # Entrée refusée avant tout envoi : ni enregistrement ni échec compté.
    with pytest.raises(ValueError):
        client.add_chunks(did, [])
    assert len(ledger.records) == 1 and ledger.errors == 0

    # Échecs (permanent, puis issue incertaine d'une écriture) : comptés, non détaillés.
    fake.inject("POST", "/v1/documents/*/chunks", 422, 502)
    with pytest.raises(AlbertPermanentError):
        client.add_chunks(did, ["texte"])
    with pytest.raises(AlbertTransientError):
        client.add_chunks(did, ["texte"])
    assert len(ledger.records) == 1 and ledger.errors == 2

    line = ledger.summary_line()
    assert "1 appel(s)" in line and "2 échec(s)" in line
    out = tmp_path / "albert_usage.jsonl"
    assert ledger.write_jsonl(str(out)) == str(out)
    text = out.read_text(encoding="utf-8")
    rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    assert [r["role"] for r in rows] == ["push"]
    assert FAKE_ALBERT_KEY not in text


def test_preflight_expired_raises(caplog):
    expired = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    fake = FakeAlbert(me_overrides={"expires": expired})
    client = _client(fake)
    with pytest.raises(AlbertAuthError) as excinfo:
        albert_preflight.run_preflight(client, roles=("recode",), today=FIXED_TODAY, clock=lambda: FIXED_NOW)
    assert excinfo.value.reason == "account_expired"
    assert excinfo.value.credential_required == "albert_api_key"

    # Expiration proche (moins de 30 jours) : avertissement, pas d'erreur.
    soon = int(FIXED_NOW + 10 * 86400)
    fake = FakeAlbert(me_overrides={"expires": soon})
    client = _client(fake)
    caplog.set_level(logging.WARNING)
    result = albert_preflight.run_preflight(client, roles=("recode", "embed"), today=FIXED_TODAY,
                                            clock=lambda: FIXED_NOW)
    assert any("expire" in w for w in result.warnings)
    assert result.primary("recode") == PRIMARY_RECODE
    assert result.primary("embed") == EMBED_MODEL
    assert {"/v1/me", "/v1/models"} <= {c.path for c in fake.calls}

    # Compte valide sans expiration : aucune erreur.
    fake = FakeAlbert()
    client = _client(fake)
    albert_preflight.run_preflight(client, roles=("recode",), today=FIXED_TODAY, clock=lambda: FIXED_NOW)

    # Modèle épinglé disparu du compte : erreur explicite, jamais de repli silencieux.
    models = [m for m in load_fixture("P2_models.json")["body"]["data"] if m["id"] != PRIMARY_RECODE]
    fake = FakeAlbert(models=models)
    client = _client(fake)
    with pytest.raises(AlbertPermanentError) as excinfo:
        albert_preflight.run_preflight(client, roles=("recode",), today=FIXED_TODAY, clock=lambda: FIXED_NOW)
    assert excinfo.value.reason == "not_found"


def test_preflight_experimentation_regime_warning(caplog):
    # D1 : /v1/me par défaut (P1) a un plus petit rpd de 1000 → régime d'expérimentation signalé.
    caplog.set_level(logging.WARNING)
    fake = FakeAlbert()
    result = albert_preflight.run_preflight(_client(fake), roles=("recode",), today=FIXED_TODAY,
                                            clock=lambda: FIXED_NOW)
    assert result.regime == "experimentation"
    assert result.daily_request_cap == 1000
    regime_warnings = [w for w in result.warnings if "expérimentation" in w]
    assert len(regime_warnings) == 1
    assert "1000" in regime_warnings[0]
    assert regime_warnings[0] in caplog.text

    # Plus petit rpd au-dessus de 1000 : production, sans avertissement de régime.
    caplog.clear()
    limits = [
        {"router_id": 1, "type": "rpm", "value": 500},
        {"router_id": 1, "type": "rpd", "value": 50000},
        {"router_id": 2, "type": "rpd", "value": 20000},
        {"router_id": 2, "type": "rpd", "value": None},
        {"router_id": 2, "type": "tpm", "value": 246000},
    ]
    fake = FakeAlbert(me_overrides={"limits": limits})
    result = albert_preflight.run_preflight(_client(fake), roles=("recode",), today=FIXED_TODAY,
                                            clock=lambda: FIXED_NOW)
    assert result.regime == "production"
    assert result.daily_request_cap == 20000
    assert not any("expérimentation" in w for w in result.warnings)
    assert "expérimentation" not in caplog.text

    # Aucune limite rpd : régime inconnu, aucun plafond, aucun avertissement de régime.
    fake = FakeAlbert(me_overrides={"limits": []})
    result = albert_preflight.run_preflight(_client(fake), roles=("recode",), today=FIXED_TODAY,
                                            clock=lambda: FIXED_NOW)
    assert result.regime == "inconnu"
    assert result.daily_request_cap is None
    assert not any("expérimentation" in w for w in result.warnings)


def test_health_is_at_root():
    fake = FakeAlbert()
    client = _client(fake)
    assert client.health() == {"status": "ok"}
    (call,) = fake.calls
    assert call.url == FAKE_ROOT_URL + "/health"
    assert call.path == "/health"


def test_fake_albert_covers_all_client_endpoints():
    # Statique : chaque point d'accès déclaré par le client est servi par le faux.
    for method, template in albert_client_mod.ENDPOINTS:
        assert (method, template) in ROUTES, (method, template)
    # Dynamique : chaque requête réellement émise tombe sur une route du faux, et
    # l'exercice complet du client touche chaque point d'accès déclaré.
    fake = FakeAlbert(ocr_access=True)
    client = _client(fake)
    _exercise_client(client)
    client.ocr_document(MINIMAL_PDF, pages=[0])
    albert_preflight.run_preflight(client, roles=("recode",), today=FIXED_TODAY, clock=lambda: FIXED_NOW)
    seen = set()
    for call in fake.calls:
        assert _route_matches(call.method, call.path), (call.method, call.path)
        assert call.status not in (404, 405), (call.method, call.path, call.status)
        assert call.url.startswith(FAKE_ROOT_URL)
        for method, template in albert_client_mod.ENDPOINTS:
            if method == call.method and _template_regex(template).match(call.path):
                seen.add((method, template))
    assert seen == set(albert_client_mod.ENDPOINTS)
    assert albert_catalog.check_endpoint_type(OCR_DOC_MODEL, "ocr") is not None
