"""Client Albert : écritures non rejouées, alias envoyés en id, un seul analyseur Retry-After, signature D3."""

import dataclasses
import json
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest

from scripts.rad_albert import catalog
from scripts.rad_albert import errors as albert_errors
from scripts.rad_albert import limiter as albert_limiter
from scripts.rad_albert import preflight as albert_preflight
from scripts.rad_albert import retry as albert_retry
from scripts.rad_albert.client import AlbertClient
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_albert.errors import AlbertPermanentError, AlbertUncertainWriteError
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert, load_fixture

PROJECT_ROOT = Path(__file__).resolve().parent
FIXTURES = Path("tests/fixtures/albert")
FIXED_TODAY = date(2026, 9, 26)
MESSAGES = [{"role": "user", "content": "Réponds seulement : OK."}]
SMALL_CHAIN_ROLES = ("recode", "citation", "book_structure", "long_context")
MINISTRAL = "ministral-3-8b-instruct-2512"
MISTRAL_SMALL = "mistral-small-3-2-24b-instruct-2506"
GEMMA = "gemma-4-31b-it"



class SleepRecorder:
    """Sommeil factice : enregistre les durées demandées sans attendre."""

    def __init__(self):
        """Crée un enregistreur vide."""
        self.calls = []

    def __call__(self, seconds):
        """Enregistre une demande de sommeil."""
        self.calls.append(float(seconds))


@pytest.fixture(autouse=True)
def _fresh_process_state():
    """Oublie les limiteurs partagés et le cache du preflight."""
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    yield
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


def _cfg(**overrides):
    """Configuration Albert ON avec surcharges."""
    return dataclasses.replace(AlbertConfig(enabled=True), **overrides)


def _client(fake, cfg=None, sleep=None):
    """Client sur ``fake`` sans limiteur."""
    return AlbertClient(cfg or _cfg(), FAKE_ALBERT_KEY, transport=fake.transport,
                        sleep=sleep or SleepRecorder(), use_limiter=False)


@pytest.mark.parametrize("name, expected_path", [
    ("openai/gpt-oss-120b", "/v1/models/gpt-oss-120b"),
    ("albert/BAAI/bge-m3", "/v1/models/bge-m3"),
    ("../me", "/v1/models/..%2Fme"),
    ("a?b=1", "/v1/models/a%3Fb%3D1"),
    ("x#y", "/v1/models/x%23y"),
])
def test_model_info_single_path_segment(name, expected_path):
    fake = FakeAlbert()
    client = _client(fake)
    try:
        client.model_info(name)
    except AlbertPermanentError:
        pass
    (call,) = fake.calls
    assert httpx.URL(call.url).raw_path.decode() == expected_path
    assert call.params == {}
    assert not fake.calls_to("GET", "/v1/me")


@pytest.mark.parametrize("name", ["", "  ", ".", "..", "albert/.."])
def test_model_info_rejects_empty_or_dots(name):
    fake = FakeAlbert()
    with pytest.raises(AlbertPermanentError):
        _client(fake).model_info(name)
    assert fake.calls == []


def _document(client):
    """Crée une collection et un document ; renvoie leurs ids."""
    cid = client.create_collection("coll")
    return cid, client.create_document(cid, "doc")


@pytest.mark.parametrize("injected", [httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, 500, 502, 504])
def test_create_collection_never_resent_when_outcome_unknown(injected):
    fake = FakeAlbert()
    fake.inject("POST", "/v1/collections", injected)
    with pytest.raises(AlbertUncertainWriteError) as excinfo:
        _client(fake).create_collection("coll")
    assert len(fake.calls_to("POST", "/v1/collections")) == 1
    assert excinfo.value.retryable is False
    assert "list_chunks" in str(excinfo.value)


def test_create_document_never_resent_on_500():
    fake = FakeAlbert()
    client = _client(fake)
    cid = client.create_collection("coll")
    fake.inject("POST", "/v1/documents", 500)
    with pytest.raises(AlbertUncertainWriteError):
        client.create_document(cid, "doc")
    assert len(fake.calls_to("POST", "/v1/documents")) == 1


def test_add_chunks_never_resent_on_502():
    fake = FakeAlbert()
    client = _client(fake)
    _cid, did = _document(client)
    fake.inject("POST", "/v1/documents/*/chunks", 502)
    with pytest.raises(AlbertUncertainWriteError):
        client.add_chunks(did, ["texte"])
    assert len(fake.calls_to("POST", "/v1/documents/*/chunks")) == 1


@pytest.mark.parametrize("injected", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, 503, 429])
def test_creations_resent_when_request_surely_not_processed(injected):
    fake = FakeAlbert()
    client = _client(fake)
    _cid, did = _document(client)
    fake.inject("POST", "/v1/documents/*/chunks", injected)
    assert client.add_chunks(did, ["texte"])
    assert len(fake.calls_to("POST", "/v1/documents/*/chunks")) == 2


def test_reads_still_retried_after_read_timeout():
    fake = FakeAlbert()
    fake.inject("GET", "/v1/models", httpx.ReadTimeout)
    assert _client(fake).models()
    assert len(fake.calls_to("GET", "/v1/models")) == 2


def test_alias_sent_as_id_once_listing_known():
    fake = FakeAlbert()
    client = _client(fake)
    albert_preflight.run_preflight(client, roles=("embed",), today=FIXED_TODAY)
    first = client.chat(MESSAGES, "openweight-small", max_tokens=16)
    second = client.chat(MESSAGES, "albert/openweight-large", max_tokens=16)
    sent = [c.json["model"] for c in fake.calls_to("POST", "/v1/chat/completions")]
    assert sent == [MINISTRAL, "gpt-oss-120b"]
    assert (first.model, second.model) == (MINISTRAL, "gpt-oss-120b")

    # Preflight servi par le cache : un nouveau client reçoit aussi toutes les résolutions.
    other_fake = FakeAlbert()
    other = AlbertClient(_cfg(), FAKE_ALBERT_KEY, transport=other_fake.transport, sleep=SleepRecorder(),
                         use_limiter=False)
    other._scope = client._scope  # même portée de cache (même clé, même URL)
    cached = albert_preflight.run_preflight(other, roles=("embed",), today=FIXED_TODAY)
    assert cached.cached is True
    other.chat(MESSAGES, "openweight-small", max_tokens=16)
    assert other_fake.calls_to("POST", "/v1/chat/completions")[0].json["model"] == MINISTRAL


def test_alias_unchanged_without_listing():
    fake = FakeAlbert()
    _client(fake).chat(MESSAGES, "openweight-small", max_tokens=16)
    assert fake.calls_to("POST", "/v1/chat/completions")[0].json["model"] == "openweight-small"


def test_max_completion_tokens_never_sent():
    fake = FakeAlbert()
    client = _client(fake)
    client.chat(MESSAGES, "gpt-oss-120b", max_tokens=100, max_completion_tokens=100)
    client.chat(MESSAGES, "gpt-oss-120b", extra_body={"max_completion_tokens": 50})
    first, second = [c.json for c in fake.calls_to("POST", "/v1/chat/completions")]
    assert "max_completion_tokens" not in first and "max_completion_tokens" not in second
    headroom = AlbertConfig(enabled=True).reasoning_headroom
    assert first["max_tokens"] == 100 + headroom
    assert second["max_tokens"] == 50 + headroom


def test_virtual_time_is_explicit_or_mock_transport_only():
    wrapper = lambda seconds: time.sleep(seconds)  # noqa: E731 - sommeil réel enveloppé
    real = AlbertClient(_cfg(), FAKE_ALBERT_KEY, sleep=wrapper)
    assert real._limiter_for("recode") is albert_limiter.get_limiter("recode", real.cfg)
    real.close()
    fake = FakeAlbert()
    mocked = AlbertClient(_cfg(), FAKE_ALBERT_KEY, transport=fake.transport, sleep=SleepRecorder())
    assert mocked._limiter_for("recode") is not albert_limiter.get_limiter("recode", mocked.cfg)
    forced = AlbertClient(_cfg(), FAKE_ALBERT_KEY, transport=fake.transport, sleep=SleepRecorder(),
                          virtual_time=False)
    assert forced._limiter_for("recode") is albert_limiter.get_limiter("recode", forced.cfg)


def test_virtual_clock_counts_a_real_sleep_once(monkeypatch):
    # Le sommeil injecté « attend vraiment » : l'horloge monotone avance d'autant.
    # 2 RPM, rafale de départ 2 (concurrence recode) prise sur la première minute :
    # la 3e requête attend 60 s (déficit d'une requête + avance d'une requête à
    # rembourser), la 4e attend 30 s (un jeton au débit de 2 RPM).
    # Compté deux fois (monotone + décalage virtuel), le seau se remplirait deux
    # fois trop vite et la 4e requête partirait sans attendre (sommeils [60.0]).
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    slept = []

    def real_wrapper(seconds):
        """Sommeil réel simulé : note la durée et fait avancer l'horloge monotone."""
        slept.append(seconds)
        now[0] += seconds

    fake = FakeAlbert()
    client = AlbertClient(_cfg(recode_rpm=2), FAKE_ALBERT_KEY, transport=fake.transport, sleep=real_wrapper,
                          virtual_time=True)
    for _ in range(4):
        client.chat(MESSAGES, role="recode", max_tokens=16, today=FIXED_TODAY)
    assert slept == pytest.approx([60.0, 30.0], abs=1e-6)


def test_single_retry_after_parser():
    assert albert_retry.parse_retry_after is albert_errors.parse_retry_after
    assert not hasattr(albert_errors, "_retry_after_from_headers")
    err = albert_errors.classify_http_error(429, "slow down", {"Retry-After": "7"}, endpoint="/v1/embeddings")
    assert err.retry_after == 7.0
    err = albert_errors.classify_http_error(429, "slow down", {"retry-after": "3600"}, endpoint="/v1/embeddings")
    assert isinstance(err, albert_errors.AlbertQuotaExhausted)


def test_ocr_no_access_d3_signature():
    fixture = load_fixture("P3_ocr_page0.json")
    err = albert_errors.classify_http_error(
        fixture["status"], json.dumps(fixture["body"]), fixture["headers"], endpoint="/v1/ocr"
    )
    assert (err.reason, err.status) == ("not_found", 404)
    assert albert_errors.is_ocr_access_denied(err) is True
    assert albert_errors.is_ocr_access_denied(
        albert_errors.classify_http_error(403, '{"detail":"Insufficient permissions."}', {}, endpoint="/v1/ocr")
    ) is True
    assert albert_errors.is_ocr_access_denied(
        albert_errors.classify_http_error(404, '{"detail":"Not Found"}', {}, endpoint="/v1/ocr")
    ) is False
    assert albert_errors.is_ocr_access_denied(
        albert_errors.classify_http_error(404, json.dumps(fixture["body"]), {}, endpoint="/v1/chat/completions")
    ) is False

    fake = FakeAlbert()
    with pytest.raises(AlbertPermanentError) as excinfo:
        _client(fake).ocr_document(b"%PDF-1.4 x")
    assert albert_errors.is_ocr_access_denied(excinfo.value) is True
