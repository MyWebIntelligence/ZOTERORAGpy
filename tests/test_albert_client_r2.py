"""Tests des méthodes du client Albert ajoutées au sprint R2 : usage et transcription audio
(rerank, chat streamé et collections publiques retirés le 2026-10-03 avec les réponses sourcées).

Tout passe par ``tests/albert_fakes.FakeAlbert`` (``httpx.MockTransport``) ; la fixture réelle
``P23_stream_ministral.json`` (sonde live du 2026-10-02) est rejouée telle quelle pour vérifier
l'analyse SSE. Aucun appel réseau, clé factice seulement, sommeil injecté.
"""

from __future__ import annotations

import dataclasses
import io
import wave

import pytest

from scripts.rad_albert import catalog as albert_catalog
from scripts.rad_albert.client import (
    MAX_AUDIO_BYTES,
    AlbertClient,
    AlbertInputError,
    AlbertResponseError,
)
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_albert.errors import AlbertPermanentError
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert

STANDARD_HEADERS = frozenset({"host", "content-length", "content-type", "authorization", "user-agent"})


def _cfg(**overrides):
    """Configuration Albert ON avec surcharges."""
    return dataclasses.replace(AlbertConfig(enabled=True), **overrides)


def _client(fake, cfg=None, **kwargs):
    """Client branché sur le faux, sans limiteur ni attente réelle."""
    kwargs.setdefault("sleep", lambda _s: None)
    kwargs.setdefault("use_limiter", False)
    return AlbertClient(cfg or _cfg(), FAKE_ALBERT_KEY, transport=fake.transport, **kwargs)


def _wav(seconds=1.0):
    """WAV mono de silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * int(8000 * seconds))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Catalogue et configuration
# ---------------------------------------------------------------------------
def test_catalog_knows_rerank_and_audio_roles():
    assert albert_catalog.fallback_chain("rerank", today="2026-10-02") == ["bge-reranker-v2-m3"]
    assert albert_catalog.fallback_chain("audio", today="2026-10-02") == ["whisper-large-v3"]
    assert albert_catalog.fallback_chain("answer", today="2026-10-02") == [
        "gpt-oss-120b", "ministral-3-8b-instruct-2512"]
    assert albert_catalog.canonical_id("openweight-rerank") == "bge-reranker-v2-m3"
    assert albert_catalog.canonical_id("openweight-audio") == "whisper-large-v3"
    with pytest.raises(AlbertPermanentError):
        albert_catalog.check_endpoint_type("bge-m3", "rerank")
    assert albert_catalog.check_endpoint_type("whisper-large-v3", "/v1/audio/transcriptions") == \
        "automatic-speech-recognition"


def test_config_r2_defaults_are_off_and_conservative():
    cfg = AlbertConfig.from_env({})
    assert (cfg.audio_enabled, cfg.limiter_require_redis) == (False, False)
    assert cfg.data_policy == "compatible" and cfg.strict is False
    assert not any(name.startswith(("rag_", "rerank_")) for name in vars(cfg))  # fonctions retirées
    bad = AlbertConfig.from_env({"ALBERT_DATA_POLICY": "tout",
                                 "ALBERT_AUDIO_SEGMENT_SECONDS": "5", "ALBERT_AUDIO_LANGUAGE": "fr-FR!"})
    assert bad.data_policy == "albert_only"  # valeur inconnue : échec fermé
    assert bad.audio_segment_seconds == 600
    assert bad.audio_language == "fr"


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
def test_usage_sends_explicit_dates_and_paginates():
    fake = FakeAlbert()
    fake.usage_buckets = [{"object": "usage.bucket", "start_time": 1000 + i * 86400, "end_time": 1000 + (i + 1) * 86400,
                           "requests": i, "prompt_tokens": i, "completion_tokens": 0, "total_tokens": i, "cost": 0.0}
                          for i in range(3)]
    buckets = _client(fake).usage(1000, 1000 + 3 * 86400, endpoint="/v1/chat/completions", page_size=2)
    calls = fake.calls_to("GET", "/v1/usage")
    assert len(buckets) == 3 and len(calls) == 2
    assert calls[0].params["start_time"] == "1000" and calls[0].params["endpoint"] == "/v1/chat/completions"


@pytest.mark.parametrize("start,end", [(10, 5), (-1, 5), ("abc", 5)])
def test_usage_refuses_invalid_dates(start, end):
    fake = FakeAlbert()
    with pytest.raises(AlbertInputError):
        _client(fake).usage(start, end)
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Transcription audio
# ---------------------------------------------------------------------------
def test_transcribe_sends_multipart_and_normalises_segments():
    fake = FakeAlbert()
    fake.audio_reply = "Bonjour à toutes et à tous, bienvenue au séminaire."
    client = _client(fake)
    out = client.transcribe(_wav(), "entretien.wav", prompt="séminaire")
    call = fake.calls_to("POST", "/v1/audio/transcriptions")[0]
    assert call.form["model"] == "whisper-large-v3"
    assert call.form["language"] == "fr" and call.form["response_format"] == "verbose_json"
    assert call.files["file"][0] == "entretien.wav"
    assert set(call.headers) <= STANDARD_HEADERS
    assert out["text"] == "Bonjour à toutes et à tous, bienvenue au séminaire."
    assert out["segments"] and all(s["end"] >= s["start"] for s in out["segments"])
    assert out["model"] == "whisper-large-v3"
    assert client.ledger.records[-1]["role"] == "audio"


def test_transcribe_auto_language_omits_field():
    fake = FakeAlbert()
    _client(fake, _cfg(audio_language="")).transcribe(_wav(), "a.mp3")
    assert "language" not in fake.calls_to("POST", "/v1/audio/transcriptions")[0].form


@pytest.mark.parametrize("data,name,fmt", [
    (b"", "a.wav", "verbose_json"),
    (b"x" * (MAX_AUDIO_BYTES + 1), "a.wav", "verbose_json"),
    (b"RIFF", "a.m4a", "verbose_json"),
    (b"RIFF", "a.wav", "srt"),
])
def test_transcribe_refuses_before_sending(data, name, fmt):
    fake = FakeAlbert()
    with pytest.raises(AlbertInputError):
        _client(fake).transcribe(data, name, response_format=fmt)
    assert fake.calls == []


def test_transcribe_response_without_text_is_an_error():
    fake = FakeAlbert()
    fake.inject("POST", "/v1/audio/transcriptions", (200, {"id": "t", "segments": []}))
    with pytest.raises(AlbertResponseError):
        _client(fake).transcribe(_wav(), "a.wav")


def test_r2_endpoints_are_served_by_the_fake():
    from scripts.rad_albert.client import ENDPOINTS
    from tests.albert_fakes import ROUTES

    for endpoint in (("GET", "/v1/usage"), ("POST", "/v1/audio/transcriptions")):
        assert endpoint in ENDPOINTS and endpoint in ROUTES
    assert ("POST", "/v1/rerank") not in ENDPOINTS and ("POST", "/v1/search") not in ENDPOINTS
