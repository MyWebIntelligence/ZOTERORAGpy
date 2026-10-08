"""Tests des routes d'import audio Albert (``app/routes/albert_audio.py``, sprint R2, E1).

Harnais des goldens et de ``tests/test_albert_routes.py`` : base temporaire, sous-processus
capturés (aucun script réel), clés factices.
"""

from __future__ import annotations

import os

import pytest

from app.routes.albert_audio import parse_audio_logs
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401
from tests.test_albert_routes import _argv_value, _parse_events, albert_env, albert_on  # noqa: F401


@pytest.fixture
def audio_on(monkeypatch):
    """Capacité audio activée."""
    monkeypatch.setenv("ALBERT_AUDIO_ENABLED", "1")


def _upload(env, persona, names, data=b"RIFF0000WAVEfmt "):
    """POST /upload_audio avec des fichiers factices."""
    files = [("files", (name, data, "application/octet-stream")) for name in names]
    return env.client.post("/upload_audio", files=files, headers=env.headers[persona])


def test_audio_routes_are_unknown_unless_enabled(albert_env, monkeypatch):
    env = albert_env
    monkeypatch.setenv("ALBERT_ENABLED", "0")
    monkeypatch.setenv("ALBERT_AUDIO_ENABLED", "1")
    assert _upload(env, "member_albert", ["a.wav"]).status_code == 404
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_AUDIO_ENABLED", "0")
    assert _upload(env, "member_albert", ["a.wav"]).status_code == 404
    resp = env.client.post("/process_audio_sse", data={"path": "x"}, headers=env.headers["member_albert"])
    assert resp.status_code == 404
    paths = env.client.get("/openapi.json").json()["paths"]
    assert "/upload_audio" not in paths and "/process_audio_sse" not in paths


def test_upload_stores_server_generated_names(albert_env, albert_on, audio_on):
    env = albert_env
    resp = _upload(env, "member_albert", ["../../Entretien n°1.wav", "séminaire.m4a"])
    body = resp.json()
    assert resp.status_code == 200, body
    assert body["tree"][0].startswith("audio/01_") and body["tree"][0].endswith(".wav")
    assert body["tree"][1].startswith("audio/02_") and body["tree"][1].endswith(".m4a")
    session = env.uploads / body["path"]
    stored = sorted(os.listdir(session / "audio"))
    assert len(stored) == 2 and all(".." not in name and "/" not in name for name in stored)


def test_upload_refuses_unsupported_formats(albert_env, albert_on, audio_on):
    env = albert_env
    resp = _upload(env, "member_albert", ["a.wav", "script.exe"])
    assert resp.status_code == 400
    assert not [p for p in os.listdir(env.uploads) if "script" in p]


def test_process_audio_launches_script_with_albert_key_only(albert_env, albert_on, audio_on):
    env = albert_env
    path = _upload(env, "member_albert", ["a.wav"]).json()["path"]
    env.capture.take()
    resp = env.client.post("/process_audio_sse", data={"path": path, "language": "fr", "prompt": "DINUM, RGPD"},
                           headers=env.headers["member_albert"])
    assert resp.status_code == 200
    launches, _ = env.capture.take()
    assert len(launches) == 1
    launch = launches[0]
    assert launch["script"] == "rad_audio.py"
    assert _argv_value(launch["argv"], "--language") == "fr"
    assert "--prompt=DINUM, RGPD" in launch["argv"]
    assert launch["env"]["ALBERT_API_KEY"] == "member_albert.albert_api_key"
    assert launch["env"]["OPENAI_API_KEY"] is None
    assert launch["timeout"] == 21600


def test_process_audio_refusals(albert_env, albert_on, audio_on):
    env = albert_env
    path = _upload(env, "member_albert", ["a.wav"]).json()["path"]
    env.capture.take()
    bad = env.client.post("/process_audio_sse", data={"path": path, "language": "français!"},
                          headers=env.headers["member_albert"])
    assert _parse_events(bad.text)[0]["type"] == "error"
    nokey_path = _upload(env, "member_mistral", ["b.wav"]).json()["path"]
    nokey = env.client.post("/process_audio_sse", data={"path": nokey_path}, headers=env.headers["member_mistral"])
    events = _parse_events(nokey.text)
    assert events[0]["credential_required"] == "albert_api_key"
    foreign = env.client.post("/process_audio_sse", data={"path": path}, headers=env.headers["member_mistral"])
    assert _parse_events(foreign.text)[0]["type"] == "error"
    assert env.capture.take() == ([], [])


def test_parse_audio_logs():
    assert parse_audio_logs("Fichier 2/5 : entretien.wav") == {
        "type": "progress", "current": 2, "total": 5, "message": "Fichier 2/5 : entretien.wav"}
    assert parse_audio_logs("  Segment 3/7 transcrit")["type"] == "log"
    abort = parse_audio_logs("Albert abort: kind=account reason=quota_exhausted credential_required=albert_api_key")
    assert abort["type"] == "error" and abort["credential_required"] == "albert_api_key"
    assert parse_audio_logs("ligne quelconque") is None


def test_upload_total_size_is_capped(albert_env, albert_on, audio_on, monkeypatch):
    env = albert_env
    monkeypatch.setenv("UPLOAD_MAX_UNZIPPED_MB", "1")
    chunk = b"\x00" * (600 * 1024)
    resp = _upload(env, "member_albert", ["a.wav", "b.wav"], data=chunk)
    assert resp.status_code == 413
    assert not [p for p in os.listdir(env.uploads) if p.endswith("_a")]
