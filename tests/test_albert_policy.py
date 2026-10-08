"""Tests de la politique d'inférence ``ALBERT_DATA_POLICY`` (sprint Albert R2, lot L6).

* bibliothèque ``scripts/rad_albert/policy.py`` et couche web ``app/core/albert_policy.py`` ;
* routes du pipeline (harnais des goldens et de ``tests/test_albert_routes.py``) : en
  ``albert_only``, aucun lancement vers OpenAI, OpenRouter ou Mistral, modèle vide remplacé
  par le défaut Albert, clés externes retirées de l'environnement des sous-processus ;
* voie Celery (``runner.build_task_env``), filtre de citations, utilitaires de notes ;
* scripts : ``rad_chunk.apply_inference_policy`` et chaîne OCR de ``rad_dataframe``.

Aucun réseau ; politique ``compatible`` = comportement inchangé.
"""

from __future__ import annotations

from datetime import date

import fitz
import pytest

import scripts.rad_dataframe as rad
from app.core import albert_policy as web_policy
from scripts.rad_albert import policy
from scripts.rad_albert.config import AlbertConfig
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401
from tests.test_albert_routes import (  # noqa: F401
    ALBERT_CHAT_MODEL,
    SESSION,
    _argv_value,
    _fresh_ocr_session,
    _parse_events,
    _post,
    albert_env,
    albert_on,
)

STRICT = AlbertConfig(enabled=True, data_policy="albert_only")
COMPATIBLE = AlbertConfig(enabled=True)
TODAY = date(2026, 10, 2)


# ---------------------------------------------------------------------------
# Bibliothèque
# ---------------------------------------------------------------------------
def test_compatible_policy_changes_nothing():
    assert policy.check_chat_model("gpt-4o-mini", COMPATIBLE) == "gpt-4o-mini"
    assert policy.check_chat_model(None, COMPATIBLE) is None
    assert policy.check_embedding_provider("openai", COMPATIBLE) == "openai"
    assert policy.ocr_external_allowed(COMPATIBLE) is True
    assert policy.describe(COMPATIBLE)["blocked_providers"] == []


def test_strict_policy_defaults_and_refusals():
    assert policy.check_chat_model("", STRICT, role="recode", today=TODAY) == "albert/ministral-3-8b-instruct-2512"
    assert policy.check_chat_model(None, STRICT, role="notes", today=TODAY) == "albert/gpt-oss-120b"
    assert policy.check_chat_model("ALBERT/gpt-oss-120b", STRICT) == "ALBERT/gpt-oss-120b"
    for model in ("gpt-4o-mini", "google/gemini-2.5-flash", "openai/gpt-4o"):
        with pytest.raises(policy.AlbertPolicyError) as info:
            policy.check_chat_model(model, STRICT)
        assert info.value.reason == "policy_violation" and info.value.capability == "chat"
    assert policy.check_embedding_provider("", STRICT) == "albert"
    with pytest.raises(policy.AlbertPolicyError):
        policy.check_embedding_provider("openai", STRICT)
    assert policy.ocr_external_allowed(STRICT) is False
    assert policy.describe(STRICT)["blocked_providers"] == ["openai", "openrouter", "mistral"]


def test_strict_policy_requires_albert():
    off = AlbertConfig(enabled=False, data_policy="albert_only")
    with pytest.raises(policy.AlbertPolicyError) as info:
        policy.check_chat_model("albert/gpt-oss-120b", off)
    assert info.value.reason == "policy_requires_albert"
    assert isinstance(info.value, ValueError)


def test_restrict_subprocess_env(monkeypatch):
    env = {"OPENAI_API_KEY": "k1", "MISTRAL_API_KEY": "k2", "ALBERT_API_KEY": "k3", "RAGPY_DOTENV_DENY": "PINECONE_API_KEY"}
    monkeypatch.delenv("ALBERT_DATA_POLICY", raising=False)
    assert web_policy.restrict_subprocess_env(dict(env)) == env
    monkeypatch.setenv("ALBERT_DATA_POLICY", "albert_only")
    out = web_policy.restrict_subprocess_env(dict(env))
    assert "OPENAI_API_KEY" not in out and "MISTRAL_API_KEY" not in out and out["ALBERT_API_KEY"] == "k3"
    assert out["RAGPY_DOTENV_DENY"] == "MISTRAL_API_KEY,OPENAI_API_KEY,OPENROUTER_API_KEY,PINECONE_API_KEY"


def test_require_albert_model_for_stored_configurations(monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.delenv("ALBERT_DATA_POLICY", raising=False)
    web_policy.require_albert_model("gpt-4o-mini")
    monkeypatch.setenv("ALBERT_DATA_POLICY", "albert_only")
    web_policy.require_albert_model("albert/ministral-3-8b-instruct-2512")
    for model in ("gpt-4o-mini", "", None):
        with pytest.raises(ValueError):
            web_policy.require_albert_model(model)


def test_citation_resolver_and_note_utilities_refuse_external_models(monkeypatch):
    from app.routes.citations import _resolve_citation_model
    from app.utils.llm_note_generator import resolve_llm_route

    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_DATA_POLICY", "albert_only")
    with pytest.raises(ValueError):
        _resolve_citation_model("gpt-4o-mini")
    with pytest.raises(ValueError):
        resolve_llm_route("google/gemini-2.5-flash")
    assert _resolve_citation_model(ALBERT_CHAT_MODEL).provider == "albert"
    monkeypatch.setenv("ALBERT_DATA_POLICY", "compatible")
    assert _resolve_citation_model("gpt-4o-mini").provider == "openai"


# ---------------------------------------------------------------------------
# Routes du pipeline
# ---------------------------------------------------------------------------
@pytest.fixture
def strict(monkeypatch):
    """Politique albert_only (Albert activé par ``albert_on``)."""
    monkeypatch.setenv("ALBERT_DATA_POLICY", "albert_only")


def test_strict_chunking_refuses_external_model_before_launch(albert_env, albert_on, strict):
    env = albert_env
    resp, errors = _post(env, "member_albert", "/initial_text_chunking", {"path": SESSION, "model": "gpt-4o-mini"})
    assert resp.status_code == 400 and errors == []
    assert resp.json()["reason"] == "policy_violation" and resp.json()["policy"] == "albert_only"
    resp, errors = _post(env, "member_albert", "/initial_text_chunking_sse", {"path": SESSION, "model": "gpt-4o-mini"})
    events = _parse_events(resp.text)
    assert len(events) == 1 and events[0]["type"] == "error" and events[0]["reason"] == "policy_violation"
    assert env.capture.take() == ([], [])


def test_strict_chunking_empty_model_uses_albert_default_and_strips_external_keys(albert_env, albert_on, strict):
    env = albert_env
    for url in ("/initial_text_chunking", "/initial_text_chunking_sse"):
        resp, errors = _post(env, "member_albert", url, {"path": SESSION})
        assert resp.status_code == 200 and errors == [], (url, resp.text[:300])
        launches, _ = env.capture.take()
        assert _argv_value(launches[0]["argv"], "--model").startswith("albert/")
        facts = launches[0]["env"]
        assert facts["OPENAI_API_KEY"] is None and facts["MISTRAL_API_KEY"] is None
        assert {"OPENAI_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY"} <= set(facts["deny"])
    # Admin : ses clés .env externes ne parviennent pas non plus au script.
    resp, errors = _post(env, "admin", "/initial_text_chunking", {"path": SESSION, "model": ALBERT_CHAT_MODEL})
    assert resp.status_code == 200 and errors == []
    launches, _ = env.capture.take()
    assert launches[0]["env"]["OPENAI_API_KEY"] is None
    assert "OPENAI_API_KEY" in launches[0]["env"]["deny"]


def test_strict_dense_refuses_openai_and_defaults_to_albert(albert_env, albert_on, strict):
    env = albert_env
    resp, _ = _post(env, "member_albert", "/dense_embedding_generation", {"path": SESSION, "embedding_provider": "openai"})
    assert resp.status_code == 400 and "bge-m3" in resp.json()["error"]
    assert env.capture.take() == ([], [])
    resp, _ = _post(env, "member_albert", "/dense_embedding_generation", {"path": SESSION})
    assert resp.status_code == 200, resp.text[:300]
    launches, _ = env.capture.take()
    assert launches[0]["env"]["ALBERT_API_KEY"] == "member_albert.albert_api_key"
    assert launches[0]["env"]["OPENAI_API_KEY"] is None


def test_strict_ocr_needs_no_mistral_key(albert_env, albert_on, strict, monkeypatch):
    env = albert_env
    monkeypatch.setenv("OCR_ENABLE_ALBERT", "1")
    path = _fresh_ocr_session(env, "gsess-ocr-strict-albert")
    resp, errors = _post(env, "member_albert", "/process_dataframe", {"path": path})
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    facts = env.capture.take()[0][0]["env"]
    assert facts["ALBERT_API_KEY"] == "member_albert.albert_api_key" and facts["MISTRAL_API_KEY"] is None
    # Utilisateur qui n'a qu'une clé Mistral : OCR local seulement, la clé Mistral n'est jamais transmise.
    path = _fresh_ocr_session(env, "gsess-ocr-strict-mistral")
    resp, errors = _post(env, "member_mistral", "/process_dataframe", {"path": path})
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    facts = env.capture.take()[0][0]["env"]
    assert facts["MISTRAL_API_KEY"] is None and "MISTRAL_API_KEY" in facts["deny"]


def test_strict_policy_with_albert_disabled_refuses_every_launch(albert_env, monkeypatch, strict):
    env = albert_env
    monkeypatch.setenv("ALBERT_ENABLED", "0")
    resp, _ = _post(env, "member_keys", "/initial_text_chunking", {"path": SESSION, "model": "gpt-4o-mini"})
    assert resp.status_code == 400 and resp.json()["reason"] == "policy_requires_albert"
    resp, _ = _post(env, "member_keys", "/dense_embedding_generation", {"path": SESSION})
    assert resp.status_code == 400
    path = _fresh_ocr_session(env, "gsess-ocr-strict-off")
    resp, _ = _post(env, "member_mistral", "/process_dataframe", {"path": path})
    assert resp.status_code == 400 and resp.json()["reason"] == "policy_requires_albert"
    assert env.capture.take() == ([], [])


def test_compatible_policy_keeps_historical_launch(albert_env, albert_on, monkeypatch):
    env = albert_env
    monkeypatch.setenv("ALBERT_DATA_POLICY", "compatible")
    resp, errors = _post(env, "member_keys", "/initial_text_chunking", {"path": SESSION})
    assert resp.status_code == 200 and errors == []
    launches, _ = env.capture.take()
    assert _argv_value(launches[0]["argv"], "--model") == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Celery
# ---------------------------------------------------------------------------
def test_runner_strict_env_and_model(monkeypatch):
    from types import SimpleNamespace

    from app.tasks import runner

    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_DATA_POLICY", "albert_only")
    assert runner.policy_chat_model("").startswith("albert/")
    with pytest.raises(ValueError):
        runner.policy_chat_model("gpt-4o-mini")
    assert runner.resolve_embedding_provider("") == "albert"
    with pytest.raises(ValueError):
        runner.resolve_embedding_provider("openai")

    captured = {}

    def fake_build(user, required_keys=None):
        """Environnement factice contenant des clés externes."""
        captured["keys"] = list(required_keys or [])
        return {"OPENAI_API_KEY": "x", "MISTRAL_API_KEY": "y", "ALBERT_API_KEY": "z"}

    monkeypatch.setattr(runner, "build_subprocess_env", fake_build)
    monkeypatch.setattr(runner, "ocr_albert_active", lambda user: True)
    env = runner.build_task_env(SimpleNamespace(is_admin=False, id=1), runner.STAGE_EXTRACTION)
    assert captured["keys"] == ["albert_api_key"]
    assert "OPENAI_API_KEY" not in env and "MISTRAL_API_KEY" not in env and env["ALBERT_API_KEY"] == "z"


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------
def test_rad_chunk_cli_policy():
    import scripts.rad_chunk as rc

    assert rc.apply_inference_policy("initial", None, False, env={}) == ("gpt-4o-mini", None)
    assert rc.apply_inference_policy("initial", "x/y", True, env={})[0] == "x/y"
    env = {"ALBERT_ENABLED": "1", "ALBERT_DATA_POLICY": "albert_only"}
    model, exc = rc.apply_inference_policy("all", None, False, env=env)
    assert exc is None and model.startswith("albert/") and env["EMBEDDING_PROVIDER"] == "albert"
    model, exc = rc.apply_inference_policy("initial", "gpt-4o-mini", True, env=dict(env))
    assert exc is not None and rc.albert_abort_marker(exc) == "Albert abort: kind=config reason=policy_violation"
    _model, exc = rc.apply_inference_policy("dense", None, False, env=dict(env, EMBEDDING_PROVIDER="openai"))
    assert exc is not None
    _model, exc = rc.apply_inference_policy("initial", None, False, env={"ALBERT_DATA_POLICY": "albert_only"})
    assert rc.albert_abort_marker(exc) == "Albert abort: kind=config reason=policy_requires_albert"
    assert "albert_whisper" in rc.RECODE_SKIP_PROVIDERS


def _text_pdf(path):
    """PDF d'une page avec une couche texte suffisante pour l'extraction legacy."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 60), "Texte de la page. " * 40, fontsize=9)
    doc.save(str(path))
    doc.close()
    return str(path)


def test_rad_dataframe_strict_policy_skips_mistral_and_openai(monkeypatch, tmp_path):
    calls = []

    def mistral(*args, **kwargs):
        """Maillon Mistral interdit en albert_only."""
        calls.append("mistral")
        raise AssertionError("Mistral appelé malgré la politique albert_only")

    monkeypatch.setattr(rad, "ALBERT_STRICT_POLICY", True)
    monkeypatch.setattr(rad, "OCR_ENABLE_ALBERT", False)
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", "fake-mistral")
    monkeypatch.setattr(rad, "_extract_text_with_mistral", mistral)
    monkeypatch.setattr(rad, "OCR_ENABLE_OPENAI_FALLBACK", True)
    monkeypatch.setattr(rad, "_extract_text_with_openai", mistral)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai")
    monkeypatch.setattr(rad, "_local_ocr_available", lambda *a, **k: False)
    result = rad.extract_text_with_ocr(_text_pdf(tmp_path / "a.pdf"), return_details=True)
    assert result.provider == "legacy" and calls == []


def test_rad_dataframe_compatible_policy_still_uses_mistral(monkeypatch, tmp_path):
    from collections import namedtuple

    outcome = namedtuple("Outcome", "text partial pages_done pages_total error")
    monkeypatch.setattr(rad, "ALBERT_STRICT_POLICY", False)
    monkeypatch.setattr(rad, "OCR_ENABLE_ALBERT", False)
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", "fake-mistral")
    monkeypatch.setattr(rad, "_extract_text_with_mistral",
                        lambda *a, **k: outcome("Texte Mistral " * 80, False, 1, 1, None))
    result = rad.extract_text_with_ocr(_text_pdf(tmp_path / "b.pdf"), return_details=True)
    assert result.provider == "mistral"


def test_image_attachments_only_with_albert_ocr(monkeypatch):
    monkeypatch.setattr(rad, "OCR_ENABLE_ALBERT", False)
    assert ".png" not in rad.supported_attachment_extensions()
    monkeypatch.setattr(rad, "OCR_ENABLE_ALBERT", True)
    monkeypatch.setattr(rad, "ALBERT_API_KEY", "fake-albert-key-0001")
    assert {".png", ".jpg", ".tiff"} <= set(rad.supported_attachment_extensions())
