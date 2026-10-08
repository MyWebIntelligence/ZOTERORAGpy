"""Embeddings et transcription : couple serveur + modèle (sprint « configuration unifiée », lot L6).

* ``declared_choice`` : rien de déclaré ou mode historique → ``None`` (règle
  historique) ; un modèle seul utilise le serveur par défaut ;
* ``albert_overlay`` / ``AlbertConfig.from_env`` : un modèle déclaré active la
  transcription (Albert seulement) ; rerank et réponses sourcées retirés le 2026-10-03 ;
* ``EmbeddingConfig.from_env`` : OpenAI (espace historique à l'identique pour
  ``text-embedding-3-large``), Albert, serveur compatible (dimension mesurée),
  politique ``albert_only`` ;
* phase dense : G3 inchangé à configuration équivalente, client compatible
  seul appelé, champs d'espace écrits, aucun état gardé d'un run à l'autre ;
* routes : couple transmis au script, clé du seul serveur retenu, refus avant
  lancement d'un serveur non déclaré.
"""

from __future__ import annotations

import json

import pytest

from scripts import rad_providers
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_albert.errors import AlbertDisabledError
from scripts.rad_settings.capabilities import albert_overlay, declared_choice
from scripts.rad_settings.models import ServiceConfigError
from tests import test_albert_off_golden as golden
from tests.test_albert_embeddings import (  # noqa: F401
    _assert_matches_golden_bytes, _assert_matches_golden_json, _clean_env, chunk_env,
)

rc = golden.rc

OPENROUTER = "https://openrouter.ai/api/v1"
OPENAI = "https://api.openai.com/v1"
MISTRAL = "https://api.mistral.ai/v1"
ALBERT = "https://albert.api.etalab.gouv.fr/v1"
LOCAL = "http://localhost:11434/v1"

UNIFIED = {
    "OPENROUTER_API_BASE_URL": OPENROUTER,
    "OPENAI_API_BASE_URL": OPENAI,
    "MISTRAL_API_BASE_URL": MISTRAL,
    "ALBERT_BASE_URL": ALBERT,
    "LOCAL_API_BASE_URL": LOCAL,
    "LLM_DEFAULT_SERVER": OPENROUTER,
    "LLM_DEFAULT_MODEL": "google/gemini-3.8-flash",
}


def _env(**extra):
    return dict(UNIFIED, **extra)


@pytest.fixture
def unified(monkeypatch):
    """Mode unifié dans l'environnement du processus."""
    for name, value in UNIFIED.items():
        monkeypatch.setenv(name, value)


# ---------------------------------------------------------------------------
# Règle du couple
# ---------------------------------------------------------------------------
def test_nothing_declared_or_historical_mode_keeps_the_old_rule():
    """Aucune variable du couple, ou pas de ``LLM_DEFAULT_SERVER`` : ``None``."""
    assert declared_choice("embedding", _env()) is None
    assert declared_choice("audio", {"AUDIO_MODEL": "whisper-large-v3"}) is None
    assert albert_overlay({"AUDIO_MODEL": "whisper-large-v3"}) == {}


def test_model_alone_uses_the_default_server():
    """Un modèle seul : serveur par défaut (OpenRouter sait faire des embeddings, pas de transcription)."""
    choice = declared_choice("embedding", _env(EMBEDDING_MODEL="openai/text-embedding-3-small"))
    assert choice.server.key == "openrouter" and choice.model == "openai/text-embedding-3-small"
    with pytest.raises(ServiceConfigError):
        declared_choice("audio", _env(AUDIO_MODEL="whisper-large-v3"))


def test_step_fields_override_the_env():
    """Champs de l'étape prioritaires sur le ``.env``."""
    values = _env(EMBEDDING_SERVER=OPENAI, EMBEDDING_MODEL="text-embedding-3-large")
    choice = declared_choice("embedding", values, override_server=MISTRAL, override_model="mistral-embed")
    assert (choice.server.key, choice.model) == ("mistral", "mistral-embed")


def test_audio_model_enables_the_albert_switch():
    """Un modèle déclaré sur Albert active la transcription (D11) ; l'interrupteur historique est ignoré."""
    values = _env(ALBERT_ENABLED="1", AUDIO_SERVER=ALBERT, AUDIO_MODEL="whisper-large-v3", ALBERT_AUDIO_ENABLED="0")
    assert albert_overlay(values) == {"ALBERT_AUDIO_MODEL": "whisper-large-v3", "ALBERT_AUDIO_ENABLED": "1"}
    cfg = AlbertConfig.from_env(values)
    assert cfg.audio_enabled is True and cfg.audio_model == "whisper-large-v3"


def test_audio_elsewhere_than_albert_is_refused():
    """Transcription sur le serveur local : refus explicite (Albert seulement)."""
    with pytest.raises(ServiceConfigError, match="Albert"):
        albert_overlay(_env(AUDIO_SERVER=LOCAL, AUDIO_MODEL="whisper"))


# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------
def test_openai_couple_gives_the_historical_space():
    """OpenAI officiel + text-embedding-3-large : espace historique, à l'identique."""
    cfg = rad_providers.EmbeddingConfig.from_env(_env(EMBEDDING_SERVER=OPENAI, EMBEDDING_MODEL="text-embedding-3-large"))
    assert cfg.space is rad_providers.OPENAI_DEFAULT and cfg.compat_server == ""
    small = rad_providers.EmbeddingConfig.from_env(_env(EMBEDDING_SERVER=OPENAI, EMBEDDING_MODEL="text-embedding-3-small"))
    assert (small.space.dim, small.space.is_default, small.provider) == (1536, False, "openai")
    with pytest.raises(ValueError, match="non pris en charge"):
        rad_providers.EmbeddingConfig.from_env(_env(EMBEDDING_SERVER=OPENAI, EMBEDDING_MODEL="text-embedding-9"))


def test_albert_couple_needs_albert_enabled():
    """Albert : espace bge-m3 si Albert est activé, sinon refus."""
    values = _env(EMBEDDING_SERVER=ALBERT, EMBEDDING_MODEL="bge-m3")
    with pytest.raises(AlbertDisabledError):
        rad_providers.EmbeddingConfig.from_env(values)
    cfg = rad_providers.EmbeddingConfig.from_env(dict(values, ALBERT_ENABLED="1"))
    assert cfg.provider == "albert" and cfg.space.dim == 1024


def test_compatible_server_is_measured_later():
    """Mistral : client compatible, clé du serveur, dimension inconnue jusqu'au premier appel."""
    cfg = rad_providers.EmbeddingConfig.from_env(_env(EMBEDDING_SERVER=MISTRAL, EMBEDDING_MODEL="mistral-embed"))
    assert (cfg.provider, cfg.compat_server, cfg.key_var, cfg.space.dim) == ("mistral", "mistral", "MISTRAL_API_KEY", 0)


def test_albert_only_policy_refuses_other_servers():
    """``albert_only`` : embeddings sur un autre serveur qu'Albert refusés."""
    values = _env(ALBERT_ENABLED="1", ALBERT_DATA_POLICY="albert_only", EMBEDDING_SERVER=MISTRAL,
                  EMBEDDING_MODEL="mistral-embed")
    with pytest.raises(ValueError, match="albert_only"):
        rad_providers.EmbeddingConfig.from_env(values)


def test_historical_mode_ignores_the_couple():
    """Sans ``LLM_DEFAULT_SERVER`` : ``EMBEDDING_PROVIDER`` seul compte."""
    cfg = rad_providers.EmbeddingConfig.from_env({"EMBEDDING_SERVER": MISTRAL, "EMBEDDING_MODEL": "mistral-embed"})
    assert cfg.space is rad_providers.OPENAI_DEFAULT


# ---------------------------------------------------------------------------
# Phase dense
# ---------------------------------------------------------------------------
def test_g3_unchanged_with_the_equivalent_couple(chunk_env, unified, monkeypatch, tmp_path):
    """Couple OpenAI + text-embedding-3-large : fichier et requêtes du golden G3, à l'octet."""
    monkeypatch.setenv("EMBEDDING_SERVER", OPENAI)
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    output_path = str(tmp_path / "output_chunks_with_embeddings.json")
    assert rc.generate_and_save_embeddings(golden._write_chunks_file(tmp_path), output_path) == output_path
    with open(output_path, "rb") as fh:
        _assert_matches_golden_bytes("g3_dense_output.json", fh.read())
    _assert_matches_golden_json("g3_embeddings_create_kwargs.json", chunk_env.calls)


def test_compatible_server_embeds_with_its_own_client(chunk_env, unified, monkeypatch, tmp_path):
    """Mistral : dimension mesurée, client compatible seul appelé, champs d'espace écrits."""
    monkeypatch.setenv("EMBEDDING_SERVER", MISTRAL)
    monkeypatch.setenv("EMBEDDING_MODEL", "mistral-embed")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-mistral-embed-0001")
    made = []

    def fake_make(provider, api_key, env=None):
        made.append((provider, api_key))
        return golden._FakeLLMClient("mistral", chunk_env.calls)

    monkeypatch.setattr(rc, "make_compat_client", fake_make)
    assert rc.albert_embed_startup_exception() is None
    output_path = str(tmp_path / "out.json")
    rc.generate_and_save_embeddings(golden._write_chunks_file(tmp_path), output_path)
    assert made == [("compat:mistral", "fake-mistral-embed-0001")]
    assert {c["client"] for c in chunk_env.calls} == {"mistral"}
    assert chunk_env.calls[0]["kwargs"]["input"] == ["dimension"]
    dim = len(golden._fake_vector("x"))
    chunks = json.loads(open(output_path, encoding="utf-8").read())
    embedded = [c for c in chunks if c.get("embedding")]
    assert embedded and all((c["embedding_provider"], c["embedding_model"], c["embedding_dim"]) == ("mistral", "mistral-embed", dim)
                            for c in embedded)
    # Run suivant sans couple : client OpenAI historique, aucun état gardé.
    chunk_env.calls.clear()
    monkeypatch.delenv("EMBEDDING_SERVER")
    monkeypatch.delenv("EMBEDDING_MODEL")
    rc.generate_and_save_embeddings(golden._write_chunks_file(tmp_path), str(tmp_path / "out2.json"))
    assert {c["client"] for c in chunk_env.calls} == {"openai"}


def test_compatible_server_without_key_stops_before_any_call(chunk_env, unified, monkeypatch):
    """Clé du serveur absente : erreur de démarrage (sortie 2), sauf pour le serveur local."""
    monkeypatch.setenv("EMBEDDING_SERVER", MISTRAL)
    monkeypatch.setenv("EMBEDDING_MODEL", "mistral-embed")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert "MISTRAL_API_KEY" in str(rc.albert_embed_startup_exception())
    monkeypatch.setenv("EMBEDDING_SERVER", LOCAL)
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    assert rc.albert_embed_startup_exception() is None
    assert rc._dense_requires_openai() is False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: E402,F401
from tests.test_albert_routes import SESSION, _post, albert_env  # noqa: E402,F401


def test_route_sends_the_couple_with_the_server_key(albert_env, unified, monkeypatch):
    """Couple Mistral : clé Mistral exigée, couple transmis au script ; sans clé : 403."""
    from app.services import embedding_target

    built = []
    original = embedding_target.explicit_embedding_env
    monkeypatch.setattr(embedding_target, "explicit_embedding_env",
                        lambda user, choice: built.append(original(user, choice)) or built[-1])
    monkeypatch.setenv("EMBEDDING_SERVER", MISTRAL)
    monkeypatch.setenv("EMBEDDING_MODEL", "mistral-embed")
    env = albert_env
    resp, errors = _post(env, "member_mistral", "/dense_embedding_generation", {"path": SESSION})
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    launches, _ = env.capture.take()
    assert launches[0]["env"]["MISTRAL_API_KEY"] == "member_mistral.mistral_api_key"
    assert built[0]["EMBEDDING_SERVER"] == MISTRAL and built[0]["EMBEDDING_MODEL"] == "mistral-embed"
    resp, _ = _post(env, "member_openrouter", "/dense_embedding_generation", {"path": SESSION})
    assert resp.status_code == 403 and resp.json()["credential_required"] == "mistral_api_key"


def test_route_step_fields_and_undeclared_server(albert_env, unified):
    """Champs de l'étape : couple retenu ; adresse non déclarée : 400 avant tout lancement."""
    env = albert_env
    form = {"path": SESSION, "server": OPENAI, "model": "text-embedding-3-large"}
    resp, errors = _post(env, "member_keys", "/dense_embedding_generation", form)
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    env.capture.take()
    form = {"path": SESSION, "server": "https://evil.example.org/v1", "model": "x"}
    resp, _ = _post(env, "admin", "/dense_embedding_generation", form)
    assert resp.status_code == 400 and resp.json()["model_not_configured"] is True
    assert env.capture.take() == ([], [])


def test_celery_env_carries_the_couple(albert_env, unified, monkeypatch):
    """Celery : même couple, même clé que la route."""
    from app.tasks import runner

    monkeypatch.setenv("EMBEDDING_SERVER", MISTRAL)
    monkeypatch.setenv("EMBEDDING_MODEL", "mistral-embed")
    env = runner.build_task_env(albert_env.users["member_mistral"], runner.STAGE_DENSE)
    assert env["EMBEDDING_SERVER"] == MISTRAL and env["EMBEDDING_MODEL"] == "mistral-embed"
    with pytest.raises(Exception) as caught:
        runner.build_task_env(albert_env.users["member_openrouter"], runner.STAGE_DENSE)
    assert getattr(caught.value, "credential_key", None) == "mistral_api_key"
