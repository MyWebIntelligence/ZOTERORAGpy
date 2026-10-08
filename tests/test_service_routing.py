"""Serveur + modèle par service, de bout en bout (sprint « configuration unifiée », lot L4).

* ``routed_model_for`` : mode historique inchangé, traduction des couples en
  formes routées (historiques pour OpenAI/OpenRouter/Albert, ``@<serveur>:<modèle>``
  pour les autres serveurs) ;
* ``rad_providers`` : la forme ``@`` donne un fournisseur ``compat:<serveur>``,
  son identifiant de clé et un client sur l'adresse déclarée ;
* notes, citations et recodage utilisent le client du serveur déclaré avec sa
  clé, envoient le nom du modèle tel quel, sans repli ;
* routes : modèle routé dans l'argv, clé du serveur exigée (403 sinon),
  configuration invalide refusée (400) avant tout lancement.
"""

from __future__ import annotations

import pytest

from scripts import rad_providers
from scripts.rad_settings.chat import legacy_mode, mistral_api_root, routed_model_for, route_model
from scripts.rad_settings.models import ServiceConfigError, resolve_service
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401
from tests.test_albert_routes import SESSION, _argv_value, _post, albert_env  # noqa: F401

OPENROUTER = "https://openrouter.ai/api/v1"
OPENAI = "https://api.openai.com/v1"
MISTRAL = "https://api.mistral.ai/v1"
ALBERT = "https://albert.api.etalab.gouv.fr/v1"
ANTHROPIC = "https://api.anthropic.com/v1"
LOCAL = "http://localhost:11434/v1"

UNIFIED = {
    "OPENROUTER_API_BASE_URL": OPENROUTER,
    "OPENAI_API_BASE_URL": OPENAI,
    "MISTRAL_API_BASE_URL": "https://api.mistral.ai",
    "ALBERT_BASE_URL": ALBERT,
    "ANTHROPIC_API_BASE_URL": ANTHROPIC,
    "LOCAL_API_BASE_URL": LOCAL,
    "LLM_DEFAULT_SERVER": OPENROUTER,
    "LLM_DEFAULT_MODEL": "google/gemini-3.8-flash",
}


# ---------------------------------------------------------------------------
# Traduction
# ---------------------------------------------------------------------------
def test_historical_mode_translates_nothing():
    """Sans ``LLM_DEFAULT_SERVER`` : aucun modèle routé, l'ancien chemin s'applique."""
    assert legacy_mode({}) is True
    assert routed_model_for("recode", "google/gemini-2.5-flash", env={}) is None
    assert routed_model_for("notes", None, env={"OPENROUTER_DEFAULT_MODEL": "x/y"}) is None


@pytest.mark.parametrize("service_values, expected", [
    ({}, "google/gemini-3.8-flash"),
    ({"LLM_RECODE_MODEL": "openai/gpt-4o-mini"}, "openai/gpt-4o-mini"),
    ({"LLM_RECODE_SERVER": OPENAI, "LLM_RECODE_MODEL": "gpt-4o-mini"}, "gpt-4o-mini"),
    ({"LLM_RECODE_SERVER": ALBERT, "LLM_RECODE_MODEL": "ministral-3-8b-instruct-2512"},
     "albert/ministral-3-8b-instruct-2512"),
    ({"LLM_RECODE_SERVER": MISTRAL, "LLM_RECODE_MODEL": "mistral-small-2603"}, "@mistral:mistral-small-2603"),
    ({"LLM_RECODE_SERVER": ANTHROPIC, "LLM_RECODE_MODEL": "claude-sonnet-5-5"}, "@anthropic:claude-sonnet-5-5"),
    ({"LLM_RECODE_SERVER": LOCAL, "LLM_RECODE_MODEL": "meta/llama-4"}, "@local:meta/llama-4"),
])
def test_couples_are_routed(service_values, expected):
    """Formes historiques pour OpenAI, OpenRouter et Albert ; ``@`` pour les autres."""
    routed = routed_model_for("recode", env=dict(UNIFIED, **service_values))
    assert routed.routed_model == expected


def test_official_servers_with_unusual_names_use_the_compat_form():
    """OpenRouter sans ``/`` ou OpenAI à une autre adresse : forme ``@``, jamais un mauvais client."""
    values = dict(UNIFIED, LLM_NOTES_MODEL="auto")
    assert routed_model_for("notes", env=values).routed_model == "@openrouter:auto"
    proxy = "https://proxy.example.org/v1"
    values = dict(UNIFIED, OPENAI_API_BASE_URL=proxy, LLM_NOTES_SERVER=proxy, LLM_NOTES_MODEL="gpt-4o-mini")
    assert routed_model_for("notes", env=values).routed_model == "@openai:gpt-4o-mini"


def test_run_override_and_albert_alias():
    """Champ de l'étape prioritaire ; ``albert/<id>`` saisi sans serveur désigne Albert."""
    assert routed_model_for("notes", "openai/gpt-5.6-luna-pro", env=UNIFIED).routed_model == "openai/gpt-5.6-luna-pro"
    assert routed_model_for("notes", "albert/gpt-oss-120b", env=UNIFIED).routed_model == "albert/gpt-oss-120b"
    assert routed_model_for("notes", "mistral-small-2603", MISTRAL, env=UNIFIED).routed_model == "@mistral:mistral-small-2603"


def test_book_service_has_its_own_couple():
    """Fiches de lecture : ``LLM_BOOK_*``, sinon le défaut."""
    values = dict(UNIFIED, LLM_BOOK_SERVER=MISTRAL, LLM_BOOK_MODEL="mistral-large-latest")
    assert routed_model_for("book", env=values).routed_model == "@mistral:mistral-large-latest"
    assert routed_model_for("notes", env=values).routed_model == "google/gemini-3.8-flash"


def test_unified_mode_refuses_unknown_server():
    """Adresse non déclarée au bloc 1 : refus (la clé ne part jamais ailleurs)."""
    with pytest.raises(ServiceConfigError):
        routed_model_for("recode", env=dict(UNIFIED, LLM_RECODE_SERVER="https://evil.example.org/v1"))


def test_local_keyword_is_ocr_only():
    """``local`` (moteurs OCR) refusé pour un service de langage."""
    values = dict(UNIFIED, LLM_RECODE_SERVER="local", LLM_RECODE_MODEL="docling")
    with pytest.raises(ServiceConfigError):
        routed_model_for("recode", env=values)


def test_mistral_address_with_or_without_v1():
    """``MISTRAL_API_BASE_URL`` s'écrit avec ou sans ``/v1`` : même serveur, racine OCR identique."""
    for written in ("https://api.mistral.ai", "https://api.mistral.ai/v1", "https://api.mistral.ai/v1/"):
        values = dict(UNIFIED, MISTRAL_API_BASE_URL=written, LLM_NOTES_SERVER=MISTRAL, LLM_NOTES_MODEL="m")
        assert routed_model_for("notes", env=values).routed_model == "@mistral:m"
        assert mistral_api_root(written) == "https://api.mistral.ai"


def test_route_model_on_resolved_choice():
    """``route_model`` sur un couple résolu directement."""
    choice = resolve_service("notes", dict(UNIFIED, LLM_NOTES_SERVER=ALBERT, LLM_NOTES_MODEL="gpt-oss-120b"))
    assert route_model(choice) == "albert/gpt-oss-120b"


# ---------------------------------------------------------------------------
# Résolveur et client
# ---------------------------------------------------------------------------
def test_resolver_understands_the_compat_form():
    """``@mistral:x`` → ``compat:mistral``, modèle ``x``, clé ``mistral_api_key``."""
    res = rad_providers.resolve_llm_provider("@mistral:mistral-small-2603", albert_enabled=False)
    assert res == ("compat:mistral", "mistral-small-2603", "mistral_api_key")
    assert rad_providers.legacy_provider("@local:meta/llama") == "compat:local"
    assert rad_providers.legacy_cache_provider_label("@local:meta/llama", prefer_openai=True) == "compat:local"
    assert rad_providers.resolve_llm_provider("google/x", albert_enabled=False).provider == "openrouter"
    with pytest.raises(ValueError):
        rad_providers.resolve_llm_provider("@nowhere:x", albert_enabled=False)


def test_compat_client_uses_declared_address_and_key(monkeypatch):
    """Client sur l'adresse déclarée, avec la clé donnée ; clé absente : refus explicite."""
    monkeypatch.setenv("ANTHROPIC_API_BASE_URL", ANTHROPIC)
    client = rad_providers.make_compat_client("compat:anthropic", "fake-anthropic-key-0001")
    assert str(client.base_url).rstrip("/") == ANTHROPIC
    assert client.api_key == "fake-anthropic-key-0001"
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        rad_providers.make_compat_client("compat:anthropic", None)
    monkeypatch.delenv("ANTHROPIC_API_BASE_URL")
    with pytest.raises(ValueError, match="non déclarée"):
        rad_providers.make_compat_client("compat:anthropic", "k")


class _FakeCompletions:
    """``chat.completions`` factice : enregistre les appels."""

    def __init__(self, calls):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Msg:
            content = "<p>réponse du serveur déclaré</p>"

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]
            usage = None

        return _Resp()


class _FakeClient:
    """Client compatible OpenAI factice."""

    def __init__(self, calls):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(calls)})()


def test_notes_use_the_declared_server_client(monkeypatch):
    """Notes : client du serveur déclaré, clé transmise, modèle tel quel, aucun repli."""
    from app.utils import llm_note_generator as gen

    calls, seen = [], {}

    def fake_factory(provider, api_key, env=None):
        seen.update(provider=provider, api_key=api_key)
        return _FakeClient(calls)

    monkeypatch.setattr(gen, "make_compat_client", fake_factory)
    out = gen._generate_with_llm("prompt", model="@mistral:mistral-small-2603", server_api_key="fake-mistral-key-9")
    assert "serveur déclaré" in out
    assert seen == {"provider": "compat:mistral", "api_key": "fake-mistral-key-9"}
    assert calls[0]["model"] == "mistral-small-2603"


def test_citations_use_the_declared_server_client(monkeypatch):
    """Citations : même règle."""
    from app.utils import citation_filter as cf

    calls, seen = [], {}

    def fake_factory(provider, api_key, env=None):
        seen.update(provider=provider, api_key=api_key)
        return _FakeClient(calls)

    monkeypatch.setattr(cf, "make_compat_client", fake_factory)
    out = cf._call_llm_api("prompt", "@anthropic:claude-sonnet-5-5", server_api_key="fake-anthropic-key-9")
    assert "serveur déclaré" in out
    assert seen == {"provider": "compat:anthropic", "api_key": "fake-anthropic-key-9"}
    assert calls[0]["model"] == "claude-sonnet-5-5"


def test_recode_uses_the_declared_server_client(monkeypatch):
    """Recodage : client du serveur déclaré (clé de l'environnement), sans ``seed`` ni repli."""
    from scripts import rad_chunk

    calls = []
    monkeypatch.setenv("LOCAL_API_BASE_URL", LOCAL)
    monkeypatch.setenv("LOCAL_API_KEY", "fake-local-key-9")
    seen = {}

    def fake_factory(provider, api_key, env=None):
        seen.update(provider=provider, api_key=api_key)
        return _FakeClient(calls)

    monkeypatch.setattr(rad_chunk, "make_compat_client", fake_factory)
    monkeypatch.setattr(rad_chunk, "_COMPAT_CLIENTS", {})
    texts, statuses = rad_chunk.gpt_recode_batch(["texte brut"], "consigne", model="@local:meta/llama-4")
    assert statuses == ["recoded"] and texts == ["<p>réponse du serveur déclaré</p>"]
    assert seen == {"provider": "compat:local", "api_key": "fake-local-key-9"}
    assert calls[0]["model"] == "meta/llama-4" and "seed" not in calls[0]
    assert rad_chunk.missing_llm_client_for_phase("initial", "@local:meta/llama-4") is False
    monkeypatch.delenv("LOCAL_API_KEY")
    assert rad_chunk.missing_llm_client_for_phase("initial", "@local:meta/llama-4") is True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@pytest.fixture
def unified(monkeypatch):
    """Mode unifié : serveurs et défauts déclarés dans l'environnement du serveur."""
    for name, value in UNIFIED.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-server-0001")


def test_route_sends_the_routed_model_with_the_server_key(albert_env, unified):
    """Recodage sur Mistral : argv routé, clé Mistral personnelle transmise."""
    env = albert_env
    form = {"path": SESSION, "server": MISTRAL, "model": "mistral-small-2603"}
    resp, errors = _post(env, "member_mistral", "/initial_text_chunking", form)
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    launches, _ = env.capture.take()
    assert _argv_value(launches[0]["argv"], "--model") == "@mistral:mistral-small-2603"
    assert launches[0]["env"]["MISTRAL_API_KEY"] == "member_mistral.mistral_api_key"


def test_route_default_couple_needs_the_default_server_key(albert_env, unified):
    """Couple par défaut (OpenRouter) : clé OpenRouter exigée, modèle routé tel quel."""
    env = albert_env
    resp, _ = _post(env, "member_mistral", "/initial_text_chunking", {"path": SESSION})
    assert resp.status_code == 403 and resp.json()["credential_required"] == "openrouter_api_key"
    resp, errors = _post(env, "member_openrouter", "/initial_text_chunking", {"path": SESSION})
    assert resp.status_code == 200 and errors == []
    launches, _ = env.capture.take()
    assert _argv_value(launches[0]["argv"], "--model") == "google/gemini-3.8-flash"


def test_route_new_server_keys_need_a_personal_key_for_members(albert_env, unified):
    """Anthropic : clé du .env pour l'administrateur ; un compte sans clé personnelle est refusé (403)."""
    env = albert_env
    form = {"path": SESSION, "server": ANTHROPIC, "model": "claude-sonnet-5-5"}
    resp, errors = _post(env, "admin", "/initial_text_chunking", form)
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    launches, _ = env.capture.take()
    assert _argv_value(launches[0]["argv"], "--model") == "@anthropic:claude-sonnet-5-5"
    resp, _ = _post(env, "member_keys", "/initial_text_chunking", form)
    assert resp.status_code == 403 and resp.json()["credential_required"] == "anthropic_api_key"
    assert env.capture.take() == ([], [])


def test_route_refuses_an_undeclared_server_before_launch(albert_env, unified):
    """Adresse inconnue : 400 ``model_not_configured``, rien n'est lancé."""
    env = albert_env
    form = {"path": SESSION, "server": "https://evil.example.org/v1", "model": "x"}
    resp, _ = _post(env, "admin", "/initial_text_chunking", form)
    assert resp.status_code == 400 and resp.json()["model_not_configured"] is True
    assert env.capture.take() == ([], [])
