"""Choix personnels de serveur et de modèle (sprint « configuration unifiée », lot L9).

* ``GET/PUT /users/me/settings`` : choix validés (serveur déclaré au bloc 1,
  capable du service, Albert pour les réponses sourcées, modèle sans blanc),
  refus global, valeur vide = héritage du ``.env``, choix propres à chaque compte ;
* application : le champ de l'étape l'emporte, puis le choix personnel, puis le
  ``.env`` (recodage, OCR explicite) ;
* clés personnelles des serveurs ajoutés (Anthropic, Google, DeepSeek, Qwen,
  GLM) : listées et acceptées seulement quand l'adresse est déclarée, retirées
  des sous-processus non-admin et interdites de rechargement depuis le ``.env`` ;
* pages : carte « Mes choix de modèles » et champ serveur des étapes en mode
  unifié seulement (en mode historique, les goldens G9 figent les pages).
"""

from __future__ import annotations

import pytest

from app.core import credentials as creds
from app.models.user_setting import UserSetting
from app.services import ocr_target
from app.services.user_settings import effective_services, get_user_settings, validate_user_settings
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401
from tests.test_albert_routes import SESSION, _argv_value, _fresh_ocr_session, _post, albert_env  # noqa: F401

OPENROUTER = "https://openrouter.ai/api/v1"
MISTRAL = "https://api.mistral.ai/v1"
ALBERT = "https://albert.api.etalab.gouv.fr/v1"
ANTHROPIC = "https://api.anthropic.com/v1"

UNIFIED = {
    "OPENROUTER_API_BASE_URL": OPENROUTER,
    "MISTRAL_API_BASE_URL": MISTRAL,
    "ALBERT_BASE_URL": ALBERT,
    "ANTHROPIC_API_BASE_URL": ANTHROPIC,
    "LLM_DEFAULT_SERVER": OPENROUTER,
    "LLM_DEFAULT_MODEL": "google/gemini-3.8-flash",
}


@pytest.fixture
def unified(monkeypatch):
    """Mode unifié : serveurs et défauts déclarés dans l'environnement du serveur."""
    for name, value in UNIFIED.items():
        monkeypatch.setenv(name, value)


def _get(env, persona):
    return env.client.get("/users/me/settings", headers=env.headers[persona])


def _put(env, persona, changes):
    return env.client.put("/users/me/settings", json={"changes": changes}, headers=env.headers[persona])


# ---------------------------------------------------------------------------
# API des choix personnels
# ---------------------------------------------------------------------------
def test_historical_mode_reports_no_unified_choice(albert_env):
    """Sans ``LLM_DEFAULT_SERVER`` : mode historique annoncé, aucun choix enregistré."""
    body = _get(albert_env, "member_keys").json()
    assert body["unified"] is False
    assert set(body["settings"].values()) == {""}
    assert body["services"]["recode"]["legacy"] is True


def test_choices_are_validated_stored_and_effective(albert_env, unified):
    """Choix valides : enregistrés (adresse normalisée), couple effectif recalculé, audit sans valeur."""
    env = albert_env
    resp = _put(env, "member_keys", {"LLM_RECODE_SERVER": MISTRAL + "/", "LLM_RECODE_MODEL": "mistral-small-2603",
                                     "LLM_NOTES_MODEL": "openai/gpt-4o-mini"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["unified"] is True and body["changed"] == ["LLM_NOTES_MODEL", "LLM_RECODE_MODEL", "LLM_RECODE_SERVER"]
    assert body["settings"]["LLM_RECODE_SERVER"] == MISTRAL
    assert body["services"]["recode"] == {"server": MISTRAL, "server_label": "Mistral", "model": "mistral-small-2603",
                                          "legacy": False, "error": None}
    # Modèle seul : serveur par défaut.
    assert body["services"]["notes"]["server"] == OPENROUTER and body["services"]["notes"]["model"] == "openai/gpt-4o-mini"
    # Rien de déclaré : défaut du .env.
    assert body["services"]["citations"]["model"] == "google/gemini-3.8-flash"
    assert {"key": "anthropic", "label": "Anthropic", "url": ANTHROPIC, "families": ["chat"]} in body["servers"]


@pytest.mark.parametrize("changes, field", [
    ({"LLM_RECODE_SERVER": "https://evil.example.org/v1"}, "LLM_RECODE_SERVER"),
    ({"OCR_SERVER": OPENROUTER}, "OCR_SERVER"),
    ({"LLM_RAG_SERVER": ALBERT}, "LLM_RAG_SERVER"),
    ({"LLM_NOTES_SERVER": "local"}, "LLM_NOTES_SERVER"),
    ({"LLM_NOTES_MODEL": "gpt 4o"}, "LLM_NOTES_MODEL"),
    ({"LLM_NOTES_MODEL": "a\nOCR_MODEL=x"}, "LLM_NOTES_MODEL"),
    ({"LLM_DEFAULT_MODEL": "x"}, "LLM_DEFAULT_MODEL"),
    ({"OPENAI_API_KEY": "sk-x"}, "OPENAI_API_KEY"),
])
def test_invalid_choices_are_refused_and_nothing_is_stored(albert_env, unified, changes, field):
    """Une seule valeur refusée : 400 qui nomme le champ, rien n'est enregistré."""
    env = albert_env
    resp = _put(env, "member_keys", dict(changes, LLM_CITATIONS_MODEL="openai/gpt-4o-mini"))
    assert resp.status_code == 400 and field in resp.json()["errors"]
    assert get_user_settings(env.db, env.users["member_keys"]) == {}


def test_empty_value_inherits_and_choices_are_per_account(albert_env, unified):
    """Valeur vide : choix effacé ; un autre compte ne voit jamais ces choix."""
    env = albert_env
    assert _put(env, "member_keys", {"LLM_NOTES_MODEL": "openai/gpt-4o-mini"}).status_code == 200
    assert _get(env, "member_mistral").json()["settings"]["LLM_NOTES_MODEL"] == ""
    assert _put(env, "member_keys", {"LLM_NOTES_MODEL": ""}).status_code == 200
    rows = env.db.query(UserSetting).filter(UserSetting.user_id == env.users["member_keys"].id).all()
    assert rows == []


def test_ocr_local_keyword_and_albert_notes_are_accepted(unified):
    """``local`` pour l'OCR, Albert pour les notes : acceptés."""
    accepted, errors = validate_user_settings({"OCR_SERVER": "LOCAL", "OCR_MODEL": "docling",
                                               "LLM_NOTES_SERVER": ALBERT, "LLM_NOTES_MODEL": "gpt-oss-120b"})
    assert errors == {} and accepted["OCR_SERVER"] == "local" and accepted["LLM_NOTES_SERVER"] == ALBERT


def test_sourced_answers_are_no_longer_a_personal_choice(unified):
    """Réponses sourcées retirées (2026-10-03) : ni service effectif ni choix personnel."""
    assert "rag" not in effective_services({})
    _accepted, errors = validate_user_settings({"LLM_RAG_MODEL": "gpt-oss-120b"})
    assert "LLM_RAG_MODEL" in errors


# ---------------------------------------------------------------------------
# Application dans les routes
# ---------------------------------------------------------------------------
def test_personal_choice_applies_and_the_step_field_wins(albert_env, unified):
    """Recodage : choix personnel (Mistral) appliqué ; le champ de l'étape l'emporte."""
    env = albert_env
    assert _put(env, "member_mistral", {"LLM_RECODE_SERVER": MISTRAL, "LLM_RECODE_MODEL": "mistral-small-2603"}).status_code == 200
    resp, errors = _post(env, "member_mistral", "/initial_text_chunking", {"path": SESSION})
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    launches, _ = env.capture.take()
    assert _argv_value(launches[0]["argv"], "--model") == "@mistral:mistral-small-2603"
    assert launches[0]["env"]["MISTRAL_API_KEY"] == "member_mistral.mistral_api_key"
    resp, _ = _post(env, "member_mistral", "/initial_text_chunking", {"path": SESSION, "model": "mistral-large-latest"})
    assert resp.status_code == 200
    launches, _ = env.capture.take()
    assert _argv_value(launches[0]["argv"], "--model") == "@mistral:mistral-large-latest"
    # Un autre compte garde le défaut du .env (OpenRouter).
    resp, _ = _post(env, "member_openrouter", "/initial_text_chunking", {"path": SESSION})
    launches, _ = env.capture.take()
    assert _argv_value(launches[0]["argv"], "--model") == "google/gemini-3.8-flash"


def test_personal_ocr_choice_reaches_the_script(albert_env, unified, monkeypatch):
    """OCR personnel (Mistral) sans OCR déclaré au .env : couple et clé transmis au script."""
    env = albert_env
    built = []
    original = ocr_target.explicit_ocr_env
    monkeypatch.setattr(ocr_target, "explicit_ocr_env", lambda user, choice: built.append(original(user, choice)) or built[-1])
    assert _put(env, "member_mistral", {"OCR_SERVER": MISTRAL, "OCR_MODEL": "mistral-ocr-latest"}).status_code == 200
    path = _fresh_ocr_session(env, "gsess-ocr-personal-mistral")
    resp, errors = _post(env, "member_mistral", "/process_dataframe", {"path": path})
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    facts = env.capture.take()[0][0]["env"]
    assert facts["MISTRAL_API_KEY"] == "member_mistral.mistral_api_key"
    assert built and built[0]["OCR_SERVER"] == MISTRAL and built[0]["OCR_MODEL"] == "mistral-ocr-latest"
    # Un compte sans choix personnel garde la chaîne historique (aucun OCR déclaré au .env).
    built.clear()
    path = _fresh_ocr_session(env, "gsess-ocr-personal-none")
    _post(env, "member_keys", "/process_dataframe", {"path": path})
    assert built == []


def test_explicit_ocr_target_prefers_the_personal_choice(unified, monkeypatch):
    """``explicit_ocr_target`` : choix personnel avant le ``.env``, ``None`` sans aucun modèle."""
    assert ocr_target.explicit_ocr_target({}) is None
    monkeypatch.setenv("OCR_SERVER", MISTRAL)
    monkeypatch.setenv("OCR_MODEL", "mistral-ocr-latest")
    choice = ocr_target.explicit_ocr_target({"OCR_SERVER": "local", "OCR_MODEL": "pymupdf"})
    assert choice.server_url == "local" and choice.model == "pymupdf"
    assert ocr_target.explicit_ocr_required_keys(choice) == []


# ---------------------------------------------------------------------------
# Clés personnelles des serveurs ajoutés
# ---------------------------------------------------------------------------
NEW_KEYS = ("anthropic_api_key", "google_api_key", "deepseek_api_key", "qwen_api_key", "glm_api_key")


def test_new_server_keys_are_listed_only_when_declared(monkeypatch):
    """Clé listée seulement si l'adresse de son serveur est déclarée au bloc 1."""
    for var in creds.SERVER_CREDENTIAL_BASE_URLS.values():
        monkeypatch.delenv(var, raising=False)
    assert not set(NEW_KEYS) & set(creds.visible_credential_keys())
    monkeypatch.setenv("ANTHROPIC_API_BASE_URL", ANTHROPIC)
    visible = creds.visible_credential_keys()
    assert "anthropic_api_key" in visible and "google_api_key" not in visible
    assert set(NEW_KEYS) <= set(creds.CREDENTIAL_KEYS)
    assert not {creds.CREDENTIAL_ENV_MAPPING[k] for k in NEW_KEYS} & set(creds.SERVER_SECRET_ENV_VARS)


def test_personal_key_is_saved_only_for_a_declared_server(albert_env, unified):
    """``PUT /users/me/credentials`` : clé Anthropic acceptée (adresse déclarée), Google ignorée."""
    env = albert_env
    resp = env.client.put("/users/me/credentials", headers=env.headers["member_keys"],
                          json={"anthropic_api_key": "fake-anthropic-member-0001", "google_api_key": "fake-google-0001"})
    assert resp.status_code == 200 and resp.json()["updated"] == ["anthropic_api_key"]
    listed = env.client.get("/users/me/credentials", headers=env.headers["member_keys"]).json()
    assert listed["anthropic_api_key"]["has_value"] is True and "fake-anthropic" not in str(listed)
    assert "google_api_key" not in listed


def test_member_uses_a_personal_anthropic_key(albert_env, unified, monkeypatch):
    """Un compte non-admin avec sa clé Anthropic : lancé avec sa clé, jamais celle du .env."""
    env = albert_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-server-0001")
    creds.update_user_credentials(env.users["member_keys"], {"anthropic_api_key": "fake-anthropic-member-0002"}, env.db)
    form = {"path": SESSION, "server": ANTHROPIC, "model": "claude-sonnet-5-5"}
    resp, errors = _post(env, "member_keys", "/initial_text_chunking", form)
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    launches, _ = env.capture.take()
    assert _argv_value(launches[0]["argv"], "--model") == "@anthropic:claude-sonnet-5-5"
    sub = creds.build_subprocess_env(env.users["member_keys"], required_keys=["anthropic_api_key"])
    assert sub["ANTHROPIC_API_KEY"] == "fake-anthropic-member-0002"
    assert "ANTHROPIC_API_KEY" not in sub[creds.DOTENV_DENY_ENV_VAR].split(",")


def test_non_admin_subprocess_never_reloads_new_server_keys(albert_env, monkeypatch):
    """Non-admin sans clé personnelle : clés retirées et inscrites dans ``RAGPY_DOTENV_DENY``."""
    env = albert_env
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-server-0001")
    sub = creds.build_subprocess_env(env.users["member_nokeys"])
    assert "GOOGLE_API_KEY" not in sub
    denied = set(sub[creds.DOTENV_DENY_ENV_VAR].split(","))
    assert {"ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GLM_API_KEY", "GOOGLE_API_KEY", "QWEN_API_KEY"} <= denied
    assert creds.build_subprocess_env(env.users["admin"])["GOOGLE_API_KEY"] == "fake-google-server-0001"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def test_pages_show_the_choices_in_unified_mode_only(albert_env, unified, monkeypatch):
    """Profil et pipeline : carte des choix et champ serveur en mode unifié ; absents sinon."""
    env = albert_env
    profile = env.client.get("/profile", headers=env.headers["member_keys"]).text
    assert 'id="modelChoicesForm"' in profile and 'id="cred_anthropic_api_key"' in profile
    pipeline = env.client.get("/pipeline", headers=env.headers["member_keys"]).text
    assert 'id="recodingServerInput"' in pipeline and 'id="zoteroServerInput"' in pipeline
    monkeypatch.delenv("LLM_DEFAULT_SERVER")
    monkeypatch.delenv("ANTHROPIC_API_BASE_URL")
    profile = env.client.get("/profile", headers=env.headers["member_keys"]).text
    assert 'id="modelChoicesForm"' not in profile and "anthropic" not in profile
    pipeline = env.client.get("/pipeline", headers=env.headers["member_keys"]).text
    assert "recodingServerInput" not in pipeline


# ---------------------------------------------------------------------------
# Adresses personnelles (surface SSRF) et effacement d'une valeur
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url, admitted", [
    ("https://xyz.weaviate.network", True),
    ("https://api.mistral.ai", True),
    ("https://8.8.8.8:6333", True),
    ("http://xyz.weaviate.network", False),
    ("https://localhost:6333", False),
    ("https://qdrant:6333", False),
    ("https://redis", False),
    ("https://127.0.0.1", False),
    ("https://10.0.0.5", False),
    ("https://169.254.169.254/latest/meta-data", False),
    ("https://[::1]:8000", False),
    ("https://0x7f.1", False),
    ("https://user:pw@xyz.weaviate.network", False),
    ("https://db.internal", False),
    ("https://xyz.weaviate.network:99999", False),
])
def test_personal_url_policy(url, admitted):
    """``https`` public seulement : ni réseau interne, ni nom court, ni IP privée ou non standard."""
    from app.core.url_policy import personal_url_error

    assert (personal_url_error(url) is None) is admitted


def test_member_cannot_save_an_internal_address_but_admin_can(albert_env):
    """Non-admin : adresse interne refusée (400, rien n'est écrit) ; administrateur : admise."""
    env = albert_env
    before = creds.get_user_credentials(env.users["member_keys"])
    resp = env.client.put("/users/me/credentials", headers=env.headers["member_keys"],
                          json={"qdrant_url": "https://qdrant:6333", "openai_api_key": "fake-openai-put-0009"})
    assert resp.status_code == 400 and resp.json()["invalid_keys"] == ["qdrant_url"]
    env.db.refresh(env.users["member_keys"])
    assert creds.get_user_credentials(env.users["member_keys"]) == before
    resp = env.client.put("/users/me/credentials", headers=env.headers["admin"], json={"qdrant_url": "http://qdrant:6333"})
    assert resp.status_code == 200


def test_empty_value_clears_a_personal_credential(albert_env):
    """Une chaîne vide efface la valeur enregistrée (bouton « Vider » du profil)."""
    env = albert_env
    user = env.users["member_keys"]
    assert creds.get_user_credentials(user).get("openai_api_key")
    resp = env.client.put("/users/me/credentials", headers=env.headers["member_keys"], json={"openai_api_key": ""})
    assert resp.status_code == 200
    env.db.refresh(user)
    assert "openai_api_key" not in creds.get_user_credentials(user)
