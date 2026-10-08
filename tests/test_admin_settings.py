"""Page administrateur « Paramètres » (sprint « configuration unifiée », lot L8).

* administrateurs seuls ;
* toutes les variables actives, dans l'ordre du registre, secrets masqués ;
* écriture sur place (commentaires et ordre conservés), sauvegarde, journal
  d'audit sans valeur, application à chaud ;
* refus global : variable verrouillée, valeur invalide, retour à la ligne,
  serveur non déclaré ;
* test d'un serveur : clé jamais renvoyée.
"""

from __future__ import annotations

import os

import pytest

from app.models.audit import AuditLog
from app.routes import admin_settings
from scripts.rad_settings import access
from scripts.rad_settings import registry as reg
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401
from tests.test_albert_routes import albert_env  # noqa: F401

SECRET = "sk-or-v1-" + "S" * 36 + "9876"
ENV_TEXT = f"""# mon commentaire en tête
OPENROUTER_API_KEY={SECRET}
# commentaire collé
MISTRAL_MAX_PAGES=950
OPENROUTER_API_BASE_URL=https://openrouter.ai/api/v1
MISTRAL_API_BASE_URL=https://api.mistral.ai/v1
JWT_SECRET_KEY={"j" * 48}
"""


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """``.env`` temporaire ; l'environnement et l'état d'``access`` sont restaurés à la fin."""
    saved_environ = dict(os.environ)
    saved_state = dict(access._STATE)
    path = tmp_path / ".env"
    path.write_text(ENV_TEXT, encoding="utf-8")
    monkeypatch.setenv(access.ENV_FILE_VAR, str(path))
    monkeypatch.setattr(admin_settings, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(admin_settings, "LOCK_PATH", tmp_path / "env.lock")
    yield path
    os.environ.clear()
    os.environ.update(saved_environ)
    access._STATE.clear()
    access._STATE.update(saved_state)


def _get(env, persona="admin"):
    return env.client.get("/api/admin/settings", headers=env.headers[persona])


def _put(env, changes, persona="admin"):
    return env.client.put("/api/admin/settings", json={"changes": changes}, headers=env.headers[persona])


def _flat(payload):
    return [s for b in payload["blocks"] for sub in b["subblocks"] for s in sub["settings"]]


def test_admin_only(albert_env, env_file):
    """Un compte non administrateur est refusé (lecture comme écriture)."""
    env = albert_env
    assert _get(env, "member_keys").status_code == 403
    assert _put(env, {"MISTRAL_MAX_PAGES": "10"}, "member_keys").status_code == 403
    assert env_file.read_text(encoding="utf-8") == ENV_TEXT


def test_lists_every_active_variable_in_registry_order(albert_env, env_file):
    """Toutes les variables actives, dans l'ordre du registre ; jamais une prévue."""
    payload = _get(albert_env).json()
    names = [s["name"] for s in _flat(payload)]
    assert names == [s.name for s in reg.SETTINGS if s.status == reg.ACTIVE]
    by_name = {s["name"]: s for s in _flat(payload)}
    assert by_name["MISTRAL_MAX_PAGES"]["value"] == "950" and by_name["MISTRAL_MAX_PAGES"]["declared"] is True
    assert by_name["JWT_SECRET_KEY"]["locked"]
    assert {"key": "openrouter", "label": "OpenRouter", "url": "https://openrouter.ai/api/v1"} in payload["servers"]


def test_secrets_are_never_returned(albert_env, env_file):
    """Une clé n'est jamais renvoyée : masque ``••••`` + quatre derniers caractères."""
    resp = _get(albert_env)
    assert SECRET not in resp.text and ("j" * 48) not in resp.text
    by_name = {s["name"]: s for s in _flat(resp.json())}
    assert by_name["OPENROUTER_API_KEY"]["value"] == admin_settings.MASK_PREFIX + "9876"


def test_update_writes_in_place_backs_up_audits_and_applies(albert_env, env_file):
    """Écriture sur place, commentaires gardés, sauvegarde, audit sans valeur, prise à chaud."""
    env = albert_env
    resp = _put(env, {"MISTRAL_MAX_PAGES": "700", "OPENROUTER_API_KEY": admin_settings.MASK_PREFIX + "9876"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["changed"] == ["MISTRAL_MAX_PAGES"] and body["restart_required"] == []
    text = env_file.read_text(encoding="utf-8")
    assert "# mon commentaire en tête" in text and "# commentaire collé\nMISTRAL_MAX_PAGES=700" in text
    assert f"OPENROUTER_API_KEY={SECRET}" in text
    assert os.listdir(env_file.parent / "backups")
    assert os.environ.get("MISTRAL_MAX_PAGES") == "700"
    log = env.db.query(AuditLog).filter(AuditLog.action == admin_settings.AUDIT_ACTION).order_by(AuditLog.id.desc()).first()
    assert log.details == {"variables": ["MISTRAL_MAX_PAGES"]}


def test_new_variable_is_added_and_file_tidied(albert_env, env_file):
    """Variable absente du fichier : ajoutée et rangée à sa place."""
    resp = _put(albert_env, {"LLM_DEFAULT_SERVER": "https://openrouter.ai/api/v1",
                             "LLM_DEFAULT_MODEL": "google/gemini-3.8-flash"})
    assert resp.status_code == 200, resp.text
    text = env_file.read_text(encoding="utf-8")
    assert "# --- 2.1 Par défaut ---\nLLM_DEFAULT_SERVER=https://openrouter.ai/api/v1\nLLM_DEFAULT_MODEL=google/gemini-3.8-flash" in text


@pytest.mark.parametrize("changes, field", [
    ({"JWT_SECRET_KEY": "x" * 48}, "JWT_SECRET_KEY"),
    ({"MISTRAL_MAX_PAGES": "beaucoup"}, "MISTRAL_MAX_PAGES"),
    ({"RESEND_FROM_EMAIL": "a\nINJECTED=1"}, "RESEND_FROM_EMAIL"),
    ({"LLM_NOTES_SERVER": "https://evil.example.org/v1"}, "LLM_NOTES_SERVER"),
    ({"NOT_A_VARIABLE": "1"}, "NOT_A_VARIABLE"),
])
def test_invalid_changes_are_refused_and_nothing_is_written(albert_env, env_file, changes, field):
    """Une seule valeur refusée : rien n'est écrit, l'erreur nomme le champ."""
    resp = _put(albert_env, dict(changes, MISTRAL_MAX_PAGES=changes.get("MISTRAL_MAX_PAGES", "700")))
    assert resp.status_code == 400 and field in resp.json()["errors"]
    assert env_file.read_text(encoding="utf-8") == ENV_TEXT
    assert "INJECTED" not in env_file.read_text(encoding="utf-8")


def test_restart_required_is_reported(albert_env, env_file):
    """Variable lue au démarrage : la réponse l'annonce."""
    resp = _put(albert_env, {"BOOK_NOTE_VERSION": "v1"})
    assert resp.status_code == 200 and resp.json()["restart_required"] == ["BOOK_NOTE_VERSION"]


def test_server_test_never_returns_the_key(albert_env, env_file, monkeypatch):
    """« Tester » : réponse lisible, clé jamais renvoyée."""
    seen = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "google/gemini-3.8-flash"}, {"id": "openai/gpt-4o-mini"}]}

    def fake_get(url, headers=None, timeout=None):
        seen.update(url=url, auth=(headers or {}).get("Authorization", ""))
        return _Resp()

    monkeypatch.setattr(admin_settings.requests, "get", fake_get)
    access.refresh_environ(force=True)
    resp = albert_env.client.post("/api/admin/settings/test-server", json={"server": "openrouter"},
                                  headers=albert_env.headers["admin"])
    assert resp.status_code == 200 and resp.json()["ok"] is True and resp.json()["models"] == 2
    assert SECRET not in resp.text and seen["url"] == "https://openrouter.ai/api/v1/models"
    models = albert_env.client.get("/api/admin/settings/models", params={"server": "https://openrouter.ai/api/v1"},
                                   headers=albert_env.headers["admin"])
    assert models.json()["models"] == ["google/gemini-3.8-flash", "openai/gpt-4o-mini"]


def test_settings_page_renders_for_admins_only(albert_env, env_file):
    """La page s'affiche pour un administrateur ; un autre compte est renvoyé à l'accueil."""
    env = albert_env
    page = env.client.get("/admin/settings", headers=env.headers["admin"])
    assert page.status_code == 200 and 'id="settingsBlocks"' in page.text
    other = env.client.get("/admin/settings", headers=env.headers["member_keys"], follow_redirects=False)
    assert other.status_code in (302, 303, 307)


def test_server_must_render_the_service(albert_env, env_file):
    """Lot L6 : un serveur incapable du service de la variable est refusé (rien n'est écrit)."""
    resp = _put(albert_env, {"OCR_SERVER": "https://openrouter.ai/api/v1", "OCR_MODEL": "mistral-ocr-latest"})
    assert resp.status_code == 400 and "OCR_SERVER" in resp.json()["errors"]
    assert env_file.read_text(encoding="utf-8") == ENV_TEXT
    resp = _put(albert_env, {"EMBEDDING_SERVER": "https://api.mistral.ai/v1", "EMBEDDING_MODEL": "mistral-embed"})
    assert resp.status_code == 200, resp.text
