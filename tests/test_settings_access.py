"""Relecture du ``.env`` à chaud et démarrage strict (sprint « configuration unifiée », lot L3).

* le ``.env`` est chargé sans écraser l'environnement réel (shell, Compose, tests) ;
* une modification du fichier est reprise au rafraîchissement suivant (ajout,
  changement, retrait), sauf pour les variables de déploiement ;
* le démarrage strict refuse un ``.env`` incomplet et nomme la commande ;
* ``env_tool sync`` écrit les variables obligatoires absentes avec la valeur
  actuelle du code, sans toucher aux autres ;
* toute variable lue à l'import d'un module de ``app/`` est marquée « à
  redémarrer » dans le registre.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from scripts import env_tool
from scripts.rad_settings import access, registry

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_access(tmp_path, monkeypatch):
    """État d'``access`` isolé : fichier temporaire, photographie remise à zéro puis restaurée."""
    saved = dict(access._STATE)
    access._reset_for_tests()
    env_file = tmp_path / ".env"
    monkeypatch.setenv(access.ENV_FILE_VAR, str(env_file))
    yield env_file
    access._STATE.clear()
    access._STATE.update(saved)


def _write(path: Path, text: str) -> None:
    """Écrit ``text`` en changeant sûrement la date de modification."""
    previous = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text(text, encoding="utf-8")
    if path.stat().st_mtime_ns == previous:
        os.utime(path, ns=(previous + 1_000_000, previous + 1_000_000))


def test_load_does_not_override_the_real_environment(isolated_access):
    """Premier chargement : le fichier complète l'environnement, sans l'écraser."""
    _write(isolated_access, "MISTRAL_OCR_TIMEOUT=111\nMISTRAL_OCR_RETRIES=9\n")
    env = {"MISTRAL_OCR_RETRIES": "2", access.ENV_FILE_VAR: str(isolated_access)}
    access.load_into_environ(env)
    assert env["MISTRAL_OCR_TIMEOUT"] == "111" and env["MISTRAL_OCR_RETRIES"] == "2"


def test_refresh_picks_up_changes_additions_and_removals(isolated_access):
    """Fichier modifié : valeur changée reprise, ajout pris, retrait retiré."""
    _write(isolated_access, "MISTRAL_OCR_TIMEOUT=111\nMISTRAL_MAX_PAGES=900\n")
    env = {access.ENV_FILE_VAR: str(isolated_access)}
    access.load_into_environ(env)
    _write(isolated_access, "MISTRAL_OCR_TIMEOUT=222\nMISTRAL_SPLIT_PART_MB=20\n")
    access.refresh_environ(env)
    assert env["MISTRAL_OCR_TIMEOUT"] == "222"
    assert env["MISTRAL_SPLIT_PART_MB"] == "20"
    assert "MISTRAL_MAX_PAGES" not in env


def test_real_environment_always_wins(isolated_access):
    """Une variable de l'environnement réel n'est jamais modifiée ni retirée."""
    _write(isolated_access, "MISTRAL_OCR_TIMEOUT=111\n")
    env = {"MISTRAL_OCR_TIMEOUT": "5", access.ENV_FILE_VAR: str(isolated_access)}
    access.load_into_environ(env)
    _write(isolated_access, "OTHER=1\n")
    access.refresh_environ(env)
    assert env["MISTRAL_OCR_TIMEOUT"] == "5"


def test_deploy_variables_are_not_changed_on_the_fly(isolated_access):
    """Déploiement (``deploy``) : chargé au démarrage, jamais modifié ensuite."""
    _write(isolated_access, "CORS_ORIGINS=https://a.example\n")
    env = {access.ENV_FILE_VAR: str(isolated_access)}
    access.load_into_environ(env)
    _write(isolated_access, "CORS_ORIGINS=https://b.example\n")
    access.refresh_environ(env)
    assert env["CORS_ORIGINS"] == "https://a.example"
    assert registry.BY_NAME["CORS_ORIGINS"].apply != registry.HOT


def test_unchanged_file_is_not_reread(isolated_access, monkeypatch):
    """Sans changement de date ni de taille : aucune relecture (un ``stat`` seulement)."""
    _write(isolated_access, "MISTRAL_OCR_TIMEOUT=111\n")
    env = {access.ENV_FILE_VAR: str(isolated_access)}
    access.load_into_environ(env)
    calls = {"n": 0}
    real = access.read_env_file

    def counting(path=None):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(access, "read_env_file", counting)
    for _ in range(5):
        access.refresh_environ(env)
    assert calls["n"] == 0


def test_missing_file_means_no_values(isolated_access):
    """Fichier absent (image Docker, CI) : rien n'est chargé, aucune erreur."""
    env = {access.ENV_FILE_VAR: str(isolated_access)}
    assert access.load_into_environ(env) == {}


def test_strict_start_refuses_an_incomplete_env(isolated_access, monkeypatch):
    """Démarrage strict : variables obligatoires manquantes → refus avec la commande."""
    _write(isolated_access, "MISTRAL_OCR_TIMEOUT=111\n")
    env = {access.ENV_FILE_VAR: str(isolated_access), access.STRICT_VAR: "1"}
    with pytest.raises(access.SettingsIncompleteError) as info:
        access.enforce_complete(env)
    assert "env_tool.py sync --apply" in str(info.value)
    assert "MISTRAL_OCR_TIMEOUT" not in info.value.missing
    assert "OPENROUTER_API_KEY" in info.value.missing
    env[access.STRICT_VAR] = "0"
    access.enforce_complete(env)


def test_strict_start_accepts_a_synced_env(isolated_access, monkeypatch):
    """Après ``env_tool sync --apply``, le démarrage strict passe."""
    monkeypatch.setattr(env_tool, "BACKUP_DIR", isolated_access.parent / "backups")
    monkeypatch.setattr(env_tool, "LOCK_PATH", isolated_access.parent / "env.lock")
    _write(isolated_access, "")
    assert env_tool.main(["sync", "--env", str(isolated_access), "--apply"]) == 0
    env = {access.ENV_FILE_VAR: str(isolated_access), access.STRICT_VAR: "1"}
    access.enforce_complete(env)


def test_sync_writes_code_defaults_and_keeps_existing_values(tmp_path, monkeypatch, capsys):
    """``sync`` : valeurs actuelles du code ; valeurs existantes et secrets intacts."""
    monkeypatch.setattr(env_tool, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(env_tool, "LOCK_PATH", tmp_path / "env.lock")
    path = tmp_path / ".env"
    secret = "sk-or-v1-" + "Z" * 40
    path.write_text(f"OPENROUTER_API_KEY={secret}\nMISTRAL_MAX_PAGES=700\n", encoding="utf-8")
    assert env_tool.main(["sync", "--env", str(path)]) == 0
    assert path.read_text(encoding="utf-8").count("\n") == 2
    assert env_tool.main(["sync", "--env", str(path), "--apply"]) == 0
    values = dotenv_values(path)
    assert values["OPENROUTER_API_KEY"] == secret and values["MISTRAL_MAX_PAGES"] == "700"
    assert values["MISTRAL_OCR_RETRIES"] == registry.BY_NAME["MISTRAL_OCR_RETRIES"].default
    assert values["USERS_SANDBOX"] == "FALSE"
    assert values["DEFAULT_MAX_WORKERS"] == str(max(1, (os.cpu_count() or 2) - 1))
    assert "ALBERT_LIMITER_BACKEND" not in values
    assert all(values.get(name) is not None for name in access.required_names())
    assert secret not in capsys.readouterr().out
    assert env_tool.main(["sync", "--env", str(path), "--apply"]) == 0
    assert len(os.listdir(tmp_path / "backups")) == 1


def test_settings_refresh_middleware_refreshes_http_only(monkeypatch):
    """Le middleware rafraîchit avant chaque requête HTTP, pas pour le cycle de vie."""
    from app import main as app_main

    calls = []
    monkeypatch.setattr(app_main, "refresh_environ", lambda: calls.append(1))
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    middleware = app_main.SettingsRefreshMiddleware(inner)
    import anyio

    anyio.run(middleware, {"type": "http"}, None, None)
    anyio.run(middleware, {"type": "lifespan"}, None, None)
    assert seen == ["http", "lifespan"] and calls == [1]


def test_build_subprocess_env_refreshes_first(monkeypatch):
    """``build_subprocess_env`` relit le ``.env`` avant de copier l'environnement."""
    from app.core import credentials

    calls = []
    monkeypatch.setattr(credentials, "refresh_environ", lambda: calls.append(1))

    class Admin:
        is_admin = True
        api_credentials = None
        id = 1

    monkeypatch.setattr(credentials, "get_user_credentials", lambda user: {})
    credentials.build_subprocess_env(Admin())
    assert calls == [1]


def _module_level_reads(tree: ast.AST):
    """Noms de variables lus hors fonction (module ou corps de classe)."""
    def reads(node):
        for sub in ast.walk(node):
            name = ""
            if isinstance(sub, ast.Call) and sub.args and isinstance(sub.args[0], ast.Constant):
                func = sub.func
                callee = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                receiver = ast.unparse(func.value).lower() if isinstance(func, ast.Attribute) else ""
                if callee == "getenv" or (callee == "get" and "env" in receiver):
                    name = sub.args[0].value
            if isinstance(name, str) and name in registry.BY_NAME:
                yield name

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bodies = [s for s in stmt.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))] \
            if isinstance(stmt, ast.ClassDef) else [stmt]
        for body in bodies:
            yield from reads(body)


def test_import_time_reads_are_marked_restart():
    """Une variable lue à l'import d'un module de ``app/`` n'est pas déclarée « à chaud »."""
    wrong = {}
    for path in sorted((REPO / "app").rglob("*.py")):
        if "__pycache__" in path.parts or path.name.startswith("test_"):
            continue
        for name in _module_level_reads(ast.parse(path.read_text(encoding="utf-8"))):
            if registry.BY_NAME[name].apply == registry.HOT:
                wrong.setdefault(name, str(path.relative_to(REPO)))
    assert not wrong, f"lues à l'import mais marquées hot : {wrong}"


def test_env_example_still_matches_registry():
    """``.env.example`` reste le rendu du registre après les changements du lot L3."""
    from scripts.rad_settings.render import render_example

    assert (REPO / ".env.example").read_text(encoding="utf-8") == render_example()
