"""Outil du ``.env`` (sprint « configuration unifiée », lot L2).

* ``.env.example`` est exactement le rendu du registre ;
* la lecture de ``dotenv_file`` concorde avec python-dotenv ;
* le rangement ne change aucune valeur, garde les commentaires collés, isole
  les noms hors registre et est idempotent ;
* l'écriture se fait sur place (même inode), après sauvegarde en ``600`` ;
* aucune commande n'affiche une valeur.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from dotenv import dotenv_values

from scripts import env_tool
from scripts.rad_settings import dotenv_file, registry
from scripts.rad_settings.render import TIDY_TITLE, check, render_example, tidy_layout, validate

REPO = Path(__file__).resolve().parents[1]

SECRET = "sk-or-v1-" + "Q" * 40

SAMPLE = f"""# ===== ANCIEN EN-TÊTE =====
OPENAI_API_KEY=plain-value
# note collée au-dessus de la clé OpenRouter
OPENROUTER_API_KEY={SECRET}
export MISTRAL_API_BASE_URL=https://api.mistral.ai
DEFAULT_MAX_WORKERS=8   # commentaire de fin de ligne

# commentaire isolé

RESEND_FROM_EMAIL="Nom avec espaces <x@example.org>"
OPENAI_OCR_PROMPT='Transcris # sans commentaire'
JWT_SECRET_KEY="multi
ligne"
MY_CUSTOM_VAR=1
ALBERT_LIVE=1
MAX_ACTIVE_SESSIONS=5
DEBUG=false
DEBUG=true
EMPTY_ONE=
"""


def _tidy(text: str):
    return dotenv_file.tidy(dotenv_file.parse(text), tidy_layout(), registry.classify, TIDY_TITLE)


def test_env_example_is_the_registry_rendering():
    """``.env.example`` est le rendu exact du registre (régénérer avec ``env_tool example``)."""
    assert (REPO / ".env.example").read_text(encoding="utf-8") == render_example()


def test_env_example_shows_active_and_commented_settings_only():
    """Variables actives ou commentées présentes une fois ; variables cachées absentes."""
    lines = dotenv_file.parse(render_example())
    active = [line.name for line in lines if line.kind == dotenv_file.ASSIGN]
    commented = [line.commented_name for line in lines if line.commented_name]
    for setting in registry.SETTINGS:
        expected = {"active": (1, 0), "commented": (0, 1), "hidden": (0, 0)}[setting.render]
        assert (active.count(setting.name), commented.count(setting.name)) == expected, setting.name


def test_only_openrouter_and_mistral_servers_are_prefilled():
    """Demande d'Amar : serveurs OpenRouter et Mistral préremplis, les autres en commentaire."""
    values = dotenv_values(REPO / ".env.example")
    assert values["OPENROUTER_API_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert values["MISTRAL_API_BASE_URL"] == "https://api.mistral.ai/v1"
    for name in ("OPENAI_API_KEY", "OPENAI_API_BASE_URL", "ALBERT_API_KEY", "ALBERT_BASE_URL", "ANTHROPIC_API_BASE_URL",
                 "GOOGLE_API_BASE_URL", "DEEPSEEK_API_BASE_URL", "QWEN_API_BASE_URL", "GLM_API_BASE_URL",
                 "LOCAL_API_BASE_URL"):
        assert name not in values, name
        assert registry.BY_NAME[name].render == "commented", name


def test_parser_matches_python_dotenv(tmp_path):
    """Valeurs identiques à python-dotenv (citations, export, commentaires, multiligne, vide)."""
    path = tmp_path / ".env"
    path.write_text(SAMPLE, encoding="utf-8")
    ours = dotenv_file.values(dotenv_file.parse(SAMPLE))
    assert ours == {k: (v or "") for k, v in dotenv_values(path).items()}
    assert dotenv_file.duplicates(dotenv_file.parse(SAMPLE)) == ["DEBUG"]


def test_tidy_keeps_values_and_sorts_by_registry(tmp_path):
    """Rangement : valeurs identiques (python-dotenv), ordre du registre, groupes finaux."""
    new_text, report = _tidy(SAMPLE)
    before, after = tmp_path / "before", tmp_path / "after"
    before.write_text(SAMPLE, encoding="utf-8")
    after.write_text(new_text, encoding="utf-8")
    assert dict(dotenv_values(after)) == dict(dotenv_values(before))
    order = [line.name for line in dotenv_file.parse(new_text) if line.kind == dotenv_file.ASSIGN]
    registered = [name for name in order if name in registry.BY_NAME]
    assert registered == sorted(registered, key=lambda n: registry.names().index(n))
    assert report.unknown == ["MY_CUSTOM_VAR", "EMPTY_ONE"]
    assert report.internal == ["ALBERT_LIVE"] and report.obsolete == ["MAX_ACTIVE_SESSIONS"]
    assert report.duplicates == ["DEBUG"] and order.count("DEBUG") == 1
    assert "# ===== ANCIEN EN-TÊTE =====" not in new_text and "# commentaire isolé" not in new_text
    assert "# note collée au-dessus de la clé OpenRouter\nOPENROUTER_API_KEY=" in new_text
    assert "DEFAULT_MAX_WORKERS=8   # commentaire de fin de ligne" in new_text
    assert "# --- 1.1 OpenRouter ---" in new_text and "# ===== 1. SERVEURS D'API" in new_text


def test_tidy_is_idempotent():
    """Ranger un fichier déjà rangé ne change rien."""
    once, _ = _tidy(SAMPLE)
    twice, _ = _tidy(once)
    assert twice == once


def test_write_in_place_keeps_inode_and_backs_up(tmp_path):
    """Écriture sur place : même inode, sauvegarde en 600, contenu exact."""
    path = tmp_path / ".env"
    path.write_text("A=1\n", encoding="utf-8")
    inode = path.stat().st_ino
    backup = dotenv_file.write_in_place(path, "A=2\n", backup_dir=tmp_path / "bk", lock_path=tmp_path / "lock")
    assert path.read_text(encoding="utf-8") == "A=2\n"
    assert path.stat().st_ino == inode
    assert backup is not None and backup.read_text(encoding="utf-8") == "A=1\n"
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    created = dotenv_file.write_in_place(tmp_path / "new.env", "B=1\n", backup_dir=tmp_path / "bk")
    assert created is None and stat.S_IMODE((tmp_path / "new.env").stat().st_mode) == 0o600


def test_write_in_place_restores_on_mismatch(tmp_path, monkeypatch):
    """Relecture différente : sauvegarde restaurée et erreur levée."""
    path = tmp_path / ".env"
    path.write_text("A=1\n", encoding="utf-8")
    real_read = Path.read_text
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        if self == path:
            calls["n"] += 1
            if calls["n"] == 1:
                return "corrompu"
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)
    with pytest.raises(dotenv_file.EnvWriteError):
        dotenv_file.write_in_place(path, "A=2\n", backup_dir=tmp_path / "bk")
    monkeypatch.undo()
    assert path.read_text(encoding="utf-8") == "A=1\n"


@pytest.mark.parametrize("value, line", [
    ("abc", "X=abc"),
    ("", "X="),
    ("a b", 'X="a b"'),
    ("a#b", 'X="a#b"'),
    ('say "hi"', 'X="say \\"hi\\""'),
])
def test_format_assignment_quotes_when_needed(value, line, tmp_path):
    """Valeur citée seulement si nécessaire, relue identique par python-dotenv."""
    assert dotenv_file.format_assignment("X", value) == line
    path = tmp_path / ".env"
    path.write_text(line + "\n", encoding="utf-8")
    assert (dotenv_values(path)["X"] or "") == value


@pytest.mark.parametrize("bad", ["a\nb", "a\rb", "a\x00b", "a b"])
def test_format_assignment_refuses_line_breaks(bad):
    """Retour à la ligne ou NUL : refus (injection de lignes)."""
    with pytest.raises(ValueError):
        dotenv_file.format_assignment("X", bad)


def test_check_never_prints_values(tmp_path, capsys):
    """``env_tool check`` : noms et raisons seulement, jamais une valeur."""
    path = tmp_path / ".env"
    path.write_text(SAMPLE + "MISTRAL_MAX_PAGES=sentinelleA7\nALBERT_OCR_MODE=sentinelleB9\n", encoding="utf-8")
    code = env_tool.main(["check", "--env", str(path), "--absent"])
    out = capsys.readouterr().out
    assert code == 1
    for value in (SECRET, "plain-value", "sentinelleA7", "sentinelleB9", "Nom avec espaces"):
        assert value not in out, value
    assert "MY_CUSTOM_VAR" in out and "MISTRAL_MAX_PAGES" in out and "ALBERT_OCR_MODE" in out


def test_check_categories():
    """Catégories du bilan : invalides, inconnues, internes, obsolètes, prévues."""
    report = check({"MISTRAL_MAX_PAGES": "x", "FOO_BAR": "1", "ALBERT_LIVE": "1", "MAX_ACTIVE_SESSIONS": "2",
                    "EMBEDDING_SERVER": "https://api.openai.com/v1", "DEBUG": "1"}, [])
    names = {key: [name for name, _ in items] for key, items in report.items()}
    assert names["error"] == ["MISTRAL_MAX_PAGES"] and names["unknown"] == ["FOO_BAR"]
    assert names["internal"] == ["ALBERT_LIVE"] and names["obsolete"] == ["MAX_ACTIVE_SESSIONS"]
    # Lot L6 : plus aucune variable prévue (EMBEDDING_SERVER est active).
    assert names.get("planned", []) == [] and names["warning"] == ["DEBUG"]
    assert all("EMBEDDING_SERVER" not in found for key, found in names.items() if key != "absent")
    assert "OPENAI_API_KEY" in names["absent"]


def test_validate_by_kind():
    """Contrôle de forme par nature."""
    get = registry.BY_NAME.__getitem__
    assert validate(get("MISTRAL_MAX_UPLOAD_MB"), "45.5")[0] == "ok"
    assert validate(get("MISTRAL_MAX_PAGES"), "9.5")[0] == "error"
    assert validate(get("ALBERT_DATA_POLICY"), "albert_only")[0] == "ok"
    assert validate(get("ALBERT_DATA_POLICY"), "strict")[0] == "error"
    assert validate(get("USERS_SANDBOX"), "TRUE")[0] == "ok"
    assert validate(get("USERS_SANDBOX"), "true")[0] == "warning"
    assert validate(get("APP_URL"), "localhost:8000")[0] == "error"
    assert validate(get("APP_URL"), "")[0] == "ok"


def test_cli_tidy_apply(tmp_path, monkeypatch, capsys):
    """``tidy --apply`` : fichier rangé sur place, sauvegarde, valeurs identiques."""
    monkeypatch.setattr(env_tool, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(env_tool, "LOCK_PATH", tmp_path / "env.lock")
    path = tmp_path / ".env"
    path.write_text(SAMPLE, encoding="utf-8")
    reference = dict(dotenv_values(path))
    assert env_tool.main(["tidy", "--env", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == SAMPLE
    assert env_tool.main(["tidy", "--env", str(path), "--apply"]) == 0
    assert dict(dotenv_values(path)) == reference
    assert len(os.listdir(tmp_path / "backups")) == 1
    out = capsys.readouterr().out
    assert SECRET not in out and "plain-value" not in out
    assert env_tool.main(["tidy", "--env", str(path), "--apply"]) == 0
    assert len(os.listdir(tmp_path / "backups")) == 1


def test_cli_example_check_detects_drift(tmp_path):
    """``example --check`` : code 1 si le fichier diffère du registre."""
    target = tmp_path / ".env.example"
    target.write_text("ancien\n", encoding="utf-8")
    assert env_tool.main(["example", "--output", str(target), "--check"]) == 1
    assert env_tool.main(["example", "--output", str(target)]) == 0
    assert env_tool.main(["example", "--output", str(target), "--check"]) == 0


# ---------------------------------------------------------------------------
# Variables historiques sans effet (mode unifié)
# ---------------------------------------------------------------------------
PRUNE_SAMPLE = """# --- 2.1 Par défaut ---
LLM_DEFAULT_SERVER=https://openrouter.ai/api/v1
LLM_DEFAULT_MODEL=google/gemini-3.8-flash
# snapshot du durcissement
RECODE_MODEL=gpt-4o-mini-2024-07-18
RECODE_PREFER_OPENAI=1
OCR_ENABLE_ALBERT=1
EMBEDDING_PROVIDER=albert
ALBERT_AUDIO_ENABLED=0
LLM_RAG_MODEL=gpt-oss-120b
"""


def test_ignored_historical_follows_the_couples():
    """Mode unifié : recodage et défaut toujours remplacés ; OCR et embeddings seulement si le modèle est déclaré."""
    from scripts.rad_settings.render import ignored_historical

    values = dict(dotenv_file.values(dotenv_file.parse(PRUNE_SAMPLE)))
    # ALBERT_AUDIO_ENABLED=0 : valeur par défaut du code, sans effet même sans couple.
    assert [n for n, _ in ignored_historical(values)] == ["RECODE_MODEL", "RECODE_PREFER_OPENAI", "ALBERT_AUDIO_ENABLED"]
    values.update(OCR_MODEL="lightonocr-2-1b", EMBEDDING_MODEL="bge-m3")
    assert [n for n, _ in ignored_historical(values)] == ["RECODE_MODEL", "RECODE_PREFER_OPENAI", "OCR_ENABLE_ALBERT",
                                                          "EMBEDDING_PROVIDER", "ALBERT_AUDIO_ENABLED"]
    del values["LLM_DEFAULT_SERVER"]
    assert ignored_historical(values) == []
    assert [n for n, _ in check(values, [])["ignored"]] == []
    # Réponses sourcées retirées (2026-10-03) : variables obsolètes.
    assert [n for n, _ in check(values, [])["obsolete"]] == ["LLM_RAG_MODEL"]


def test_cli_prune(tmp_path, monkeypatch, capsys):
    """``prune`` : aperçu, refus d'une variable encore lue, retrait ciblé avec ses commentaires collés."""
    monkeypatch.setattr(env_tool, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(env_tool, "LOCK_PATH", tmp_path / "env.lock")
    path = tmp_path / ".env"
    path.write_text(PRUNE_SAMPLE, encoding="utf-8")
    assert env_tool.main(["prune", "--env", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == PRUNE_SAMPLE
    assert env_tool.main(["prune", "--env", str(path), "--apply", "EMBEDDING_PROVIDER"]) == 1
    assert env_tool.main(["prune", "--env", str(path), "--apply", "RECODE_MODEL", "LLM_RAG_MODEL"]) == 0
    text = path.read_text(encoding="utf-8")
    assert "RECODE_MODEL" not in text and "snapshot du durcissement" not in text and "RECODE_PREFER_OPENAI=1" in text
    assert "LLM_RAG_MODEL" not in text
    values = dict(dotenv_values(path))
    assert values["EMBEDDING_PROVIDER"] == "albert" and values["OCR_ENABLE_ALBERT"] == "1"
    assert len(os.listdir(tmp_path / "backups")) == 1
    assert "gpt-4o-mini-2024-07-18" not in capsys.readouterr().out


def test_prefer_openai_has_no_effect_in_unified_mode(monkeypatch):
    """Durcissement + mode unifié : le serveur déclaré décide (slug OpenRouter), pas RECODE_PREFER_OPENAI."""
    from scripts import rad_chunk
    from scripts.rad_recode_cache import RecodeConfig

    cfg = RecodeConfig(harden_enabled=True, prefer_openai=True)
    monkeypatch.delenv("LLM_DEFAULT_SERVER", raising=False)
    assert rad_chunk.recode_uses_openrouter("openai/gpt-4o-mini", cfg) is False
    monkeypatch.setenv("LLM_DEFAULT_SERVER", "https://openrouter.ai/api/v1")
    assert rad_chunk.recode_uses_openrouter("openai/gpt-4o-mini", cfg) is True
    assert rad_chunk.recode_uses_openrouter("gpt-4o-mini", cfg) is False
