"""Isolation ``.env`` des sous-processus non-admin (``scripts/rad_env.py``).

Invariants 6, 7 et 8 du sprint Albert :

* un sous-processus lancé avec l'env non-admin de ``build_subprocess_env`` ne
  recharge jamais depuis le ``.env`` les secrets ``*_API_KEY`` retirés, et un
  ``override=True`` n'y remplace jamais une clé personnelle réinjectée ;
* sans ``RAGPY_DOTENV_DENY``, ``load_dotenv_guarded`` est exactement
  ``dotenv.load_dotenv`` ; la configuration non secrète reste relue ;
* plus aucun ``load_dotenv`` nu dans les scripts lancés en sous-processus.

Les sous-processus tournent avec ``cwd=tmp_path`` et ``python -c`` : ``find_dotenv``
cherche alors dans le répertoire courant, où seul le ``.env`` factice existe.
Seuls des noms et des booléens sont vérifiés, jamais des valeurs de clé.
"""

import ast
import json
import os
import re
import subprocess
import sys

import dotenv
import pytest

RAGPY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(RAGPY_ROOT, "scripts")
for _p in (RAGPY_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import rad_env  # noqa: E402
from tests.albert_fakes import FAKE_ALBERT_KEY  # noqa: E402


@pytest.fixture(autouse=True)
def _real_dotenv_lookup(monkeypatch):
    """Ces tests éprouvent la recherche du ``.env`` elle-même : sans le fichier
    neutre que le conftest désigne par ``RAGPY_ENV_FILE`` (tests hermétiques)."""
    monkeypatch.delenv("RAGPY_ENV_FILE", raising=False)


DENY_VAR = "RAGPY_DOTENV_DENY"

# Contenu du .env factice : deux secrets et une variable de configuration.
FAKE_DOTENV = {
    "OPENAI_API_KEY": "fake-openai-dotenv-0001",
    "ALBERT_API_KEY": FAKE_ALBERT_KEY,
    "MISTRAL_OCR_MODEL": "fake-mistral-ocr-model-dotenv-0001",
}
CHECKED_NAMES = sorted(FAKE_DOTENV)

# Scripts lancés en sous-processus par les routes et les tâches (liste littérale ;
# complétée par ceux que le code des routes et des tâches nomme).
SUBPROCESS_ENTRY_SCRIPTS = ("rad_dataframe.py", "rad_chunk.py", "rad_vectordb.py", "rad_clustering.py")
# Le seul module autorisé à appeler dotenv.load_dotenv directement.
GUARD_MODULE = os.path.join(SCRIPTS_DIR, "rad_env.py")


# ----------------------------------------------------------------------
# Aides
# ----------------------------------------------------------------------
@pytest.fixture
def restore_environ():
    """Restaure ``os.environ`` à l'identique après le test (les chargements dotenv y écrivent)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _write_dotenv(directory, values):
    """Écrit un fichier ``.env`` (``NOM=valeur`` par ligne) dans ``directory`` et renvoie son chemin."""
    path = directory / ".env"
    path.write_text("".join(f"{name}={value}\n" for name, value in values.items()), encoding="utf-8")
    return str(path)


def _clear(names):
    """Retire ``names`` de ``os.environ`` (le test tourne sous ``restore_environ``)."""
    for name in names:
        os.environ.pop(name, None)


def _presence(names):
    """Présence (booléen) de chaque nom dans ``os.environ``."""
    return {name: name in os.environ for name in names}


def _user(is_admin, personal=None):
    """Utilisateur transitoire (ADMIN ou USER) avec des identifiants personnels chiffrés."""
    import app.models  # noqa: F401  (enregistre tous les mappers utilisés par User)
    import app.models.pipeline_session  # noqa: F401
    from app.core import credentials as creds
    from app.models.user import User

    return User(
        email="admin@example.test" if is_admin else "user@example.test",
        hashed_password="x",
        roles=["USER", "ADMIN"] if is_admin else ["USER"],
        is_active=True,
        is_verified=True,
        api_credentials=creds.encrypt_credentials(personal) if personal else None,
    )


def _run_presence_probe(env, cwd, loader):
    """Lance ``python -c`` : chargement ``.env`` par ``loader`` puis présence des noms vérifiés.

    ``loader`` vaut ``'guarded'`` (``rad_env.load_dotenv_guarded``) ou ``'plain'``
    (``dotenv.load_dotenv``, témoin). Renvoie le dictionnaire nom → booléen.
    """
    call = "rad_env.load_dotenv_guarded()" if loader == "guarded" else "dotenv.load_dotenv()"
    code = (
        "import sys, os, json\n"
        f"sys.path.insert(0, {SCRIPTS_DIR!r})\n"
        "import dotenv, rad_env\n"
        f"{call}\n"
        f"print(json.dumps({{n: n in os.environ for n in {CHECKED_NAMES!r}}}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ----------------------------------------------------------------------
# Sous-processus non-admin
# ----------------------------------------------------------------------
def test_subprocess_cannot_refill_secrets_from_dotenv(tmp_path):
    from app.core import credentials as creds

    _write_dotenv(tmp_path, FAKE_DOTENV)
    env = creds.build_subprocess_env(_user(is_admin=False))
    denied = rad_env.parse_deny_list(env.get(DENY_VAR))
    assert {"OPENAI_API_KEY", "ALBERT_API_KEY"} <= denied
    assert "MISTRAL_OCR_MODEL" not in denied
    # Point de départ : aucun des trois noms dans l'env du sous-processus.
    for name in CHECKED_NAMES:
        env.pop(name, None)

    guarded = _run_presence_probe(env, tmp_path, "guarded")
    assert guarded == {"ALBERT_API_KEY": False, "MISTRAL_OCR_MODEL": True, "OPENAI_API_KEY": False}

    # Témoin : un load_dotenv nu recharge bien les secrets depuis ce même .env.
    plain = _run_presence_probe(env, tmp_path, "plain")
    assert plain == {"ALBERT_API_KEY": True, "MISTRAL_OCR_MODEL": True, "OPENAI_API_KEY": True}


def test_admin_subprocess_env_reloads_dotenv_as_before(tmp_path):
    from app.core import credentials as creds

    _write_dotenv(tmp_path, FAKE_DOTENV)
    env = creds.build_subprocess_env(_user(is_admin=True))
    assert DENY_VAR not in env
    for name in CHECKED_NAMES:
        env.pop(name, None)
    assert _run_presence_probe(env, tmp_path, "guarded") == _run_presence_probe(env, tmp_path, "plain")


def test_rad_chunk_subprocess_does_not_refill_openai_key(tmp_path):
    from app.core import credentials as creds

    _write_dotenv(tmp_path, FAKE_DOTENV)
    env = creds.build_subprocess_env(_user(is_admin=False))
    for name in CHECKED_NAMES + ["OPENROUTER_API_KEY"]:
        env.pop(name, None)
    code = (
        "import sys, os\n"
        f"sys.path.insert(0, {SCRIPTS_DIR!r})\n"
        "import rad_chunk\n"
        "print('RESULT', rad_chunk.client is None, 'OPENAI_API_KEY' in os.environ,"
        " 'MISTRAL_OCR_MODEL' in os.environ)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path), env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "RESULT True False True"


# ----------------------------------------------------------------------
# load_dotenv_guarded en processus
# ----------------------------------------------------------------------
def test_guard_is_plain_load_dotenv_without_marker(tmp_path, monkeypatch, restore_environ):
    path = _write_dotenv(tmp_path, FAKE_DOTENV)

    # Marqueur absent : appel direct. Marqueur vide (toutes les clés réinjectées) :
    # sans override, l'environnement obtenu est le même que celui de load_dotenv.
    for marker in (None, ""):
        if marker is None:
            os.environ.pop(DENY_VAR, None)
        else:
            os.environ[DENY_VAR] = marker
        _clear(CHECKED_NAMES)
        plain_result = dotenv.load_dotenv(path)
        plain_env = dict(os.environ)
        _clear(CHECKED_NAMES)
        guarded_result = rad_env.load_dotenv_guarded(path)
        assert guarded_result is plain_result is True
        assert dict(os.environ) == plain_env

    # Arguments transmis tels quels, valeur de retour relayée.
    os.environ.pop(DENY_VAR, None)
    calls = []

    def spy(*args, **kwargs):
        """Enregistre l'appel à ``dotenv.load_dotenv`` et renvoie une valeur reconnaissable."""
        calls.append((args, kwargs))
        return "sentinel"

    monkeypatch.setattr(rad_env.dotenv, "load_dotenv", spy)
    assert rad_env.load_dotenv_guarded("a.env", override=True, verbose=False) == "sentinel"
    assert rad_env.load_dotenv_guarded() == "sentinel"
    assert calls == [(("a.env",), {"override": True, "verbose": False}), ((), {})]


def test_config_vars_still_reloaded(tmp_path, restore_environ):
    values = dict(FAKE_DOTENV, OPENROUTER_DEFAULT_MODEL="fake-default-model-dotenv-0001")
    path = _write_dotenv(tmp_path, values)
    _clear(values)
    os.environ[DENY_VAR] = "ALBERT_API_KEY,OPENAI_API_KEY"

    assert rad_env.load_dotenv_guarded(path) is True
    assert _presence(sorted(values)) == {
        "ALBERT_API_KEY": False,
        "MISTRAL_OCR_MODEL": True,
        "OPENAI_API_KEY": False,
        "OPENROUTER_DEFAULT_MODEL": True,
    }
    assert os.environ["MISTRAL_OCR_MODEL"] == FAKE_DOTENV["MISTRAL_OCR_MODEL"]

    # Un nom refusé déjà présent (clé personnelle réinjectée) garde sa valeur,
    # même avec override=True.
    _clear(values)
    os.environ["OPENAI_API_KEY"] = "fake-openai-personal-0002"
    rad_env.load_dotenv_guarded(path, override=True)
    assert os.environ["OPENAI_API_KEY"] == "fake-openai-personal-0002"
    assert "ALBERT_API_KEY" not in os.environ
    assert os.environ["MISTRAL_OCR_MODEL"] == FAKE_DOTENV["MISTRAL_OCR_MODEL"]


@pytest.mark.parametrize("personal", [
    {"openai_api_key": "fake-openai-personal-0004"},
    # Toutes les clés *_API_KEY réinjectées : la liste de refus est vide mais présente.
    "all",
], ids=["one-key", "all-keys-empty-deny"])
def test_override_true_keeps_reinjected_personal_keys(tmp_path, restore_environ, personal):
    from app.core import credentials as creds

    if personal == "all":
        personal = {
            cred_key: f"fake-{cred_key.replace('_', '-')}-personal-0004"
            for cred_key, env_key in creds.CREDENTIAL_ENV_MAPPING.items()
            if env_key.endswith("_API_KEY")
        }
    env = creds.build_subprocess_env(_user(is_admin=False, personal=personal))
    injected = {creds.CREDENTIAL_ENV_MAPPING[cred_key] for cred_key in personal}
    assert "OPENAI_API_KEY" in injected and "OPENAI_API_KEY" not in rad_env.parse_deny_list(env[DENY_VAR])
    if len(personal) > 1:
        assert env[DENY_VAR] == ""

    # .env admin factice : une autre valeur pour chaque clé réinjectée, un secret
    # non réinjecté et une configuration déjà présente dans l'env.
    dotenv_values = {name: f"fake-admin-dotenv-{index:04d}" for index, name in enumerate(sorted(injected))}
    dotenv_values.update(ALBERT_API_KEY=FAKE_ALBERT_KEY, MISTRAL_OCR_MODEL=FAKE_DOTENV["MISTRAL_OCR_MODEL"])
    path = _write_dotenv(tmp_path, dotenv_values)
    env["MISTRAL_OCR_MODEL"] = "fake-mistral-ocr-model-env-0004"
    os.environ.clear()
    os.environ.update(env)

    assert rad_env.load_dotenv_guarded(path, override=True) is True
    kept = {name: os.environ.get(name) == env[name] for name in sorted(injected)}
    assert kept == {name: True for name in sorted(injected)}
    if "ALBERT_API_KEY" not in injected:
        assert "ALBERT_API_KEY" not in os.environ
    # La configuration non secrète suit toujours l'override demandé.
    assert os.environ["MISTRAL_OCR_MODEL"] == FAKE_DOTENV["MISTRAL_OCR_MODEL"]

    # Témoin : sans la garde, override=True remplace bien la clé personnelle.
    os.environ.clear()
    os.environ.update(env)
    dotenv.load_dotenv(path, override=True)
    assert (os.environ.get("OPENAI_API_KEY") == env["OPENAI_API_KEY"]) is False


def test_deny_list_parsing_tolerates_spaces_and_empty_items():
    assert rad_env.parse_deny_list(None) == frozenset()
    assert rad_env.parse_deny_list("") == frozenset()
    assert rad_env.parse_deny_list("A_API_KEY,B_API_KEY") == {"A_API_KEY", "B_API_KEY"}
    assert rad_env.parse_deny_list(" A_API_KEY , ,B_API_KEY,") == {"A_API_KEY", "B_API_KEY"}


# ----------------------------------------------------------------------
# Contrat avec app/core/credentials.py
# ----------------------------------------------------------------------
def test_deny_literal_equals_credentials():
    from app.core import credentials as creds

    assert rad_env.DOTENV_DENY_ENV_VAR == DENY_VAR
    assert creds.DOTENV_DENY_ENV_VAR == rad_env.DOTENV_DENY_ENV_VAR

    api_key_names = {name for name in creds.CREDENTIAL_ENV_MAPPING.values() if name.endswith("_API_KEY")}
    env = creds.build_subprocess_env(_user(is_admin=False))
    raw = env[DENY_VAR]
    assert re.fullmatch(r"[A-Z0-9_]+(,[A-Z0-9_]+)*", raw)
    assert raw.split(",") == sorted(raw.split(","))
    assert rad_env.parse_deny_list(raw) == api_key_names

    personal = {"openai_api_key": "fake-openai-user-0003", "albert_api_key": FAKE_ALBERT_KEY}
    env = creds.build_subprocess_env(_user(is_admin=False, personal=personal))
    assert rad_env.parse_deny_list(env[DENY_VAR]) == api_key_names - {"OPENAI_API_KEY", "ALBERT_API_KEY"}

    assert DENY_VAR not in creds.build_subprocess_env(_user(is_admin=True))


# ----------------------------------------------------------------------
# Aucun load_dotenv nu dans les scripts lancés en sous-processus (AST)
# ----------------------------------------------------------------------
def _launched_scripts():
    """Scripts de ``scripts/`` lancés en sous-processus : liste littérale plus ceux nommés par l'app."""
    names = set(SUBPROCESS_ENTRY_SCRIPTS)
    pattern = re.compile(r"\b(rad_[a-z0-9_]+\.py)\b")
    for folder in ("routes", "tasks"):
        root = os.path.join(RAGPY_ROOT, "app", folder)
        for filename in sorted(os.listdir(root)):
            if filename.endswith(".py"):
                with open(os.path.join(root, filename), encoding="utf-8") as fh:
                    names.update(pattern.findall(fh.read()))
    return sorted(name for name in names if os.path.isfile(os.path.join(SCRIPTS_DIR, name)))


def _local_module_path(dotted):
    """Fichier de ``scripts/`` correspondant au module ``dotted`` (``scripts.x`` ou ``x``), sinon ``None``."""
    parts = dotted.split(".")
    if parts[0] == "scripts":
        parts = parts[1:]
    if not parts:
        return None
    base = os.path.join(SCRIPTS_DIR, *parts)
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(candidate):
            return candidate
    return None


def _imported_local_modules(tree):
    """Fichiers de ``scripts/`` importés par l'arbre ``tree`` (imports absolus seulement)."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dotted_names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            dotted_names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        else:
            continue
        for dotted in dotted_names:
            parts = dotted.split(".")
            # Un import de ``a.b.c`` exécute aussi les paquets ``a`` et ``a.b``.
            for end in range(1, len(parts) + 1):
                path = _local_module_path(".".join(parts[:end]))
                if path:
                    found.add(path)
    return found


def _bare_load_dotenv_uses(tree):
    """Lignes où ``load_dotenv`` est importé depuis dotenv ou appelé (nom ou attribut)."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "dotenv":
            if any(alias.name == "load_dotenv" for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "load_dotenv":
                lines.append(node.lineno)
    return lines


def test_no_bare_load_dotenv_in_subprocess_scripts():
    entries = _launched_scripts()
    assert {"rad_chunk.py", "rad_dataframe.py", "rad_vectordb.py"} <= set(entries)

    to_visit = [os.path.join(SCRIPTS_DIR, name) for name in entries]
    seen = set()
    offenders = []
    guarded_callers = set()
    while to_visit:
        path = to_visit.pop()
        if path in seen:
            continue
        seen.add(path)
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        if os.path.samefile(path, GUARD_MODULE):
            continue
        offenders += [f"{os.path.relpath(path, RAGPY_ROOT)}:{line}" for line in _bare_load_dotenv_uses(tree)]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_dotenv_guarded":
                guarded_callers.add(os.path.basename(path))
        to_visit.extend(_imported_local_modules(tree) - seen)

    assert GUARD_MODULE in seen
    assert offenders == []
    # Les deux scripts qui chargent le .env le font par la garde.
    assert {"rad_chunk.py", "rad_dataframe.py"} <= guarded_callers
