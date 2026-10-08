"""Registre unique des variables (sprint « configuration unifiée », lot L1).

Le registre ``scripts/rad_settings/registry.py`` doit rester **complet** et
**honnête** :

* toute variable lue par le code applicatif (``app/``, ``scripts/``,
  ``ingestion/``, ``core/``) y figure, ou dans ``INTERNAL_NAMES`` ;
* une variable ``planned`` n'est encore lue nulle part (sinon elle est active) ;
* les défauts Albert sont ceux de ``ENV_REGISTRY`` ;
* les variables personnelles (portée ``user``) sont exactement celles de
  ``CREDENTIAL_ENV_MAPPING``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from scripts.rad_albert.config import ENV_REGISTRY, FIELD_ENV_NAMES
from scripts.rad_settings import registry as reg

REPO = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = ("app", "scripts", "ingestion", "core")
READ_METHODS = {"get", "pop", "setdefault"}
WRITE_PREFIXES = ("set", "del", "unset")


def _literal(node: ast.AST) -> str:
    """Chaîne littérale portée par ``node`` (``''`` sinon)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _looks_like_env_name(text: str) -> bool:
    """Vrai pour un nom en majuscules de trois caractères ou plus."""
    return len(text) >= 3 and text[0].isalpha() and text.upper() == text and text.replace("_", "").isalnum()


def _is_env_read(node: ast.Call) -> bool:
    """Vrai si l'appel lit une variable : ``getenv``, ``environ.get``,
    ``env.get``, ``dotenv_values(...).get`` ou un utilitaire ``*env*``."""
    func = node.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
    if name == "getenv":
        return True
    if name in READ_METHODS and isinstance(func, ast.Attribute):
        receiver = ast.unparse(func.value).lower()
        return "env" in receiver
    lowered = name.lower()
    return "env" in lowered and not lowered.startswith(WRITE_PREFIXES)


def _reads(tree: ast.AST) -> Iterator[Tuple[str, int]]:
    """Noms de variables lus dans un module : ``(nom, ligne)``."""
    for node in ast.walk(tree):
        name = ""
        if isinstance(node, ast.Call) and node.args and _is_env_read(node):
            name = _literal(node.args[0])
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if "env" in ast.unparse(node.value).lower():
                name = _literal(node.slice)
        elif isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            if node.comparators and "env" in ast.unparse(node.comparators[0]).lower():
                name = _literal(node.left)
        if name and _looks_like_env_name(name):
            yield name, getattr(node, "lineno", 0)


def _collect_reads() -> Dict[str, List[str]]:
    """Table nom → emplacements des lectures dans le code applicatif."""
    found: Dict[str, List[str]] = {}
    for root in SCANNED_ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            if "__pycache__" in path.parts or path.name.startswith("test_") or path.name == "conftest.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for name, line in _reads(tree):
                found.setdefault(name, []).append(f"{path.relative_to(REPO)}:{line}")
    for env_name in FIELD_ENV_NAMES.values():
        found.setdefault(env_name, []).append("scripts/rad_albert/config.py:FIELD_ENV_NAMES")
    return found


READS = _collect_reads()


def test_every_variable_read_by_the_code_is_registered():
    """Une variable lue par le code figure au registre (ou parmi les internes)."""
    missing = {
        name: places for name, places in READS.items()
        if name not in reg.BY_NAME and name not in reg.INTERNAL_NAMES
    }
    assert not missing, f"variables lues mais absentes du registre : {missing}"


def test_planned_variables_are_not_read_yet():
    """Une variable ``planned`` n'est lue nulle part : sinon, la passer ``active``."""
    read_planned = {name: READS[name] for name in reg.names(reg.PLANNED) if name in READS}
    assert not read_planned, f"variables prévues déjà lues : {read_planned}"


def test_albert_defaults_match_env_registry():
    """Chaque variable d'``ENV_REGISTRY`` est au registre, avec le même défaut."""
    for name, default, _description in ENV_REGISTRY:
        setting = reg.get(name)
        assert setting is not None, name
        assert setting.default == default, (name, setting.default, default)
        assert setting.status == reg.ACTIVE, name


def test_user_scope_matches_credential_mapping():
    """Les variables personnelles actives sont exactement celles des identifiants."""
    from app.core.credentials import CREDENTIAL_ENV_MAPPING

    user_scoped = {s.name for s in reg.SETTINGS if s.scope == reg.USER and s.status == reg.ACTIVE}
    assert user_scoped == set(CREDENTIAL_ENV_MAPPING.values())


def test_layout_keys_are_ordered_and_nested():
    """Blocs numérotés dans l'ordre, sous-blocs préfixés par leur bloc et croissants."""
    block_numbers = [int(block.key) for block in reg.LAYOUT]
    assert block_numbers == list(range(1, len(block_numbers) + 1))
    for block in reg.LAYOUT:
        sub_numbers = []
        for sub in block.subblocks:
            prefix, _, rank = sub.key.partition(".")
            assert prefix == block.key, sub.key
            sub_numbers.append(int(rank))
        assert sub_numbers == list(range(1, len(sub_numbers) + 1)), block.key


def test_every_setting_has_flattened_position():
    """L'aplatissement renseigne bloc et sous-bloc, dans l'ordre de ``LAYOUT``."""
    order = [(b.key, s.key, x.name) for b, s, x in reg.iter_layout()]
    assert [(x.block, x.subblock, x.name) for x in reg.SETTINGS] == order


def test_secrets_never_have_a_default_or_real_example():
    """Une clé n'a aucun défaut ; l'exemple est vide ou un gabarit en X."""
    for setting in reg.SETTINGS:
        if not setting.is_secret:
            continue
        assert setting.default == "", setting.name
        example = setting.example_value
        assert example == "" or set(example.split("_")[-1].split("-")[-1]) == {"X"}, setting.name


def test_security_sensitive_variables_are_locked():
    """Les variables dont la modification depuis le web serait dangereuse sont verrouillées."""
    must_lock = {
        "JWT_SECRET_KEY", "JWT_SECRET_KEY_PREVIOUS", "RAGPY_ENV", "DEBUG", "CORS_ORIGINS", "DATABASE_URL",
        "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND", "ALBERT_LIMITER_REDIS_URL", "ALBERT_BASE_URL",
        "ALBERT_ALLOW_CUSTOM_HOST", "LOCAL_OCR_PYTHON", "FLOWER_USER", "FLOWER_PASSWORD",
    }
    unlocked = sorted(name for name in must_lock if not reg.BY_NAME[name].locked)
    assert not unlocked


def test_classify_names():
    """Classement des noms rencontrés dans un ``.env``."""
    assert reg.classify("MISTRAL_API_KEY") == "registered"
    assert reg.classify("ALBERT_LIVE") == "internal"
    assert reg.classify("MAX_ACTIVE_SESSIONS") == "obsolete"
    assert reg.classify("NOT_A_RAGPY_VARIABLE") == "unknown"


def test_planned_couples_cover_every_service():
    """Chaque service de ``models.SERVICES`` a ses deux variables au registre."""
    from scripts.rad_settings.models import DEFAULT_MODEL_VAR, DEFAULT_SERVER_VAR, SERVICES, SERVER_DEFS

    for service in SERVICES:
        assert service.server_var in reg.BY_NAME, service.server_var
        assert service.model_var in reg.BY_NAME, service.model_var
    assert DEFAULT_SERVER_VAR in reg.BY_NAME and DEFAULT_MODEL_VAR in reg.BY_NAME
    for server in SERVER_DEFS:
        for var in server.base_url_vars + (server.key_var,):
            assert var in reg.BY_NAME, var
