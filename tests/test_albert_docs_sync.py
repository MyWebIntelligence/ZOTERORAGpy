"""Synchronisation entre le code Albert, les ``.env.example`` et la documentation.

Invariants couverts (``.claude/tasks/SPRINT_albert.md``, lot 8) :

* 32 : ``ALBERT_LIVE`` n'apparaît dans aucun fichier d'environnement d'exemple
  (interrupteur réservé au shell) ;
* 33 : toute variable Albert lue par le code est enregistrée dans
  ``ENV_REGISTRY``, présente dans le bloc Albert du ``.env.example`` racine
  avec son défaut, désactivée par défaut, et documentée dans
  ``.claude/docs/albert.md``.

S'y ajoutent des contrôles de cohérence entre le guide et le code (points
relevés au lot 8) : exceptions du bloc Albert quand ``ALBERT_ENABLED=0``,
absence d'invite interactive de clé dans ``rad_chunk.py``, ``.env`` non chargé
par ``rad_vectordb.py``, journal d'usage non écrit par les routes en processus
et assouplissement du journal des notes dans l'E2E.

Aucun appel réseau ; le ``.env`` réel n'est jamais lu. Les assertions portent
sur des noms et des booléens (les valeurs comparées sont des défauts publics).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

from dotenv import dotenv_values

from scripts.rad_albert.config import ENV_REGISTRY, FIELD_ENV_NAMES, AlbertConfig

REPO = Path(__file__).resolve().parents[1]
ROOT_ENV_EXAMPLE = REPO / ".env.example"
SCRIPTS_ENV_EXAMPLE = REPO / "scripts" / ".env.example"
ALBERT_DOC = REPO / ".claude" / "docs" / "albert.md"

ALBERT_BLOCK_HEADER = "# ===== ALBERT API (DINUM) — OPT-IN, OFF PAR DÉFAUT ====="
FLOWER_LINE = "# Flower monitoring credentials"

LIVE_SWITCH = "ALBERT_" + "LIVE"
"""Interrupteur des tests live : shell uniquement, jamais enregistré ni écrit."""

EXTRA_ALBERT_NAMES = frozenset({"OCR_ENABLE_ALBERT", "EMBEDDING_PROVIDER", "DEDUP_SIM_THRESHOLD_BGE_M3"})
"""Variables du sprint Albert dont le nom ne commence pas par ``ALBERT_``."""

READ_METHODS = frozenset({"getenv", "get", "pop", "setdefault"})
"""Méthodes dont le premier argument littéral est un nom de variable lu."""

WRITE_PREFIXES = ("set", "del", "unset")
"""Préfixes des fonctions d'écriture d'env (``setenv``, ``delenv``…), ignorées."""

SCANNED_ROOTS = ("scripts", "app")
"""Racines du code applicatif parcourues par l'analyse AST."""


def _registry_names() -> List[str]:
    """Noms de ``ENV_REGISTRY`` dans l'ordre d'enregistrement."""
    return [name for name, _default, _meaning in ENV_REGISTRY]


def _is_albert_name(name: str) -> bool:
    """Vrai si ``name`` désigne une variable d'environnement du sprint Albert."""
    return name.startswith("ALBERT_") or name in EXTRA_ALBERT_NAMES


def _literal_name(node: ast.AST) -> str:
    """Chaîne littérale portée par ``node`` (``''`` si ce n'est pas une chaîne)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _callee_name(func: ast.AST) -> str:
    """Nom de la fonction appelée (``Name.id`` ou ``Attribute.attr``, sinon ``''``)."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_env_read_call(node: ast.Call) -> bool:
    """Vrai si l'appel lit une variable : ``getenv``/``get``/``pop``/``setdefault``
    ou un utilitaire d'env (``_env_bool``…), à l'exclusion des écritures."""
    callee = _callee_name(node.func)
    if callee in READ_METHODS:
        return True
    lowered = callee.lower()
    return "env" in lowered and not lowered.startswith(WRITE_PREFIXES)


def _env_reads(tree: ast.AST) -> Iterator[Tuple[str, int]]:
    """Noms Albert lus dans un module : ``(nom, ligne)`` pour chaque lecture.

    Formes reconnues : ``os.getenv("X")``, ``os.environ.get("X")``,
    ``env.get("X")``, ``dotenv_values(...).get("X")``, un utilitaire d'env
    appelé avec ``"X"`` en premier argument, ``environ["X"]`` en lecture et
    ``"X" in environ``. Les affectations (``env["X"] = …``) et les écritures
    (``setenv``, ``delenv``) ne comptent pas.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and _is_env_read_call(node):
            name = _literal_name(node.args[0])
            if _is_albert_name(name):
                yield name, node.lineno
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            name = _literal_name(node.slice)
            if _is_albert_name(name):
                yield name, node.lineno
        elif isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            name = _literal_name(node.left)
            if _is_albert_name(name):
                yield name, node.lineno


_NAME_RE = r"""['"]((?:ALBERT_[A-Z0-9_]+)|OCR_ENABLE_ALBERT|EMBEDDING_PROVIDER|DEDUP_SIM_THRESHOLD_BGE_M3)['"]"""
_REGEX_READS = (
    re.compile(r"\b(?!set(?!default)|del|unset)\w*(?:getenv|get|pop|setdefault|env\w*)\s*\(\s*" + _NAME_RE, re.IGNORECASE),
    re.compile(r"\[\s*" + _NAME_RE + r"\s*\](?!\s*=[^=])"),
    re.compile(_NAME_RE + r"\s+(?:not\s+)?in\b"),
)
"""Repli par expressions régulières (mêmes formes que ``_env_reads``)."""


def _regex_reads(source: str) -> Iterator[Tuple[str, int]]:
    """Lectures Albert d'un module que ``ast`` ne sait pas analyser (erreur de
    syntaxe préexistante) : même couverture que ``_env_reads``, par grep."""
    for lineno, line in enumerate(source.splitlines(), 1):
        for pattern in _REGEX_READS:
            for match in pattern.finditer(line):
                yield match.group(1), lineno


def _code_files() -> Iterator[Path]:
    """Modules Python du code applicatif (``scripts/`` et ``app/``), tests exclus."""
    for root in SCANNED_ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if path.name.startswith("test_") or path.name == "conftest.py":
                continue
            yield path


def _collect_reads() -> Dict[str, List[str]]:
    """Table nom → emplacements ``chemin:ligne`` des lectures Albert du code."""
    reads: Dict[str, List[str]] = {}
    for path in _code_files():
        source = path.read_text(encoding="utf-8")
        try:
            found = list(_env_reads(ast.parse(source, filename=str(path))))
        except SyntaxError:
            found = list(_regex_reads(source))
        for name, lineno in found:
            reads.setdefault(name, []).append(f"{path.relative_to(REPO)}:{lineno}")
    return reads


def _env_example_lines() -> List[str]:
    """Lignes du ``.env.example`` racine, sans fin de ligne."""
    return ROOT_ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()


def _assignment_index(lines: List[str]) -> Dict[str, int]:
    """Index de la première affectation ``NOM=`` de chaque variable du fichier."""
    pattern = re.compile(r"^([A-Z][A-Z0-9_]*)=")
    index: Dict[str, int] = {}
    for position, line in enumerate(lines):
        match = pattern.match(line)
        if match and match.group(1) not in index:
            index[match.group(1)] = position
    return index


def test_every_registered_var_in_env_example():
    """Chaque variable de ``ENV_REGISTRY`` figure une seule fois dans le bloc
    Albert du ``.env.example`` racine (avant ``# Flower…``), avec son défaut."""
    lines = _env_example_lines()
    assert lines.count(ALBERT_BLOCK_HEADER) == 1
    assert lines.count(FLOWER_LINE) == 1
    header_at = lines.index(ALBERT_BLOCK_HEADER)
    flower_at = lines.index(FLOWER_LINE)
    assert header_at < flower_at

    positions = _assignment_index(lines)
    values = dotenv_values(ROOT_ENV_EXAMPLE)
    missing = [name for name in _registry_names() if name not in positions]
    assert missing == []
    outside = [name for name in _registry_names() if not header_at < positions[name] < flower_at]
    assert outside == []
    duplicated = [
        name for name in _registry_names()
        if sum(1 for line in lines if line.startswith(name + "=")) != 1
    ]
    assert duplicated == []
    wrong_default = [name for name, default, _meaning in ENV_REGISTRY if values.get(name) != default]
    assert wrong_default == []


def test_every_albert_env_read_in_code_is_registered():
    """Toute lecture littérale d'une variable Albert dans ``scripts/`` et ``app/``
    vise un nom de ``ENV_REGISTRY`` (``ALBERT_LIVE`` excepté : shell seulement)."""
    registered: Set[str] = set(_registry_names())
    reads = _collect_reads()
    # Garde-fou : l'analyse trouve bien les lectures connues du code.
    assert {"ALBERT_API_KEY", "OCR_ENABLE_ALBERT", "EMBEDDING_PROVIDER"} <= set(reads)
    unregistered = sorted(
        f"{name} ({', '.join(places)})"
        for name, places in reads.items()
        if name != LIVE_SWITCH and name not in registered
    )
    assert unregistered == []
    # Lectures dynamiques d'AlbertConfig.from_env : toutes enregistrées.
    assert set(FIELD_ENV_NAMES.values()) <= registered
    assert LIVE_SWITCH not in registered


def test_albert_live_absent_from_env_examples():
    """``ALBERT_LIVE`` n'apparaît dans aucun des deux ``.env.example``, sous
    aucune casse (invariant 32)."""
    for path in (ROOT_ENV_EXAMPLE, SCRIPTS_ENV_EXAMPLE):
        text = path.read_text(encoding="utf-8")
        assert LIVE_SWITCH.lower() not in text.lower(), path.name


def test_albert_defaults_off():
    """Albert est désactivé par défaut : exemples, registre et configuration."""
    values = dotenv_values(ROOT_ENV_EXAMPLE)
    assert values.get("ALBERT_ENABLED") == "0"
    assert values.get("OCR_ENABLE_ALBERT") == "0"
    assert values.get("ALBERT_API_KEY") == ""
    assert values.get("EMBEDDING_PROVIDER") == "openai"
    assert values.get("DEDUP_SIM_THRESHOLD_BGE_M3") == ""

    defaults = {name: default for name, default, _meaning in ENV_REGISTRY}
    assert defaults["ALBERT_ENABLED"] == "0"
    assert defaults["OCR_ENABLE_ALBERT"] == "0"
    assert defaults["ALBERT_API_KEY"] == ""

    cfg = AlbertConfig.from_env({})
    assert cfg.enabled is False
    assert cfg.ocr_enabled is False
    cfg_example = AlbertConfig.from_env({k: v for k, v in values.items() if v is not None})
    assert cfg_example.enabled is False
    assert cfg_example.ocr_enabled is False

    scripts_lines = SCRIPTS_ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    assert scripts_lines[0].startswith("#")
    assert ".env.example" in scripts_lines[0]
    assert not any(re.match(r"^(ALBERT_|OCR_ENABLE_ALBERT=|EMBEDDING_PROVIDER=)", line) for line in scripts_lines)


def test_docs_mention_required_topics():
    """``.claude/docs/albert.md`` couvre les sujets imposés par le lot 8."""
    text = ALBERT_DOC.read_text(encoding="utf-8")
    required = (
        "RGPD",
        "expérimentation",
        "2026-10-01",
        "2026-12-01",
        "data/albert_manifests",
        "albert.api@numerique.gouv.fr",
        "ALBERT_SUBPROCESS_TIMEOUT",
        "DEDUP_SIM_THRESHOLD_BGE_M3",
        "RAGPY_DOTENV_DENY",
        "OCR_PROVIDER_FALLBACK",
        "ALBERT_OCR_SKIP_RECODE",
        "-m albert_live",
        ".venv/bin/python -m uvicorn",
        "docker compose up -d --build",
    )
    absent = [topic for topic in required if topic not in text]
    assert absent == []


def test_every_registered_var_documented_in_albert_md():
    """Chaque variable de ``ENV_REGISTRY`` est citée dans ``.claude/docs/albert.md``."""
    text = ALBERT_DOC.read_text(encoding="utf-8")
    undocumented = [name for name in _registry_names() if f"`{name}`" not in text]
    assert undocumented == []


def test_albert_doc_commands_on_one_line():
    """Aucune commande de ``.claude/docs/albert.md`` n'est coupée par une
    continuation ``\\`` en fin de ligne."""
    lines = ALBERT_DOC.read_text(encoding="utf-8").splitlines()
    continued = [number for number, line in enumerate(lines, 1) if line.rstrip().endswith("\\")]
    assert continued == []


PROJECT_GUIDES = (REPO / "CLAUDE.md", REPO / ".claude" / "CLAUDE.md")
"""Guides projet qui décrivent le comportement de ``rad_chunk.py``."""

VECTORDB_SCRIPT = REPO / "scripts" / "rad_vectordb.py"
VECTORDB_NO_DOTENV_NOTE = "ne charge pas le `.env`"
"""Phrase du guide Albert signalant que ``rad_vectordb.py`` ne lit pas le ``.env``."""

IN_PROCESS_LEDGER_NOTE = "n'écrivent pas encore"
"""Phrase du guide Albert signalant les routes en processus sans journal d'usage."""

LEDGER_WRITE_MARKERS = ("write_jsonl(", "USAGE_FILENAME", "albert_usage.jsonl")
"""Indices d'une écriture du journal ``albert_usage.jsonl`` dans le code."""


def _albert_block_preamble() -> List[str]:
    """Lignes de commentaire du bloc Albert du ``.env.example``, avant ``ALBERT_ENABLED=``."""
    lines = _env_example_lines()
    start = lines.index(ALBERT_BLOCK_HEADER)
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("ALBERT_ENABLED="))
    return lines[start:end]


def _app_writes_usage_ledger() -> bool:
    """Vrai si un module de ``app/`` (tests exclus) écrit le journal d'usage Albert."""
    for path in sorted((REPO / "app").rglob("*.py")):
        if "__pycache__" in path.parts or path.name.startswith("test_"):
            continue
        source = path.read_text(encoding="utf-8")
        if any(marker in source for marker in LEDGER_WRITE_MARKERS):
            return True
    return False


def test_env_example_preamble_names_off_exceptions():
    """L'en-tête du bloc Albert ne prétend pas que tout est ignoré avec Albert
    désactivé : il nomme les deux exceptions (``EMBEDDING_PROVIDER=albert``
    refusé, ``DEDUP_SIM_THRESHOLD_BGE_M3`` appliqué à tout fichier bge-m3)."""
    preamble = "\n".join(_albert_block_preamble())
    assert "toutes les variables de ce bloc sont ignorées" not in preamble
    assert "EMBEDDING_PROVIDER=albert" in preamble
    assert "DEDUP_SIM_THRESHOLD_BGE_M3" in preamble


def test_guides_do_not_promise_interactive_key_prompt():
    """``rad_chunk.py`` ne demande plus la clé OpenAI (lot 2) : les guides projet
    ne décrivent plus d'invite interactive."""
    stale = ("prompt interactif", "le script la demande", "prompts for missing OpenAI keys")
    found = [
        f"{path.relative_to(REPO)}: {phrase}"
        for path in PROJECT_GUIDES
        for phrase in stale
        if phrase in path.read_text(encoding="utf-8")
    ]
    assert found == []
    rad_chunk_source = (REPO / "scripts" / "rad_chunk.py").read_text(encoding="utf-8")
    assert "input(" not in rad_chunk_source


def test_rad_vectordb_dotenv_note_matches_code():
    """Le guide Albert dit que ``rad_vectordb.py`` ne charge pas le ``.env`` si,
    et seulement si, le script ne le charge effectivement pas."""
    loads_dotenv = "load_dotenv" in VECTORDB_SCRIPT.read_text(encoding="utf-8")
    documented = VECTORDB_NO_DOTENV_NOTE in ALBERT_DOC.read_text(encoding="utf-8")
    assert documented is (not loads_dotenv)


def test_in_process_usage_ledger_gap_documented():
    """Tant qu'aucun module de ``app/`` n'écrit ``albert_usage.jsonl``, le guide
    Albert signale que les routes en processus (notes, fiches, citations) ne
    l'écrivent pas encore (correction attendue au lot 9, qui met aussi le guide
    à jour)."""
    if _app_writes_usage_ledger():
        return
    assert IN_PROCESS_LEDGER_NOTE in ALBERT_DOC.read_text(encoding="utf-8")


E2E_SMOKE_SCRIPT = REPO / "scripts" / "albert_e2e_smoke.py"
NOTES_LEDGER_FLAG_RE = re.compile(r"^NOTES_LEDGER_REQUIRED\s*=\s*(True|False)\b", re.MULTILINE)
"""Affectation de la constante qui rend obligatoire le journal des notes (contrôle on6)."""


def test_e2e_notes_ledger_relaxation_documented():
    """Si l'E2E rend informatif le journal des notes (``NOTES_LEDGER_REQUIRED =
    False``), le guide Albert le dit : un ``E2E ALBERT OK`` ne prouve alors pas la
    traçabilité de l'usage des notes."""
    if not E2E_SMOKE_SCRIPT.is_file():
        return
    match = NOTES_LEDGER_FLAG_RE.search(E2E_SMOKE_SCRIPT.read_text(encoding="utf-8"))
    if match is None or match.group(1) == "True":
        return
    assert "NOTES_LEDGER_REQUIRED = False" in ALBERT_DOC.read_text(encoding="utf-8")
