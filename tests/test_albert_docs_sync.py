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

Lot 9 : cache tiktoken persistant posé par le ``conftest.py`` racine et documenté
(section 10) ; comportements du limiteur, des réessais sur 429, de la liste
``/v1/models`` et de ``/save_credentials`` décrits par le guide (sections 3.2,
3.5, 3.6 et 8) et confrontés au code ; changements visibles, tests de
régression et journal d'exécution du lot 9 consignés dans ``SPRINT_albert.md``.

Aucun appel réseau ; le ``.env`` réel n'est jamais lu. Les assertions portent
sur des noms et des booléens (les valeurs comparées sont des défauts publics).
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
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


# ---------------------------------------------------------------------------
# Lot 9 : cache tiktoken, limiteur, réessais, /v1/models, /save_credentials
# ---------------------------------------------------------------------------
SPRINT_DOC = REPO / ".claude" / "tasks" / "SPRINT_albert.md"
ROOT_CONFTEST = REPO / "conftest.py"

TIKTOKEN_WARM_COMMAND = (
    "TIKTOKEN_CACHE_DIR=$PWD/data/tiktoken_cache .venv/bin/python -c \"import tiktoken; "
    "tiktoken.encoding_for_model('text-embedding-3-large'); tiktoken.get_encoding('o200k_base')\""
)
"""Commande (une ligne) qui préchauffe le cache tiktoken persistant des tests."""

TIKTOKEN_CACHE_NAMES = ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR")
"""Variables lues par tiktoken pour situer son cache (la première l'emporte)."""

_CONFTEST_PROBE = (
    "import json, os, runpy, sys; runpy.run_path('conftest.py'); "
    "print(json.dumps({'value': os.environ.get('TIKTOKEN_CACHE_DIR'), "
    "'tiktoken_imported': 'tiktoken' in sys.modules}))"
)
"""Exécute une copie du conftest racine puis rapporte la variable posée (JSON)."""

LINE_SEPARATORS = tuple(
    chr(code) for code in range(0x110000) if len(("a" + chr(code) + "b").splitlines()) > 1
)
"""Séparateurs de ligne au sens de ``str.splitlines`` (refusés par ``/save_credentials``)."""


def _doc_section(text: str, heading: str) -> str:
    """Texte de la section dont le titre commence par ``heading`` (jusqu'au titre
    suivant de même niveau ou de niveau supérieur, ou jusqu'à un ``---``)."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(heading))
    level = len(heading) - len(heading.lstrip("#"))
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        depth = len(line) - len(line.lstrip("#"))
        if line.strip() == "---" or (depth and depth <= level and line[depth:depth + 1] == " "):
            end = index
            break
    return "\n".join(lines[start:end])


def _bullet_block(text: str, marker: str) -> str:
    """Puce de premier niveau commençant par ``marker`` et ses sous-puces."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(marker))
    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith(("- ", "#"))), len(lines))
    return "\n".join(lines[start:end])


def _run_conftest_copy(tmp_path: Path, extra_env: Dict[str, str]) -> Dict[str, object]:
    """Importe une copie du conftest racine dans un sous-processus (dossier
    temporaire, sans ``.env``, proxy mort) et renvoie ce qu'il a posé."""
    shutil.copyfile(ROOT_CONFTEST, tmp_path / "conftest.py")
    env = {
        name: value for name, value in os.environ.items()
        if name not in TIKTOKEN_CACHE_NAMES and name != "PYTHONPATH" and not name.startswith("ALBERT_")
    }
    env.update({
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "127.0.0.1,localhost,testserver",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    env.update(extra_env)
    proc = subprocess.run([sys.executable, "-c", _CONFTEST_PROBE], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_conftest_defaults_tiktoken_cache_dir_to_repo_data(tmp_path):
    """Sans ``TIKTOKEN_CACHE_DIR``, le conftest racine pose, dès son import, le
    cache persistant ``<dépôt>/data/tiktoken_cache`` (le cache par défaut, sous
    le dossier temporaire du système, peut être purgé) ; il n'importe pas
    tiktoken (aucun téléchargement possible à ce stade)."""
    seen = _run_conftest_copy(tmp_path, {})
    expected = os.path.join(str(tmp_path), "data", "tiktoken_cache")
    assert seen["value"] is not None
    assert os.path.realpath(seen["value"]) == os.path.realpath(expected)
    assert seen["tiktoken_imported"] is False


def test_conftest_keeps_an_explicit_tiktoken_cache_location(tmp_path):
    """Un emplacement choisi par l'appelant (``TIKTOKEN_CACHE_DIR``, même vide, ou
    l'ancien ``DATA_GYM_CACHE_DIR``) n'est jamais remplacé par le conftest."""
    chosen = str(tmp_path / "cache-choisi")
    assert _run_conftest_copy(tmp_path, {"TIKTOKEN_CACHE_DIR": chosen})["value"] == chosen
    assert _run_conftest_copy(tmp_path, {"TIKTOKEN_CACHE_DIR": ""})["value"] == ""
    assert _run_conftest_copy(tmp_path, {"DATA_GYM_CACHE_DIR": chosen})["value"] is None


def test_albert_doc_tiktoken_cache_note():
    """La section des tests du guide nomme le cache persistant et donne, sur une
    ligne, la commande qui le préchauffe."""
    section = _doc_section(ALBERT_DOC.read_text(encoding="utf-8"), "## 10.")
    assert "data/tiktoken_cache" in section
    assert TIKTOKEN_WARM_COMMAND in section


LIMITER_REQUEST_BOUND = (
    "au plus RPM × part + rafale − 1 requêtes sur toute minute glissante, "
    "la rafale valant au plus la concurrence du rôle"
)
"""Borne exacte du seau local (requêtes), telle que le guide et le sprint l'énoncent."""

LIMITER_TOKEN_BOUND = "au plus TPM × part + rafale de tokens − 1 tokens sur toute minute glissante"
"""Borne exacte du seau local (tokens), telle que le guide l'énonce."""

FALSE_SLIDING_CLAIM = re.compile(r"jamais plus de[^.;]*minute glissante")
"""Ancienne affirmation fausse : « jamais plus de RPM … sur une minute glissante »."""


def _saturated_stamps(limiter, clock: List[float], count: int) -> List[float]:
    """Instants de ``count`` acquisitions enchaînées sur l'horloge factice ``clock``."""
    stamps = []
    for _ in range(count):
        limiter.acquire(0)
        stamps.append(clock[0])
    return stamps


def _max_per_sliding_minute(stamps: List[float], tolerance: float = 1e-6) -> int:
    """Plus grand nombre d'instants dans une fenêtre glissante ``[t, t + 60)``."""
    return max(sum(1 for t in stamps[i:] if t < start + 60.0 - tolerance) for i, start in enumerate(stamps))


def test_albert_doc_limiter_section_matches_code():
    """Section 3.2 : budget ``notes`` de gpt-oss sous un rôle du budget ``recode``
    (CLI et web), rafale bornée par la concurrence du rôle, prise sur la première
    minute et plafond du niveau accumulé (borne exacte sur toute minute glissante,
    requêtes et tokens), Redis retenté après une panne."""
    from scripts.rad_albert import limiter as albert_limiter

    section = _doc_section(ALBERT_DOC.read_text(encoding="utf-8"), "### 3.2")

    for role in ("recode", "citation", "book_structure", "long_context"):
        assert albert_limiter.limiter_role(role, "albert/gpt-oss-120b") == "notes", role
    assert albert_limiter.limiter_role("recode", "ministral-3-8b-instruct-2512") == "recode"
    for phrase in ("gpt-oss", "budget `notes`", "CLI", "web"):
        assert phrase in section, phrase

    cfg = dataclasses.replace(AlbertConfig(enabled=True), recode_concurrency=3)
    limiter = albert_limiter.get_limiter("recode", cfg, clock=lambda: 0.0, sleep=lambda seconds: None)
    waits = [limiter.reserve(0) for _ in range(4)]
    assert waits[:3] == [0.0] * 3 and waits[3] > 0
    for phrase in ("rafale de départ", "concurrence", "première minute", LIMITER_REQUEST_BOUND,
                   LIMITER_TOKEN_BOUND):
        assert phrase in section, phrase
    assert FALSE_SLIDING_CLAIM.search(section) is None

    # Borne annoncée = borne du code : saturé, deux minutes d'inactivité, saturé de
    # nouveau ; la minute glissante qui suit l'inactivité atteint exactement
    # RPM × part + rafale − 1 (46 à 45 par minute avec la rafale de 2 du recodage).
    defaults = AlbertConfig(enabled=True)
    now = [0.0]

    def _advance(seconds):
        """Sommeil factice : avance l'horloge."""
        now[0] += max(0.0, float(seconds))

    bucket = albert_limiter.get_limiter("recode", defaults, clock=lambda: now[0], sleep=_advance)
    stamps = _saturated_stamps(bucket, now, 3 * defaults.recode_rpm)
    now[0] += 120.0
    after_idle = _saturated_stamps(bucket, now, 3 * defaults.recode_rpm)
    bound = defaults.recode_rpm * defaults.process_share + defaults.recode_concurrency - 1
    assert _max_per_sliding_minute(stamps + after_idle) == bound == 46
    assert "46 requêtes à 45 par minute" in section

    # Rafale de tokens = même fraction de minute (rafale × TPM / RPM), valeurs du guide.
    # La part du processus s'applique aux deux débits : elle s'annule dans le rapport.
    for role, rpm, burst in (("recode", defaults.recode_rpm, defaults.recode_concurrency),
                             ("notes", defaults.notes_rpm, defaults.notes_concurrency)):
        tokens = albert_limiter.get_limiter(role, defaults, clock=lambda: 0.0, sleep=lambda seconds: None)
        expected = min(defaults.chat_tpm * defaults.process_share, burst * defaults.chat_tpm / rpm)
        assert abs(tokens.token_burst - expected) < 1e-6, role
    for figure in ("5 100", "25 600"):
        assert figure in section, figure

    retry_every = f"{albert_limiter.REDIS_RETRY_SECONDS:g} s"
    assert retry_every == "30 s"
    assert "Redis" in section and retry_every in section


def test_albert_doc_quota_section_matches_retry():
    """Sections 3.2 et 3.5 : un 429 sans ``Retry-After`` ne devient un quota
    épuisé qu'après une fenêtre complète d'une minute depuis le premier 429."""
    from scripts.rad_albert import retry as albert_retry

    assert albert_retry.RATE_WINDOW_SECONDS == 60.0
    text = ALBERT_DOC.read_text(encoding="utf-8")
    for heading in ("### 3.2", "### 3.5"):
        section = _doc_section(text, heading)
        for phrase in ("sans `Retry-After`", "fenêtre complète"):
            assert phrase in section, (heading, phrase)


def test_albert_doc_models_listing_section():
    """Section 3.6 : liste ``/v1/models`` du compte (cache de 600 s), alias
    résolus par elle et replis absents retirés, en CLI comme sur le web ; sur le
    web, sans liste en cache, elle est chargée juste avant le premier repli."""
    from scripts.rad_albert import preflight as albert_preflight

    assert albert_preflight.CACHE_TTL_SECONDS == 600.0
    assert _notes_module().ALBERT_LISTING_TTL_SECONDS == 600.0
    section = _doc_section(ALBERT_DOC.read_text(encoding="utf-8"), "### 3.6")
    for phrase in ("`/v1/models`", "600 s", "CLI", "web", "alias", "repli", "set_model_listing",
                   "juste avant le premier repli"):
        assert phrase in section, phrase
    assert "aucun alias résolu récemment sur le web" not in section


def _notes_module():
    """Module des notes (import paresseux : l'application n'est chargée que pour ce contrôle)."""
    from app.utils import llm_note_generator

    return llm_note_generator


def test_albert_doc_save_credentials_matches_code():
    """Section 8 : ``/save_credentials`` ne refuse que les séparateurs de ligne (au
    sens de ``str.splitlines``) et NUL ; une tabulation est acceptée, comme le
    fait le code."""
    from app.routes import settings as settings_routes

    for char in LINE_SEPARATORS + ("\x00",):
        assert settings_routes._has_forbidden_chars(f"a{char}b") is True, repr(char)
    assert settings_routes._has_forbidden_chars("a\tb") is False
    section = _doc_section(ALBERT_DOC.read_text(encoding="utf-8"), "## 8.")
    for phrase in ("`POST /save_credentials`", "séparateur de ligne", "str.splitlines", "NUL", "tabulation"):
        assert phrase in section, phrase


LOT9_VISIBLE_CHANGES = (
    "/upload_stage_file", "/upload_zip", "/upload_csv", "/cluster_documents_sse",
    "uploads/", "401", "403", "400", "/save_credentials",
)
"""Routes et statuts des changements visibles du lot 9 (chemin désactivé)."""

LOT9_REGRESSION_FILES = (
    "tests/test_albert_review_regressions_f1.py",
    "tests/test_albert_review_regressions_f2.py",
    "tests/test_albert_review_regressions_f3.py",
    "tests/test_albert_review_regressions_f4.py",
    "tests/test_albert_review_regressions_g1.py",
    "tests/test_session_ownership.py",
    "tests/test_albert_blank_chunks.py",
    "tests/test_albert_web_ledger.py",
)
"""Fichiers des tests de régression du lot 9 (au lieu d'un fichier unique)."""

LOT9_COMMITS = ("a84e941", "0dd3e23", "be3693d")
"""Commits du lot 9 consignés dans le journal d'exécution."""


def test_sprint_records_lot9():
    """``SPRINT_albert.md`` consigne le lot 9 : changements visibles acceptés,
    fichiers des tests de régression et entrées du journal d'exécution."""
    text = SPRINT_DOC.read_text(encoding="utf-8")
    derogations = _bullet_block(text, "- **Dérogations et changements visibles acceptés.**")
    for marker in LOT9_VISIBLE_CHANGES:
        assert marker in derogations, marker

    risks = _bullet_block(text, "- **Rafale initiale du seau local.**")
    assert LIMITER_REQUEST_BOUND in risks
    assert FALSE_SLIDING_CLAIM.search(risks) is None

    lot9 = _doc_section(text, "### Lot 9")
    journal = _doc_section(text, "## Journal d'exécution")
    for path in LOT9_REGRESSION_FILES:
        assert path in lot9, path
    for commit in LOT9_COMMITS:
        assert commit in journal, commit
    assert "commit final du lot 9 en attente" in journal
