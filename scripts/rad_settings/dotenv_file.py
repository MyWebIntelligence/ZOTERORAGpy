"""Lecture et écriture conservatrices d'un fichier ``.env`` (lot L2).

Contrairement à l'ancien rédacteur de ``/save_credentials`` (qui réécrivait le
fichier en ``k=v`` et perdait commentaires et lignes vides), ce module :

* découpe le fichier en lignes typées en gardant le **texte brut** de chaque
  affectation (guillemets, ``export``, commentaire de fin de ligne compris) :
  ranger un ``.env`` ne fait que déplacer des lignes, les valeurs restent
  identiques à l'octet ;
* écrit **sur place** (même inode : compatible avec un ``.env`` monté seul dans
  un conteneur, où ``os.replace`` échoue), sous verrou, après une sauvegarde
  horodatée en ``600``, avec ``fsync`` et relecture de contrôle (restauration
  automatique en cas d'écart).

Bibliothèque standard seulement. La sémantique des valeurs suit python-dotenv
pour les cas courants (guillemets simples et doubles, ``export``, commentaire
après une valeur non citée précédé d'un blanc, valeur citée sur plusieurs
lignes) ; ``env_tool`` vérifie l'égalité exacte avec ``dotenv_values``.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:  # verrou inter-processus (POSIX) ; absent sous Windows : écriture sans verrou
    import fcntl
except ImportError:  # pragma: no cover - plateforme sans fcntl
    fcntl = None  # type: ignore[assignment]

_ASSIGN_RE = re.compile(r"^\s*(export\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*)$")
_COMMENTED_ASSIGN_RE = re.compile(r"^\s*#\s*([A-Z][A-Z0-9_]+)=(.*)$")
_HEADER_RE = re.compile(r"^\s*#\s*(?:={3,}|-{3,})")

BLANK = "blank"
COMMENT = "comment"
ASSIGN = "assign"
OTHER = "other"


@dataclass
class EnvLine:
    """Une ligne logique d'un ``.env`` (une affectation citée peut couvrir
    plusieurs lignes physiques).

    Attributes:
        raw: Texte brut, sans fin de ligne finale (``\\n`` internes pour une
            valeur sur plusieurs lignes).
        kind: ``blank``, ``comment``, ``assign`` ou ``other``.
        name: Nom de la variable (affectation seulement).
        value: Valeur interprétée (affectation seulement).
        lineno: Numéro de la première ligne physique (1 pour la première).
    """

    raw: str
    kind: str
    name: str = ""
    value: str = ""
    lineno: int = 0

    @property
    def is_header(self) -> bool:
        """Vrai pour un commentaire d'en-tête (``# =====`` ou ``# ---``)."""
        return self.kind == COMMENT and bool(_HEADER_RE.match(self.raw))

    @property
    def commented_name(self) -> str:
        """Nom d'une variable commentée (``# NOM=valeur``), sinon ``''``."""
        if self.kind != COMMENT:
            return ""
        match = _COMMENTED_ASSIGN_RE.match(self.raw)
        return match.group(1) if match else ""


def _unquote_double(body: str) -> str:
    """Interprète les échappements d'une valeur entre guillemets doubles."""
    escapes = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\", "'": "'"}
    out: List[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] in escapes:
            out.append(escapes[body[i + 1]])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _closing_quote(text: str, quote: str) -> int:
    """Indice du guillemet fermant ``quote`` dans ``text`` (après l'ouvrant), ou -1."""
    i = 1
    while i < len(text):
        if quote == '"' and text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i
        i += 1
    return -1


def parse_value(rest: str) -> str:
    """Valeur d'une affectation à partir du texte après ``=``.

    Args:
        rest: Texte brut après le signe ``=`` (éventuellement sur plusieurs
            lignes pour une valeur citée).

    Returns:
        La valeur interprétée, comme python-dotenv pour les cas courants.
    """
    text = rest.strip()
    if text[:1] in ("'", '"'):
        quote = text[0]
        end = _closing_quote(text, quote)
        if end > 0:
            body = text[1:end]
            return _unquote_double(body) if quote == '"' else body
    # Valeur non citée : un commentaire commence à « # » précédé d'un blanc.
    match = re.search(r"\s#", rest)
    value = rest[:match.start()] if match else rest
    return value.strip()


def parse(text: str) -> List[EnvLine]:
    """Découpe le contenu d'un ``.env`` en lignes logiques.

    Args:
        text: Contenu du fichier.

    Returns:
        Les lignes, dans l'ordre du fichier.
    """
    physical = text.splitlines()
    lines: List[EnvLine] = []
    i = 0
    while i < len(physical):
        raw = physical[i]
        lineno = i + 1
        stripped = raw.strip()
        if not stripped:
            lines.append(EnvLine(raw, BLANK, lineno=lineno))
            i += 1
            continue
        if stripped.startswith("#"):
            lines.append(EnvLine(raw, COMMENT, lineno=lineno))
            i += 1
            continue
        match = _ASSIGN_RE.match(raw)
        if not match:
            lines.append(EnvLine(raw, OTHER, lineno=lineno))
            i += 1
            continue
        name, rest = match.group(2), match.group(3)
        block = [raw]
        opener = rest.lstrip()[:1]
        if opener in ("'", '"'):
            joined = rest.lstrip()
            while _closing_quote(joined, opener) < 0 and i + 1 < len(physical):
                i += 1
                block.append(physical[i])
                joined += "\n" + physical[i]
            rest = joined
        lines.append(EnvLine("\n".join(block), ASSIGN, name=name, value=parse_value(rest), lineno=lineno))
        i += 1
    return lines


def values(lines: Iterable[EnvLine]) -> Dict[str, str]:
    """Valeurs effectives (la dernière affectation d'un nom l'emporte)."""
    out: Dict[str, str] = {}
    for line in lines:
        if line.kind == ASSIGN:
            out[line.name] = line.value
    return out


def duplicates(lines: Iterable[EnvLine]) -> List[str]:
    """Noms affectés plusieurs fois, dans l'ordre de leur première apparition."""
    seen: Dict[str, int] = {}
    for line in lines:
        if line.kind == ASSIGN:
            seen[line.name] = seen.get(line.name, 0) + 1
    return [name for name, count in seen.items() if count > 1]


def attached_comments(lines: List[EnvLine]) -> Dict[str, List[str]]:
    """Commentaires collés au-dessus d'une affectation (sans ligne vide entre).

    Les en-têtes (``# =====``, ``# ---``) et les variables commentées
    (``# NOM=valeur``) ne sont jamais rattachés. Pour un nom affecté plusieurs
    fois, seuls les commentaires de la dernière affectation sont gardés.

    Returns:
        Table nom → lignes de commentaire brutes, dans l'ordre du fichier.
    """
    attached: Dict[str, List[str]] = {}
    pending: List[str] = []
    for line in lines:
        if line.kind == COMMENT and not line.is_header and not line.commented_name:
            pending.append(line.raw)
            continue
        if line.kind == ASSIGN:
            attached[line.name] = pending
        pending = []
    return attached


def format_assignment(name: str, value: str) -> str:
    """Ligne ``NOM=valeur``, citée seulement si la valeur l'exige.

    Une valeur avec blanc en bord, ``#``, guillemet, antislash ou ``$`` est
    écrite entre guillemets doubles avec échappements.

    Raises:
        ValueError: Nom invalide, ou valeur contenant un saut de ligne ou NUL
            (refus hérité de ``/save_credentials``).
    """
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name or ""):
        raise ValueError(f"nom de variable invalide : {name!r}")
    if any(sep in value for sep in ("\x00", "\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85",
                                    " ", " ")):
        raise ValueError(f"{name} : retour à la ligne ou caractère NUL interdit")
    needs_quotes = value != value.strip() or any(ch in value for ch in "#'\"\\$ ")
    if not needs_quotes:
        return f"{name}={value}"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{name}="{escaped}"'


class EnvWriteError(RuntimeError):
    """Écriture du ``.env`` impossible ou non vérifiée (fichier restauré)."""


def _lock(lock_path: Optional[Path]):
    """Ouvre et verrouille ``lock_path`` (``None`` : pas de verrou)."""
    if lock_path is None or fcntl is None:
        return None
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def backup_file(path: Path, backup_dir: Path) -> Optional[Path]:
    """Copie ``path`` dans ``backup_dir`` (droits 600), ``None`` s'il n'existe pas.

    Le nom est ``env-AAAAMMJJ-HHMMSS`` (suffixe ``-N`` en cas de collision).
    """
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(backup_dir, 0o700)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"env-{stamp}"
    counter = 1
    while target.exists():
        counter += 1
        target = backup_dir / f"env-{stamp}-{counter}"
    shutil.copyfile(path, target)
    os.chmod(target, 0o600)
    return target


def write_in_place(path: Path, text: str, *, backup_dir: Path, lock_path: Optional[Path] = None) -> Optional[Path]:
    """Écrit ``text`` dans ``path`` sur place, après sauvegarde, sous verrou.

    Le fichier garde son inode et ses droits (compatible avec un ``.env`` monté
    seul dans un conteneur). Après écriture, le contenu est relu ; en cas
    d'écart, la sauvegarde est restaurée et ``EnvWriteError`` est levée.

    Args:
        path: Fichier ``.env`` à écrire (créé en ``600`` s'il n'existe pas).
        text: Nouveau contenu complet.
        backup_dir: Dossier des sauvegardes horodatées.
        lock_path: Fichier de verrou inter-processus (``None`` : pas de verrou).

    Returns:
        Le chemin de la sauvegarde, ou ``None`` si le fichier n'existait pas.

    Raises:
        EnvWriteError: Contenu relu différent de ``text`` (fichier restauré).
    """
    handle = _lock(lock_path)
    try:
        backup = backup_file(path, backup_dir)
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        with open(path, "r+", encoding="utf-8", newline="") as fh:
            fh.seek(0)
            fh.write(text)
            fh.truncate()
            fh.flush()
            os.fsync(fh.fileno())
        if path.read_text(encoding="utf-8") != text:
            if backup is not None:
                with open(path, "r+", encoding="utf-8", newline="") as fh:
                    fh.seek(0)
                    fh.write(backup.read_text(encoding="utf-8"))
                    fh.truncate()
                    fh.flush()
                    os.fsync(fh.fileno())
            raise EnvWriteError(f"contenu relu différent après écriture de {path} ; sauvegarde restaurée")
        return backup
    finally:
        if handle is not None:
            handle.close()


@dataclass
class TidyReport:
    """Bilan d'un rangement (noms seulement, jamais de valeur)."""

    placed: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    internal: List[str] = field(default_factory=list)
    obsolete: List[str] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    dropped_comments: int = 0
    kept_comments: int = 0
    other_lines: List[int] = field(default_factory=list)


def tidy(lines: List[EnvLine], layout: Iterable[Tuple[str, str, Iterable[Tuple[str, str, Iterable[str]]]]],
         classify, title_lines: Iterable[str]) -> Tuple[str, TidyReport]:
    """Range les affectations d'un ``.env`` selon ``layout``.

    Chaque affectation garde sa ligne brute (valeur identique à l'octet) et ses
    commentaires collés ; les en-têtes et commentaires isolés sont remplacés
    par les en-têtes du registre. Les noms hors registre partent dans des blocs
    finaux (à trier, internes, obsolètes).

    Args:
        lines: Lignes du ``.env`` (``parse``).
        layout: ``[(clé du bloc, titre, [(clé du sous-bloc, titre, [noms])])]``.
        classify: Fonction nom → ``registered``/``internal``/``obsolete``/``unknown``.
        title_lines: Lignes de l'en-tête du fichier rangé.

    Returns:
        ``(nouveau contenu, bilan)``.
    """
    report = TidyReport()
    report.duplicates = duplicates(lines)
    last: Dict[str, EnvLine] = {}
    for line in lines:
        if line.kind == ASSIGN:
            last[line.name] = line
        elif line.kind == OTHER:
            report.other_lines.append(line.lineno)
    comments = attached_comments(lines)
    report.kept_comments = sum(len(comments.get(name, [])) for name in last)
    report.dropped_comments = sum(1 for line in lines if line.kind == COMMENT) - report.kept_comments

    out: List[str] = list(title_lines)

    def emit(name: str) -> None:
        """Écrit l'affectation ``name`` précédée de ses commentaires collés."""
        out.extend(comments.get(name, []))
        out.append(last[name].raw)

    placed = set()
    for block_key, block_title, subblocks in layout:
        block_lines: List[str] = []
        for sub_key, sub_title, sub_names in subblocks:
            present = [name for name in sub_names if name in last]
            if not present:
                continue
            block_lines.append(f"# --- {sub_key} {sub_title} ---")
            for name in present:
                block_lines.extend(comments.get(name, []))
                block_lines.append(last[name].raw)
                placed.add(name)
        if block_lines:
            out.extend(["", f"# ===== {block_key}. {block_title} =====", *block_lines])
    report.placed = [name for name in last if name in placed]

    groups = (
        ("unknown", "À TRIER : variables inconnues du registre (scripts/rad_settings/registry.py)"),
        ("internal", "INTERNES : à retirer du .env (shell, tests ou système)"),
        ("obsolete", "OBSOLÈTES : jamais lues par le code"),
    )
    for category, title in groups:
        names = [name for name in last if name not in placed and classify(name) in (category,) +
                 (("registered",) if category == "unknown" else ())]
        if not names:
            continue
        getattr(report, category).extend(names)
        out.extend(["", f"# ===== {title} ====="])
        for name in names:
            emit(name)
    return "\n".join(out) + "\n", report
