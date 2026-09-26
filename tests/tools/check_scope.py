#!/usr/bin/env python3
"""Check that the working tree or the index only touches owned paths.

Usage::

    check_scope.py --owned FILE [--preexisting FILE] [--emit-add-list OUT]
                   [--allow-owned-preexisting FILE]
                   [--verify-snapshot SNAP [--watch FILE]]
    check_scope.py --owned FILE --staged [--restrict FILE]
    check_scope.py --snapshot SNAP [--watch FILE] [--preexisting FILE]

Worktree mode reads ``git status --porcelain --untracked-files=all -z``:
every changed or untracked path must be owned, preexisting or ignored
(``.DS_Store``, ``Icon\\r``, ``._*``). A path both owned and preexisting
holds the user's uncommitted work: it is reported as ``OWNED_PREEXISTING``
and left out of the add list unless ``--allow-owned-preexisting`` lists it.
Staged mode reads ``git diff --cached --name-status -z``: every staged path
must be owned. Drive conflict copies (a path component such as
``name (1).ext``, ``dir (1)`` or containing CONFLIT / conflict) are always
reported as ``CONFLICT: path``.

Restrict lines (``--restrict``) are ``path=sym1,sym2`` or
``path=@docstrings``. Symbols are top-level names bound by a definition, an
assignment or an import (each import alias on its own); statements binding
no name (such as ``app.include_router(...)``) are grouped under the symbol
``<toplevel>``, which a restrict line may list. In ``@docstrings`` mode a
body reduced to a lone ``pass`` or ``...`` equals an empty body.

Git status cannot see ignored files (``.env``, ``data/``). ``--snapshot``
writes ``sha256<TAB>path`` (never content) for the ``--watch`` entries
(literal paths, globs, ``dir/`` prefixes) and the ``--preexisting`` paths;
keep the snapshot out of the lot's reach. ``--verify-snapshot`` then
reports ``WATCHED_CHANGED: path`` (hash differs, or a new ``--watch``
match) and ``PREEXISTING_CHANGED: path`` (hash or git status differs).

The last line is ``SCOPE_OK`` (exit 0) or ``SCOPE_VIOLATION n`` (exit 1),
or ``SNAPSHOT_OK n`` in snapshot mode; usage errors exit 2.
"""

from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import os
import re
import subprocess
import sys
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

IGNORED_RE = re.compile(r"(^|/)\.DS_Store$|(^|/)Icon\r$|(^|/)\._")
CONFLICT_COPY_RE = re.compile(r" \(\d+\)(\.|$)")
CONFLICT_WORDS = ("conflit", "conflict")
DOCSTRINGS_ONLY = "@docstrings"
OTHER_SYMBOL = "<toplevel>"
MISSING_DIGEST = "-"


class UsageError(Exception):
    """Raised for invalid arguments, unreadable files or git failures."""


def _nfc(path: str) -> str:
    """Return ``path`` in Unicode NFC form for comparisons."""
    return unicodedata.normalize("NFC", path)


def _decode(raw: bytes) -> str:
    """Decode a git path, keeping undecodable bytes reversible."""
    return raw.decode("utf-8", "surrogateescape")


def is_ignored(path: str) -> bool:
    """Tell whether ``path`` is OS noise that is never in scope nor reported."""
    return bool(IGNORED_RE.search(path))


def is_conflict_copy(path: str) -> bool:
    """Tell whether ``path`` is, or lies under, a Google Drive conflict copy.

    Every path component is tested, so a file inside a conflict copy of a
    directory (``dir (1)/a.json``) is caught as well.
    """
    for part in path.rstrip("/").split("/"):
        if CONFLICT_COPY_RE.search(part):
            return True
        lowered = part.lower()
        if any(word in lowered for word in CONFLICT_WORDS):
            return True
    return False


def git(args: Sequence[str], cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run git with ``args`` and return the completed process (bytes output)."""
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    if check and proc.returncode != 0:
        raise UsageError(f"git {args[0]} failed (exit {proc.returncode})")
    return proc


def repo_root() -> str:
    """Return the top-level directory of the current git repository."""
    proc = git(["rev-parse", "--show-toplevel"])
    return _decode(proc.stdout).rstrip("\n")


class Owned:
    """Set of owned exact paths and directory prefixes."""

    def __init__(self, lines: Iterable[str]):
        """Build the matcher from the lines of an owned file."""
        self.exact: Set[str] = set()
        self.prefixes: List[str] = []
        for line in lines:
            entry = _nfc(line.rstrip("\r\n"))
            if not entry.strip():
                continue
            while entry.startswith("./"):
                entry = entry[2:]
            if entry.endswith("/"):
                self.prefixes.append(entry)
            else:
                self.exact.add(entry)

    def __contains__(self, path: str) -> bool:
        """Tell whether ``path`` is owned exactly or under an owned directory."""
        norm = _nfc(path)
        return norm in self.exact or any(norm.startswith(prefix) for prefix in self.prefixes)


def read_owned(path: str) -> Owned:
    """Load an owned file."""
    try:
        with open(path, encoding="utf-8") as handle:
            return Owned(handle.read().split("\n"))
    except OSError as exc:
        raise UsageError(f"cannot read owned file {path}") from exc


_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}


def _read_quoted(text: str, start: int) -> Tuple[str, int]:
    """Decode a C-quoted git path starting at ``text[start] == '"'``.

    Returns the decoded path and the index just after the closing quote.
    """
    out = bytearray()
    i = start + 1
    while i < len(text):
        char = text[i]
        if char == '"':
            return _decode(bytes(out)), i + 1
        if char == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in "01234567" and re.match(r"[0-7]{3}", text[i + 1:i + 4]):
                out.append(int(text[i + 1:i + 4], 8))
                i += 4
                continue
            out.append(_ESCAPES.get(nxt, ord(nxt)))
            i += 2
            continue
        out.extend(char.encode("utf-8", "surrogateescape"))
        i += 1
    raise ValueError("unterminated quoted path")


def _split_status_paths(rest: str, rename: bool) -> List[str]:
    """Split the path part of a plain porcelain line into paths.

    For renames and copies (``old -> new``) the new path comes first.
    """
    if not rename:
        return [_read_quoted(rest, 0)[0] if rest.startswith('"') else rest]
    if rest.startswith('"'):
        old, end = _read_quoted(rest, 0)
        remainder = rest[end:]
        if not remainder.startswith(" -> "):
            raise ValueError("malformed rename entry")
        new_part = remainder[4:]
    else:
        old, sep, new_part = rest.partition(" -> ")
        if not sep:
            return [rest]
    new = _read_quoted(new_part, 0)[0] if new_part.startswith('"') else new_part
    return [new, old]


def parse_porcelain_text(data: bytes) -> List[Tuple[str, str]]:
    """Return ``(status, path)`` for every path of a saved porcelain output.

    Accepts the plain format (quoted paths, ``old -> new`` renames) and the
    NUL-separated ``-z`` format.
    """
    if b"\0" in data:
        return [(status, path) for status, paths in parse_porcelain_z(data) for path in paths]
    pairs: List[Tuple[str, str]] = []
    text = _decode(data)
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if len(line) < 4:
            continue
        status, rest = line[:2], line[3:]
        rename = "R" in status or "C" in status
        try:
            pairs.extend((status, path) for path in _split_status_paths(rest, rename))
        except ValueError:
            pairs.append((status, rest))
    return pairs


def parse_porcelain_z(data: bytes) -> List[Tuple[str, List[str]]]:
    """Parse ``git status --porcelain -z`` output into (status, paths) pairs.

    For renames and copies the list holds the new path first, then the
    original path.
    """
    fields = data.split(b"\0")
    entries: List[Tuple[str, List[str]]] = []
    i = 0
    while i < len(fields):
        record = fields[i]
        i += 1
        if len(record) < 4:
            continue
        status = record[:2].decode("ascii", "replace")
        paths = [_decode(record[3:])]
        if ("R" in status or "C" in status) and i < len(fields):
            paths.append(_decode(fields[i]))
            i += 1
        entries.append((status, paths))
    return entries


def parse_name_status_z(data: bytes) -> List[Tuple[str, List[str]]]:
    """Parse ``git diff --name-status -z`` output into (status, paths) pairs."""
    fields = data.split(b"\0")
    entries: List[Tuple[str, List[str]]] = []
    i = 0
    while i < len(fields):
        status = fields[i].decode("ascii", "replace")
        i += 1
        if not status:
            continue
        count = 2 if status[:1] in ("R", "C") else 1
        paths = [_decode(field) for field in fields[i:i + count]]
        i += count
        entries.append((status, paths))
    return entries


class Preexisting:
    """Paths of a saved porcelain status, with their saved status codes."""

    def __init__(self, pairs: Iterable[Tuple[str, str]] = ()):
        """Index ``(status, path)`` pairs by NFC path, keeping raw paths."""
        self.statuses: Dict[str, str] = {}
        self.raw: Dict[str, str] = {}
        for status, path in pairs:
            key = _nfc(path)
            self.statuses.setdefault(key, status)
            self.raw.setdefault(key, path)

    def __contains__(self, path: str) -> bool:
        """Tell whether ``path`` was listed in the saved status."""
        return _nfc(path) in self.statuses


def read_preexisting(path: Optional[str]) -> Preexisting:
    """Load the paths (and status codes) listed in a saved porcelain file."""
    if not path:
        return Preexisting()
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise UsageError(f"cannot read preexisting file {path}") from exc
    return Preexisting(parse_porcelain_text(data))


def read_path_list(path: Optional[str], what: str) -> List[str]:
    """Read one path (or pattern) per line, skipping blank lines."""
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().split("\n")
    except OSError as exc:
        raise UsageError(f"cannot read {what} file {path}") from exc
    entries = []
    for line in lines:
        entry = line.rstrip("\r")
        if not entry.strip():
            continue
        while entry.startswith("./"):
            entry = entry[2:]
        entries.append(entry)
    return entries


# --- snapshots of watched and preexisting paths (hashes only, never content) ------


def file_digest(path: str) -> str:
    """Return the sha256 hex digest of a file, ``dir`` for a directory, ``-`` if absent."""
    if os.path.isdir(path):
        return "dir"
    if not os.path.isfile(path):
        return MISSING_DIGEST
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            block = handle.read(1 << 16)
            while block:
                digest.update(block)
                block = handle.read(1 << 16)
    except OSError as exc:
        raise UsageError(f"cannot read watched path {path}") from exc
    return digest.hexdigest()


def _real(path: Optional[str]) -> Optional[str]:
    """Return the resolved absolute form of ``path`` (None stays None)."""
    return os.path.realpath(path) if path else None


def expand_watch(root: str, entries: Sequence[str], exclude: Set[str]) -> Dict[str, str]:
    """Expand watch entries into ``{nfc path: repo-relative path}``.

    An entry is a literal path (kept even when absent, so its creation is
    seen), a glob (``*``, ``?``, ``[``; ``**`` recursive) or a directory
    prefix ending with ``/`` (every file under it). Paths whose resolved
    location is in ``exclude`` (the snapshot and add-list files) are left out.
    """
    found: Dict[str, str] = {}

    def keep(rel: str) -> None:
        """Record ``rel`` unless it designates an excluded file."""
        if os.path.realpath(os.path.join(root, rel)) not in exclude:
            found.setdefault(_nfc(rel), rel)

    for entry in entries:
        if entry.endswith("/"):
            base = os.path.join(root, entry)
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames.sort()
                for name in sorted(filenames):
                    full = os.path.join(dirpath, name)
                    keep(os.path.relpath(full, root).replace(os.sep, "/"))
        elif any(char in entry for char in "*?["):
            for rel in sorted(glob.glob(entry, root_dir=root, recursive=True)):
                if os.path.isfile(os.path.join(root, rel)):
                    keep(rel.replace(os.sep, "/"))
        else:
            keep(entry)
    return found


def write_snapshot(root: str, out: str, watch: Sequence[str], preexisting: Preexisting) -> int:
    """Write ``sha256<TAB>path`` for every watched and preexisting path.

    Returns the number of entries written. OS noise paths are skipped.
    """
    exclude = {_real(out)}
    paths = expand_watch(root, watch, exclude)
    for key, raw in preexisting.raw.items():
        if not is_ignored(raw):
            paths.setdefault(key, raw)
    lines = []
    for key in sorted(paths):
        rel = paths[key]
        if "\t" in rel or "\n" in rel:
            raise UsageError("cannot snapshot a path holding a tab or a newline")
        lines.append(f"{file_digest(os.path.join(root, rel))}\t{rel}\n")
    try:
        with open(out, "w", encoding="utf-8", errors="surrogateescape") as handle:
            handle.writelines(lines)
    except OSError as exc:
        raise UsageError(f"cannot write snapshot {out}") from exc
    return len(lines)


def read_snapshot(path: str) -> Dict[str, Tuple[str, str]]:
    """Load a snapshot into ``{nfc path: (digest, repo-relative path)}``."""
    try:
        with open(path, encoding="utf-8", errors="surrogateescape") as handle:
            lines = handle.read().split("\n")
    except OSError as exc:
        raise UsageError(f"cannot read snapshot {path}") from exc
    entries: Dict[str, Tuple[str, str]] = {}
    for line in lines:
        if not line:
            continue
        digest, sep, rel = line.partition("\t")
        if not sep or not rel:
            raise UsageError(f"malformed snapshot {path}")
        entries[_nfc(rel)] = (digest, rel)
    return entries


def snapshot_violations(root: str, snapshot: Dict[str, Tuple[str, str]], watch: Sequence[str],
                        exclude: Set[str], preexisting: Preexisting, allowed: Owned,
                        current_status: Dict[str, str]) -> List[str]:
    """Compare the snapshot with the disk and return the violation lines.

    A preexisting path (not explicitly allowed) is ``PREEXISTING_CHANGED``
    when its content hash or its git status differs from the saved ones;
    any other snapshot path is ``WATCHED_CHANGED`` when its hash differs.
    With watch entries, a newly matching path is ``WATCHED_CHANGED`` too.
    """
    violations: List[str] = []
    for key in sorted(snapshot):
        digest, rel = snapshot[key]
        if key in preexisting.statuses and rel in allowed:
            continue
        now = file_digest(os.path.join(root, rel))
        if key in preexisting.statuses:
            if now != digest or current_status.get(key) != preexisting.statuses[key]:
                violations.append(f"PREEXISTING_CHANGED: {rel}")
        elif now != digest:
            violations.append(f"WATCHED_CHANGED: {rel}")
    for key, rel in sorted(expand_watch(root, watch, exclude).items()):
        if key not in snapshot:
            violations.append(f"WATCHED_CHANGED: {rel}")
    return violations


# --- restrict (AST comparison HEAD vs index) -------------------------------------


def _docstring_node(body: list) -> bool:
    """Tell whether the first statement of ``body`` is a docstring."""
    return bool(
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )


def _is_placeholder(stmt: ast.stmt) -> bool:
    """Tell whether ``stmt`` is a placeholder body (``pass`` or ``...``)."""
    if isinstance(stmt, ast.Pass):
        return True
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    )


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove module, class and function docstrings from ``tree`` in place.

    A body left empty or holding a lone ``pass`` / ``...`` is normalised to
    an empty body, so replacing a placeholder with a docstring is allowed.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if _docstring_node(node.body):
                node.body = node.body[1:]
            if len(node.body) == 1 and _is_placeholder(node.body[0]):
                node.body = []
    return tree


def _target_names(target: ast.AST) -> List[str]:
    """Return the names bound by an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for elt in target.elts:
            names.extend(_target_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    node = target
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return [node.id] if isinstance(node, ast.Name) else []


def _node_symbols(node: ast.stmt) -> List[str]:
    """Return the top-level symbol names a statement defines."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        names: List[str] = []
        for target in node.targets:
            names.extend(_target_names(target))
        return names
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _target_names(node.target)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [_alias_name(alias) for alias in node.names]
    return []


def _alias_name(alias: ast.alias) -> str:
    """Return the name an import alias binds in the module namespace."""
    return alias.asname or alias.name.split(".")[0]


def symbol_map(tree: ast.Module) -> Dict[str, List[str]]:
    """Map each top-level symbol to the dumps of the statements binding it.

    The module docstring is skipped; statements binding no name are grouped
    under ``<toplevel>``. Each alias of an import gets its own single-alias
    dump, so extending ``from m import a, b`` with ``c`` only changes ``c``.
    """
    body = tree.body[1:] if _docstring_node(tree.body) else tree.body
    result: Dict[str, List[str]] = {}
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                dump = ast.dump(ast.Import(names=[alias]))
                result.setdefault(_alias_name(alias), []).append(dump)
            continue
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                dump = ast.dump(ast.ImportFrom(module=node.module, names=[alias], level=node.level))
                result.setdefault(_alias_name(alias), []).append(dump)
            continue
        dump = ast.dump(node)
        for name in _node_symbols(node) or [OTHER_SYMBOL]:
            result.setdefault(name, []).append(dump)
    return result


def _git_blob(root: str, spec: str) -> bytes:
    """Return the content of ``git show spec`` or empty bytes if absent."""
    proc = git(["show", spec], cwd=root, check=False)
    return proc.stdout if proc.returncode == 0 else b""


def read_restrict(path: str) -> Dict[str, Set[str]]:
    """Load ``path=sym1,sym2`` / ``path=@docstrings`` restriction lines."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        raise UsageError(f"cannot read restrict file {path}") from exc
    rules: Dict[str, Set[str]] = {}
    for line in lines:
        if not line.strip():
            continue
        target, sep, symbols = line.rpartition("=")
        if not sep or not target:
            raise UsageError(f"malformed restrict line: {line!r}")
        while target.startswith("./"):
            target = target[2:]
        names = {s.strip() for s in symbols.split(",") if s.strip()}
        rules.setdefault(target, set()).update(names)
    return rules


def restrict_violations(root: str, path: str, allowed: Set[str]) -> List[str]:
    """Return the symbols of ``path`` that differ between HEAD and the index
    without being allowed by the restriction."""
    head_src = _git_blob(root, f"HEAD:{path}")
    index_src = _git_blob(root, f":{path}")
    try:
        head_tree = ast.parse(head_src)
        index_tree = ast.parse(index_src)
    except (SyntaxError, ValueError):
        return ["<syntax-error>"]
    if DOCSTRINGS_ONLY in allowed:
        same = ast.dump(strip_docstrings(head_tree)) == ast.dump(strip_docstrings(index_tree))
        return [] if same else [DOCSTRINGS_ONLY]
    head_map = symbol_map(head_tree)
    index_map = symbol_map(index_tree)
    changed = [
        name for name in sorted(set(head_map) | set(index_map))
        if head_map.get(name) != index_map.get(name)
    ]
    return [name for name in changed if name not in allowed]


# --- modes -----------------------------------------------------------------------


def check_worktree(root: str, owned: Owned, preexisting: Preexisting,
                   emit_path: Optional[str], allowed: Optional[Owned] = None,
                   snapshot_path: Optional[str] = None,
                   watch: Sequence[str] = ()) -> List[str]:
    """Check the working tree and return the violation lines.

    An owned path that is also preexisting holds uncommitted work of the
    user: it is reported as ``OWNED_PREEXISTING`` and kept out of the add
    list unless ``allowed`` lists it.
    """
    allowed = allowed or Owned([])
    data = git(["status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=root).stdout
    entries = parse_porcelain_z(data)
    violations: List[str] = []
    to_add: List[str] = []
    seen: Set[str] = set()
    for status, paths in entries:
        for position, path in enumerate(paths):
            if path in seen:
                continue
            seen.add(path)
            if is_ignored(path):
                continue
            pre = path in preexisting
            if pre and path not in owned:
                continue
            if is_conflict_copy(path):
                if not pre:
                    violations.append(f"CONFLICT: {path}")
                continue
            if path in owned:
                if pre and path not in allowed:
                    violations.append(f"OWNED_PREEXISTING: {path}")
                    continue
                full = os.path.join(root, path)
                rename_source = position > 0
                if not rename_source and (os.path.lexists(full) or status[1:2] == "D"):
                    to_add.append(path)
                continue
            violations.append(f"OUT_OF_SCOPE: {path}")
    if snapshot_path:
        current_status = {
            _nfc(path): status for status, paths in entries for path in paths
        }
        exclude = {_real(snapshot_path), _real(emit_path)}
        violations.extend(snapshot_violations(
            root, read_snapshot(snapshot_path), watch, exclude, preexisting, allowed,
            current_status,
        ))
    if emit_path:
        try:
            with open(emit_path, "w", encoding="utf-8", errors="surrogateescape") as handle:
                handle.writelines(f"{path}\n" for path in sorted(to_add))
        except OSError as exc:
            raise UsageError(f"cannot write add list {emit_path}") from exc
    return violations


def check_staged(root: str, owned: Owned, restrict: Dict[str, Set[str]]) -> List[str]:
    """Check the index and return the violation lines."""
    data = git(["diff", "--cached", "--name-status", "-z", "--no-renames"], cwd=root).stdout
    violations: List[str] = []
    for _status, paths in parse_name_status_z(data):
        for path in paths:
            if is_conflict_copy(path):
                violations.append(f"CONFLICT: {path}")
            elif path not in owned:
                violations.append(f"OUT_OF_SCOPE: {path}")
    for path in sorted(restrict):
        for symbol in restrict_violations(root, path, restrict[path]):
            violations.append(f"RESTRICT_VIOLATION: {path}: {symbol}")
    return violations


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Check that changes stay in scope.")
    parser.add_argument("--owned", help="owned paths, one per line (required to check)")
    parser.add_argument("--preexisting", help="saved porcelain status whose paths are ignored")
    parser.add_argument("--emit-add-list", metavar="OUT", help="write changed owned paths to OUT")
    parser.add_argument("--staged", action="store_true", help="check the index instead")
    parser.add_argument("--restrict", help="per-file symbol restrictions (with --staged)")
    parser.add_argument("--allow-owned-preexisting", metavar="FILE",
                        help="owned paths allowed to be preexisting too (committed with the lot)")
    parser.add_argument("--watch", metavar="FILE",
                        help="sensitive paths, globs or dir/ prefixes, one per line")
    parser.add_argument("--snapshot", metavar="OUT",
                        help="write sha256<TAB>path for watched and preexisting paths, then stop")
    parser.add_argument("--verify-snapshot", metavar="FILE",
                        help="report watched or preexisting paths changed since the snapshot")
    return parser


def _validate(args: argparse.Namespace) -> None:
    """Reject option combinations that make no sense."""
    if args.snapshot:
        extra = [args.owned, args.staged, args.restrict, args.emit_add_list,
                 args.verify_snapshot, args.allow_owned_preexisting]
        if any(extra):
            raise UsageError("--snapshot only combines with --watch and --preexisting")
        if not args.watch and not args.preexisting:
            raise UsageError("--snapshot needs --watch and/or --preexisting")
        return
    if not args.owned:
        raise UsageError("--owned is required (or use --snapshot)")
    if args.restrict and not args.staged:
        raise UsageError("--restrict requires --staged")
    if args.staged and (args.emit_add_list or args.verify_snapshot or args.watch
                        or args.allow_owned_preexisting):
        raise UsageError("--emit-add-list, --watch, --verify-snapshot and "
                         "--allow-owned-preexisting are worktree mode options")
    if args.watch and not args.verify_snapshot:
        raise UsageError("--watch requires --snapshot or --verify-snapshot")


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: run the requested check and print the verdict."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        _validate(args)
        root = repo_root()
        preexisting = read_preexisting(args.preexisting)
        watch = read_path_list(args.watch, "watch")
        if args.snapshot:
            count = write_snapshot(root, args.snapshot, watch, preexisting)
            print(f"SNAPSHOT_OK {count}")
            return 0
        owned = read_owned(args.owned)
        if args.staged:
            restrict = read_restrict(args.restrict) if args.restrict else {}
            violations = check_staged(root, owned, restrict)
        else:
            allowed = Owned(read_path_list(args.allow_owned_preexisting, "allow"))
            violations = check_worktree(root, owned, preexisting, args.emit_add_list,
                                        allowed, args.verify_snapshot, watch)
    except UsageError as exc:
        print(f"USAGE_ERROR: {exc}", file=sys.stderr)
        return 2
    for line in violations:
        print(line)
    if violations:
        print(f"SCOPE_VIOLATION {len(violations)}")
        return 1
    print("SCOPE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
