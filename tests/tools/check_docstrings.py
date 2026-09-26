#!/usr/bin/env python3
"""Require docstrings on new Python functions and methods.

Usage::

    check_docstrings.py --staged [--strict]
    check_docstrings.py --worktree FILE [FILE ...] [--strict]

The HEAD version of each file (empty when new) is compared with the index
(``--staged``) or the working tree (``--worktree``). A function, async
function or method, ``__init__`` included, needs a docstring when its
qualified name is absent from HEAD and it is defined at module level or
directly in a class body (nested classes included).

Exempt unless ``--strict``: functions defined inside another function,
lambdas, and ``test_*`` functions of test files (basename ``test_*`` or a
path under ``tests/``).

Prints ``MISSING: path:qualname:lineno`` per offender, ``PARSE_ERROR: path``
for unparsable files, ``NOT_FOUND: path`` for a ``--worktree`` file that
does not exist (unless it is a tracked file deleted from the working tree),
then ``MISSING=n``. Exit 1 on any finding, 2 on usage error.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from collections import Counter
from typing import Dict, List, NamedTuple, Optional, Tuple

FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


class UsageError(Exception):
    """Raised for invalid arguments or git failures."""


class FunctionInfo(NamedTuple):
    """One function definition found in a module."""

    qualname: str
    name: str
    lineno: int
    has_docstring: bool
    nested: bool


class ModuleInfo(NamedTuple):
    """Functions and lambdas of one parsed module."""

    functions: List[FunctionInfo]
    lambdas: List[Tuple[str, str, int]]


def _child_bodies(node: ast.AST) -> List[list]:
    """Return the statement lists directly held by a compound statement."""
    bodies = []
    for field in ("body", "orelse", "finalbody"):
        value = getattr(node, field, None)
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            bodies.append(value)
    for handler in getattr(node, "handlers", []) or []:
        bodies.append(handler.body)
    for case in getattr(node, "cases", []) or []:
        bodies.append(case.body)
    return bodies


def analyse_module(tree: ast.Module) -> ModuleInfo:
    """Collect every function (with its qualified name) and lambda of ``tree``."""
    functions: List[FunctionInfo] = []
    qualnames: Dict[int, str] = {}

    def visit(body: list, prefix: str, in_function: bool) -> None:
        """Walk a statement list, recording definitions and their qualnames."""
        for node in body:
            if isinstance(node, FUNCTION_TYPES):
                qualname = prefix + node.name
                qualnames[id(node)] = qualname
                functions.append(FunctionInfo(
                    qualname, node.name, node.lineno,
                    ast.get_docstring(node, clean=False) is not None, in_function,
                ))
                visit(node.body, qualname + ".<locals>.", True)
            elif isinstance(node, ast.ClassDef):
                qualname = prefix + node.name
                qualnames[id(node)] = qualname
                visit(node.body, qualname + ".", in_function)
            else:
                for child in _child_bodies(node):
                    visit(child, prefix, in_function)

    visit(tree.body, "", False)
    lambdas: List[Tuple[str, str, int]] = []
    stack: List[Tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        node, scope = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Lambda):
                lambdas.append((scope, ast.dump(child), child.lineno))
            stack.append((child, qualnames.get(id(child), scope)))
    return ModuleInfo(functions, lambdas)


def is_test_file(path: str) -> bool:
    """Tell whether ``path`` is a test file for the ``test_*`` exemption."""
    norm = path.replace("\\", "/")
    return os.path.basename(norm).startswith("test_") or "/tests/" in "/" + norm


def find_missing(path: str, head_src: bytes, new_src: bytes, strict: bool) -> List[Tuple[str, int]]:
    """Return ``(qualname, lineno)`` of new definitions lacking a docstring.

    Raises SyntaxError when the new source cannot be parsed. An unparsable
    HEAD version counts as empty.
    """
    new_info = analyse_module(ast.parse(new_src))
    try:
        head_info = analyse_module(ast.parse(head_src))
    except (SyntaxError, ValueError):
        head_info = ModuleInfo([], [])
    head_names = {f.qualname for f in head_info.functions}
    test_file = is_test_file(path)
    missing: List[Tuple[str, int]] = []
    for func in new_info.functions:
        if func.qualname in head_names or func.has_docstring:
            continue
        if not strict and (func.nested or (test_file and func.name.startswith("test_"))):
            continue
        missing.append((func.qualname, func.lineno))
    if strict:
        remaining = Counter((scope, dump) for scope, dump, _line in head_info.lambdas)
        for scope, dump, lineno in new_info.lambdas:
            if remaining[(scope, dump)] > 0:
                remaining[(scope, dump)] -= 1
                continue
            missing.append((f"{scope}.<lambda>" if scope else "<lambda>", lineno))
    return sorted(missing, key=lambda item: item[1])


def _git(args: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run git and return the completed process (bytes output)."""
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True)


def repo_root() -> Optional[str]:
    """Return the repository top level, or None outside a repository."""
    proc = _git(["rev-parse", "--show-toplevel"])
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "surrogateescape").rstrip("\n")


def git_blob(root: Optional[str], spec: str) -> bytes:
    """Return ``git show spec`` output, or empty bytes when it does not exist."""
    if root is None:
        return b""
    proc = _git(["show", spec], cwd=root)
    return proc.stdout if proc.returncode == 0 else b""


def in_head(root: Optional[str], rel: str) -> bool:
    """Tell whether the repository-relative path ``rel`` exists in HEAD."""
    if root is None:
        return False
    return _git(["cat-file", "-e", f"HEAD:{rel}"], cwd=root).returncode == 0


def staged_targets(root: str) -> List[Tuple[str, str]]:
    """Return ``(head_path, index_path)`` for staged added or modified ``.py`` files."""
    proc = _git(["diff", "--cached", "--name-status", "-z", "-M", "--diff-filter=ACMRT"], cwd=root)
    if proc.returncode != 0:
        raise UsageError("git diff --cached failed")
    fields = proc.stdout.split(b"\0")
    targets: List[Tuple[str, str]] = []
    i = 0
    while i < len(fields):
        status = fields[i].decode("ascii", "replace")
        i += 1
        if not status:
            continue
        if status[:1] in ("R", "C"):
            old, new = (f.decode("utf-8", "surrogateescape") for f in fields[i:i + 2])
            i += 2
            head_path = old if status[:1] == "R" else new
        else:
            new = fields[i].decode("utf-8", "surrogateescape")
            i += 1
            head_path = new
        if new.endswith(".py"):
            targets.append((head_path, new))
    return targets


def _repo_relative(root: Optional[str], path: str) -> Optional[str]:
    """Return ``path`` relative to the repository root, or None if outside."""
    if root is None:
        return None
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return None
    return rel.replace(os.sep, "/")


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Require docstrings on new functions.")
    parser.add_argument("--staged", action="store_true", help="compare HEAD with the index")
    parser.add_argument("--worktree", nargs="+", metavar="FILE", default=[],
                        help="compare HEAD with these working-tree files")
    parser.add_argument("--strict", action="store_true", help="remove every exemption")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: check the requested files and print the findings."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    if not args.staged and not args.worktree:
        print("USAGE_ERROR: use --staged and/or --worktree FILE ...", file=sys.stderr)
        return 2
    root = repo_root()
    jobs: List[Tuple[str, bytes, bytes]] = []
    try:
        if args.staged:
            if root is None:
                raise UsageError("--staged needs a git repository")
            for head_path, new_path in staged_targets(root):
                jobs.append((new_path, git_blob(root, f"HEAD:{head_path}"), git_blob(root, f":{new_path}")))
    except UsageError as exc:
        print(f"USAGE_ERROR: {exc}", file=sys.stderr)
        return 2
    not_found = 0
    for path in args.worktree:
        rel = _repo_relative(root, path)
        if not os.path.isfile(path):
            if rel and in_head(root, rel):
                # Tracked file deleted in the working tree: nothing new to check.
                print(f"SKIP: {path} (deleted)", file=sys.stderr)
            else:
                print(f"NOT_FOUND: {path}")
                not_found += 1
            continue
        if not path.endswith(".py"):
            continue
        with open(path, "rb") as handle:
            new_src = handle.read()
        head_src = git_blob(root, f"HEAD:{rel}") if rel else b""
        jobs.append((rel or path, head_src, new_src))
    missing_count = 0
    parse_errors = 0
    for path, head_src, new_src in jobs:
        try:
            missing = find_missing(path, head_src, new_src, args.strict)
        except (SyntaxError, ValueError):
            print(f"PARSE_ERROR: {path}")
            parse_errors += 1
            continue
        for qualname, lineno in missing:
            print(f"MISSING: {path}:{qualname}:{lineno}")
        missing_count += len(missing)
    print(f"MISSING={missing_count}")
    return 1 if missing_count or parse_errors or not_found else 0


if __name__ == "__main__":
    sys.exit(main())
