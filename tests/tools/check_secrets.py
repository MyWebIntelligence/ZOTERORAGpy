#!/usr/bin/env python3
"""Scan staged additions and/or files for secret-looking strings.

Usage::

    check_secrets.py [--staged] [--paths P [P ...]] [--env-file PATH]

Two kinds of hits are reported:

* ``pattern``: API key shapes (``sk-``, ``pcsk_`` or ``Bearer`` followed by
  20 or more key characters); templates made of ``X`` and declared fakes
  starting with ``fake-`` are ignored;
* ``real_value:<NAME>``: a value of the env file whose name contains KEY,
  SECRET, TOKEN or PASSWORD and which is at least 12 characters long. The
  default env file is ``.env`` in the current directory, else at the top of
  the git repository; when none is found a NOTE is printed on stderr. An
  unreadable env file is a usage error.

``--staged`` scans the added lines of ``git diff --cached --text``.
``--paths`` scans files recursively; a missing root is skipped with a SKIP
note and files over 5 MB are skipped only when binary (SKIPPED note).

Only locations are printed (``HIT: <source>:<line> (<kind>)``), never the
matched value nor the line content. The last line is
``PATTERN_HITS=a REAL_VALUE_HITS=b``. Exit 1 on any hit, 2 on usage error.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

MAX_FILE_BYTES = 5 * 1024 * 1024
SNIFF_BYTES = 8192
DEFAULT_ENV_FILE = ".env"
MIN_REAL_VALUE_LEN = 12
SECRET_NAME_WORDS = ("KEY", "SECRET", "TOKEN", "PASSWORD")
SKIPPED_DIRS = {".git"}

# Regex sources are assembled from pieces so that this file never contains
# a literal key shape itself.
_KEY_CHARS = "[A-Za-z0-9_" + "-]"
_BEARER_CHARS = "[A-Za-z0-9._" + "-]"
# Only a lowercase letter right before the prefix rules a match out (words
# such as "task-" or "disk-"); digits, "%3D" or "=" before it still match.
_NO_WORD_BEFORE = "(?<![a-z])"
PATTERNS = [
    re.compile(_NO_WORD_BEFORE + "s" + "k" + "-" + "(" + _KEY_CHARS + "{20,})"),
    re.compile(_NO_WORD_BEFORE + "pc" + "sk" + "_" + "(" + _KEY_CHARS + "{20,})"),
    re.compile(r"\b" + "Bea" + "rer" + " " + "(" + _BEARER_CHARS + "{20,})"),
]
# A template is only X characters, optionally after short lowercase prefix
# segments such as ``proj-`` or ``or-v1-``.
TEMPLATE_RE = re.compile(r"(?:[a-z0-9]{1,8}[-_.]){0,3}X+(?:[-_.]X+)*[-_.]?")
# Declared test placeholders, such as the sprint's fixed fake key.
FAKE_PREFIXES = ("fake-", "fake_")
HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
ANSI_RE = re.compile("\x1b" + r"\[[0-9;]*[A-Za-z]")


class UsageError(Exception):
    """Raised for invalid arguments or git failures."""


def is_template(secret: str) -> bool:
    """Tell whether a matched secret part is a placeholder.

    Placeholders are ``X`` templates (``proj-XXXX``) and declared fakes whose
    secret part starts with ``fake-`` or ``fake_`` (any case).
    """
    return bool(TEMPLATE_RE.fullmatch(secret)) or secret.lower().startswith(FAKE_PREFIXES)


def is_template_value(value: str) -> bool:
    """Tell whether a whole env value is an ``X`` placeholder (key prefix allowed)."""
    for prefix in ("s" + "k-", "pc" + "sk_", "Bea" + "rer "):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return is_template(value)


def resolve_env_file(env_file: Optional[str]) -> Optional[str]:
    """Return the env file to read, or None when there is none.

    An explicit path is used as given. The default ``.env`` is looked up in
    the current directory, then at the top of the enclosing git repository.
    A missing file is announced on stderr since it disables the real-value
    scan.
    """
    if env_file is not None:
        if env_file and os.path.isfile(env_file):
            return env_file
    elif os.path.isfile(DEFAULT_ENV_FILE):
        return DEFAULT_ENV_FILE
    else:
        proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True)
        if proc.returncode == 0:
            top = proc.stdout.decode("utf-8", "surrogateescape").rstrip("\n")
            candidate = os.path.join(top, DEFAULT_ENV_FILE)
            if os.path.isfile(candidate):
                return candidate
    print("NOTE: env file not found, real-value scan disabled", file=sys.stderr)
    return None


def load_real_values(env_file: Optional[str]) -> Dict[str, str]:
    """Return the secret-named values of ``env_file`` long enough to scan for.

    Raises UsageError (without any content) when the file cannot be decoded
    or read.
    """
    if not env_file:
        return {}
    logging.getLogger("dotenv.main").setLevel(logging.ERROR)
    from dotenv import dotenv_values

    try:
        entries = dotenv_values(env_file)
    except (UnicodeDecodeError, OSError, ValueError):
        raise UsageError("cannot read env file") from None
    values: Dict[str, str] = {}
    for name, value in entries.items():
        if not value or len(value) < MIN_REAL_VALUE_LEN or is_template_value(value):
            continue
        if any(word in name.upper() for word in SECRET_NAME_WORDS):
            values[name] = value
    return values


class Scanner:
    """Accumulate hits over lines without ever keeping their content."""

    def __init__(self, real_values: Dict[str, str]):
        """Store the real values to look for and reset the counters."""
        self.real_values = real_values
        self.hits: List[str] = []
        self.pattern_hits = 0
        self.real_hits = 0

    def scan_line(self, source: str, text: str) -> None:
        """Record the hits found in one line attributed to ``source``."""
        for pattern in PATTERNS:
            for match in pattern.finditer(text):
                if is_template(match.group(1)):
                    continue
                self.pattern_hits += 1
                self.hits.append(f"HIT: {source} (pattern)")
        for name, value in self.real_values.items():
            if value in text:
                self.real_hits += 1
                self.hits.append(f"HIT: {source} (real_value:{name})")


def _git_stdout(args: List[str]) -> bytes:
    """Run git and return its stdout, raising UsageError on failure."""
    proc = subprocess.run(["git", *args], capture_output=True)
    if proc.returncode != 0:
        raise UsageError(f"git {args[0]} failed (exit {proc.returncode})")
    return proc.stdout


def _unquote_header_path(raw: str) -> str:
    """Decode a possibly C-quoted path of a ``+++`` diff header."""
    if not raw.startswith('"'):
        return raw
    out = bytearray()
    i = 1
    escapes = {"t": 9, "n": 10, "r": 13, '"': 34, "\\": 92, "a": 7, "b": 8, "f": 12, "v": 11}
    while i < len(raw):
        char = raw[i]
        if char == '"':
            break
        if char == "\\" and i + 1 < len(raw):
            chunk = raw[i + 1:i + 4]
            if re.fullmatch(r"[0-7]{3}", chunk):
                out.append(int(chunk, 8))
                i += 4
                continue
            out.append(escapes.get(raw[i + 1], ord(raw[i + 1])))
            i += 2
            continue
        out.extend(char.encode("utf-8", "surrogateescape"))
        i += 1
    return out.decode("utf-8", "replace")


def _header_path(value: str) -> Optional[str]:
    """Return the file path of a ``+++`` header value, or None for /dev/null."""
    value = value.split("\t", 1)[0]
    path = _unquote_header_path(value)
    if path == "/dev/null":
        return None
    return path[2:] if path.startswith("b/") else path


def iter_added_lines(lines: Iterable[str]) -> Iterator[Tuple[Optional[str], Optional[int], int, str]]:
    """Yield ``(path, new_lineno, stream_lineno, text)`` for added diff lines.

    Hunk headers are followed so that an added line whose content starts
    with ``++`` is still seen; outside hunks, ``+`` lines other than ``+++``
    headers are also treated as additions.
    """
    path: Optional[str] = None
    old_left = new_left = 0
    new_no = 0
    for index, raw in enumerate(lines, 1):
        line = ANSI_RE.sub("", raw.rstrip("\n"))
        if old_left > 0 or new_left > 0:
            tag = line[:1]
            if tag == "+":
                yield path, new_no, index, line[1:]
                new_no += 1
                new_left -= 1
                continue
            if tag == "-":
                old_left -= 1
                continue
            if tag in (" ", ""):
                old_left -= 1
                new_left -= 1
                new_no += 1
                continue
            if tag == "\\":
                continue
            old_left = new_left = 0
        match = HUNK_RE.match(line)
        if match:
            old_left = int(match.group(1)) if match.group(1) is not None else 1
            new_no = int(match.group(2))
            new_left = int(match.group(3)) if match.group(3) is not None else 1
            continue
        if line.startswith("+++ "):
            path = _header_path(line[4:])
            continue
        if line.startswith("diff "):
            path = None
            continue
        if line.startswith("+") and not line.startswith("+++"):
            yield path, None, index, line[1:]


def scan_staged(scanner: Scanner) -> None:
    """Scan the lines added in the index relative to HEAD.

    ``--text`` makes git show the added lines of files it would otherwise
    report as binary (NUL bytes, ``-diff`` attribute).
    """
    data = _git_stdout([
        "-c", "core.quotePath=false", "diff", "--cached", "-U0", "--text", "--no-color",
        "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/",
    ])
    lines = data.decode("utf-8", "replace").split("\n")
    for path, new_no, index, text in iter_added_lines(lines):
        if path is not None and new_no is not None:
            source = f"staged:{path}:{new_no}"
        else:
            source = f"staged:{index}"
        scanner.scan_line(source, text)


def _iter_files(root: str) -> Iterator[str]:
    """Yield regular files under ``root`` (or ``root`` itself if a file)."""
    if os.path.isfile(root):
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIPPED_DIRS)
        for name in sorted(filenames):
            yield os.path.join(dirpath, name)


def _same_file(left: str, right: Optional[str]) -> bool:
    """Tell whether two paths designate the same existing file."""
    if not right:
        return False
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _is_binary_prefix(path: str) -> bool:
    """Tell whether the first bytes of ``path`` contain a NUL byte."""
    with open(path, "rb") as handle:
        return b"\0" in handle.read(SNIFF_BYTES)


def scan_paths(scanner: Scanner, roots: List[str], env_file: Optional[str]) -> None:
    """Scan every readable file under ``roots``, line by line.

    A missing root is reported on stderr and skipped. Files larger than
    5 MB are skipped only when they look binary (NUL in the first 8 KiB);
    every skipped file is named on stderr, never its content.
    """
    for root in roots:
        if not os.path.exists(root):
            print(f"SKIP: {root} (not found)", file=sys.stderr)
            continue
        for path in _iter_files(root):
            if not os.path.isfile(path) or _same_file(path, env_file):
                continue
            try:
                size = os.path.getsize(path)
                if size > MAX_FILE_BYTES and _is_binary_prefix(path):
                    print(f"SKIPPED: {path} ({size} bytes, binary)", file=sys.stderr)
                    continue
                with open(path, "rb") as handle:
                    for number, raw in enumerate(handle, 1):
                        line = raw.decode("utf-8", "replace").rstrip("\n")
                        scanner.scan_line(f"{path}:{number}", line)
            except OSError:
                print(f"SKIPPED: {path} (unreadable)", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Scan for secret-looking strings.")
    parser.add_argument("--staged", action="store_true", help="scan lines added in the index")
    parser.add_argument("--paths", nargs="+", default=[], help="files or directories to scan")
    parser.add_argument("--env-file", default=None,
                        help="env file holding real values (default: .env here or at the repo top)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: scan, print locations and counters, return exit code."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        if not args.staged and not args.paths:
            raise UsageError("use --staged and/or --paths")
        env_file = resolve_env_file(args.env_file)
        scanner = Scanner(load_real_values(env_file))
        if args.staged:
            scan_staged(scanner)
        if args.paths:
            scan_paths(scanner, args.paths, env_file)
    except UsageError as exc:
        print(f"USAGE_ERROR: {exc}", file=sys.stderr)
        return 2
    for hit in scanner.hits:
        print(hit)
    print(f"PATTERN_HITS={scanner.pattern_hits} REAL_VALUE_HITS={scanner.real_hits}")
    return 1 if scanner.pattern_hits or scanner.real_hits else 0


if __name__ == "__main__":
    sys.exit(main())
