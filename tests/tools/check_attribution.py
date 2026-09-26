#!/usr/bin/env python3
"""Detect attribution trailers and tool mentions in text or added diff lines.

Usage::

    check_attribution.py --diff        # unified diff on stdin, added lines only
    check_attribution.py --stdin       # every stdin line
    check_attribution.py --file PATH   # every line of PATH

The searched words are assembled at run time, so this file never contains
them. Each line is first normalised (NFKC, invisible format characters
dropped, dash variants folded to ``-``, whitespace runs collapsed), then
existing project documentation names (the project guide file and its hidden
directory, both case-sensitive) are removed before the case-insensitive
match.

Prints ``HIT: <source>:<line>`` per offending line (never its content), then
``ATTRIBUTION_HITS=n``. Exit 1 on any hit, 2 on usage error.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from typing import Iterable, Iterator, List, Optional, Tuple

_WORDS = [
    "clau" + "de",
    "anthro" + "pic",
    "co-" + "auth" + "ored",
    "generated " + "with",
    "noreply" + "@",
    "sub" + "ag" + "ent",
    "sous-" + "ag" + "ent",
]
PATTERN = re.compile("|".join(re.escape(word) for word in _WORDS), re.IGNORECASE)
# Project documentation names that legitimately appear in paths: the guide
# file (exact, case-sensitive) and the hidden directory when it is followed
# by a slash, whitespace, a quote, a bracket, light punctuation or the end.
_DOC_FILE_TOKEN = "CLAU" + "DE.md"
_DOC_DIR_RE = re.compile(r"\." + "clau" + "de" + r"""(?=/|[\s`'"()\[\]{}<>,;:!?]|$)""")
# Dash-like characters folded to "-" before matching: U+2010..U+2015, U+2212.
_DASHES = {code: "-" for code in list(range(0x2010, 0x2016)) + [0x2212]}
_SPACES_RE = re.compile(r"\s+")
HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
ANSI_RE = re.compile("\x1b" + r"\[[0-9;]*[A-Za-z]")


def normalize(text: str) -> str:
    """Fold a line so that spacing, dash and width variants still match.

    Applies NFKC, drops invisible format characters (zero-width spaces,
    soft hyphens), maps dash-like characters to ``-`` and collapses every
    whitespace run to one space.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    return _SPACES_RE.sub(" ", text.translate(_DASHES))


def is_hit(text: str) -> bool:
    """Tell whether ``text`` contains a searched word once doc names are removed."""
    cleaned = normalize(text).replace(_DOC_FILE_TOKEN, "")
    cleaned = _DOC_DIR_RE.sub("", cleaned)
    return bool(PATTERN.search(cleaned))


def _header_path(value: str) -> Optional[str]:
    """Return the path named by a ``+++`` header value (None for /dev/null)."""
    value = value.split("\t", 1)[0]
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if value == "/dev/null":
        return None
    return value[2:] if value.startswith("b/") else value


def iter_added_lines(lines: Iterable[str]) -> Iterator[Tuple[Optional[str], Optional[int], int, str]]:
    """Yield ``(path, new_lineno, stream_lineno, text)`` for added diff lines.

    Hunk headers are followed, so removed lines, context and ``+++`` file
    headers are skipped while an added line starting with ``++`` is kept.
    Outside a hunk, ``+`` lines other than ``+++`` count as added lines.
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


def _read_lines(data: bytes) -> List[str]:
    """Decode raw bytes into lines (UTF-8, undecodable bytes replaced)."""
    text = data.decode("utf-8", "replace")
    if text.endswith("\n"):
        text = text[:-1]
    return text.split("\n") if text else []


def check_diff(lines: List[str]) -> List[str]:
    """Return the hit locations among the added lines of a diff."""
    hits = []
    for path, new_no, index, text in iter_added_lines(lines):
        if is_hit(text):
            source = f"diff:{path}:{new_no}" if path and new_no is not None else f"diff:{index}"
            hits.append(source)
    return hits


def check_all(label: str, lines: List[str]) -> List[str]:
    """Return the hit locations among every line."""
    return [f"{label}:{number}" for number, line in enumerate(lines, 1) if is_hit(line)]


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Detect attribution lines.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--diff", action="store_true", help="unified diff on stdin (added lines)")
    group.add_argument("--stdin", action="store_true", help="check every stdin line")
    group.add_argument("--file", metavar="PATH", help="check every line of PATH")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: read the input, print hit locations and the counter."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    if args.file:
        try:
            with open(args.file, "rb") as handle:
                lines = _read_lines(handle.read())
        except OSError:
            print(f"USAGE_ERROR: cannot read {args.file}", file=sys.stderr)
            return 2
        hits = check_all(args.file, lines)
    else:
        lines = _read_lines(sys.stdin.buffer.read())
        hits = check_diff(lines) if args.diff else check_all("stdin", lines)
    for hit in hits:
        print(f"HIT: {hit}")
    print(f"ATTRIBUTION_HITS={len(hits)}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
