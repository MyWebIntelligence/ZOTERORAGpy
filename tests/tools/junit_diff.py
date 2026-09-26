#!/usr/bin/env python3
"""Compare two JUnit XML reports written by ``pytest --junitxml``.

Usage::

    junit_diff.py --baseline A.xml --current B.xml [--max-failures N]
                  [--allowed-removals FILE] [--net-legacy FILE]
    junit_diff.py --summary FILE

Diff mode prints the offending test ids (``NEW: id`` / ``MISSING: id``)
followed by one summary line::

    NEW_FAILURES=a FAILURES=b ERRORS=c MISSING=d PASSED_DELTA=e NET_LEGACY=f

FAILURES and ERRORS count the ``<testcase>`` elements of the current report
holding a ``<failure>`` / ``<error>`` (as pytest's summary does). NEW lists
failing tests that passed in, or were absent from, the baseline. MISSING
lists baseline-passed tests now absent or skipped, minus allowed removals.

Exit codes: 0 = OK, 1 = regression, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, NamedTuple, Optional

DEFAULT_ALLOWED_REMOVALS = os.path.join("data", "albert_gates", "allowed_removals.txt")
DEFAULT_NET_LEGACY = os.path.join("data", "albert_gates", "net_legacy.json")

PASSED = "passed"
SKIPPED = "skipped"
FAILURE = "failure"
ERROR = "error"
FAILING = (FAILURE, ERROR)

# Higher rank wins when one test id appears several times in a report.
_RANK = {PASSED: 0, SKIPPED: 1, FAILURE: 2, ERROR: 3}


class UsageError(Exception):
    """Raised for invalid arguments or unreadable input files."""


def _case_status(case: ET.Element) -> str:
    """Return the status of one ``<testcase>`` element."""
    status = PASSED
    for child in case:
        tag = child.tag
        if tag in (ERROR, FAILURE, SKIPPED) and _RANK[tag] > _RANK[status]:
            status = tag
    return status


def _case_id(case: ET.Element) -> str:
    """Return the test id ``classname::name`` of a ``<testcase>`` element."""
    classname = case.get("classname") or ""
    name = case.get("name") or ""
    return f"{classname}::{name}" if classname else name


def _load_root(path: str) -> ET.Element:
    """Parse ``path`` as XML and return its root element."""
    try:
        return ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise UsageError(f"cannot read JUnit report {path}: {exc.__class__.__name__}") from exc


class Report(NamedTuple):
    """Parsed JUnit report: worst status per id and per-element counters."""

    statuses: Dict[str, str]
    failures: int
    errors: int


def load_report(path: str) -> Report:
    """Parse a JUnit report into statuses and failure/error counters.

    Works with a ``<testsuites>`` or a ``<testsuite>`` root and with several
    suites. When an id is repeated (pytest writes a second ``<testcase>``
    for a teardown error after a call failure), the most severe status is
    kept for the id, while the counters count every ``<testcase>`` element
    holding a ``<failure>`` or an ``<error>``, like pytest's own summary.
    """
    statuses: Dict[str, str] = {}
    failures = errors = 0
    for case in _load_root(path).iter("testcase"):
        tid = _case_id(case)
        status = _case_status(case)
        tags = {child.tag for child in case}
        failures += FAILURE in tags
        errors += ERROR in tags
        previous = statuses.get(tid)
        if previous is None or _RANK[status] > _RANK[previous]:
            statuses[tid] = status
    return Report(statuses, failures, errors)


def _classname_to_file(classname: str) -> str:
    """Convert a dotted pytest classname into a ``.py`` file path.

    The longest dotted prefix that exists on disk wins; otherwise trailing
    components starting with an uppercase letter (test classes) are dropped.
    """
    parts = [p for p in classname.split(".") if p]
    if not parts:
        return ""
    for end in range(len(parts), 0, -1):
        candidate = "/".join(parts[:end]) + ".py"
        if os.path.isfile(candidate):
            return candidate
    end = len(parts)
    while end > 1 and parts[end - 1][:1].isupper():
        end -= 1
    return "/".join(parts[:end]) + ".py"


def failing_files(path: str) -> List[str]:
    """Return the sorted list of files holding failing or erroring tests."""
    worst: Dict[str, tuple] = {}
    for case in _load_root(path).iter("testcase"):
        tid = _case_id(case)
        status = _case_status(case)
        if tid not in worst or _RANK[status] > _RANK[worst[tid][0]]:
            # A collection error has an empty classname and the dotted module
            # path as its name.
            dotted = case.get("classname") or case.get("name") or ""
            worst[tid] = (status, case.get("file"), dotted)
    files = set()
    for status, file_attr, dotted in worst.values():
        if status not in FAILING:
            continue
        if file_attr:
            files.add(file_attr.replace("\\", "/"))
        else:
            converted = _classname_to_file(dotted)
            if converted:
                files.add(converted)
    return sorted(files)


def _count(results: Dict[str, str], status: str) -> int:
    """Count the tests of ``results`` having ``status``."""
    return sum(1 for value in results.values() if value == status)


def load_allowed_removals(path: Optional[str]) -> set:
    """Read the ids allowed to disappear (one per line, blanks and ``#`` ignored)."""
    if not path:
        return set()
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        raise UsageError(f"cannot read allowed removals {path}") from exc
    return {line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")}


def load_net_legacy(path: Optional[str]) -> int:
    """Read the integer ``count`` key of the net legacy JSON file (0 if absent)."""
    if not path:
        return 0
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return int(data.get("count", 0)) if isinstance(data, dict) else 0
    except (OSError, ValueError, TypeError) as exc:
        raise UsageError(f"cannot read net legacy file {path}") from exc


def diff_reports(baseline: Report, current: Report, allowed: Iterable[str]) -> dict:
    """Compute regression metrics between two parsed reports.

    A new failure is a test failing or erroring in ``current`` that passed
    in ``baseline`` or was absent from it (a test skipped or already failing
    in the baseline is not new, though it still counts in the counters).
    """
    allowed_set = set(allowed)
    base, cur = baseline.statuses, current.statuses
    new_failures = sorted(
        tid for tid, status in cur.items()
        if status in FAILING and base.get(tid) in (None, PASSED)
    )
    missing = sorted(
        tid for tid, status in base.items()
        if status == PASSED
        and cur.get(tid) in (None, SKIPPED)
        and tid not in allowed_set
    )
    return {
        "new": new_failures,
        "missing": missing,
        "failures": current.failures,
        "errors": current.errors,
        "passed_delta": _count(cur, PASSED) - _count(base, PASSED),
    }


def _default_if_exists(value: Optional[str], default: str) -> Optional[str]:
    """Return ``value`` when given, else ``default`` if that file exists."""
    if value:
        return value
    return default if os.path.isfile(default) else None


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Compare pytest JUnit XML reports.")
    parser.add_argument("--baseline", help="reference JUnit XML")
    parser.add_argument("--current", help="JUnit XML to check")
    parser.add_argument("--max-failures", type=int, default=0,
                        help="tolerated failures + errors in the current report")
    parser.add_argument("--allowed-removals", help="ids allowed to disappear")
    parser.add_argument("--net-legacy", help="JSON file with an integer 'count'")
    parser.add_argument("--summary", metavar="FILE", help="summarise a single report")
    return parser


def run_summary(path: str) -> int:
    """Print the failure summary of a single report and return 0."""
    report = load_report(path)
    files = failing_files(path)
    print(
        f"FAILURES={report.failures} ERRORS={report.errors} "
        f"FAILING_FILES={','.join(files)}"
    )
    return 0


def run_diff(args: argparse.Namespace) -> int:
    """Compare baseline and current reports, print the result, return exit code."""
    allowed_path = _default_if_exists(args.allowed_removals, DEFAULT_ALLOWED_REMOVALS)
    legacy_path = _default_if_exists(args.net_legacy, DEFAULT_NET_LEGACY)
    allowed = load_allowed_removals(allowed_path)
    net_legacy = load_net_legacy(legacy_path)
    baseline = load_report(args.baseline)
    current = load_report(args.current)
    result = diff_reports(baseline, current, allowed)
    for tid in result["new"]:
        print(f"NEW: {tid}")
    for tid in result["missing"]:
        print(f"MISSING: {tid}")
    print(
        f"NEW_FAILURES={len(result['new'])} FAILURES={result['failures']} "
        f"ERRORS={result['errors']} MISSING={len(result['missing'])} "
        f"PASSED_DELTA={result['passed_delta']} NET_LEGACY={net_legacy}"
    )
    failed = (
        result["new"]
        or result["missing"]
        or (result["failures"] + result["errors"]) > args.max_failures
    )
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: dispatch to summary or diff mode."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.summary:
            if args.baseline or args.current:
                raise UsageError("--summary cannot be combined with --baseline/--current")
            return run_summary(args.summary)
        if not args.baseline or not args.current:
            raise UsageError("--baseline and --current are required (or use --summary)")
        if args.max_failures < 0:
            raise UsageError("--max-failures must be >= 0")
        return run_diff(args)
    except UsageError as exc:
        print(f"USAGE_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
