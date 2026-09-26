"""Tests for the gate tools of ``tests/tools`` (hermetic: throwaway git repos,
fake env files, no network)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).parent / "tools"


def _env():
    """Return an environment isolated from every inherited git setting.

    All ``GIT_*`` variables are dropped (``GIT_DIR``, ``GIT_CONFIG_PARAMETERS``,
    ``GIT_CONFIG_COUNT``/``KEY_n``/``VALUE_n``, ``GIT_COMMON_DIR``...), then
    the global and system configurations are disabled.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def run_tool(tool, *args, cwd, stdin=None):
    """Run ``tests/tools/<tool>`` as a CLI and return the completed process."""
    return subprocess.run(
        [sys.executable, str(TOOLS / tool), *[str(a) for a in args]],
        cwd=str(cwd),
        env=_env(),
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def lines_of(proc):
    """Return the stdout lines of a completed process."""
    return proc.stdout.splitlines()


def git(repo, *args):
    """Run git inside a throwaway repository and return its stdout."""
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), env=_env(), capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def write(repo, rel, content=""):
    """Write ``content`` to ``repo/rel``, creating parent directories."""
    path = Path(repo) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def commit_all(repo, message="snapshot"):
    """Stage everything and commit it in the throwaway repository."""
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path):
    """Create a throwaway git repository holding one initial commit."""
    path = tmp_path / "repo"
    path.mkdir()
    hooks = tmp_path / "no_hooks"
    hooks.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Gate Test")
    git(path, "config", "user.email", "gate@example.invalid")
    git(path, "config", "commit.gpgsign", "false")
    git(path, "config", "core.hooksPath", str(hooks))
    write(path, "README.txt", "base\n")
    commit_all(path, "init")
    return path


@pytest.fixture
def gates(tmp_path):
    """Return a directory outside the repository for gate input files."""
    path = tmp_path / "gates"
    path.mkdir()
    return path


# --- junit_diff --------------------------------------------------------------------


def junit_xml(suites, root="testsuites"):
    """Build a pytest-like JUnit document from suites of (classname, name, status[, file])."""
    parts = []
    for suite in suites:
        cases = []
        for case in suite:
            classname, name, status = case[:3]
            file_attr = f' file="{case[3]}"' if len(case) > 3 else ""
            inner = "" if status == "passed" else f'<{status} message="m">details</{status}>'
            cases.append(
                f'<testcase classname="{classname}" name="{name}"{file_attr} time="0.01">'
                f"{inner}</testcase>"
            )
        parts.append(f'<testsuite name="pytest" tests="{len(suite)}">{"".join(cases)}</testsuite>')
    body = f"<testsuites>{''.join(parts)}</testsuites>" if root == "testsuites" else parts[0]
    return '<?xml version="1.0" encoding="utf-8"?>' + body


def test_junit_diff_counts_errors_and_missing(tmp_path):
    baseline = tmp_path / "baseline.xml"
    current = tmp_path / "current.xml"
    baseline.write_text(junit_xml([[
        ("tests.test_a", "test_ok", "passed"),
        ("tests.test_a", "test_breaks", "passed"),
        ("tests.test_a", "test_gone", "passed"),
        ("tests.test_a", "test_old_fail", "failure"),
        ("tests.test_b.TestK", "test_skipped_now", "passed"),
    ]], root="testsuite"))
    current.write_text(junit_xml([
        [
            ("tests.test_a", "test_ok", "passed"),
            ("tests.test_a", "test_breaks", "error"),
            ("tests.test_a", "test_old_fail", "failure"),
        ],
        [
            ("tests.test_b.TestK", "test_skipped_now", "skipped"),
            ("tests.test_c", "test_new", "passed"),
        ],
    ]))
    proc = run_tool("junit_diff.py", "--baseline", baseline, "--current", current, cwd=tmp_path)
    out = lines_of(proc)
    assert proc.returncode == 1
    assert out[-1] == "NEW_FAILURES=1 FAILURES=1 ERRORS=1 MISSING=2 PASSED_DELTA=-2 NET_LEGACY=0"
    assert "NEW: tests.test_a::test_breaks" in out
    assert "MISSING: tests.test_a::test_gone" in out
    assert "MISSING: tests.test_b.TestK::test_skipped_now" in out
    assert not any(line.startswith("NEW: tests.test_a::test_old_fail") for line in out)

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"count": 3}))
    proc = run_tool("junit_diff.py", "--baseline", baseline, "--current", current,
                    "--max-failures", "5", "--net-legacy", legacy, cwd=tmp_path)
    assert proc.returncode == 1
    assert lines_of(proc)[-1].endswith("NET_LEGACY=3")

    proc = run_tool("junit_diff.py", "--baseline", baseline, "--current", baseline, cwd=tmp_path)
    assert proc.returncode == 1
    assert lines_of(proc)[-1] == "NEW_FAILURES=0 FAILURES=1 ERRORS=0 MISSING=0 PASSED_DELTA=0 NET_LEGACY=0"
    proc = run_tool("junit_diff.py", "--baseline", baseline, "--current", baseline,
                    "--max-failures", "1", cwd=tmp_path)
    assert proc.returncode == 0

    proc = run_tool("junit_diff.py", "--baseline", baseline, cwd=tmp_path)
    assert proc.returncode == 2

    # A call failure plus a teardown error give two <testcase> elements for
    # one id (both counted, like pytest's summary); a test skipped in the
    # baseline that fails now is counted but is not NEW.
    base2 = tmp_path / "base2.xml"
    cur2 = tmp_path / "cur2.xml"
    base2.write_text(junit_xml([[
        ("tests.test_d", "test_double", "passed"),
        ("tests.test_d", "test_was_skipped", "skipped"),
    ]]))
    cur2.write_text(junit_xml([[
        ("tests.test_d", "test_double", "failure"),
        ("tests.test_d", "test_double", "error"),
        ("tests.test_d", "test_was_skipped", "failure"),
    ]]))
    proc = run_tool("junit_diff.py", "--baseline", base2, "--current", cur2, cwd=tmp_path)
    assert proc.returncode == 1
    assert lines_of(proc) == [
        "NEW: tests.test_d::test_double",
        "NEW_FAILURES=1 FAILURES=2 ERRORS=1 MISSING=0 PASSED_DELTA=-1 NET_LEGACY=0",
    ]
    proc = run_tool("junit_diff.py", "--summary", cur2, cwd=tmp_path)
    assert lines_of(proc) == ["FAILURES=2 ERRORS=1 FAILING_FILES=tests/test_d.py"]


def test_junit_diff_summary_failing_files(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(junit_xml([
        [
            ("tests.test_x.TestY", "test_1", "failure"),
            ("tests.test_z", "test_2", "error"),
            ("tests.test_ok", "test_3", "passed"),
        ],
        [
            ("tests.test_n.TestA.TestB", "test_4", "failure"),
            ("main_module", "test_5", "failure", "app/test_main.py"),
            ("tests.test_s", "test_6", "skipped"),
            ("", "scripts.test_collect_broken", "error"),
        ],
    ]))
    proc = run_tool("junit_diff.py", "--summary", report, cwd=tmp_path)
    assert proc.returncode == 0
    assert lines_of(proc)[-1] == (
        "FAILURES=3 ERRORS=2 FAILING_FILES=app/test_main.py,scripts/test_collect_broken.py,"
        "tests/test_n.py,tests/test_x.py,tests/test_z.py"
    )


def test_junit_diff_allowed_removals(tmp_path):
    baseline = tmp_path / "baseline.xml"
    current = tmp_path / "current.xml"
    baseline.write_text(junit_xml([[("m", "test_keep", "passed"), ("m", "test_removed", "passed")]]))
    current.write_text(junit_xml([[("m", "test_keep", "passed")]]))

    proc = run_tool("junit_diff.py", "--baseline", baseline, "--current", current, cwd=tmp_path)
    assert proc.returncode == 1
    assert "MISSING: m::test_removed" in lines_of(proc)

    allowed = tmp_path / "allowed.txt"
    allowed.write_text("# retired tests\n\nm::test_removed\n")
    proc = run_tool("junit_diff.py", "--baseline", baseline, "--current", current,
                    "--allowed-removals", allowed, cwd=tmp_path)
    assert proc.returncode == 0
    assert lines_of(proc)[-1] == "NEW_FAILURES=0 FAILURES=0 ERRORS=0 MISSING=0 PASSED_DELTA=-1 NET_LEGACY=0"

    default = tmp_path / "data" / "albert_gates" / "allowed_removals.txt"
    default.parent.mkdir(parents=True)
    default.write_text("m::test_removed\n")
    proc = run_tool("junit_diff.py", "--baseline", baseline, "--current", current, cwd=tmp_path)
    assert proc.returncode == 0
    assert "MISSING=0" in lines_of(proc)[-1]


# --- check_scope -------------------------------------------------------------------


def test_check_scope_ignores_ds_store_flags_conflicts(repo, gates):
    write(repo, "src/a.py", "A = 1\n")
    write(repo, "legacy.txt", "old\n")
    commit_all(repo)
    write(repo, "legacy.txt", "changed before the lot\n")
    preexisting = gates / "preexisting.txt"
    preexisting.write_text(git(repo, "status", "--porcelain", "--untracked-files=all"))
    owned = gates / "owned.txt"
    owned.write_text("src/a.py\n")

    write(repo, "src/a.py", "A = 2\n")
    write(repo, ".DS_Store", "x")
    write(repo, "src/.DS_Store", "x")
    write(repo, "src/._a.py", "x")
    write(repo, "Icon\r", "")
    write(repo, "notes (1).txt", "copy")
    write(repo, "rapport CONFLIT.md", "copy")

    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", preexisting, cwd=repo)
    out = lines_of(proc)
    assert proc.returncode == 1
    assert "CONFLICT: notes (1).txt" in out
    assert "CONFLICT: rapport CONFLIT.md" in out
    assert out[-1] == "SCOPE_VIOLATION 2"
    assert not any("OUT_OF_SCOPE" in line for line in out)
    assert "DS_Store" not in proc.stdout and "Icon" not in proc.stdout and "._a" not in proc.stdout

    (repo / "notes (1).txt").unlink()
    (repo / "rapport CONFLIT.md").unlink()
    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", preexisting, cwd=repo)
    assert proc.returncode == 0
    assert lines_of(proc)[-1] == "SCOPE_OK"

    write(repo, "other.txt", "stray")
    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", preexisting, cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == ["OUT_OF_SCOPE: other.txt", "SCOPE_VIOLATION 1"]
    (repo / "other.txt").unlink()

    # A conflict copy of a directory under an owned prefix is still caught,
    # kept out of the add list, and refused once staged.
    owned.write_text("src/a.py\ntests/fixtures/albert/\n")
    copy = "tests/fixtures/albert/golden_off (1)/a.json"
    write(repo, "tests/fixtures/albert/golden_off/a.json", "{}\n")
    write(repo, copy, "{}\n")
    add_list = gates / "add.txt"
    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", preexisting,
                    "--emit-add-list", add_list, cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == [f"CONFLICT: {copy}", "SCOPE_VIOLATION 1"]
    assert add_list.read_text(encoding="utf-8").splitlines() == [
        "src/a.py", "tests/fixtures/albert/golden_off/a.json",
    ]
    git(repo, "add", "tests/fixtures/albert")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == [f"CONFLICT: {copy}", "SCOPE_VIOLATION 1"]


def test_check_scope_emit_add_list(repo, gates):
    write(repo, "pkg/mod.py", "X = 1\n")
    write(repo, "top.py", "Y = 1\n")
    commit_all(repo)
    write(repo, "pkg/mod.py", "X = 2\n")
    write(repo, "pkg/new.py", "Z = 1\n")
    write(repo, "pkg/sub/deep.py", "W = 1\n")
    (repo / "top.py").unlink()
    write(repo, "pkg/.DS_Store", "x")
    write(repo, "scratch.txt", "user notes")
    owned = gates / "owned.txt"
    owned.write_text("pkg/\ntop.py\n\n")
    preexisting = gates / "preexisting.txt"
    preexisting.write_text("?? scratch.txt\n")
    add_list = gates / "add.txt"

    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", preexisting,
                    "--emit-add-list", add_list, cwd=repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert lines_of(proc)[-1] == "SCOPE_OK"
    assert add_list.read_text(encoding="utf-8").splitlines() == [
        "pkg/mod.py", "pkg/new.py", "pkg/sub/deep.py", "top.py",
    ]

    git(repo, "add", f"--pathspec-from-file={add_list}")
    staged = git(repo, "diff", "--cached", "--name-status").splitlines()
    assert sorted(staged) == ["A\tpkg/new.py", "A\tpkg/sub/deep.py", "D\ttop.py", "M\tpkg/mod.py"]


def test_check_scope_staged_out_of_scope(repo, gates):
    write(repo, "a.py", "A = 1\n")
    write(repo, "b.py", "B = 1\n")
    git(repo, "add", "a.py", "b.py")
    owned = gates / "owned.txt"
    owned.write_text("a.py\n")
    preexisting = gates / "preexisting.txt"
    preexisting.write_text("?? b.py\n")

    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", preexisting, cwd=repo)
    assert proc.returncode == 0
    assert lines_of(proc)[-1] == "SCOPE_OK"

    proc = run_tool("check_scope.py", "--owned", owned, "--staged", cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == ["OUT_OF_SCOPE: b.py", "SCOPE_VIOLATION 1"]

    git(repo, "rm", "-q", "--cached", "b.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", cwd=repo)
    assert proc.returncode == 0
    assert lines_of(proc)[-1] == "SCOPE_OK"

    write(repo, ".DS_Store", "x")
    git(repo, "add", ".DS_Store")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", cwd=repo)
    assert proc.returncode == 1
    assert "OUT_OF_SCOPE: .DS_Store" in lines_of(proc)

    proc = run_tool("check_scope.py", "--owned", owned, "--restrict", owned, cwd=repo)
    assert proc.returncode == 2


MOD_HEAD = '''"""Module doc."""
import os

CONST = 1


def f():
    return 1


def g():
    return 2


class C:
    def m(self):
        return 3
'''


def test_check_scope_restrict_symbols(repo, gates):
    write(repo, "mod.py", MOD_HEAD)
    commit_all(repo)
    owned = gates / "owned.txt"
    owned.write_text("mod.py\nnew_mod.py\n")
    restrict = gates / "restrict.txt"

    changed_f = MOD_HEAD.replace("Module doc.", "Module doc, reworded.").replace("return 1", "return 10")
    write(repo, "mod.py", changed_f)
    git(repo, "add", "mod.py")
    restrict.write_text("mod.py=f\n")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 0, proc.stdout
    assert lines_of(proc)[-1] == "SCOPE_OK"

    write(repo, "mod.py", changed_f.replace("return 2", "return 20"))
    git(repo, "add", "mod.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == ["RESTRICT_VIOLATION: mod.py: g", "SCOPE_VIOLATION 1"]

    write(repo, "mod.py", changed_f + "\n\ndef h():\n    return 4\n")
    git(repo, "add", "mod.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert "RESTRICT_VIOLATION: mod.py: h" in lines_of(proc)
    restrict.write_text("mod.py=f,h\n")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 0

    write(repo, "new_mod.py", "def helper():\n    return 1\n")
    git(repo, "add", "new_mod.py")
    restrict.write_text("mod.py=f,h\nnew_mod.py=helper\n")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 0
    write(repo, "new_mod.py", "def helper():\n    return 1\n\n\nX = 2\n")
    git(repo, "add", "new_mod.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == ["RESTRICT_VIOLATION: new_mod.py: X", "SCOPE_VIOLATION 1"]


def test_check_scope_restrict_imports_and_toplevel(repo, gates):
    head = "from os.path import join, exists\nimport sys, json\n\n\ndef f():\n    return 1\n"
    write(repo, "m.py", head)
    commit_all(repo)
    owned = gates / "owned.txt"
    owned.write_text("m.py\n")
    restrict = gates / "restrict.txt"
    restrict.write_text("m.py=f,isdir\n")

    # Extending a multi-name import only changes the new name.
    extended = head.replace("join, exists", "join, exists, isdir").replace("return 1", "return 2")
    write(repo, "m.py", extended)
    git(repo, "add", "m.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 0, proc.stdout
    assert lines_of(proc) == ["SCOPE_OK"]

    # Splitting ``import sys, json`` into two statements changes nothing.
    write(repo, "m.py", extended.replace("import sys, json\n", "import sys\nimport json\n"))
    git(repo, "add", "m.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 0, proc.stdout

    # A statement binding no name is the ``<toplevel>`` symbol.
    write(repo, "m.py", extended + "\n\nprint(f())\n")
    git(repo, "add", "m.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == ["RESTRICT_VIOLATION: m.py: <toplevel>", "SCOPE_VIOLATION 1"]
    restrict.write_text("m.py=f,isdir,<toplevel>\n")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 0, proc.stdout


DOC_HEAD = '''"""Old module doc."""


def f(x):
    return x + 1


class K:
    def m(self):
        """Old."""
        return 2
'''

DOC_NEW = '''"""New module doc, longer."""


def f(x):
    """Add one to ``x``."""
    return x + 1


class K:
    """A class."""

    def m(self):
        """New wording."""
        return 2
'''


def test_check_scope_restrict_docstrings_only(repo, gates):
    write(repo, "doc.py", DOC_HEAD)
    commit_all(repo)
    owned = gates / "owned.txt"
    owned.write_text("doc.py\n")
    restrict = gates / "restrict.txt"
    restrict.write_text("doc.py=@docstrings\n")

    write(repo, "doc.py", DOC_NEW)
    git(repo, "add", "doc.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 0, proc.stdout
    assert lines_of(proc)[-1] == "SCOPE_OK"

    write(repo, "doc.py", DOC_NEW.replace("x + 1", "x + 2"))
    git(repo, "add", "doc.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == ["RESTRICT_VIOLATION: doc.py: @docstrings", "SCOPE_VIOLATION 1"]

    # Replacing a ``pass`` or ``...`` placeholder body by a docstring is allowed.
    placeholder_head = (
        "class E(Exception):\n    pass\n\n\n"
        "class P:\n    def existing(self, ids):\n        ...\n"
    )
    placeholder_new = (
        'class E(Exception):\n    """Raised on error."""\n\n\n'
        'class P:\n    def existing(self, ids):\n        """Return the ids."""\n'
    )
    write(repo, "ph.py", placeholder_head)
    git(repo, "add", "ph.py")
    git(repo, "commit", "-q", "-m", "placeholders")
    owned.write_text("doc.py\nph.py\n")
    restrict.write_text("ph.py=@docstrings\n")
    write(repo, "ph.py", placeholder_new)
    git(repo, "add", "ph.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 0, proc.stdout
    assert lines_of(proc)[-1] == "SCOPE_OK"
    write(repo, "ph.py", placeholder_new + "        return ids\n")
    git(repo, "add", "ph.py")
    proc = run_tool("check_scope.py", "--owned", owned, "--staged", "--restrict", restrict, cwd=repo)
    assert proc.returncode == 1
    assert "RESTRICT_VIOLATION: ph.py: @docstrings" in lines_of(proc)


def test_check_scope_paths_with_spaces(repo, gates):
    write(repo, "old name.txt", "a\n")
    commit_all(repo)
    git(repo, "mv", "old name.txt", "new name.txt")
    write(repo, "données été.txt", "user file\n")
    preexisting = gates / "preexisting.txt"
    status = git(repo, "status", "--porcelain", "--untracked-files=all")
    assert '"' in status  # paths are C-quoted by git
    preexisting.write_text(status, encoding="utf-8")

    owned_path = "dir with space/file é.py"
    write(repo, owned_path, "V = 1\n")
    write(repo, "autre fichier.txt", "stray\n")
    owned = gates / "owned.txt"
    owned.write_text(owned_path + "\n", encoding="utf-8")
    add_list = gates / "add.txt"

    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", preexisting,
                    "--emit-add-list", add_list, cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == ["OUT_OF_SCOPE: autre fichier.txt", "SCOPE_VIOLATION 1"]
    assert add_list.read_text(encoding="utf-8").splitlines() == [owned_path]

    (repo / "autre fichier.txt").unlink()
    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", preexisting, cwd=repo)
    assert proc.returncode == 0
    assert lines_of(proc)[-1] == "SCOPE_OK"


def test_check_scope_owned_preexisting_and_snapshot(repo, gates):
    write(repo, "tracked.md", "v1\n")
    commit_all(repo)
    write(repo, "notes.md", "user notes\n")
    write(repo, "tracked.md", "user edit\n")
    write(repo, ".DS_Store", "x")
    preexisting = gates / "preexisting.txt"
    preexisting.write_text(git(repo, "status", "--porcelain", "--untracked-files=all"))
    snapshot = gates / "snapshot.tsv"
    proc = run_tool("check_scope.py", "--snapshot", snapshot, "--preexisting", preexisting, cwd=repo)
    assert proc.returncode == 0, proc.stderr
    assert lines_of(proc) == ["SNAPSHOT_OK 2"]
    rows = [line.split("\t") for line in snapshot.read_text(encoding="utf-8").splitlines()]
    assert [row[1] for row in rows] == ["notes.md", "tracked.md"]
    assert all(len(row[0]) == 64 for row in rows)
    assert "user" not in snapshot.read_text(encoding="utf-8")

    owned = gates / "owned.txt"
    owned.write_text("src/new.py\n")
    write(repo, "src/new.py", "N = 1\n")
    args = ["--owned", owned, "--preexisting", preexisting, "--verify-snapshot", snapshot]
    proc = run_tool("check_scope.py", *args, cwd=repo)
    assert proc.returncode == 0, proc.stdout
    assert lines_of(proc) == ["SCOPE_OK"]

    # Overwriting, deleting or staging a preexisting path is a violation.
    write(repo, "notes.md", "overwritten by the lot\n")
    (repo / "tracked.md").unlink()
    proc = run_tool("check_scope.py", *args, cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == [
        "PREEXISTING_CHANGED: notes.md", "PREEXISTING_CHANGED: tracked.md", "SCOPE_VIOLATION 2",
    ]
    write(repo, "notes.md", "user notes\n")
    write(repo, "tracked.md", "user edit\n")
    proc = run_tool("check_scope.py", *args, cwd=repo)
    assert proc.returncode == 0, proc.stdout
    git(repo, "add", "notes.md")
    proc = run_tool("check_scope.py", *args, cwd=repo)
    assert lines_of(proc) == ["PREEXISTING_CHANGED: notes.md", "SCOPE_VIOLATION 1"]
    git(repo, "rm", "-q", "--cached", "notes.md")

    # An owned path that is also preexisting is flagged and never emitted,
    # unless it is explicitly allowed.
    owned.write_text("src/new.py\nnotes.md\n")
    add_list = gates / "add.txt"
    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", preexisting,
                    "--emit-add-list", add_list, cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == ["OWNED_PREEXISTING: notes.md", "SCOPE_VIOLATION 1"]
    assert add_list.read_text(encoding="utf-8").splitlines() == ["src/new.py"]
    allow = gates / "allow.txt"
    allow.write_text("notes.md\n")
    write(repo, "notes.md", "edited by the lot, as agreed\n")
    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", preexisting,
                    "--emit-add-list", add_list, "--allow-owned-preexisting", allow,
                    "--verify-snapshot", snapshot, cwd=repo)
    assert proc.returncode == 0, proc.stdout
    assert add_list.read_text(encoding="utf-8").splitlines() == ["notes.md", "src/new.py"]


def test_check_scope_watch_ignored_paths(repo, gates):
    write(repo, ".gitignore", ".env\n.env.*\ndata/\n")
    commit_all(repo)
    write(repo, ".env", "ALBERT_LIVE=0\n")
    write(repo, "data/albert_gates/owned_L1.txt", "src/a.py\n")
    write(repo, "data/albert_gates/scratch.xml", "<x/>\n")
    watch = gates / "watch.txt"
    watch.write_text(".env\n.env.local\ndata/albert_gates/owned_*.txt\n")
    snapshot = repo / "data" / "albert_gates" / "snapshot.tsv"
    owned = gates / "owned.txt"
    owned.write_text("src/a.py\n")

    proc = run_tool("check_scope.py", "--snapshot", snapshot, "--watch", watch, cwd=repo)
    assert proc.returncode == 0, proc.stderr
    assert lines_of(proc) == ["SNAPSHOT_OK 3"]
    text = snapshot.read_text(encoding="utf-8")
    assert "ALBERT_LIVE" not in text and "src/a.py" not in text
    assert [line.split("\t")[1] for line in text.splitlines()] == [
        ".env", ".env.local", "data/albert_gates/owned_L1.txt",
    ]
    assert text.splitlines()[1].split("\t")[0] == "-"

    args = ["--owned", owned, "--watch", watch, "--verify-snapshot", snapshot]
    proc = run_tool("check_scope.py", *args, cwd=repo)
    assert proc.returncode == 0, proc.stdout
    assert lines_of(proc) == ["SCOPE_OK"]

    # Ignored files are invisible to git status, but not to the snapshot.
    write(repo, "data/albert_gates/scratch.xml", "<y/>\n")
    proc = run_tool("check_scope.py", "--owned", owned, "--preexisting", gates / "none.txt",
                    cwd=repo)
    assert proc.returncode == 2
    proc = run_tool("check_scope.py", "--owned", owned, cwd=repo)
    assert lines_of(proc) == ["SCOPE_OK"]
    write(repo, ".env", "ALBERT_LIVE=1\n")
    write(repo, "data/albert_gates/owned_L1.txt", "src/a.py\nscripts/\n")
    write(repo, "data/albert_gates/owned_L2.txt", "x.py\n")
    write(repo, ".env.local", "X=1\n")
    proc = run_tool("check_scope.py", *args, cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == [
        "WATCHED_CHANGED: .env",
        "WATCHED_CHANGED: .env.local",
        "WATCHED_CHANGED: data/albert_gates/owned_L1.txt",
        "WATCHED_CHANGED: data/albert_gates/owned_L2.txt",
        "SCOPE_VIOLATION 4",
    ]
    assert "ALBERT_LIVE" not in proc.stdout + proc.stderr

    # Emptying the watch list does not hide changes to snapshotted paths.
    watch.write_text("")
    proc = run_tool("check_scope.py", *args, cwd=repo)
    assert proc.returncode == 1
    assert "WATCHED_CHANGED: .env" in lines_of(proc)

    assert run_tool("check_scope.py", "--snapshot", snapshot, cwd=repo).returncode == 2
    assert run_tool("check_scope.py", "--owned", owned, "--watch", watch, cwd=repo).returncode == 2
    assert run_tool("check_scope.py", "--owned", owned, "--staged", "--verify-snapshot",
                    snapshot, cwd=repo).returncode == 2


# --- check_secrets -----------------------------------------------------------------


def fake_secrets():
    """Build fake secret strings at run time (never written as literals)."""
    return {
        "pattern_openai": "s" + "k-" + "A" * 24,
        "pattern_pinecone": "pc" + "sk_" + "B" * 24,
        "pattern_bearer": "Bea" + "rer " + "C" * 24,
        "real": "fakevalue" + "Q" * 16,
    }


def test_check_secrets_never_prints_values(repo):
    secrets = fake_secrets()
    write(repo, ".env", f"OPENAI_API_KEY={secrets['real']}\nLOG_LEVEL=INFO\n")
    write(repo, "data/leak.txt", "\n".join([
        "x = 1",
        f"key = {secrets['pattern_openai']}",
        f"pc = {secrets['pattern_pinecone']}",
        f"auth: {secrets['pattern_bearer']}",
        f"real: {secrets['real']}",
    ]) + "\n")

    proc = run_tool("check_secrets.py", "--paths", "data", cwd=repo)
    out = lines_of(proc)
    assert proc.returncode == 1
    assert out[-1] == "PATTERN_HITS=3 REAL_VALUE_HITS=1"
    assert "HIT: data/leak.txt:2 (pattern)" in out
    assert "HIT: data/leak.txt:5 (real_value:OPENAI_API_KEY)" in out

    write(repo, "staged.py", f"A = 1\nB = '{secrets['pattern_openai']}'\nC = '{secrets['real']}'\n")
    git(repo, "add", "staged.py")
    staged = run_tool("check_secrets.py", "--staged", cwd=repo)
    assert staged.returncode == 1
    assert "HIT: staged:staged.py:2 (pattern)" in lines_of(staged)
    assert "HIT: staged:staged.py:3 (real_value:OPENAI_API_KEY)" in lines_of(staged)
    assert lines_of(staged)[-1] == "PATTERN_HITS=1 REAL_VALUE_HITS=1"

    combined = run_tool("check_secrets.py", "--staged", "--paths", ".", cwd=repo)
    assert combined.returncode == 1
    assert ".env" not in combined.stdout

    for result in (proc, staged, combined):
        blob = result.stdout + result.stderr
        for value in secrets.values():
            assert value not in blob
        assert "A" * 24 not in blob and "Q" * 16 not in blob


def test_check_secrets_ignores_x_templates(tmp_path):
    (tmp_path / ".env").write_text(
        "SHORT_PASSWORD=abc\nMODEL_NAME=some-long-model-identifier\n"
        "UNUSED_API_KEY=" + "s" + "k-proj-" + "X" * 40 + "\n",
        encoding="utf-8",
    )
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "env.example").write_text("\n".join([
        "OPENAI_API_KEY=" + "s" + "k-proj-" + "X" * 40,
        "OPENROUTER_API_KEY=" + "s" + "k-or-v1-" + "X" * 40,
        "PINECONE_API_KEY=" + "pc" + "sk_" + "X" * 40,
        "Authorization: " + "Bea" + "rer " + "X" * 30,
        "Authorization: " + "Bea" + "rer " + "fake" + "-albert-key-0001",
        "ALBERT_API_KEY=" + "fake" + "-albert-key-0001",
        "plain " + "s" + "k-" + "X" * 24,
        "password: abc",
        "model: some-long-model-identifier",
        "ri" + "sk" + "-assessment-for-the-whole-project-plan",
    ]) + "\n", encoding="utf-8")
    proc = run_tool("check_secrets.py", "--paths", "templates", cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout
    assert lines_of(proc) == ["PATTERN_HITS=0 REAL_VALUE_HITS=0"]

    (templates / "almost.txt").write_text("value " + "s" + "k-" + "X" * 20 + "Y7" + "\n")
    proc = run_tool("check_secrets.py", "--paths", "templates", cwd=tmp_path)
    assert proc.returncode == 1
    assert lines_of(proc)[-1] == "PATTERN_HITS=1 REAL_VALUE_HITS=0"

    # Keys after a URL-encoded "=" or a digit are still keys.
    (templates / "encoded.txt").write_text(
        "https://h.invalid/?api_key%3D" + "s" + "k-" + "D" * 24 + "\n"
        + "id1" + "pc" + "sk_" + "E" * 24 + "\n"
    )
    proc = run_tool("check_secrets.py", "--paths", "templates", cwd=tmp_path)
    assert proc.returncode == 1
    assert "HIT: templates/encoded.txt:1 (pattern)" in lines_of(proc)
    assert "HIT: templates/encoded.txt:2 (pattern)" in lines_of(proc)
    assert lines_of(proc)[-1] == "PATTERN_HITS=3 REAL_VALUE_HITS=0"
    assert "D" * 24 not in proc.stdout + proc.stderr

    proc = run_tool("check_secrets.py", cwd=tmp_path)
    assert proc.returncode == 2


def test_check_secrets_skips_missing_root(tmp_path):
    (tmp_path / "data" / "albert_gates").mkdir(parents=True)
    (tmp_path / "data" / "albert_gates" / "owned.txt").write_text("a.py\n")
    proc = run_tool("check_secrets.py", "--paths", "data/albert_gates", "tests/fixtures/albert",
                    "--env-file", tmp_path / "absent.env", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert lines_of(proc) == ["PATTERN_HITS=0 REAL_VALUE_HITS=0"]
    assert "SKIP: tests/fixtures/albert (not found)" in proc.stderr
    assert "NOTE: env file not found" in proc.stderr


def test_check_secrets_binary_staged_and_large_files(repo):
    secrets = fake_secrets()
    write(repo, ".env", f"OPENAI_API_KEY={secrets['real']}\n")
    (repo / "blob.bin").write_bytes(b"head\0\0\n" + secrets["real"].encode() + b"\n")
    write(repo, ".gitattributes", "*.json -diff\n")
    write(repo, "data.json", '{"k": "' + secrets["pattern_openai"] + '"}\n')
    git(repo, "add", "blob.bin", ".gitattributes", "data.json")
    proc = run_tool("check_secrets.py", "--staged", cwd=repo)
    assert proc.returncode == 1
    assert "HIT: staged:blob.bin:2 (real_value:OPENAI_API_KEY)" in lines_of(proc)
    assert "HIT: staged:data.json:1 (pattern)" in lines_of(proc)
    assert lines_of(proc)[-1] == "PATTERN_HITS=1 REAL_VALUE_HITS=1"
    git(repo, "rm", "-q", "--cached", "blob.bin", ".gitattributes", "data.json")

    # A text file over 5 MB is scanned; a binary one is skipped visibly.
    filler = ("y" * 999 + "\n") * 6000
    write(repo, "big/report.json", filler + "tail " + secrets["real"] + "\n")
    (repo / "big" / "blob.db").write_bytes(b"\0" * 16 + filler.encode() + secrets["real"].encode())
    proc = run_tool("check_secrets.py", "--paths", "big", cwd=repo)
    assert proc.returncode == 1
    assert lines_of(proc) == [
        "HIT: big/report.json:6001 (real_value:OPENAI_API_KEY)",
        "PATTERN_HITS=0 REAL_VALUE_HITS=1",
    ]
    assert "SKIPPED: big/blob.db" in proc.stderr
    assert secrets["real"] not in proc.stdout + proc.stderr


def test_check_secrets_env_file_resolution(repo):
    secrets = fake_secrets()
    write(repo, ".env", f"OPENAI_API_KEY={secrets['real']}\n")
    write(repo, "sub/dir/leak.txt", f"v = {secrets['real']}\n")
    # From a subdirectory, the default .env is found at the repository top.
    proc = run_tool("check_secrets.py", "--paths", ".", cwd=repo / "sub")
    assert proc.returncode == 1
    assert lines_of(proc) == [
        "HIT: ./dir/leak.txt:1 (real_value:OPENAI_API_KEY)",
        "PATTERN_HITS=0 REAL_VALUE_HITS=1",
    ]
    assert "NOTE" not in proc.stderr

    # An undecodable env file is a usage error that prints no content.
    (repo / ".env").write_bytes(
        b"OPENAI_API_KEY=" + secrets["real"].encode() + b"\n\xff\xfe broken\n"
    )
    proc = run_tool("check_secrets.py", "--paths", "sub", cwd=repo)
    assert proc.returncode == 2
    assert "USAGE_ERROR: cannot read env file" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert secrets["real"] not in proc.stdout + proc.stderr


# --- check_docstrings --------------------------------------------------------------


LEGACY_HEAD = '''def old_fn():
    return 1


class Old:
    def old_method(self):
        return 2
'''

LEGACY_NEW = '''def old_fn():
    return 1


def new_fn():
    return 2


def documented():
    """Has a docstring."""

    def inner():
        return 3

    return inner


async def new_async():
    return 4


class Old:
    def old_method(self):
        return 2

    def new_method(self):
        return 5


class Fresh:
    """Class doc."""

    def __init__(self):
        self.x = 1

    class Inner:
        def deep(self):
            return 6


square = lambda v: v * v
'''

SAMPLE_TEST = '''def helper():
    return 1


def test_one():
    assert helper() == 1


class TestGroup:
    def test_two(self):
        assert True
'''


def missing_names(proc):
    """Return the ``path:qualname`` part of each MISSING line."""
    return {
        line[len("MISSING: "):].rsplit(":", 1)[0]
        for line in lines_of(proc) if line.startswith("MISSING: ")
    }


def test_check_docstrings_policy(repo):
    write(repo, "legacy.py", LEGACY_HEAD)
    commit_all(repo)
    write(repo, "legacy.py", LEGACY_NEW)
    write(repo, "tests/test_sample.py", SAMPLE_TEST)
    expected = {
        "legacy.py:new_fn",
        "legacy.py:new_async",
        "legacy.py:Old.new_method",
        "legacy.py:Fresh.__init__",
        "legacy.py:Fresh.Inner.deep",
        "tests/test_sample.py:helper",
    }
    strict_extra = {
        "legacy.py:documented.<locals>.inner",
        "legacy.py:<lambda>",
        "tests/test_sample.py:test_one",
        "tests/test_sample.py:TestGroup.test_two",
    }

    proc = run_tool("check_docstrings.py", "--worktree", "legacy.py", "tests/test_sample.py", cwd=repo)
    assert proc.returncode == 1
    assert missing_names(proc) == expected
    assert "MISSING: legacy.py:new_fn:5" in lines_of(proc)
    assert lines_of(proc)[-1] == f"MISSING={len(expected)}"

    strict = run_tool("check_docstrings.py", "--worktree", "legacy.py", "tests/test_sample.py",
                      "--strict", cwd=repo)
    assert strict.returncode == 1
    assert missing_names(strict) == expected | strict_extra

    git(repo, "add", "legacy.py", "tests/test_sample.py")
    staged = run_tool("check_docstrings.py", "--staged", cwd=repo)
    assert staged.returncode == 1
    assert missing_names(staged) == expected

    commit_all(repo)
    clean = run_tool("check_docstrings.py", "--staged", cwd=repo)
    assert clean.returncode == 0
    assert lines_of(clean) == ["MISSING=0"]

    write(repo, "broken.py", "def x(:\n    pass\n")
    broken = run_tool("check_docstrings.py", "--worktree", "broken.py", cwd=repo)
    assert broken.returncode == 1
    assert "PARSE_ERROR: broken.py" in lines_of(broken)

    assert run_tool("check_docstrings.py", cwd=repo).returncode == 2

    # A mistyped --worktree path is a finding, not a silent pass; a tracked
    # file deleted from the working tree has nothing new to check.
    typo = run_tool("check_docstrings.py", "--worktree", "legacy.py", "legacy_typo.py", cwd=repo)
    assert typo.returncode == 1
    assert lines_of(typo) == ["NOT_FOUND: legacy_typo.py", "MISSING=0"]
    (repo / "legacy.py").unlink()
    deleted = run_tool("check_docstrings.py", "--worktree", "legacy.py", cwd=repo)
    assert deleted.returncode == 0
    assert lines_of(deleted) == ["MISSING=0"]


# --- check_attribution -------------------------------------------------------------


def attribution_words():
    """Build the searched words and project doc names at run time."""
    brand = "Clau" + "de"
    return {
        "brand": brand,
        "doc_file": brand.upper() + ".md",
        "doc_dir": "." + brand.lower() + "/",
        "bare_dir": "." + brand.lower(),
        "trailer": "Co-" + "Auth" + "ored-By: Someone <someone@example.invalid>",
        "generated": "Gener" + "ated with some tool",
    }


def test_check_attribution_flags_trailers_ignores_project_doc_names(tmp_path):
    words = attribution_words()
    message = "\n".join([
        "Fix the parser",
        "",
        f"See {words['doc_file']} and {words['doc_dir']}tasks/plan.md for the plan.",
        words["trailer"],
    ]) + "\n"
    proc = run_tool("check_attribution.py", "--stdin", cwd=tmp_path, stdin=message)
    assert proc.returncode == 1
    assert lines_of(proc) == ["HIT: stdin:4", "ATTRIBUTION_HITS=1"]
    assert "Someone" not in proc.stdout

    clean_msg = tmp_path / "msg.txt"
    clean_msg.write_text(
        f"Update {words['doc_file']}\nTouch {words['doc_dir']}tasks/x.md\n", encoding="utf-8"
    )
    proc = run_tool("check_attribution.py", "--file", clean_msg, cwd=tmp_path)
    assert proc.returncode == 0
    assert lines_of(proc) == ["ATTRIBUTION_HITS=0"]

    header_path = words["brand"].lower() + "_notes.md"
    diff = "\n".join([
        f"diff --git a/{header_path} b/{header_path}",
        f"--- a/{header_path}",
        f"+++ b/{header_path}",
        "@@ -1,2 +1,3 @@",
        f"-old line naming {words['brand']}",
        f"+new line about {words['doc_dir']}tasks/x.md",
        f"+{words['generated']}",
        f" context naming {words['brand']}",
    ]) + "\n"
    proc = run_tool("check_attribution.py", "--diff", cwd=tmp_path, stdin=diff)
    assert proc.returncode == 1
    assert lines_of(proc) == [f"HIT: diff:{header_path}:2", "ATTRIBUTION_HITS=1"]
    assert "some tool" not in proc.stdout

    removed_only = "\n".join([
        "--- a/x.txt",
        "+++ b/x.txt",
        "@@ -1 +0,0 @@",
        "-" + words["trailer"],
    ]) + "\n"
    proc = run_tool("check_attribution.py", "--diff", cwd=tmp_path, stdin=removed_only)
    assert proc.returncode == 0
    assert lines_of(proc) == ["ATTRIBUTION_HITS=0"]

    no_hunk = "--- a/x.txt\n+++ b/x.txt\n+" + words["trailer"] + "\n"
    proc = run_tool("check_attribution.py", "--diff", cwd=tmp_path, stdin=no_hunk)
    assert proc.returncode == 1
    assert lines_of(proc)[-1] == "ATTRIBUTION_HITS=1"


def test_check_attribution_normalises_variants_and_bare_doc_dir(tmp_path):
    words = attribution_words()
    variants = "\n".join([
        "Gener" + "ated  with two spaces",
        "Gener" + "ated\twith a tab",
        "Co\u2011" + "Auth" + "ored\u2011By: Someone",
        "Co\u2010" + "auth" + "ored-by: Someone",
        "\uff23" + "lau" + "de in full width",
        "Clau" + "\u200bde with a zero width space",
        "see code" + words["bare_dir"] + ".com for the docs",
    ]) + "\n"
    proc = run_tool("check_attribution.py", "--stdin", cwd=tmp_path, stdin=variants)
    assert proc.returncode == 1
    assert lines_of(proc)[-1] == "ATTRIBUTION_HITS=7"

    bare = "\n".join([
        f"le dossier {words['bare_dir']} est ignoré",
        words["bare_dir"],
        f"paths: ({words['bare_dir']}), '{words['bare_dir']}', {words['bare_dir']}: done",
        f"~/{words['doc_dir']}settings.json and {words['doc_file']}",
    ]) + "\n"
    proc = run_tool("check_attribution.py", "--stdin", cwd=tmp_path, stdin=bare)
    assert proc.returncode == 0, proc.stdout
    assert lines_of(proc) == ["ATTRIBUTION_HITS=0"]
