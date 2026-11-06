"""The local gate forwards the selection and parallelises only on request.

The suite is the slow part of the gate, so an explicit ``--parallel`` opt-in
can use every core. Serial stays the default because CI and ordering-sensitive
tests need a reproducible reference; parallelism must never be implicit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GATE = _ROOT / "tools" / "check.py"

# The temporary test checks the two facts the gate controls: which ``src`` the
# imports resolve to, and whether xdist workers exist only when requested.
_TEMP_TEST = """\
import os
from pathlib import Path

import polyx


def test_imports_resolve_to_the_gate_worktree():
    source = Path(os.environ["POLYX_EXPECTED_SRC"]).resolve()
    assert Path(polyx.__file__).resolve().is_relative_to(source)


def test_parallelism_matches_the_request():
    expected = os.environ.get("POLYX_EXPECT_PARALLEL") == "1"
    assert ("PYTEST_XDIST_WORKER" in os.environ) is expected
"""


def _run_gate(tmp_path: Path, *args: str, parallel: bool):
    tests = tmp_path / "test_gate_seam.py"
    tests.write_text(_TEMP_TEST, encoding="utf-8")
    env = os.environ.copy()
    # An enclosing xdist run exports these; the inner run must decide for itself.
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("PYTEST_XDIST_WORKER_COUNT", None)
    env["POLYX_EXPECTED_SRC"] = str(_ROOT / "src")
    env["POLYX_EXPECT_PARALLEL"] = "1" if parallel else "0"
    command = [
        sys.executable,
        str(_GATE),
        str(tests),
        "-vv",
        "-p",
        "no:cacheprovider",
        *args,
    ]
    return subprocess.run(
        command,
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _assert_only_selection_ran(result: subprocess.CompletedProcess[str], expected: int) -> None:
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASSED") == expected
    assert "test_gate_seam.py" in result.stdout


def test_selection_runs_serially_by_default(tmp_path):
    result = _run_gate(tmp_path, parallel=False)

    _assert_only_selection_ran(result, expected=2)


def test_parallel_is_an_explicit_opt_in(tmp_path):
    result = _run_gate(tmp_path, "--parallel", parallel=True)

    _assert_only_selection_ran(result, expected=2)


def test_parallel_accepts_an_explicit_worker_count(tmp_path):
    result = _run_gate(tmp_path, "--parallel=1", parallel=True)

    _assert_only_selection_ran(result, expected=2)
