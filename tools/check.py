"""Run the project's local checks from any checkout or worktree.

The project ``.venv`` is installed editable against the main checkout, so
running pytest or ty from a git worktree would otherwise import the wrong
``src/``. This script locates that interpreter, points ``PYTHONPATH`` at the
worktree it is run from, and runs the same gate as CI: the suite, ruff and the
ty engine check.

A worktree's own ``.venv`` is used only when it can import pytest and PySide6.
A partial environment created by ``uv run`` without the GUI extra is skipped in
favour of the main checkout's interpreter, with a warning.

Usage:
    python tools/check.py [--parallel[=N]] [pytest args...]

``--parallel`` is an explicit opt-in: it runs the tests under pytest-xdist
with ``-n auto``, or ``-n N`` when given as ``--parallel=N``. Omit it and the
suite runs serially, as CI does. Re-run a parallel failure serially before
treating it as real. Every other argument is forwarded to pytest untouched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _main_worktree() -> Path | None:
    """The first worktree git lists, which is the main checkout."""
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(
        [git, "-C", str(ROOT), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            return Path(line.removeprefix("worktree "))
    return None


def _can_run_suite(python: Path) -> bool:
    """Whether this interpreter has the suite's runtime dependencies."""
    result = subprocess.run(
        [str(python), "-c", "import pytest, PySide6"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _select_interpreter(
    candidates: list[Path], usable: Callable[[Path], bool]
) -> tuple[Path | None, list[Path]]:
    """The first usable interpreter, plus the existing ones that were skipped."""
    skipped: list[Path] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if usable(candidate):
            return candidate, skipped
        skipped.append(candidate)
    return None, skipped


def _python() -> Path:
    """The project interpreter, or ``POLYX_PYTHON`` when injected."""
    override = os.environ.get("POLYX_PYTHON")
    if override:
        python = Path(override).expanduser()
        if not python.is_file() or not _can_run_suite(python):
            raise SystemExit(f"POLYX_PYTHON={python} cannot import pytest and PySide6.")
        return python

    candidates = [
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    main = _main_worktree()
    if main is not None and main != ROOT:
        candidates += [
            main / ".venv" / "bin" / "python",
            main / ".venv" / "Scripts" / "python.exe",
        ]

    chosen, skipped = _select_interpreter(candidates, _can_run_suite)
    for candidate in skipped:
        print(
            f"Ignoring {candidate}: cannot import pytest and PySide6. "
            "Run `uv sync --extra gui` in this checkout, or remove its .venv.",
            file=sys.stderr,
        )
    if chosen is not None:
        return chosen
    raise SystemExit(
        "No project interpreter with pytest and PySide6 found: create the dev "
        "env with `uv sync --extra gui`, or set POLYX_PYTHON."
    )


def _run(command: list[str], env: dict[str, str]) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def _split_parallel(argv: list[str]) -> tuple[list[str], list[str]]:
    """Separate the opt-in ``--parallel`` flag from the args pytest receives."""
    forwarded: list[str] = []
    worker_args: list[str] = []
    for argument in argv:
        if argument == "--parallel":
            worker_args = ["-n", "auto"]
        elif argument.startswith("--parallel="):
            worker_args = ["-n", argument.partition("=")[2] or "auto"]
        else:
            forwarded.append(argument)
    return forwarded, worker_args


def main(argv: list[str]) -> int:
    pytest_args, xdist_args = _split_parallel(argv)
    python = _python()
    env = os.environ.copy()
    source = str(ROOT / "src")
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source + (os.pathsep + inherited if inherited else "")
    # The suite opens no windows; PySide6 only asks for a platform.
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["VIRTUAL_ENV"] = str(python.parent.parent)

    commands = [
        [str(python), "-m", "pytest", "-q", *xdist_args, *pytest_args],
        [str(python), "-m", "ruff", "check", "src", "tests", "tools", "install"],
        [str(python), "-m", "ruff", "format", "--check", "src", "tests", "tools", "install"],
        [str(python), "-m", "ty", "check", "src/polyx/core", "src/polyx/cli"],
    ]
    for command in commands:
        code = _run(command, env)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
