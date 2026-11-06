"""Run the project's local checks from any checkout or worktree.

The project ``.venv`` is installed editable against the main checkout, so
running pytest or ty from a git worktree would otherwise import the wrong
``src/``. This script locates that interpreter, points ``PYTHONPATH`` at the
worktree it is run from, and runs the same gate as CI: the suite, ruff and the
ty engine check.

Usage:
    python tools/check.py [pytest args...]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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


def _python() -> Path:
    """The project interpreter, or ``POLYX_PYTHON`` when injected."""
    override = os.environ.get("POLYX_PYTHON")
    if override:
        return Path(override).expanduser()

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
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "No project interpreter found: create the dev env with "
        "`uv sync --extra gui`, or set POLYX_PYTHON."
    )


def _run(command: list[str], env: dict[str, str]) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def main(argv: list[str]) -> int:
    python = _python()
    env = os.environ.copy()
    source = str(ROOT / "src")
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source + (os.pathsep + inherited if inherited else "")
    # The suite opens no windows; PySide6 only asks for a platform.
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["VIRTUAL_ENV"] = str(python.parent.parent)

    commands = [
        [str(python), "-m", "pytest", "-q", *argv],
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
