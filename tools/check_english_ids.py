#!/usr/bin/env python3
"""Reject Spanish in the added lines of Python files, and in all of ``src/``.

This is the command-line layer over :mod:`tests.english_scan`, the single
vocabulary the ``*_english`` pytest gates use. Pre-commit passes the staged
files; this script asks ``git diff --cached`` which of their lines are new and
prints every finding as ``path:line: finding``.

A staged commit reports untouched lines nowhere, so Spanish that the gates
intentionally grandfather (legacy test helpers) can stay in a file while new
code is held to the English rule. On a whole-tree run (``pre-commit run
--all-files``) there may be no staged diff at all: a file under ``src/`` is then
scanned in full, because production code has no grandfathered Spanish, while a
file under ``tests/`` keeps the added-lines-only rule so those legacy names
still pass.

The hook runs as a ``language: python`` local hook: pre-commit provides the
interpreter, and git is found through normal executable lookup. There is no
shell, no absolute path and no regex vocabulary of its own.
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
import sys
import tokenize
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.english_scan import scan_python_source  # noqa: E402

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _git() -> str:
    git = shutil.which("git")
    if git is None:
        raise SystemExit("git not found on PATH")
    return git


def added_lines(path: str) -> set[int]:
    """Return the line numbers ``path`` adds to the staged tree."""
    return set(_staged_additions().get(Path(path).as_posix(), set()))


@functools.lru_cache(maxsize=1)
def _staged_additions() -> dict[str, set[int]]:
    """Map each staged path to the line numbers it adds.

    The whole diff is read once, without a pathspec. Limiting the diff to one
    path disables rename detection against the old path, so a pure rename is
    reported as a brand-new file and every grandfathered Spanish line in a
    renamed test helper would be scanned. ``git diff`` prints no hunks for a
    rename whose content did not change, which is exactly what the hook wants.
    """
    git_cmd = _git()
    result = subprocess.run(
        [git_cmd, "diff", "--cached", "--unified=0", "--no-color", "--find-renames"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"git diff failed: {result.stderr.strip()}")

    additions: dict[str, set[int]] = {}
    current: str | None = None
    line = 0
    for raw in result.stdout.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:]
            current = None if target == "/dev/null" else target.removeprefix("b/")
            if current is not None:
                additions.setdefault(current, set())
            continue
        match = _HUNK.match(raw)
        if match is not None:
            line = int(match.group(1))
            continue
        if current is not None and raw.startswith("+") and not raw.startswith("+++"):
            additions[current].add(line)
            line += 1
    return additions


def _scope(path: Path, added: set[int]) -> set[int] | None:
    """Lines to scan: the staged additions, or the whole of a ``src/`` file."""
    if added:
        return added
    if path.parts and path.parts[0] == "src":
        return None
    return set()


def _error_line(exc: BaseException) -> int:
    if isinstance(exc, SyntaxError) and exc.lineno:
        return exc.lineno
    if len(exc.args) > 1 and isinstance(exc.args[1], tuple):
        return int(exc.args[1][0])
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    seen: set[tuple[str, int, str]] = set()
    failures = False

    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8")
            found = scan_python_source(source, _scope(path, added_lines(raw)))
        except (OSError, UnicodeDecodeError, tokenize.TokenError, SyntaxError) as exc:
            print(f"{raw}:{_error_line(exc)}: scanner-error: {exc}")
            failures = True
            continue
        for line, finding in sorted(found):
            key = (raw, line, finding)
            if key in seen:
                continue
            seen.add(key)
            print(f"{raw}:{line}: {finding}")
            failures = True

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
