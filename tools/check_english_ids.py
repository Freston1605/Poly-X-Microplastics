#!/usr/bin/env python3
"""Reject Spanish in the lines a commit adds to Python files.

This is the command-line layer over :mod:`tests.english_scan`, the single
vocabulary the ``*_ingles`` pytest gates use. Pre-commit passes the staged
files; this script asks ``git diff --cached`` which of their lines are new and
prints every finding as ``path:line: finding``. Untouched lines are never
reported, so Spanish that the gates intentionally grandfather (legacy test
helpers) can stay in a file while new code is held to the English rule.

The hook runs as a ``language: python`` local hook: pre-commit provides the
interpreter, and git is found through normal executable lookup. There is no
shell, no absolute path and no regex vocabulary of its own.
"""

from __future__ import annotations

import ast
import io
import re
import shutil
import subprocess
import sys
import tokenize
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.english_scan import (  # noqa: E402
    _identifiers_from_source,
    _strip_inline_code,
    comment_blocks,
    identifier_occurrences,
    is_protected_comment,
    spanish_comment_findings,
    spanish_findings,
    spanish_identifier_findings,
    split_comment,
)

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _git() -> str:
    git = shutil.which("git")
    if git is None:
        raise SystemExit("git not found on PATH")
    return git


def added_lines(path: str) -> set[int]:
    """Return the line numbers ``path`` adds to the staged tree."""
    git_cmd = _git()
    result = subprocess.run(
        [git_cmd, "diff", "--cached", "--unified=0", "--no-color", "--", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{path}: git diff failed: {result.stderr.strip()}")

    added: set[int] = set()
    line = 0
    for raw in result.stdout.splitlines():
        match = _HUNK.match(raw)
        if match is not None:
            line = int(match.group(1))
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            added.add(line)
            line += 1
    return added


def _docstring_spans(tree: ast.AST) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            spans.append((first.value.lineno, first.value.end_lineno or first.value.lineno))
    return spans


def _added_runs(lines: Iterable[int], added: set[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for line in sorted(set(lines)):
        if line not in added:
            continue
        if current and line == current[-1] + 1:
            current.append(line)
        else:
            if current:
                runs.append(current)
            current = [line]
    if current:
        runs.append(current)
    return runs


def findings_in(source: str, added: set[int]) -> list[tuple[int, str]]:
    """Return ``(line, finding)`` for the added lines of one source file."""
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    identifiers = _identifiers_from_source(source)
    found: list[tuple[int, str]] = []

    comment_tokens = [
        token for token in tokens if token.type == tokenize.COMMENT and token.start[0] in added
    ]
    for token in comment_tokens:
        if is_protected_comment(token.string):
            continue
        body = split_comment(token.string)[1]
        found.extend(
            (token.start[0], finding) for finding in spanish_comment_findings(body, identifiers)
        )

    # The gates also judge a block of consecutive comment lines as one prose
    # unit; only fully-added blocks can be attributed to this commit.
    for block in comment_blocks(comment_tokens):
        bodies = [
            split_comment(token.string)[1]
            for token in block
            if not is_protected_comment(token.string)
        ]
        if len(bodies) < 2:
            continue
        found.extend(
            (block[0].start[0], finding)
            for finding in spanish_comment_findings(" ".join(bodies), identifiers)
        )

    for line, name in identifier_occurrences(source):
        if line not in added:
            continue
        found.extend((line, finding) for finding in spanish_identifier_findings(name))

    source_lines = source.splitlines()
    for start, end in _docstring_spans(ast.parse(source)):
        for run in _added_runs(range(start, end + 1), added):
            prose = "\n".join(source_lines[number - 1] for number in run)
            found.extend(
                (run[0], finding) for finding in spanish_findings(_strip_inline_code(prose))
            )

    return found


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
            found = findings_in(source, added_lines(raw))
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
