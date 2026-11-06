"""Reject hard-coded fallback colours in ``CLASS_COLOR_HEX`` lookups.

``CLASS_COLOR_HEX.get(name, "#rrggbb")`` forks the unknown-class colour away
from its one documented owner, ``FALLBACK_COLOR_HEX``. The pre-commit hook runs
this check so a literal fallback cannot slip in silently.

The two legacy Qt canvases are explicitly deferred by spec #29: changing their
palettes is the "unify the Qt canvases" card, not this guardrail's job.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Deferred by spec #29 (unify the Qt canvases), not forgotten.
ALLOWED = {
    "src/polyx/gui/labeler/canvas.py",
    "src/polyx/gui/viewer/canvas.py",
}


def _is_palette(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "CLASS_COLOR_HEX") or (
        isinstance(node, ast.Attribute) and node.attr == "CLASS_COLOR_HEX"
    )


def violations_in(path: Path) -> list[tuple[int, str]]:
    """Return ``(line, expression)`` for every lookup with a literal fallback."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "get" or not _is_palette(node.func.value):
            continue
        if len(node.args) < 2:
            continue
        fallback = node.args[1]
        if isinstance(fallback, ast.Constant) and isinstance(fallback.value, str):
            found.append((node.lineno, ast.unparse(node)))
    return sorted(found)


def main() -> int:
    failures: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in ALLOWED:
            continue
        for line, expression in violations_in(path):
            failures.append(f"{relative}:{line}: {expression}")

    if not failures:
        return 0
    print("Hard-coded fallback colours — use FALLBACK_COLOR_HEX:", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
