"""Report interface strings that have no English translation.

Walks the syntax tree of every module looking for ``tr()`` calls with a string
literal and checks them against the dictionary. AST rather than a regular
expression because many strings are split across several pieces between
parentheses, and a regex would cut them in half.

Usage:
    python scripts/audit_translations.py          # summary
    python scripts/audit_translations.py --list   # plus the missing strings
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from polyx.core.translations_en import EN, NO_TRANSLATE  # noqa: E402


def tr_strings(path: Path) -> list[str]:
    """Every string literal passed to tr() in that file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else (func.attr if isinstance(func, ast.Attribute) else "")
        )
        if name != "tr" or not node.args:
            continue
        arg = node.args[0]
        # Literals only: tr(variable) cannot be audited statically, and
        # tr(f"...{x}") is not a usable key because it changes per call.
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append(arg.value)
    return out


def package_files() -> list[Path]:
    """Every module under ``src/polyx``, excluding legacy and caches."""
    return [
        path
        for path in sorted((ROOT / "src" / "polyx").rglob("*.py"))
        if "legacy" not in path.parts and "__pycache__" not in path.parts
    ]


def main() -> int:
    # The Windows console is cp1252 and many strings carry arrows or emoji;
    # without this the auditor dies while printing instead of auditing.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list", action="store_true", help="print the strings that are not translated"
    )
    args = parser.parse_args()

    missing: dict[str, list[Path]] = {}
    total = 0
    unique: set[str] = set()
    for path in package_files():
        for text in tr_strings(path):
            total += 1
            unique.add(text)
            if text in EN or text in NO_TRANSLATE:
                continue
            missing.setdefault(text, []).append(path.relative_to(ROOT))

    translated = len(unique) - len(missing)
    print(f"tr() strings:  {total} ({len(unique)} distinct)")
    print(f"translated:    {translated}")
    print(
        f"UNTRANSLATED:  {len(missing)}"
        + (f"   ({100 * len(missing) / len(unique):.1f} %)" if unique else "")
    )

    if missing:
        by_file: dict[str, int] = {}
        for paths in missing.values():
            for path in paths:
                key = str(path)
                by_file[key] = by_file.get(key, 0) + 1
        print("\nby file:")
        for name, count in sorted(by_file.items(), key=lambda item: -item[1]):
            print(f"  {count:>4}  {name}")

    if args.list and missing:
        print("\nuntranslated strings:")
        for text in sorted(missing):
            one_line = text.replace("\n", " ")
            print(f"  {one_line[:110]}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
