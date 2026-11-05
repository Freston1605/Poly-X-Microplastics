#!/usr/bin/env python3
"""Pre-commit check: no Spanish identifiers in staged Python code."""

from __future__ import annotations

import re
import subprocess

SPANISH_WORDS = re.compile(
    r"\b(ruta|estado|eventos|placa|modelo|carpeta|imagen|archivo)\b", re.IGNORECASE
)


def main() -> int:
    result = subprocess.run(
        ["/usr/bin/git", "diff", "--cached", "--unified=0", "--", "src/", "tests/"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if "pre-commit-config" in line:
            continue
        if SPANISH_WORDS.search(line):
            print(f"Spanish identifier detected: {line.strip()[:80]}")
            print("Use English identifiers per AGENTS.md/CLAUDE.md.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
