#!/usr/bin/env python3
"""Fail when the gettext catalogs drift from the source tree.

This is the pre-commit and local-gate entry point. It prints one line per
finding — ``path:line: kind: detail`` — and exits non-zero, so the person
fixing it does not have to guess which string moved.

Standard library only: pre-commit installs no third-party package for a local
hook, and reading a ``.po`` does not need one.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.i18n import audit  # noqa: E402


def main() -> int:
    findings = audit.audit_tree(ROOT)
    if findings:
        audit.report(findings)
        print(f"{len(findings)} translation finding(s)", file=sys.stderr)
        return 1
    print("translation audit clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
