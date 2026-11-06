#!/usr/bin/env python3
"""The gettext workflow: extract, update, compile, audit and pseudo-locale.

Run from anywhere in the checkout::

    python tools/i18n/cli.py extract   # regenerate src/polyx/locales/messages.pot
    python tools/i18n/cli.py update    # merge the template into every catalog
    python tools/i18n/cli.py compile   # write the .mo files the runtime reads
    python tools/i18n/cli.py audit     # the gate: fail on any drift
    python tools/i18n/cli.py pseudo    # generate the development pseudo-locale

Extraction, update and compile need Babel (a build dependency). The audit is
standard-library only, so pre-commit can run it in its isolated environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.i18n import audit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Dispatch one subcommand."""
    parser = argparse.ArgumentParser(prog="tools/i18n/cli.py", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("extract", help="regenerate the committed template")
    subparsers.add_parser("update", help="merge the template into every catalog")
    compile_parser = subparsers.add_parser("compile", help="compile every catalog")
    compile_parser.add_argument(
        "--pseudo", action="store_true", help="also compile the development pseudo-locale"
    )
    subparsers.add_parser("audit", help="fail on any catalog drift")
    subparsers.add_parser("pseudo", help="generate the development pseudo-locale")
    args = parser.parse_args(argv)

    if args.command == "audit":
        return _audit(ROOT)
    if args.command == "extract":
        from tools.i18n import catalog

        print(catalog.write_template(ROOT))
        return 0
    if args.command == "update":
        from tools.i18n import catalog

        for path in catalog.update_catalogs(ROOT):
            print(path)
        return 0
    if args.command == "compile":
        from tools.i18n import catalog

        for path in catalog.compile_catalogs(ROOT, include_pseudo=args.pseudo):
            print(path)
        return 0
    if args.command == "pseudo":
        from tools.i18n import pseudo

        print(pseudo.generate(ROOT))
        print(f"Select it with POLYX_LOCALE={pseudo.TAG}")
        return 0
    return 2


def _audit(root: Path) -> int:
    findings = audit.audit_tree(root)
    if findings:
        audit.report(findings)
        print(f"{len(findings)} translation finding(s)", file=sys.stderr)
        return 1
    print("translation audit clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
