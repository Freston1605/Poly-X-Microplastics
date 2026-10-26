"""Generate a compact HTML report from a headless Poly-X run folder.

Usage:
    polyx-report --run ./run --output ./run/report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyx.core.report_simple import generate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate compact HTML report from a run folder")
    p.add_argument("--run", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    try:
        out = generate(Path(args.run), Path(args.output) if args.output else None)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"WROTE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
