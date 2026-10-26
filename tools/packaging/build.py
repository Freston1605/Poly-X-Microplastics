#!/usr/bin/env python3
"""Build Poly-X with PyInstaller.

Usage:
    python tools/packaging/build.py [--onefile] [--clean]

Requires PyInstaller in the active environment:
    pip install pyinstaller
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).resolve().parent / "polyx.spec"
DIST = ROOT / "dist"
WORK = ROOT / "build"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Poly-X with PyInstaller")
    parser.add_argument("--clean", action="store_true", help="remove build/dist first")
    args = parser.parse_args()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed. Run: pip install pyinstaller", file=sys.stderr)
        return 1

    if args.clean:
        for d in (DIST, WORK):
            if d.exists():
                shutil.rmtree(d)

    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm",
           "--distpath", str(DIST), "--workpath", str(WORK)]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())