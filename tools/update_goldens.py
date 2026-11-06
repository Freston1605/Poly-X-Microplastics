"""Helper to regenerate characterization golden hashes.

Usage:
    python tools/update_goldens.py <test_file> [--apply]

By default, prints the actual hashes found in test failures. With --apply,
rewrites the golden constants in the test file. Use with care: only apply
when you have verified the new output is correct (e.g., after an intentional
style change).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate golden hashes")
    parser.add_argument("test_file", help="Test file containing golden hashes")
    parser.add_argument("--apply", action="store_true", help="Apply new hashes")
    args = parser.parse_args()

    test_file = Path(args.test_file)
    if not test_file.is_file():
        print(f"File not found: {test_file}", file=sys.stderr)
        return 1

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q", "--no-header"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Extract actual hashes from assertion errors
    actual_hashes = re.findall(r"\+ ([0-9a-f]{64})", result.stdout + result.stderr)
    if not actual_hashes:
        print("No hash mismatches found — goldens are up to date.")
        return 0

    print(f"Found {len(actual_hashes)} hash mismatch(es):")
    for h in actual_hashes:
        print(f"  {h}")

    if args.apply:
        content = test_file.read_text(encoding="utf-8")
        # Replace old hashes (64 hex chars) with new ones in order
        old_hashes = re.findall(r'"([0-9a-f]{64})"', content)
        for old, new in zip(old_hashes, actual_hashes, strict=False):
            content = content.replace(f'"{old}"', f'"{new}"', 1)
        test_file.write_text(content, encoding="utf-8")
        print(f"Updated {len(actual_hashes)} hash(es) in {test_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
