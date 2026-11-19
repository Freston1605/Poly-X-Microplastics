"""Stage and verify the pinned model weights before a job uses them.

Compute nodes have no internet access, so a job verifies the staged directory
against its ``sha256sum`` manifest before training starts. On a networked node
(login node) the optional URL lets the same command fetch the missing files
first; on a compute node the URL is unset and a missing file is a hard failure
that names it. This is the staging step the pipeline's weight resolver expects:
training then only ever sees local, verified files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from polyx.core.i18n import _
from polyx.hpc.hashing import HashError, verify_manifest
from polyx.hpc.weights import fetch_weights


def main(argv: list[str] | None = None) -> int:
    """Verify the staged weights, optionally fetch them, and report the outcome."""
    parser = argparse.ArgumentParser(description=_("Stage and verify pinned model weights"))
    parser.add_argument(
        "--weights-dir",
        default=os.environ.get("POLYX_WEIGHTS_DIR", ""),
        help=_("directory holding the weight files (default: POLYX_WEIGHTS_DIR)"),
    )
    parser.add_argument(
        "--manifest",
        default=os.environ.get("POLYX_WEIGHTS_MANIFEST", ""),
        help=_("sha256sum-style manifest (default: POLYX_WEIGHTS_MANIFEST)"),
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("POLYX_WEIGHTS_URL", ""),
        help=_(
            "base URL to fetch missing files from (default: POLYX_WEIGHTS_URL); "
            "leave unset on a compute node"
        ),
    )
    args = parser.parse_args(argv)

    if not args.weights_dir or not args.manifest:
        print(
            "ERROR: "
            + _("set POLYX_WEIGHTS_DIR and POLYX_WEIGHTS_MANIFEST before staging weights"),
            file=sys.stderr,
        )
        return 1
    directory = Path(args.weights_dir).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()

    problems = _problems(directory, manifest)
    if problems and args.url:
        print(f"STAGING: fetching {len(problems)} missing weight(s) from {args.url}")
        try:
            fetch_weights(manifest, directory, base_url=args.url)
        except HashError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        problems = _problems(directory, manifest)
    if problems:
        for problem in problems:
            print(f"WEIGHTS: {problem}", file=sys.stderr)
        print(
            "ERROR: "
            + _(
                "pinned weights are missing or unverified; stage them with "
                "--url on a networked node"
            ),
            file=sys.stderr,
        )
        return 1

    print(f"WEIGHTS: {directory}")
    print(f"MANIFEST: {manifest}")
    print("VERIFIED: true")
    return 0


def _problems(directory: Path, manifest: Path) -> list[str]:
    """Every missing or mismatched entry, or an empty list when all are staged."""
    try:
        return verify_manifest(directory, manifest)
    except (OSError, HashError) as exc:
        return [str(exc)]


if __name__ == "__main__":
    raise SystemExit(main())
