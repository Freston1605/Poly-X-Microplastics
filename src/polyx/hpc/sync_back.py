"""Sync only the small scientific artifacts back to the local machine.

Datasets, environments, intermediate predictions and full checkpoint series
stay on the cluster. The allow-list is the execution's agreed artifact manifest,
so this helper cannot silently grow to copy a whole run directory. An artifact
larger than the cap is an error rather than a partial copy.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from polyx.pipeline.artifacts import ArtifactLayout, artifact_manifest

DEFAULT_MAX_BYTES = 1_000_000


class SyncError(RuntimeError):
    """An artifact could not be synced back safely."""


def sync_back(source: Path, destination: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> list[Path]:
    """Copy the small artifacts of ``source`` into ``destination``.

    Returns the relative paths that were copied. Missing artifacts are skipped
    so an interrupted execution can still sync what it produced; an oversized
    allow-listed artifact fails loudly instead of being truncated.
    """
    source = Path(source)
    destination = Path(destination)
    layout = ArtifactLayout(source)
    copied: list[Path] = []
    for absolute in artifact_manifest(layout).values():
        artifact = Path(absolute)
        if not artifact.is_file():
            continue
        size = artifact.stat().st_size
        if size > max_bytes:
            raise SyncError(f"{artifact.name}: artifact too large to sync back ({size} bytes)")
        relative = artifact.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, target)
        copied.append(relative)
    return copied


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the submission script after a run."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m polyx.hpc.sync_back SOURCE DESTINATION", file=sys.stderr)
        return 2
    try:
        copied = sync_back(Path(args[0]), Path(args[1]))
    except (SyncError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for relative in copied:
        print(relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
