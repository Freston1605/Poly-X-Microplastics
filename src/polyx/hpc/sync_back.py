"""Sync only the small scientific artifacts back to the local machine.

Datasets, environments, intermediate predictions and full checkpoint series
stay on the cluster. The allow-list is the execution's agreed artifact manifest,
so this helper cannot silently grow to copy a whole run directory. The small
artifacts share one cap; the kept checkpoint has its own, larger cap because a
trained detector legitimately weighs tens of megabytes and is a required
artifact. An artifact above its cap is an error rather than a partial copy.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

from polyx.pipeline.artifacts import ArtifactLayout, artifact_manifest

DEFAULT_MAX_BYTES = 1_000_000
#: A kept checkpoint is expected to be large; this cap catches a runaway file.
DEFAULT_CHECKPOINT_MAX_BYTES = 512 * 1024 * 1024


class SyncError(RuntimeError):
    """An artifact could not be synced back safely."""


def sync_back(
    source: Path,
    destination: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    checkpoint_max_bytes: int = DEFAULT_CHECKPOINT_MAX_BYTES,
) -> list[Path]:
    """Copy the small artifacts of ``source`` into ``destination``.

    Returns the relative paths that were copied. Missing artifacts are skipped
    so an interrupted execution can still sync what it produced; an oversized
    allow-listed artifact fails loudly instead of being truncated. The manifest's
    ``best_checkpoint`` entry uses the checkpoint cap; every other artifact uses
    the small-artifact cap.
    """
    source = Path(source)
    destination = Path(destination)
    layout = ArtifactLayout(source)
    copied: list[Path] = []
    for name, absolute in artifact_manifest(layout).items():
        artifact = Path(absolute)
        if not artifact.is_file():
            continue
        size = artifact.stat().st_size
        cap = checkpoint_max_bytes if name == "best_checkpoint" else max_bytes
        if size > cap:
            raise SyncError(f"{artifact.name}: artifact too large to sync back ({size} bytes)")
        relative = artifact.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, target)
        copied.append(relative)
    return copied


def sync_campaign(
    source: Path,
    destination: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    checkpoint_max_bytes: int = DEFAULT_CHECKPOINT_MAX_BYTES,
) -> dict[str, list[Path]]:
    """Sync every family execution under one campaign root.

    Each immediate subdirectory that holds a resolved ``config.yaml`` is one
    training execution; its artifacts land under
    ``destination/<family name>`` so three families cannot overwrite each
    other's ``metrics.json`` or ``report.md``.
    """
    source = Path(source)
    destination = Path(destination)
    synced: dict[str, list[Path]] = {}
    for run in sorted(path for path in source.iterdir() if path.is_dir()):
        config_file = run / "config.yaml"
        if not config_file.is_file():
            continue
        try:
            config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise SyncError(f"unreadable campaign configuration {config_file}: {exc}") from exc
        family = str((config.get("family") or {}).get("name") or run.name)
        synced[family] = sync_back(
            run,
            destination / family,
            max_bytes=max_bytes,
            checkpoint_max_bytes=checkpoint_max_bytes,
        )
    return synced


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the submission scripts after a run or campaign."""
    args = list(sys.argv[1:] if argv is None else argv)
    campaign = bool(args) and args[0] == "--campaign"
    if campaign:
        args = args[1:]
        if len(args) != 2:
            print(
                "usage: python -m polyx.hpc.sync_back --campaign CAMPAIGN_ROOT DESTINATION",
                file=sys.stderr,
            )
            return 2
    elif len(args) != 2:
        print("usage: python -m polyx.hpc.sync_back SOURCE DESTINATION", file=sys.stderr)
        return 2
    try:
        if campaign:
            summary = sync_campaign(Path(args[0]), Path(args[1]))
            for family, copied in sorted(summary.items()):
                for relative in copied:
                    print(f"{family}/{relative}")
            return 0
        copied = sync_back(Path(args[0]), Path(args[1]))
    except (SyncError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for relative in copied:
        print(relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
