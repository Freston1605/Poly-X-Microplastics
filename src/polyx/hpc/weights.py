"""Manifest-driven, hash-verified model weights for offline HPC jobs.

The manifest is a ``sha256sum`` style file of pinned digests. Every download
is checked against it before the file is trusted, and the check is mandatory:
an empty or missing manifest is an error rather than an invitation to fetch
whatever a URL happens to serve today. Digests are published with the release;
this module never invents or auto-accepts one.

No URL, username, token or personal path is stored here. The download base is
supplied by the caller, normally from an environment variable in a gitignored
local configuration (see ``docs/hpc-environment.md``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from polyx.hpc.hashing import HashError, parse_manifest, verify_file

#: The weight path reuses the CLI's streaming downloader; the hash decision is
#: made here so the CLI's optional ``--sha256`` flag cannot weaken this path.
WeightHashError = HashError

Downloader = Callable[[str, Path], None]


def _default_downloader(url: str, destination: Path) -> None:
    from polyx.cli.download_models import download

    download(url, destination)


def fetch_weights(
    manifest: Path,
    destination: Path,
    *,
    base_url: str,
    downloader: Downloader | None = None,
) -> list[Path]:
    """Download and verify every weight declared in ``manifest``.

    Cached files are re-verified too. A bad artifact is removed, so a later
    run cannot pick up an unverified file and mistake it for a good one.
    """
    entries = parse_manifest(Path(manifest))
    if not entries:
        raise WeightHashError(f"{manifest}: no weights are pinned; refusing an unverified download")

    fetch = downloader or _default_downloader
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    base = base_url.rstrip("/")

    for name, expected in entries.items():
        final = target / name
        pending = target / f"{name}.part"
        if final.is_file():
            try:
                verify_file(final, expected)
            except HashError as exc:
                final.unlink(missing_ok=True)
                raise WeightHashError(str(exc)) from exc
            results.append(final)
            continue

        pending.unlink(missing_ok=True)
        fetch(f"{base}/{name}", pending)
        try:
            verify_file(pending, expected)
        except HashError as exc:
            pending.unlink(missing_ok=True)
            raise WeightHashError(str(exc)) from exc
        pending.replace(final)
        results.append(final)

    return results
