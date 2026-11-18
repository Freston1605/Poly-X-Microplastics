"""SHA-256 helpers shared by the wheelhouse and the weight fetcher.

The manifest format is the one ``sha256sum`` writes: one ``<digest>  <name>``
pair per line. Keeping it textual and tool-agnostic means a maintainer can
produce or inspect it without installing anything from this project.

No Qt, no third-party imports and no user-facing strings: callers decide how
to report a mismatch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

CHUNK = 1024 * 1024
HEX_DIGEST_LENGTH = 64


class HashError(RuntimeError):
    """A manifest entry or an artifact failed verification."""


def sha256_bytes(payload: bytes) -> str:
    """Digest of an in-memory payload; handy for tests and small manifests."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Digest of a file, read in chunks so large wheel files stay cheap."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    """Raise :class:`HashError` unless ``path`` matches the expected digest."""
    computed = sha256_file(path)
    if computed.lower() != expected.strip().lower():
        raise HashError(f"SHA-256 mismatch: got {computed}, expected {expected}")


def parse_manifest(manifest: Path) -> dict[str, str]:
    """Return ``{name: digest}`` from a ``sha256sum`` style manifest.

    Blank lines and ``#`` comments are ignored. A malformed line or a digest
    that is not 64 hex characters fails loudly: a partial manifest must never
    be treated as complete.
    """
    entries: dict[str, str] = {}
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise HashError(f"{manifest}:{number}: expected '<digest>  <name>'")
        digest, name = parts[0].strip().lower(), parts[1].strip()
        if len(digest) != HEX_DIGEST_LENGTH or any(c not in "0123456789abcdef" for c in digest):
            raise HashError(f"{manifest}:{number}: invalid SHA-256 digest {digest!r}")
        if not name or "/" in name or "\\" in name:
            raise HashError(f"{manifest}:{number}: invalid artifact name {name!r}")
        if name in entries:
            raise HashError(f"{manifest}:{number}: duplicate artifact name {name!r}")
        entries[name] = digest
    return entries


def write_manifest(manifest: Path, files: Iterable[Path]) -> None:
    """Write a ``sha256sum`` style manifest for ``files``, sorted by name."""
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{sha256_file(path)}  {path.name}" for path in sorted(files, key=lambda p: p.name)]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_manifest(base: Path, manifest: Path) -> list[str]:
    """Problems found when verifying every entry of ``manifest`` under ``base``.

    A missing file is a problem too; an empty result means every declared
    artifact is present and matches its digest.
    """
    problems: list[str] = []
    for name, expected in parse_manifest(manifest).items():
        path = base / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        try:
            verify_file(path, expected)
        except HashError as exc:
            problems.append(f"{name}: {exc}")
    return problems
