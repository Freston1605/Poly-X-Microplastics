"""Build the desktop installation ZIP and its SHA-256 checksum.

The ZIP is what a user downloads from the fork's Releases page. It carries the
program tree plus the installer scripts the desktop surface expects, and it
deliberately leaves out the environment, caches, local research work and every
build artifact: the installer recreates the environment on the user's machine.

Usage:
    python tools/packaging/desktop_release.py --out dist/release [--source .]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

#: Everything the installer and the launcher need at the top of the tree.
ROOT_FILES = (
    "LEEME.txt",
    "LEEME_macOS.md",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "NOTICE",
    "SECURITY.md",
    "requirements.txt",
    "pyproject.toml",
    "uv.lock",
)
ROOT_DIRS = ("docs", "install", "src")

#: Local state and research work at the top of the tree. Matched only at the
#: root: `src/polyx/core/data/` is a package, not the local dataset folder.
TOP_LEVEL_EXCLUDED = {
    ".git",
    ".opencode",
    ".project",
    ".ua",
    ".venv",
    "agent",
    "cluster",
    "container",
    "data",
    "legacy",
    "manuscript-workspace",
    "models",
    "notebooks",
    "runs",
    "runs_train",
    "tests",
    "tools",
}
#: Build residue, excluded wherever it appears.
ANY_DEPTH_EXCLUDED = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".log", ".pdf")


def _include(path: Path, root: Path) -> bool:
    """Whether ``path`` belongs in the ZIP, judged by its path under ``root``."""
    relative = path.relative_to(root)
    if relative.parts[0] in TOP_LEVEL_EXCLUDED and len(relative.parts) > 1:
        return False
    if any(part in ANY_DEPTH_EXCLUDED for part in relative.parts[:-1]):
        return False
    return path.is_file() and not path.name.endswith(EXCLUDED_SUFFIXES)


def _files(source: Path) -> list[Path]:
    files = [source / name for name in ROOT_FILES if (source / name).is_file()]
    for directory in ROOT_DIRS:
        base = source / directory
        if base.is_dir():
            files.extend(sorted(base.rglob("*")))
    return [path for path in files if _include(path, source)]


def build(source: Path, out: Path, name: str) -> tuple[Path, Path]:
    """Write ``<name>.zip`` and ``<name>.zip.sha256`` under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in _files(source):
            zipped.write(path, path.relative_to(source).as_posix())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = out / f"{archive.name}.sha256"
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return archive, checksum


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build the Poly-X desktop release ZIP")
    parser.add_argument("--out", default="dist/release", help="output directory")
    parser.add_argument("--source", default=".", help="project root to package")
    parser.add_argument("--name", default="polyx-desktop", help="archive name, without .zip")
    args = parser.parse_args(argv)

    archive, checksum = build(Path(args.source).resolve(), Path(args.out).resolve(), args.name)
    print(f"{archive} ({archive.stat().st_size} bytes)")
    print(checksum.read_text(encoding="ascii").rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
