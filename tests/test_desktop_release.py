"""The desktop release ZIP is downloadable, complete and self-checking."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
from tools.packaging.desktop_release import TOP_LEVEL_EXCLUDED, build

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def release(tmp_path: Path):
    source = tmp_path / "source"
    for relative in (
        "README.md",
        "requirements.txt",
        "install/SETUP.bat",
        "install/bootstrap.py",
        "install/update.py",
        "docs/manual/Manual_PolyX.html",
        "src/polyx/__init__.py",
        "src/polyx/locales/es/LC_MESSAGES/messages.po",
        "tests/test_install.py",
        "tools/check.py",
        "models/best.pt",
        "data/labels/one.txt",
        "runs/old/run.json",
        "legacy/v1.py",
        "CLAUDE.md",
        ".venv/bin/python",
        "src/polyx/__pycache__/cached.pyc",
        "docs/manual/Manual_PolyX.pdf",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")

    archive, checksum = build(source, tmp_path / "dist", "polyx-desktop")
    with zipfile.ZipFile(archive) as zipped:
        return archive, checksum, set(zipped.namelist())


def test_release_carries_the_program_and_the_installer(release):
    _archive, _checksum, names = release

    assert {
        "README.md",
        "requirements.txt",
        "install/SETUP.bat",
        "install/bootstrap.py",
        "install/update.py",
        "docs/manual/Manual_PolyX.html",
        "src/polyx/__init__.py",
        "src/polyx/locales/es/LC_MESSAGES/messages.po",
    } <= names


def test_release_leaves_out_local_state_and_build_artifacts(release):
    _archive, _checksum, names = release

    assert not names & {"tests/test_install.py", "tools/check.py", "CLAUDE.md"}
    assert not {name for name in names if name.split("/")[0] in TOP_LEVEL_EXCLUDED}
    assert not {name for name in names if name.endswith((".pyc", ".log", ".pdf"))}


def test_checksum_names_the_archive_and_matches_its_bytes(release):
    archive, checksum, _names = release

    digest, filename = checksum.read_text(encoding="ascii").split()
    assert filename == archive.name
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_the_real_tree_builds_and_the_zip_opens(tmp_path: Path):
    archive, checksum = build(_REPO, tmp_path, "polyx-desktop")

    assert checksum.is_file()
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        names = set(zipped.namelist())
    assert "install/bootstrap.py" in names
    assert "src/polyx/cli/detect.py" in names
    # A package directory that shares its name with a local-data folder at the
    # repository root must survive; only the root-level folder is local state.
    assert "src/polyx/core/data/__init__.py" in names
    assert not {name for name in names if "__pycache__" in name}
