"""The consumer-distribution track stays retired (ADR-0011, spec #208).

Retiring the track was a deletion, so the test is absence: every file the
decision removed is gone, and no living surface names one.

History keeps its names on purpose — the ADRs that decided the deletion,
``CHANGELOG.md``, ``GLOSSARY.md`` (which defines the retired terms) and
``docs/specs`` — and so does the frozen user manual with the generators in
``tools/manual`` that produce it: README says the manual is out of scope until
after the paper. Everything else — package, tests, tools, workflows, packaging
metadata and the documents an agent reads — must read as if the track never
existed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: What the decision deleted. A path is retired if it is not on disk.
RETIRED_PATHS = (
    "install",
    "requirements.txt",
    "tools/packaging",
    "LEEME.txt",
    "LEEME_macOS.md",
    "src/polyx/gui/updates.py",
    "tests/test_install.py",
    "tests/test_updater_observability.py",
)

#: Living surfaces: the package, its tests and tools, CI, the container and
#: cluster paths, packaging metadata and the documents a contributor reads.
SOURCE_TREES = (
    "src/polyx",
    "tests",
    "tools",
    ".github/workflows",
    "container",
    "cluster",
)
DOCUMENT_TREES = ("docs",)
#: History records the retired names on purpose: the decisions, the frozen
#: vocabulary and the specifications that predate the deletion.
HISTORY_TREES = (Path("docs/adr"), Path("docs/specs"))
DOCUMENTS = (
    "README.md",
    "CLAUDE.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "CODING_STANDARDS.md",
    "SECURITY.md",
)
FILES = ("pyproject.toml", ".pre-commit-config.yaml", ".gitattributes", ".gitignore")

#: This module names the retired things to prove they are gone; scanning it
#: would find only the needles.
SELF = Path(__file__).resolve()

#: Frozen with the manual: the generators produce the document README keeps on
#: disk and out of scope, so their installer chapters are not living references.
FROZEN_TREES = (Path("tools/manual"), Path("docs/manual"))

#: Compiled catalogs and assets; not prose a person reads.
BINARY_SUFFIXES = frozenset({".mo", ".png", ".ico", ".pt", ".pdf", ".jpg", ".jpeg", ".zip"})

#: Retired name -> the pattern that proves it came back.
RETIRED = {
    "install/": re.compile(r"\binstall/"),
    # The root manifest, not the offline wheelhouse's own lock export
    # (``wheelhouse/requirements.txt``, ``lock-requirements.txt``).
    "requirements.txt": re.compile(r"(?<![\w/-])requirements\.txt"),
    "LEEME": re.compile(r"LEEME"),
    "name_updater": re.compile(r"\bname_updater\b"),
    "launch_updater": re.compile(r"\blaunch_updater\b"),
    "polyx.gui.updates": re.compile(r"\b(?:polyx\.)?gui\.updates\b|\bupdates\.py\b"),
    "SearchUpdates": re.compile(r"\bSearchUpdates\b"),
    "can_update": re.compile(r"\bcan_update\b"),
    "tools/packaging": re.compile(r"\btools/packaging\b|\bpolyx\.spec\b"),
    "PyInstaller": re.compile(r"\bPyInstaller\b|\bpyinstaller\b|frozen application"),
    "SETUP.bat": re.compile(r"\bSETUP\.bat\b"),
    "test_install": re.compile(r"\btest_install\b"),
    "test_updater_observability": re.compile(r"\btest_updater_observability\b"),
}


def _offline_manifest(line: str) -> bool:
    """True for the wheelhouse's own lock export, a different artifact.

    ``tools/hpc/build_wheelhouse.py`` writes ``wheelhouse/requirements.txt``
    and ``lock-requirements.txt`` from ``uv.lock``; neither is the retired
    end-user manifest, even when the path is built with ``/``.
    """
    return "wheelhouse" in line or "lock-requirements" in line


def findings(text: str) -> list[str]:
    """Retired names present in ``text``, in dictionary order."""
    found = {
        name
        for line in text.splitlines()
        for name, pattern in RETIRED.items()
        if pattern.search(line) and not (name == "requirements.txt" and _offline_manifest(line))
    }
    return [name for name in RETIRED if name in found]


def _under(relative: Path, trees: tuple[Path, ...]) -> bool:
    """True when ``relative`` is inside one of ``trees``."""
    return any(relative == tree or tree in relative.parents for tree in trees)


def _scanned(path: Path) -> bool:
    """True when a file belongs to a living surface the scan must read."""
    relative = path.relative_to(ROOT)
    if path.resolve() == SELF:
        return False
    if _under(relative, (*FROZEN_TREES, *HISTORY_TREES)):
        return False
    if "__pycache__" in relative.parts:
        return False
    return path.suffix not in BINARY_SUFFIXES


def _living_files() -> list[Path]:
    """Every file on a living surface, sorted and without duplicates."""
    files: set[Path] = {ROOT / name for name in DOCUMENTS}
    files.update(ROOT / name for name in FILES)
    for tree in (*SOURCE_TREES, *DOCUMENT_TREES):
        base = ROOT / tree
        if base.is_dir():
            files.update(candidate for candidate in base.rglob("*") if candidate.is_file())
    return sorted(path for path in files if path.is_file() and _scanned(path))


def test_the_retired_paths_are_gone() -> None:
    for name in RETIRED_PATHS:
        assert not (ROOT / name).exists(), f"{name} came back with the retired track"


def test_the_data_packager_moved_beside_the_data() -> None:
    """It packages a dataset, not the software, so it outlives ``install/``."""
    assert (ROOT / "tools" / "data" / "empaquetar_dataset_para_drive.bat").is_file()
    assert not (ROOT / "install" / "empaquetar_dataset_para_drive.bat").exists()


def test_the_frozen_manual_stays_on_disk() -> None:
    """Kept, not regenerated: README calls it out of scope until after the paper."""
    for name in ("Manual_PolyX.html", "Manual_PolyX.en.html"):
        manual = ROOT / "docs" / "manual" / name
        assert manual.is_file(), f"{name} was deleted with the track it documents"
        assert manual.stat().st_size > 0, f"{name} is empty"


def test_no_living_surface_names_a_retired_thing() -> None:
    offenders: list[str] = []
    for path in _living_files():
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        offenders.extend(f"{relative}: {name}" for name in findings(text))
    assert not offenders, "living surfaces name the retired track:\n" + "\n".join(offenders)


def test_the_scan_covers_every_living_surface() -> None:
    """Anti-vacuity: an empty tree would make the scan pass by saying nothing."""
    scanned = {path.relative_to(ROOT).as_posix() for path in _living_files()}
    for tree in SOURCE_TREES:
        assert (ROOT / tree).is_dir(), f"{tree} does not exist"
        assert any(name.startswith(f"{tree}/") for name in scanned), f"{tree} was not scanned"
    for document in DOCUMENTS:
        assert document in scanned, f"{document} was not scanned"
    for name in ("src/polyx/gui/launcher.py", "tests/test_identity.py", "pyproject.toml"):
        assert name in scanned, f"{name} was not scanned"
    assert any(name.startswith("docs/") for name in scanned), "docs/ was not scanned"


def test_the_scanner_catches_each_retired_name() -> None:
    samples = {
        "install/": "The updater lives in install/update.py.",
        "requirements.txt": "The pins live in requirements.txt.",
        "LEEME": "Open LEEME.txt after installing.",
        "name_updater": "UPDATER = ROOT / name_updater()",
        "launch_updater": "launch_updater(UPDATER)",
        "polyx.gui.updates": "from polyx.gui.updates import can_update",
        "SearchUpdates": "SearchUpdates(self).lookfor()",
        "can_update": "if not can_update(): return",
        "tools/packaging": "The spec is tools/packaging/polyx.spec.",
        "PyInstaller": "The PyInstaller bundle was never signed.",
        "SETUP.bat": "Reinstall with SETUP.bat detecting the GPU.",
        "test_install": "uv run pytest tests/test_install.py",
        "test_updater_observability": "tests/test_updater_observability.py",
    }
    for name, sample in samples.items():
        assert name in findings(sample), f"the scanner missed {name!r} in {sample!r}"


def test_the_scanner_ignores_the_offline_wheelhouse_manifest() -> None:
    assert findings("uv pip install --no-index -r wheelhouse/requirements.txt") == []
    assert findings("assert writes[0].endswith('lock-requirements.txt')") == []


def test_the_scanner_does_not_fire_on_current_vocabulary() -> None:
    current = (
        "A collaborator runs `uv sync --locked --extra gui`; a compute node runs "
        "the Apptainer image. PyPI publication is deferred. The installer is a "
        "retired term."
    )
    assert findings(current) == []
