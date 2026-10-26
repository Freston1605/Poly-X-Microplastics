"""Contract test: every polyx docstring reads English.

Encodes the English-only docstring convention as a pytest gate instead of
review discipline: every module/class/function docstring under ``polyx/`` must
contain no Spanish remnants outside inline-code backtick spans. The scanner
lives in ``tests/english_scan.py`` and is plain ``ast``, so the convention
stays enforced without carrying the one-time translation codemod.

``legacy/``, ``agent/``, ``tests/`` and user-facing scripts are out of scope
by construction: only ``polyx/**/*.py`` is walked.

Spanish is allowed to survive only inside backtick spans -- quoted UI literals
that the runtime tr() layer still serves in Spanish -- and via the explicit
module-level ``TEST_ALLOWLIST`` below. If a legitimate token trips the scan,
add it to ``TEST_ALLOWLIST``; never loosen the scanner's regex.

Anti-vacuity guards: a synthetic Spanish docstring must be flagged by the same
scanner used on the real package, the walked inventory must exceed a floor
(> 300 docstrings across enough files) and must include known real modules,
and a backticked Spanish literal must pass while the same text unbackticked
fails.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.english_scan import (
    _strip_inline_code,
    enumerate_docstrings,
    spanish_findings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tokens the scanner flags but that are legitimate in polyx docstrings.
# Extend this set, never loosen the scanner regex.
TEST_ALLOWLIST: frozenset = frozenset()

# Known real modules the walk must see (anti-vacuity: proves the scan is
# not silently passing over an empty or truncated file set).
_KNOWN_MODULES = [
    "src/polyx/core/measure/morphology.py",
    "src/polyx/gui/report_html.py",
]

# Floor for the docstring inventory (measured 410 when the convention was
# adopted; the floor only guards against the walk degenerating to near-nothing).
_DOCSTRING_FLOOR = 300
_FILE_FLOOR = 50


def _polyx_files() -> list[Path]:
    """Every ``*.py`` file under the polyx package, sorted for stability."""
    return sorted((REPO_ROOT / "src" / "polyx").rglob("*.py"))


def _allowed(findings: list[str]) -> list[str]:
    """Findings whose flagged token is in TEST_ALLOWLIST are dropped."""
    kept = []
    for finding in findings:
        _, _, token = finding.partition(":")
        if token.lower() not in TEST_ALLOWLIST:
            kept.append(finding)
    return kept


def _scan_docstring(content: str) -> list[str]:
    """Spanish findings for one docstring, backtick spans stripped first."""
    prose = _strip_inline_code(content)
    return _allowed(spanish_findings(prose))


@pytest.fixture(scope="module")
def inventory() -> list[tuple[Path, list[tuple[int, str]]]]:
    """Parse every polyx file once and enumerate its docstrings."""
    return [
        (path, enumerate_docstrings(path.read_text(encoding="utf-8")))
        for path in _polyx_files()
    ]


def test_polyx_package_walk_finds_no_spanish_docstrings(
    inventory: list[tuple[Path, list[tuple[int, str]]]],
) -> None:
    """Main contract: zero Spanish markers across all polyx docstrings."""
    offenders: list[str] = []
    for path, docs in inventory:
        for index, content in docs:
            findings = _scan_docstring(content)
            if findings:
                first = content.splitlines()[0] if content else ""
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()} #{index} {first!r}: {findings}")
    assert not offenders, "Spanish docstring remnants in polyx:\n" + "\n".join(offenders)


def test_inventory_exceeds_floor_and_covers_known_modules(
    inventory: list[tuple[Path, list[tuple[int, str]]]],
) -> None:
    """Anti-vacuity: the scan must actually see a substantial package."""
    rel_paths = {p.relative_to(REPO_ROOT).as_posix() for p, _ in inventory}
    total_docs = sum(len(docs) for _, docs in inventory)
    assert len(inventory) > _FILE_FLOOR, f"only {len(inventory)} polyx files walked"
    assert total_docs > _DOCSTRING_FLOOR, f"only {total_docs} docstrings scanned"
    for known in _KNOWN_MODULES:
        assert known in rel_paths, f"known module missing from the walk: {known}"


def test_scanner_flags_synthetic_spanish_docstring(tmp_path: Path) -> None:
    """Anti-vacuity: a Spanish docstring must be caught by the same path."""
    module = tmp_path / "synthetic.py"
    module.write_text(
        '"""Módulo de prueba: la detección de partículas genera imágenes."""\n',
        encoding="utf-8",
    )
    docs = enumerate_docstrings(module.read_text(encoding="utf-8"))
    assert len(docs) == 1
    findings = _scan_docstring(docs[0][1])
    assert findings, "scanner failed to flag a synthetic Spanish docstring"
    assert any(f.startswith("spanish-word:") for f in findings)
    assert any(f.startswith("accented:") for f in findings)


def test_backticked_spanish_literal_passes_but_bare_text_fails(
    tmp_path: Path,
) -> None:
    """The quoted-UI allowance: backticks suppress the scan, bare text not."""
    backticked = '"""Review annotations with the `Etiquetar` button."""\n'
    bare = '"""Review annotations with the Etiquetar button."""\n'
    module = tmp_path / "x.py"
    module.write_text(backticked, encoding="utf-8")
    docs = enumerate_docstrings(module.read_text(encoding="utf-8"))
    assert len(docs) == 1
    assert _scan_docstring(docs[0][1]) == [], "backticked UI literal must be allowed"
    docs_bare = enumerate_docstrings(bare)
    assert _scan_docstring(docs_bare[0][1]), "same text unbackticked must be flagged"


def test_allowlist_drops_only_listed_tokens() -> None:
    """The allowlist filter must drop exactly the tokens it names."""
    synthetic = ["spanish-word:del", "accented:Detección"]
    assert _allowed(synthetic) == synthetic, "empty allowlist must drop nothing"
