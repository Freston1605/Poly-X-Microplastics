"""Contract test: every polyx docstring reads English (M006 S02).

Encodes the M006 language convention as a pytest gate instead of review
discipline: every module/class/function docstring under ``polyx/`` must
contain no Spanish remnants outside inline-code backtick spans, using the
exact same scanner as the codemod harness (``tests/translate_code_text.py``)
so there is a single source of truth for what counts as Spanish.

``legacy/``, ``agent/``, ``tests/`` and user-facing scripts are out of scope
by construction (D021): only ``polyx/**/*.py`` is walked.

Spanish is allowed to survive only inside backtick spans — quoted UI
literals that the runtime tr() layer still serves in Spanish (S05 scope) —
and via the explicit module-level ``TEST_ALLOWLIST`` below. If a legitimate
token trips the scan, add it to ``TEST_ALLOWLIST``; never loosen the
scanner's regex.

Anti-vacuity guards (P028): a synthetic Spanish docstring must be flagged
by the same scanner used on the real package, the walked inventory must
exceed a floor (> 300 docstrings across enough files) and must include
known real modules, and a backticked Spanish literal must pass while the
same text unbackticked fails.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The codemod harness is loaded from file path (``tests/`` layout makes a
# plain import awkward, and the driver itself loads its core the same way).
_RUTA_HARNESS = Path(__file__).resolve().parent / "translate_code_text.py"
_spec = importlib.util.spec_from_file_location("translate_code_text", _RUTA_HARNESS)
tct = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("translate_code_text", tct)
_spec.loader.exec_module(tct)

# Tokens the scanner flags but that are legitimate in polyx docstrings.
# Extend this set, never loosen the scanner regex. Kept separate from the
# harness ALLOWLIST so this gate can tighten without touching the codemod.
TEST_ALLOWLIST: frozenset = frozenset()

# Known real modules the walk must see (anti-vacuity: proves the scan is
# not silently passing over an empty or truncated file set).
_KNOWN_MODULES = [
    "polyx/core/measure/morphology.py",
    "polyx/gui/report_html.py",
]

# Floor for the docstring inventory (measured 410 at S02 planning; the
# floor only guards against the walk degenerating to near-nothing).
_DOCSTRING_FLOOR = 300
_FILE_FLOOR = 50


def _polyx_files() -> list[Path]:
    """Every ``*.py`` file under the polyx package, sorted for stability."""
    return sorted((REPO_ROOT / "polyx").rglob("*.py"))


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
    prose = tct._quitar_inline_code(content)
    return _allowed(tct.hallazgos_espanoles(prose))


@pytest.fixture(scope="module")
def inventory() -> list[tuple[Path, list[tuple[int, str]]]]:
    """Parse every polyx file once and enumerate its docstrings."""
    return [
        (path, tct.enumerar_docstrings(path.read_text(encoding="utf-8")))
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
    docs = tct.enumerar_docstrings(module.read_text(encoding="utf-8"))
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
    docs = tct.enumerar_docstrings(module.read_text(encoding="utf-8"))
    assert len(docs) == 1
    assert _scan_docstring(docs[0][1]) == [], "backticked UI literal must be allowed"
    docs_bare = tct.enumerar_docstrings(bare)
    assert _scan_docstring(docs_bare[0][1]), "same text unbackticked must be flagged"


def test_allowlist_drops_only_listed_tokens() -> None:
    """The allowlist filter must drop exactly the tokens it names."""
    synthetic = ["spanish-word:del", "accented:Detección"]
    assert _allowed(synthetic) == synthetic, "empty allowlist must drop nothing"
