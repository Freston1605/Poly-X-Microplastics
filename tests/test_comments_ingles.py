"""Contract test: every polyx comment reads English (M006 S03).

Encodes the M006 language convention as a pytest gate instead of review
discipline: every comment under ``polyx/`` must contain no Spanish remnants
outside inline-code backtick spans, using the exact same scanner as the
codemod harness (``tests/translate_code_text.py``) so there is a single
source of truth for what counts as Spanish.

``legacy/``, ``agent/``, ``tests/`` and the user-facing scripts are out of
scope by construction (D021): only ``polyx/**/*.py`` is walked.

Spanish is allowed to survive only inside backtick spans -- quoted UI
literals and identifiers that the runtime tr() layer still serves in
Spanish (S05 scope) -- and via the explicit module-level ``TEST_ALLOWLIST``
below. If a legitimate token trips the scan, add it to ``TEST_ALLOWLIST``
and explain why; never loosen the scanner's regex.

The scan is not enough on its own: the detector cannot see every ASCII-only
Spanish sentence, so the S03 closeout also included a manual survivor sweep
of every comment the detector left unflagged (zero survivors found).

Anti-vacuity guards (P028): a synthetic unaccented Spanish comment must be
flagged by the same scanner used on the real package, a backticked Spanish
literal must pass while the same text bare fails, protected directives must
stay invisible to the scan, code-shaped identifiers must not be read as
prose, the walked inventory must exceed a floor and include known real
modules, and every module must round-trip through LibCST byte for byte
(D024: the HEAD audit goes vacuous once the orchestrator commits).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import libcst as cst
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The codemod harness is loaded from file path (``tests/`` layout makes a
# plain import awkward, and the driver itself loads its cores the same way).
_RUTA_HARNESS = Path(__file__).resolve().parent / "translate_code_text.py"
_spec = importlib.util.spec_from_file_location("translate_code_text", _RUTA_HARNESS)
tct = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("translate_code_text", tct)
_spec.loader.exec_module(tct)

# Tokens the scanner flags but that are legitimate in polyx comments.
# Extend this set, never loosen the scanner regex. Kept separate from the
# harness ALLOWLIST so this gate can tighten without touching the codemod.
TEST_ALLOWLIST: frozenset = frozenset()

# Known real modules the walk must see (anti-vacuity: proves the scan is
# not silently passing over an empty or truncated file set).
_KNOWN_MODULES = [
    "src/polyx/core/measure/morphology.py",
    "src/polyx/gui/report_html.py",
    "src/polyx/gui/detector/runner.py",
]

# Floor for the comment inventory (measured 1890 comments across 72 modules
# at S03 planning; the floor only guards against the walk degenerating).
_COMMENT_FLOOR = 1500
_FILE_FLOOR = 50

# The polyx package had 72 modules when the slice closed. The walk must not
# shrink below that, but adding a new English-commented module is allowed.
_MODULE_FLOOR = 72


def _polyx_files() -> list[Path]:
    """Every ``*.py`` file under the polyx package, sorted for stability."""
    return sorted((REPO_ROOT / "src" / "polyx").rglob("*.py"))


def _read_source(path: Path) -> str:
    """Read a module preserving its original line endings (no newline translation)."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _allowed(findings: list[str]) -> list[str]:
    """Findings whose flagged token is in TEST_ALLOWLIST are dropped."""
    kept = []
    for finding in findings:
        _, separator, token = finding.partition(":")
        if separator and token.lower() in TEST_ALLOWLIST:
            continue
        kept.append(finding)
    return kept


def _scan_comment_body(body: str, identifiers: tuple[str, ...] = ()) -> list[str]:
    """Spanish findings for one comment body, backtick spans stripped first."""
    return _allowed(tct.spanish_comment_findings(body, identifiers))


@pytest.fixture(scope="module")
def inventory() -> list[tuple[Path, str, list, set]]:
    """Read, tokenize and index every polyx module once."""
    modules = []
    for path in _polyx_files():
        source = _read_source(path)
        modules.append(
            (
                path,
                source,
                tct._comment_tokens(source),
                tct._identifiers_from_source(source),
            )
        )
    return modules


def test_polyx_package_walk_finds_no_spanish_comments(
    inventory: list[tuple[Path, str, list, set]],
) -> None:
    """Main contract: zero Spanish markers across all polyx comments."""
    offenders: list[str] = []
    for path, _source, tokens, identifiers in inventory:
        relative = path.relative_to(REPO_ROOT).as_posix()
        for token in tokens:
            if tct.tc.is_protected_comment(token.string):
                continue
            _prefix, body, _trailing = tct.tc.split_comment(token.string)
            if not body.strip():
                continue
            findings = _scan_comment_body(body, tuple(identifiers))
            if findings:
                offenders.append(
                    f"{relative}:{token.start[0]} {findings} :: {token.string.strip()!r}"
                )
        for block in tct.comment_blocks(tokens):
            bodies = [
                tct.tc.split_comment(item.string)[1]
                for item in block
                if not tct.tc.is_protected_comment(item.string)
            ]
            if len(bodies) < 2:
                continue
            findings = _scan_comment_body(" ".join(bodies), tuple(identifiers))
            if findings:
                label = f"{block[0].start[0]}-{block[-1].start[0]}"
                offenders.append(f"{relative}:{label} comment block {findings}")
    assert not offenders, "Spanish comment remnants in polyx:\n" + "\n".join(offenders)


def test_inventory_exceeds_floor_and_covers_known_modules(
    inventory: list[tuple[Path, str, list, set]],
) -> None:
    """Anti-vacuity: the scan must actually see a substantial package."""
    relative_paths = {path.relative_to(REPO_ROOT).as_posix() for path, *_ in inventory}
    total_comments = sum(len(tokens) for _path, _source, tokens, _ids in inventory)
    assert len(inventory) >= _MODULE_FLOOR, f"only {len(inventory)} polyx modules walked"
    assert len(relative_paths) > _FILE_FLOOR, f"only {len(relative_paths)} polyx files walked"
    assert total_comments > _COMMENT_FLOOR, f"only {total_comments} comments scanned"
    for known in _KNOWN_MODULES:
        assert known in relative_paths, f"known module missing from the walk: {known}"


def test_scanner_flags_synthetic_unaccented_spanish_comment() -> None:
    """Anti-vacuity: the ASCII-only Spanish path must still be flagged."""
    source = "# la mala clasificacion penaliza en ambos lados.\nVALUE = 1\n"
    findings = tct.scan_comments(source)
    assert findings, "scanner failed to flag an unaccented Spanish comment"
    markers = [marker for _line, items in findings for marker in items]
    assert any(marker.startswith("comment-word:clasificacion") for marker in markers)
    assert any(marker.startswith("stopword-ratio:") for marker in markers)


def test_backticked_spanish_literal_passes_but_bare_text_fails() -> None:
    """The quoted-UI allowance: backticks suppress the scan, bare text not."""
    backticked = "Review annotations with the `Etiquetar` button."
    bare = "Review annotations with the Etiquetar button."
    assert _scan_comment_body(backticked) == [], "backticked UI literal must be allowed"
    assert _scan_comment_body(bare), "same text unbackticked must be flagged"


def test_protected_directives_produce_no_findings() -> None:
    """Directives and metadata comments are never treated as prose."""
    source = (
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
        "first = 1  # noqa: F401\n"
        "second = 2  # type: ignore[attr-defined]\n"
    )
    assert tct.scan_comments(source) == []
    for raw in (
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        "# noqa: F401",
        "# type: ignore[attr-defined]",
    ):
        assert tct.tc.is_protected_comment(raw), f"not protected: {raw!r}"


def test_code_shaped_identifiers_are_not_prose() -> None:
    """Constants such as SECCIONES_INFORME are code, not Spanish prose."""
    body = "The report reads SECCIONES_INFORME and TAMANO_TOTAL from the module."
    assert _scan_comment_body(body) == []
    assert _scan_comment_body(
        body, ("SECCIONES_INFORME", "TAMANO_TOTAL")
    ) == []


def test_every_polyx_module_round_trips_through_libcst(
    inventory: list[tuple[Path, str, list, set]],
) -> None:
    """Formatting proof (D024): the codemod never rewrote code or layout."""
    mismatched = [
        path.relative_to(REPO_ROOT).as_posix()
        for path, source, _tokens, _identifiers in inventory
        if cst.parse_module(source).code != source
    ]
    assert not mismatched, "LibCST round-trip changed these modules:\n" + "\n".join(
        mismatched
    )


def test_allowlist_drops_only_listed_tokens() -> None:
    """The allowlist filter must drop exactly the tokens it names."""
    synthetic = ["spanish-word:del", "accented:Deteccion"]
    assert _allowed(synthetic) == synthetic, "empty allowlist must drop nothing"
