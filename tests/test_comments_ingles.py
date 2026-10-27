"""Contract test: every polyx comment reads English.

Encodes the English-only comment convention as a pytest gate instead of review
discipline: every comment under ``polyx/`` must contain no Spanish remnants
outside inline-code backtick spans. The scanner lives in
``tests/english_scan.py`` and is plain ``ast``/``tokenize``, so the convention
stays enforced without carrying the one-time translation codemod.

``legacy/``, ``agent/``, ``tests/`` and the user-facing scripts are out of
scope by construction: only ``polyx/**/*.py`` is walked.

Spanish is allowed to survive only inside backtick spans -- quoted UI
literals and identifiers that the runtime tr() layer still serves in
Spanish -- and via the explicit module-level ``TEST_ALLOWLIST`` below. If a
legitimate token trips the scan, add it to ``TEST_ALLOWLIST`` and explain
why; never loosen the scanner's regex.

The scan is not enough on its own: the detector cannot see every ASCII-only
Spanish sentence, so the language closeout also included a manual survivor
sweep of every comment the detector left unflagged (zero survivors found).

Anti-vacuity guards: a synthetic unaccented Spanish comment must be flagged
by the same scanner used on the real package, a backticked Spanish literal
must pass while the same text bare fails, protected directives must stay
invisible to the scan, code-shaped identifiers must not be read as prose, and
the walked inventory must exceed a floor and include known real modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.english_scan import (
    _comment_tokens,
    _identifiers_from_source,
    comment_blocks,
    is_protected_comment,
    scan_comments,
    spanish_comment_findings,
    split_comment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tokens the scanner flags but that are legitimate in polyx comments.
# Extend this set, never loosen the scanner regex.
TEST_ALLOWLIST: frozenset = frozenset()

# Known real modules the walk must see (anti-vacuity: proves the scan is
# not silently passing over an empty or truncated file set).
_KNOWN_MODULES = [
    "src/polyx/core/measure/shape.py",
    "src/polyx/gui/report_html.py",
    "src/polyx/gui/detector/runner.py",
]

# Floor for the comment inventory (measured 1890 comments across 72 modules
# when the convention was adopted; the floor only guards against the walk
# degenerating).
_COMMENT_FLOOR = 1500
_FILE_FLOOR = 50

# The polyx package had 72 modules when the convention was adopted. The walk
# must not shrink below that, but adding a new English-commented module is
# allowed.
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
    return _allowed(spanish_comment_findings(body, identifiers))


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
                _comment_tokens(source),
                _identifiers_from_source(source),
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
            if is_protected_comment(token.string):
                continue
            _prefix, body, _trailing = split_comment(token.string)
            if not body.strip():
                continue
            findings = _scan_comment_body(body, tuple(identifiers))
            if findings:
                offenders.append(
                    f"{relative}:{token.start[0]} {findings} :: {token.string.strip()!r}"
                )
        for block in comment_blocks(tokens):
            bodies = [
                split_comment(item.string)[1]
                for item in block
                if not is_protected_comment(item.string)
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
    findings = scan_comments(source)
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
    assert scan_comments(source) == []
    for raw in (
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        "# noqa: F401",
        "# type: ignore[attr-defined]",
    ):
        assert is_protected_comment(raw), f"not protected: {raw!r}"


def test_code_shaped_identifiers_are_not_prose() -> None:
    """Constants such as SECCIONES_INFORME are code, not Spanish prose."""
    body = "The report reads SECCIONES_INFORME and TAMANO_TOTAL from the module."
    assert _scan_comment_body(body) == []
    assert _scan_comment_body(body, ("SECCIONES_INFORME", "TAMANO_TOTAL")) == []


def test_allowlist_drops_only_listed_tokens() -> None:
    """The allowlist filter must drop exactly the tokens it names."""
    synthetic = ["spanish-word:del", "accented:Deteccion"]
    assert _allowed(synthetic) == synthetic, "empty allowlist must drop nothing"
