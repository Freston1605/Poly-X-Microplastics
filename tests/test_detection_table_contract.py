"""Ticket #122: one detection table, whichever door it leaves through.

`detections.csv` is a schema (ADR-0004): its headers are fixed English tokens,
because a script that reads the file must not need to know the interface
language of whoever wrote it. That contract already lived in
``DETECTION_COLUMNS`` for the run folder, but the Results page and the Viewer
exported the same kind of row under Spanish headers, so the same detection set
became three different files.

The page and the Viewer now export named subsets of the one vocabulary. This
module pins the three headers and scans for the regression: a CSV row literal
that spells a Spanish header.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core.detection_run.schema import (  # noqa: E402
    DETECTION_COLUMNS,
    PAGE_COLUMNS,
    VIEWER_COLUMNS,
)
from polyx.gui.detector.pages.results import ResultsPage  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402
from polyx.gui.viewer.export import csv_rows  # noqa: E402
from tests.match_fixtures import misclassified_result  # noqa: E402

UM = 2.0

#: Spanish header tokens the pre-#122 exports used. None may survive in a CSV.
SPANISH_HEADER_TOKENS = frozenset(
    {
        "modelo",
        "imagen",
        "clase",
        "talla_origen",
        "origen_escala",
        "largo_um",
        "metodo_largo",
        "geodesico_um",
        "largo_rect_eq_um",
        "ancho_um",
        "aspecto",
        "curvatura",
        "morfotipo",
    }
)

_CANONICAL_TOKEN = re.compile(r"[a-z][a-z0-9_]*")


def _page_with_one_result(tmp_path: Path) -> tuple[ResultsPage, object]:
    state = DetectorState()
    state.model_slots[0].alias = "m0"
    state.model_slots[0].path = tmp_path / "m0.pt"
    result = misclassified_result(tmp_path / "plate.png")
    state.results = {0: [result]}
    return ResultsPage(state), result


# ── The three headers ────────────────────────────────────────────────────


def test_page_export_headers_are_canonical(qapp, tmp_path):
    page, result = _page_with_one_result(tmp_path)

    rows = page.export_rows()

    assert rows[0] == list(PAGE_COLUMNS)
    # The page's table is a shorter view of the canonical one: same names, same
    # relative order, no second vocabulary.
    assert set(PAGE_COLUMNS) < set(DETECTION_COLUMNS)
    assert list(PAGE_COLUMNS) == [token for token in DETECTION_COLUMNS if token in PAGE_COLUMNS]
    first = dict(zip(rows[0], rows[1], strict=True))
    assert first["model"] == "m0"
    assert first["image"] == str(result.image_path)
    assert first["class"] == "PET"
    assert first["conf"] == pytest.approx(0.9)


def test_viewer_export_headers_are_canonical():
    header = csv_rows([], UM)[0]

    assert header == list(VIEWER_COLUMNS)
    # Everything the Viewer shares with the run schema uses the run's token;
    # `diam_px` is the documented box descriptor that only the Viewer has.
    assert set(header) - {"diam_px"} <= set(DETECTION_COLUMNS)
    assert "talla_origen" not in header


def test_the_published_headers_are_ascii_english_tokens():
    for header in (DETECTION_COLUMNS, PAGE_COLUMNS, VIEWER_COLUMNS):
        for token in header:
            assert token.isascii(), token
            assert _CANONICAL_TOKEN.fullmatch(token), token
            assert token not in SPANISH_HEADER_TOKENS, token


# ── ADR-0004 scan ────────────────────────────────────────────────────────


def _string_literals(node: ast.AST) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _spanish_header_findings(root: Path) -> list[str]:
    """Spanish header tokens used as literals inside a ``writerow`` call.

    The headers themselves are named constants now, and the runtime tests above
    pin their contents; this scan catches the other regression, a page that
    grows a literal header row of its own.
    """
    findings: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr != "writerow":
                continue
            findings.extend(
                f"{path}:{node.lineno}: {literal}"
                for literal in _string_literals(node)
                if literal in SPANISH_HEADER_TOKENS
            )
    return findings


def test_the_scan_catches_a_spanish_header_row(tmp_path):
    """The scan is only worth its place if it fails on the pre-#122 shape."""
    module = tmp_path / "export.py"
    module.write_text(
        'def write(w):\n    w.writerow(["modelo", "imagen", "clase"])\n',
        encoding="utf-8",
    )

    findings = _spanish_header_findings(tmp_path)

    assert findings == [
        f"{module}:2: modelo",
        f"{module}:2: imagen",
        f"{module}:2: clase",
    ]


def test_no_csv_writer_spells_a_spanish_header():
    assert _spanish_header_findings(_SRC / "polyx") == []
