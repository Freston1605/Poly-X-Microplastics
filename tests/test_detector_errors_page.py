"""Ticket #120: the Errors page lists the match the run decided.

The page used to re-match every image with box IoU at ``params.iou_tp``. A run
that matched by mask IoU then showed a different set of errors -- a false
positive and a false negative the run had never counted -- because the page
had its own rule. It now reads ``ImageResult.match``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core.detect.protocol import Instance  # noqa: E402
from polyx.core.detection_run import ImageResult  # noqa: E402
from polyx.core.detection_run.result import RunParams  # noqa: E402
from polyx.gui.detector.pages.errors import ErrorsPage  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402
from tests.match_fixtures import matched_result  # noqa: E402


def _page(image: Path) -> tuple[ErrorsPage, DetectorState]:
    state = DetectorState()
    state.model_slots[0].alias = "m0"
    state.params = RunParams()
    state.results = {0: [matched_result(image)]}
    page = ErrorsPage(state)
    page.refresh()
    return page, state


def test_errors_page_rows_equal_the_match_the_run_decided(qapp, tmp_path):
    page, state = _page(tmp_path / "plate.png")
    match = state.results[0][0].match

    assert page.table.rowCount() == (
        len(match.fp_idx) + len(match.fn_idx) + len(match.miscls_pairs)
    )
    assert page.table.rowCount() == 1
    # The one error is the stray prediction; the mask-matched true positive is
    # not an error, and no false negative is invented.
    assert page.table.item(0, 1).text() == "plate.png"
    assert page.table.item(0, 4).text() == "PP"
    assert page.table.item(0, 5).text() == "0.40"


def test_errors_page_lists_an_image_with_no_match_as_no_errors(qapp, tmp_path):
    """No match means no proven errors: the page does not invent a re-match."""
    state = DetectorState()
    state.model_slots[0].alias = "m0"
    prediction = Instance(class_id=0, class_name="PET", conf=0.9, x1=0, y1=0, x2=10, y2=10)
    gt = Instance(class_id=0, class_name="PET", conf=1.0, x1=0, y1=0, x2=10, y2=10)
    state.results = {
        0: [
            ImageResult(
                image_path=tmp_path / "plate.png",
                model_idx=0,
                predictions=[prediction],
                gt=[gt],
                has_gt=True,
                tp=1,
            )
        ]
    }
    page = ErrorsPage(state)
    page.refresh()

    assert page.table.rowCount() == 0
