"""Ticket #122: the Compare table's F1 is the run's decision, and it says which.

The summary table used to divide the counters inline under an unlabelled "F1"
header, next to pages that meant something else by the same word. It now reads
``Score.from_decision`` under the localisation criterion, and its header names
that criterion.
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

from polyx.core.detection_run.operating_point import Criterion, Score  # noqa: E402
from polyx.gui.detector.pages.compare import ComparePage  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402
from tests.match_fixtures import misclassified_run  # noqa: E402


@pytest.fixture
def english():
    from polyx.core import i18n

    previous = i18n.language()
    assert i18n.set_language("en")
    yield
    i18n.set_language(previous)


def test_compare_summary_names_its_criterion(qapp, tmp_path, english):
    run = misclassified_run(tmp_path)
    result = run.results[0][0]
    state = DetectorState()
    state.model_slots[0].alias = run.models[0].alias
    state.model_slots[0].path = run.models[0].path
    state.params = run.params
    state.results = run.results
    page = ComparePage(state)
    page.refresh()

    localisation = Score.from_decision([result], Criterion.LOCALISATION, run.params.conf)
    assert page.tbl_summary.horizontalHeaderItem(7).text() == "F1 (localisation)"
    assert page.tbl_summary.item(0, 4).text() == str(localisation.tp)
    assert page.tbl_summary.item(0, 5).text() == str(localisation.fp)
    assert page.tbl_summary.item(0, 6).text() == str(localisation.fn)
    assert page.tbl_summary.item(0, 7).text() == f"{localisation.f1:.3f}"
