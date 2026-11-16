"""Tickets #120 and #122: the Results page names the criterion it is showing.

The KPI row sums the counters the run recorded (the serialised projection of
``ImageResult.match``) under the localisation criterion, and the confidence
advice below it recommends a threshold under the with-class criterion. Both
decisions live in ``polyx.core.detection_run.operating_point`` and every figure
that reaches the page says which one it is: the two cannot diverge silently.

The confidence advice is the hypothetical sweep. It is the only place on the
page allowed to re-match, and it keeps the run's matching rule so its numbers
differ from the KPIs only in the threshold and the criterion.
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
from polyx.gui.detector.pages.results import ResultsPage  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402
from tests.match_fixtures import (  # noqa: E402
    matched_result,
    misclassified_result,
)


@pytest.fixture
def english():
    """Pin the source language: the criterion labels are the report's words."""
    from polyx.core import i18n

    previous = i18n.language()
    assert i18n.set_language("en")
    yield
    i18n.set_language(previous)


def _page_for(state: DetectorState) -> ResultsPage:
    page = ResultsPage(state)
    page.refresh()
    return page


def test_results_kpis_are_the_counters_of_the_run_s_match(qapp, tmp_path):
    """TP/FP/FN on the page equal the match the run decided."""
    state = DetectorState()
    state.model_slots[0].alias = "m0"
    state.model_slots[0].path = tmp_path / "m0.pt"
    result = matched_result(tmp_path / "plate.png")
    state.results = {0: [result]}
    page = ResultsPage(state)
    page.refresh()

    match = result.match
    assert page.kpi_tp.lbl_value.text() == str(match.tp)
    assert page.kpi_fp.lbl_value.text() == str(match.fp)
    assert page.kpi_fn.lbl_value.text() == str(match.fn)
    assert (match.tp, match.fp, match.fn) == (1, 1, 0)


def test_results_sweep_keeps_the_run_s_mask_rule(qapp, tmp_path):
    """The hypothetical sweep re-matches with the rule the run recorded.

    At the run's own confidence the sweep must reproduce the run's decision;
    matching by box here would return F1 0 instead of the mask-matched 1.
    """
    state = DetectorState()
    state.model_slots[0].alias = "m0"
    state.model_slots[0].path = tmp_path / "m0.pt"
    state.params.conf = 0.0
    result = matched_result(tmp_path / "plate.png")
    state.results = {0: [result]}
    page = ResultsPage(state)

    f1, precision, recall = page._metrics_to([result], 0.0)
    # One true positive and one false positive: box matching would find none
    # and score F1 0, so these numbers are the mask-matched ones.
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(2 / 3)


def test_results_page_kpi_and_advice_use_named_criteria(qapp, tmp_path, english):
    """The KPI says localisation; the advice says with class; neither guesses."""
    state = DetectorState()
    state.model_slots[0].alias = "m0"
    state.model_slots[0].path = tmp_path / "m0.pt"
    result = misclassified_result(tmp_path / "plate.png")
    state.results = {0: [result]}
    page = _page_for(state)

    localisation = Score.from_decision([result], Criterion.LOCALISATION, state.params.conf)
    with_class = Score.from_decision([result], Criterion.WITH_CLASS, state.params.conf)
    # The fixture disagrees between criteria on purpose: one F1 cannot be both.
    assert localisation.f1 == pytest.approx(1.0)
    assert with_class.f1 == pytest.approx(0.5)

    assert page.kpi_f1.lbl_value.text() == f"{localisation.f1:.3f}"
    assert page.kpi_f1.lbl_title.text() == "F1 (localisation)"
    assert page.lbl_prec.text() == f"Precision (localisation):  {localisation.precision:.3f}"
    assert page.lbl_rec.text() == f"Recall (localisation):  {localisation.recall:.3f}"
    assert "with class" in page.lbl_sugg.text()
    assert "localisation" not in page.lbl_sugg.text()
