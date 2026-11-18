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

from polyx.core.detect.yolo import Detection  # noqa: E402
from polyx.core.detection_run import ImageResult, RunResult  # noqa: E402
from polyx.core.detection_run.config import CalibrationPolicy  # noqa: E402
from polyx.core.detection_run.operating_point import Criterion, Score  # noqa: E402
from polyx.core.measure.calibration import (  # noqa: E402
    ORIGIN_INDEX,
    ORIGIN_MANUAL,
    ORIGIN_NONE,
    Calibration,
)
from polyx.gui.detector.pages.results import ResultsPage  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402
from tests.match_fixtures import (  # noqa: E402
    matched_result,
    matched_run,
    misclassified_result,
    misclassified_run,
)


def _page_for(state: DetectorState) -> ResultsPage:
    page = ResultsPage(state)
    page.refresh()
    return page


def _sized_result(
    image: Path, diam_um: float | None, *, class_name: str = "PET", model_idx: int = 0
) -> ImageResult:
    """One result whose single prediction carries (or lacks) a micrometre size."""
    prediction = Detection(
        class_id=0,
        class_name=class_name,
        conf=0.9,
        x1=10.0,
        y1=10.0,
        x2=20.0,
        y2=20.0,
        diam_um=diam_um,
    )
    return ImageResult(image_path=image, model_idx=model_idx, predictions=[prediction])


def _state_with_slot(tmp_path: Path) -> DetectorState:
    state = DetectorState()
    state.model_slots[0].alias = "m0"
    state.model_slots[0].path = tmp_path / "m0.pt"
    return state


# ── #123: the histogram asks the data, not the policy ────────────────────


def test_auto_calibration_with_sized_particles_shows_the_histogram(qapp, tmp_path):
    """In ``auto`` the policy's um/px is 0, but each photo has a valid scale."""
    image = tmp_path / "photos" / "plate.png"
    state = _state_with_slot(tmp_path)
    state.calibration = CalibrationPolicy(mode="auto", um_per_px=0.0)
    state.run_result = RunResult(
        run_dir=tmp_path / "run",
        calibrations={image: Calibration(um_por_px=31.0, origin=ORIGIN_INDEX)},
    )
    state.results = {0: [_sized_result(image, 42.5)]}

    page = _page_for(state)

    assert page._hist_canvas is not None
    assert not page._hist_canvas.isHidden()
    assert page._hist_placeholder.isHidden()


def test_sized_particles_show_the_histogram_even_without_a_run_record(qapp, tmp_path):
    """The gate is the data: a prediction with ``diam_um`` is enough to draw."""
    image = tmp_path / "photos" / "plate.png"
    state = _state_with_slot(tmp_path)
    state.run_result = RunResult(run_dir=tmp_path / "run")
    state.results = {0: [_sized_result(image, 12.0)]}

    page = _page_for(state)

    assert page._hist_canvas is not None
    assert not page._hist_canvas.isHidden()


def test_no_calibration_and_no_sized_detections_hide_the_histogram(qapp, tmp_path):
    image = tmp_path / "photos" / "plate.png"
    state = _state_with_slot(tmp_path)
    state.run_result = RunResult(
        run_dir=tmp_path / "run",
        calibrations={image: Calibration(origin=ORIGIN_MANUAL)},
    )
    state.results = {0: [_sized_result(image, None)]}

    page = _page_for(state)

    assert page._hist_canvas is None or page._hist_canvas.isHidden()
    assert not page._hist_placeholder.isHidden()


# ── #123: every reader asks the run by path ─────────────────────────────


def _same_name_run(tmp_path: Path) -> tuple[Path, Path, Path, RunResult]:
    """Two ``plate.png`` in different folders with different scales, plus a third."""
    first = tmp_path / "tramo_a" / "plate.png"
    second = tmp_path / "tramo_b" / "plate.png"
    ambiguous = tmp_path / "tramo_c" / "plate.png"
    run = RunResult(
        run_dir=tmp_path / "run",
        calibrations={
            first: Calibration(um_por_px=2.0, origin=ORIGIN_MANUAL),
            second: Calibration(um_por_px=3.0, origin=ORIGIN_MANUAL),
        },
    )
    return first, second, ambiguous, run


def _column(rows: list[list[object]], name: str) -> int:
    return list(rows[0]).index(name)


def test_the_csv_export_gives_each_photo_its_own_scale(qapp, tmp_path):
    first, second, _ambiguous, run = _same_name_run(tmp_path)
    state = _state_with_slot(tmp_path)
    state.run_result = run
    state.results = {
        0: [
            _sized_result(first, 20.0),
            _sized_result(second, 30.0),
        ]
    }

    rows = _page_for(state).export_rows()

    image_col, scale_col = _column(rows, "image"), _column(rows, "um_per_px")
    by_image = {Path(row[image_col]): row for row in rows[1:]}
    assert by_image[first][scale_col] == pytest.approx(2.0)
    assert by_image[second][scale_col] == pytest.approx(3.0)


def test_an_ambiguous_basename_leaves_the_export_scale_blank(qapp, tmp_path):
    _first, _second, ambiguous, run = _same_name_run(tmp_path)
    state = _state_with_slot(tmp_path)
    state.run_result = run
    state.results = {0: [_sized_result(ambiguous, 10.0)]}
    state.calibration.um_per_px = 2.0  # a policy fallback must not fill the gap

    rows = _page_for(state).export_rows()

    row = rows[1]
    assert row[_column(rows, "um_per_px")] == ""
    assert row[_column(rows, "calibration_origin")] == ORIGIN_NONE


def test_the_report_calibration_summary_agrees_with_the_page(qapp, tmp_path):
    """The run record and the page describe the same per-photo scales."""
    from polyx.core.detection_run.schema import build_run_payload

    first, second, _ambiguous, run = _same_name_run(tmp_path)
    state = _state_with_slot(tmp_path)
    state.run_result = run
    state.results = {0: [_sized_result(first, 20.0), _sized_result(second, 30.0)]}

    rows = _page_for(state).export_rows()
    scale_col = _column(rows, "um_per_px")
    page_scales = [row[scale_col] for row in rows[1:]]
    summary = build_run_payload(run)["calibration_summary"]

    assert page_scales == pytest.approx([2.0, 3.0])
    assert summary["n"] == len(page_scales) == 2
    assert summary["origins"] == {ORIGIN_MANUAL: 2}


class _FakeSignal:
    def emit(self, *_args) -> None:
        pass


class _FakeViewerState:
    def __init__(self) -> None:
        self.um_per_px = 0.0
        self.detections: list = []
        self.detections_changed = _FakeSignal()

    def load_single(self, path: Path) -> None:
        self.image = path


class _FakeViewer:
    """Stands in for ``ViewerWindow`` so the test can read the scale it got."""

    instances: list[_FakeViewer] = []

    def __init__(self) -> None:
        self.state = _FakeViewerState()
        self.shown = False
        type(self).instances.append(self)

    def show(self) -> None:
        self.shown = True

    def raise_(self) -> None:
        pass


def test_open_in_viewer_hands_each_photo_its_own_scale(qapp, tmp_path, monkeypatch):
    import polyx.gui.viewer.window as viewer_window

    first, second, ambiguous, run = _same_name_run(tmp_path)
    state = _state_with_slot(tmp_path)
    state.run_result = run
    state.calibration.um_per_px = 2.0  # a policy fallback must not fill the gap
    state.results = {
        0: [
            _sized_result(first, 20.0),
            _sized_result(second, 30.0),
            _sized_result(ambiguous, 10.0),
        ]
    }
    page = _page_for(state)
    monkeypatch.setattr(viewer_window, "ViewerWindow", _FakeViewer)
    _FakeViewer.instances.clear()

    expected = (2.0, 3.0, 0.0)
    for row, scale in enumerate(expected):
        page.table.selectRow(row)
        page._review_in_viewer()
        viewer = _FakeViewer.instances[-1]
        assert viewer.state.um_per_px == pytest.approx(scale)
    assert all(viewer.shown for viewer in _FakeViewer.instances)


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


def test_results_page_renders_the_display_name_not_the_alias(qapp, tmp_path, selected_language):
    """#124: a generated alias is a folder key; the page speaks prose."""
    from PySide6.QtCore import Qt

    selected_language("en")
    state = DetectorState()
    state.model_slots[0].alias = "model_1"
    state.model_slots[0].alias_generated = True
    state.model_slots[0].path = tmp_path / "m0.pt"
    state.results = {0: [misclassified_result(tmp_path / "plate.png")]}
    page = _page_for(state)

    assert page.table.item(0, 0).text() == "Model 1"
    # The stable alias stays in UserRole: the annotated folder is named after it.
    assert page.table.item(0, 0).data(Qt.UserRole) == "model_1"
    assert "Model 1 — conf" in page.lbl_sugg.text()
    assert "model_1" not in page.lbl_sugg.text()


def test_results_page_kpi_and_advice_use_named_criteria(qapp, tmp_path, selected_language):
    """The KPI says localisation; the advice says with class; neither guesses."""
    selected_language("en")
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


def test_report_and_page_agree_on_one_fixture(qapp, tmp_path, selected_language):
    """The report's two F1 columns and the page's KPI read one decision."""
    from polyx.gui.report.sections import _section_models

    selected_language("en")
    run = misclassified_run(tmp_path)
    result = run.results[0][0]
    state = DetectorState()
    state.model_slots[0].alias = run.models[0].alias
    state.model_slots[0].path = run.models[0].path
    state.params = run.params
    state.results = run.results
    page = _page_for(state)

    localisation = Score.from_decision([result], Criterion.LOCALISATION, run.params.conf)
    with_class = Score.from_decision([result], Criterion.WITH_CLASS, run.params.conf)
    body = _section_models(run, run.results, [result], True)

    # Both figures come from the same counters and are named where they appear.
    assert localisation.f1 == pytest.approx(1.0)
    assert with_class.f1 == pytest.approx(0.5)
    assert f"<td class='r'>{localisation.f1:.3f}</td>" in body
    assert f"<td class='r'>{with_class.f1:.3f}</td>" in body
    assert "localisation" in body and "with class" in body
    assert page.kpi_f1.lbl_value.text() == f"{localisation.f1:.3f}"


def test_results_reload_gt_is_one_core_call_with_progress(qapp, tmp_path, monkeypatch):
    """The page is a progress dialog plus one call; it measures nothing itself.

    The reload's sequence -- resolve labels, measure, match, draw, republish --
    belongs to the core. If this page ever grew its own copy again, replacing
    the core function with a stub would leave the page still working.
    """
    import polyx.gui.detector.pages.results as results_page
    from polyx.core.detection_run import ReloadSummary

    state = DetectorState()
    state.model_slots[0].alias = "m0"
    state.model_slots[0].path = tmp_path / "m0.pt"
    run = matched_run(tmp_path)
    state.run_result = run
    state.results = {0: run.results[0]}
    state.gt_folder = tmp_path / "gt"
    page = ResultsPage(state)

    calls: list[tuple] = []

    def fake_reload(run_arg, *, gt_folder=None, progress=None):
        calls.append((run_arg, gt_folder, progress))
        assert progress is not None
        progress(1, 1, "plate.png")
        return ReloadSummary(
            revisited=1, changed=[Path(run_arg.images[0])], missing_gt=[], recalculated=1
        )

    monkeypatch.setattr(results_page, "reload_ground_truth", fake_reload)
    page._reload_gt()

    assert len(calls) == 1
    assert calls[0][0] is run
    assert calls[0][1] == state.gt_folder
    assert "plate.png" in page.lbl_recharge.text()
