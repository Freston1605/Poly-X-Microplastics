"""The report consumes the canonical ``RunResult`` (ticket #63).

``generate_report`` speaks the core model vocabulary: the active models come
from ``run.models`` in dense engine order and the per-image results from
``run.results``. The GUI report page forwards its current ``run_result``, and
only a completed or aborted run is reportable. The full-report CLI path uses
the same seam; its end-to-end test lives in ``tests/test_cli_detect.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polyx.core.detect.protocol import Instance
from polyx.core.detection_run import ImageResult, ModelInfo, RunParams, RunResult, RunStatus
from polyx.core.measure.calibration import ORIGIN_MANUAL, Calibration


def make_run(
    tmp_path: Path,
    *,
    status: RunStatus | str = RunStatus.COMPLETED,
    alias: str = "best",
) -> RunResult:
    """One image, one active model, no ground truth: the smallest real run."""
    image = tmp_path / "photos" / "plate1.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"not-read-by-the-summary-section")
    detection = Instance(
        class_id=0,
        class_name="PET",
        conf=0.9,
        x1=20.0,
        y1=20.0,
        x2=60.0,
        y2=60.0,
    )
    return RunResult(
        run_dir=tmp_path / "run",
        images=[image],
        models=[
            ModelInfo(
                alias=alias,
                path=tmp_path / f"{alias}.pt",
                names={0: "PET"},
            )
        ],
        params=RunParams(conf=0.3),
        calibrations={image: Calibration(um_por_px=2.0, origin=ORIGIN_MANUAL)},
        results={
            0: [
                ImageResult(
                    image_path=image,
                    model_idx=0,
                    predictions=[detection],
                )
            ]
        },
        status=status,
    )


def test_generate_report_accepts_a_run_result(tmp_path):
    """The canonical result is the report's input, model list included."""
    from polyx.gui.report_html import generate_report

    run = make_run(tmp_path)
    output = generate_report(run, tmp_path / "report.html", sections=["resumen", "metodos"])

    html = output.read_text(encoding="utf-8")
    assert "best" in html
    assert "PET" in html


@pytest.mark.parametrize("status", [RunStatus.COMPLETED, RunStatus.ABORTED])
def test_completed_and_aborted_runs_are_reportable(tmp_path, status):
    """A partial aborted run is still a real run with results to report."""
    from polyx.gui.report_html import generate_report

    run = make_run(tmp_path, status=status)
    assert generate_report(run, tmp_path / "report.html", sections=["resumen"]).is_file()


def test_failed_runs_are_reportable_but_visibly_marked_as_partial(tmp_path):
    """Restore the old ability to report a failed batch without hiding the failure."""
    from polyx.gui.report_html import generate_report

    run = make_run(tmp_path, status=RunStatus.FAILED)
    run.error = "RuntimeError: inference exploded"
    output = generate_report(run, tmp_path / "report.html", sections=["resumen"])

    html = output.read_text(encoding="utf-8")
    assert "failed" in html
    assert "inference exploded" in html
    assert "Incomplete batch" in html or "Lote incompleto" in html


def test_assignable_round_trips_through_detections_csv_and_the_report(tmp_path):
    """Abstention must survive load_run: the report's percentage cannot change."""
    import cv2
    import numpy as np

    from polyx.core.detection_run import (
        CalibrationPolicy,
        ModelSpec,
        RunConfig,
        load_run,
        run_detection,
    )
    from polyx.gui.report_html import generate_report

    image = tmp_path / "photos" / "plate.png"
    image.parent.mkdir(parents=True)
    ok, buffer = cv2.imencode(".png", np.full((160, 160, 3), 18, np.uint8))
    assert ok
    image.write_bytes(buffer.tobytes())

    class LowConfidenceDetector:
        names = {0: "PET"}
        last_imgsz = 640
        last_fallback = False

        def predict_auto(self, image_path, **kwargs):
            return [Instance(class_id=0, class_name="PET", conf=0.2, x1=20, y1=20, x2=60, y2=60)]

    run = run_detection(
        RunConfig(
            images=[image],
            models=[ModelSpec(detector=LowConfidenceDetector(), alias="best")],
            output_dir=tmp_path / "run",
            params=RunParams(measure_shape=False, conf_assignment=0.5),
            calibration=CalibrationPolicy(mode="none"),
        )
    )
    live = generate_report(run, tmp_path / "live.html", sections=["metodos"]).read_text("utf-8")
    loaded = load_run(run.run_dir)
    rebuilt = generate_report(loaded, tmp_path / "loaded.html", sections=["metodos"]).read_text(
        "utf-8"
    )

    assert "1 / 1 (100.0 %)" in live
    assert "1 / 1 (100.0 %)" in rebuilt
    assert loaded.results[0][0].predictions[0].assignable is False


def test_report_page_forwards_its_current_run_result(qapp, tmp_path, monkeypatch):
    """The GUI page passes ``state.run_result``, not a reconstructed state."""
    pytest.importorskip("PySide6")
    from polyx.gui.detector.pages import report as report_page
    from polyx.gui.detector.state import DetectorState

    run = make_run(tmp_path)
    state = DetectorState()
    state.run_result = run
    # Deliberately empty: the report must not depend on the GUI's real-slot view.
    state.results = {}

    calls: list[RunResult] = []

    def fake_generate(report_run, output_path, **_kwargs):
        calls.append(report_run)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("report", encoding="utf-8")
        return output_path

    monkeypatch.setattr(report_page, "generate_report", fake_generate)
    page = report_page.ReportPage(state)

    page._write_html(tmp_path / "report.html")

    assert calls == [run]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RunStatus.COMPLETED, True),
        (RunStatus.ABORTED, True),
        (RunStatus.FAILED, True),
        (RunStatus.RUNNING, False),
    ],
)
def test_report_page_accepts_every_finished_run(qapp, tmp_path, monkeypatch, status, expected):
    """Completed, aborted and failed batches are reportable; a running one is not."""
    pytest.importorskip("PySide6")
    from polyx.gui.detector.pages import report as report_page
    from polyx.gui.detector.state import DetectorState

    monkeypatch.setattr(report_page.QMessageBox, "warning", lambda *_a, **_k: None)
    state = DetectorState()
    state.results = {}
    state.run_result = make_run(tmp_path, status=status)
    page = report_page.ReportPage(state)

    assert page._check_results() is expected


def test_report_page_refuses_when_there_is_no_run(qapp, monkeypatch):
    pytest.importorskip("PySide6")
    from polyx.gui.detector.pages import report as report_page
    from polyx.gui.detector.state import DetectorState

    monkeypatch.setattr(report_page.QMessageBox, "warning", lambda *_a, **_k: None)
    page = report_page.ReportPage(DetectorState())

    assert page._check_results() is False


def test_the_transitional_adapter_is_gone_and_the_legacy_loader_is_qt_free():
    """New runs never rebuild GUI state; old headless folders use core."""
    from polyx.core.detection_run import load_legacy_run

    assert callable(load_legacy_run)
    assert load_legacy_run.__module__ == "polyx.core.detection_run.legacy_reader"
