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
from polyx.core.detection_run import ImageResult, ModelInfo, RunParams, RunResult


def make_run(
    tmp_path: Path,
    *,
    status: str = "completed",
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
        params=RunParams(conf=0.3, um_per_px=2.0),
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


@pytest.mark.parametrize("status", ["completed", "aborted"])
def test_completed_and_aborted_runs_are_reportable(tmp_path, status):
    """A partial aborted run is still a real run with results to report."""
    from polyx.gui.report_html import generate_report

    run = make_run(tmp_path, status=status)
    assert generate_report(run, tmp_path / "report.html", sections=["resumen"]).is_file()


def test_failed_runs_are_not_reportable(tmp_path):
    """A failed run has no complete batch to describe; the caller decides."""
    from polyx.gui.report_html import generate_report

    run = make_run(tmp_path, status="failed")
    with pytest.raises(ValueError, match="completed or aborted"):
        generate_report(run, tmp_path / "report.html", sections=["resumen"])


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
    [("completed", True), ("aborted", True), ("failed", False)],
)
def test_report_page_requires_a_completed_or_aborted_run(
    qapp, tmp_path, monkeypatch, status, expected
):
    """No run, a running one or a failed one must not produce a report."""
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
