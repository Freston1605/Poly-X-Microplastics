"""Legacy run readers: old headless folders and old GUI run folders.

Pre-canonical headless runs wrote ``summary.json`` + ``centroids.csv`` and a
flat ``images/`` folder; the canonical engine writes ``run.json`` +
``detections.csv`` with one folder per model. Old GUI runs still have their
per-model ``labels/*.txt`` and keep opening in the Viewer. Both readers are
pinned against the frozen folders in ``tests/fixtures/legacy_runs``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "legacy_runs"
HEADLESS_RUN = FIXTURES / "headless" / "run"
HEADLESS_PHOTOS = FIXTURES / "headless" / "photos"
GUI_RUN = FIXTURES / "gui" / "detect_20260101_120000"
GUI_PHOTO = FIXTURES / "gui" / "photos" / "plate1.png"


# ── Old headless folders ───────────────────────────────────────────────


def test_legacy_headless_folder_becomes_a_run_result():
    """The old folder is read without Qt and without inventing values."""
    from polyx.core.detection_run import load_legacy_run

    run = load_legacy_run(HEADLESS_RUN, image_dir=HEADLESS_PHOTOS)

    assert run.run_dir == HEADLESS_RUN
    assert run.status == "completed"
    assert [path.name for path in run.images] == ["plate1.png", "plate2.png"]
    assert len(run.models) == 1
    assert run.models[0].alias == "model"
    assert run.models[0].model_type == "YoloModel"
    assert run.models[0].names == {0: "PET", 1: "PP"}

    assert run.params.conf == pytest.approx(0.3)
    assert run.params.iou_nms == pytest.approx(0.45)
    assert run.params.imgsz == 640
    assert run.params.device == "cpu"
    assert not hasattr(run.params, "um_per_px")
    assert run.params.measure_shape is False
    # The old batch-wide manual scale becomes the effective per-image record.
    assert {calibration.origin for calibration in run.calibrations.values()} == {"manual"}
    assert all(
        calibration.um_por_px == pytest.approx(2.5) for calibration in run.calibrations.values()
    )

    by_name = {result.image_path.name: result for result in run.results[0]}
    assert set(by_name) == {"plate1.png", "plate2.png"}

    first = by_name["plate1.png"]
    assert first.image_path == HEADLESS_PHOTOS / "plate1.png"
    assert first.annotated_path == HEADLESS_RUN / "images" / "plate1.png"
    assert [det.class_name for det in first.predictions] == ["PET", "PP"]
    assert first.predictions[0].conf == pytest.approx(0.9)
    assert first.predictions[0].diam_um == pytest.approx(40.0)
    assert (first.tp, first.fp, first.fn, first.miscls) == (1, 0, 0, 0)
    assert first.has_gt is True

    second = by_name["plate2.png"]
    assert [det.class_name for det in second.predictions] == ["PET"]
    assert (second.tp, second.fp, second.fn, second.miscls) == (0, 1, 1, 0)
    assert second.has_gt is True


def test_legacy_headless_folder_without_originals_still_loads():
    """A folder copied away from its photos stays readable: names are enough."""
    from polyx.core.detection_run import load_legacy_run

    run = load_legacy_run(HEADLESS_RUN)

    assert [path.name for path in run.images] == ["plate1.png", "plate2.png"]
    assert run.results[0][0].annotated_path == HEADLESS_RUN / "images" / "plate1.png"


def test_legacy_headless_folder_renders_the_full_report(tmp_path, selected_language):
    """The rebuilt RunResult feeds the same report seam as a canonical run."""
    selected_language("es")
    from polyx.core.detection_run import load_legacy_run
    from polyx.gui.report_html import generate_report

    run = load_legacy_run(HEADLESS_RUN, image_dir=HEADLESS_PHOTOS)
    output = generate_report(run, tmp_path / "report.html")

    html = output.read_text(encoding="utf-8")
    assert "Informe de detección" in html
    assert "model" in html
    assert "PET" in html
    # The old folder stored counts, not GT boxes: the report must say so
    # instead of rebuilding a confusion matrix that turns every legacy
    # prediction into a false positive.
    assert "no se puede reconstruir la matriz de confusión" in html
    assert "Matriz de confusión" not in html


def test_compact_report_keeps_reading_old_headless_folders(tmp_path, selected_language):
    """The Qt-free compact report still has its pre-canonical fallback."""
    selected_language("es")
    from polyx.core.report_simple import generate

    output = generate(HEADLESS_RUN, tmp_path / "report.html")

    html = output.read_text(encoding="utf-8")
    assert "Informe de ejecución de Poly-X" in html
    assert "Imágenes: 2" in html
    assert "Detecciones: 3" in html
    assert "PET" in html


def test_report_cli_renders_a_full_report_from_a_legacy_folder(tmp_path, capsys, selected_language):
    """`polyx-report --full-report` is a real consumer of the legacy loader."""
    selected_language("es")
    from polyx.cli.report import main

    output = tmp_path / "report.html"
    code = main(
        [
            "--run",
            str(HEADLESS_RUN),
            "--images",
            str(HEADLESS_PHOTOS),
            "--full-report",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert "INFORME COMPLETO:" in capsys.readouterr().out
    assert "Informe de detección" in output.read_text(encoding="utf-8")


def test_report_cli_renders_the_compact_report_from_a_legacy_folder(
    tmp_path, capsys, selected_language
):
    """`polyx-report` without --full-report also reads old headless folders."""
    selected_language("es")
    from polyx.cli.report import main

    output = tmp_path / "compact.html"
    code = main(
        [
            "--run",
            str(HEADLESS_RUN),
            "--images",
            str(HEADLESS_PHOTOS),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert "ESCRITO:" in capsys.readouterr().out
    html = output.read_text(encoding="utf-8")
    assert "Informe de ejecución de Poly-X" in html
    assert "Imágenes: 2" in html
    assert "Detecciones: 3" in html


# ── Old GUI run folders ────────────────────────────────────────────────


def test_old_gui_run_labels_are_found_per_model():
    """Each model folder contributes its own prediction label for the photo."""
    from polyx.core.annotate import find_run_labels

    found = find_run_labels(GUI_RUN, "plate1")

    assert [path.parent.parent.name for path in found] == ["alpha", "beta"]
    assert [path.name for path in found] == ["0000_plate1.txt", "0000_plate1.txt"]


def test_old_gui_run_labels_ignore_other_photos_and_files():
    from polyx.core.annotate import find_run_labels

    assert find_run_labels(GUI_RUN, "plate2") == []
    assert find_run_labels(GUI_RUN, "0000_plate1") == []


def test_viewer_loads_predictions_from_an_old_gui_run_folder(qapp, monkeypatch):
    """The Viewer offers both models and opens the chosen model's labels."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

    from polyx.gui.viewer.window import ViewerWindow

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(GUI_RUN))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("alpha", True))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    window = ViewerWindow()
    window.state.images = [GUI_PHOTO]
    window.state.current_idx = 0
    window._run_load()

    assert [det.class_name for det in window.state.detections] == ["PET"]
    assert len(window.state.detections) == 1


# ── Canonical folders ──────────────────────────────────────────────────


def test_core_exposes_the_canonical_engine_and_not_the_old_batch():
    import polyx.core as core
    import polyx.core.detection_run as detection_run

    assert not hasattr(core, "run_batch")
    assert "run_batch" not in core.__all__
    assert core.run_detection is detection_run.run_detection
    assert core.RunConfig is detection_run.RunConfig
    assert core.RunResult is detection_run.RunResult
    assert core.load_run is detection_run.load_run


def test_the_old_headless_writer_is_deleted_and_run_state_is_only_a_shim():
    """No dead second engine survives; the old name only delegates to core."""
    source = Path(__file__).resolve().parents[1] / "src" / "polyx"
    assert not (source / "core" / "detection_run" / "legacy.py").exists()
    offenders = [
        str(path.relative_to(source))
        for path in source.rglob("*.py")
        if "run_batch" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
    shim = source / "gui" / "detector" / "run_state.py"
    assert shim.is_file()
    text = shim.read_text(encoding="utf-8")
    assert "load_legacy_run" in text
    assert "from polyx.gui.detector.state" not in text
    assert "class DetectorState" not in text


def test_run_state_build_state_from_run_delegates_to_the_core_loader():
    """The compatibility name returns the canonical RunResult, not Qt state."""
    from polyx.core.detection_run import RunResult
    from polyx.gui.detector.run_state import build_state_from_run

    run = build_state_from_run(HEADLESS_RUN, HEADLESS_PHOTOS)

    assert isinstance(run, RunResult)
    assert run.run_dir == HEADLESS_RUN
    assert [path.name for path in run.images] == ["plate1.png", "plate2.png"]
    assert run.results[0][0].image_path == HEADLESS_PHOTOS / "plate1.png"


def test_canonical_folder_loads_and_renders_the_full_report(tmp_path):
    """A canonical folder reopens through load_run and feeds the report."""
    from polyx.core.detect.protocol import Instance
    from polyx.core.detection_run import (
        CalibrationPolicy,
        ModelSpec,
        RunConfig,
        RunParams,
        load_run,
        run_detection,
    )
    from polyx.gui.report_html import generate_report

    class Detector:
        names = {0: "PET"}
        last_imgsz = 640
        last_fallback = False

        def predict_auto(self, image_path, **kwargs):
            return [Instance(class_id=0, class_name="PET", conf=0.9, x1=20, y1=20, x2=40, y2=40)]

    image = HEADLESS_PHOTOS / "plate1.png"
    result = run_detection(
        RunConfig(
            images=[image],
            models=[ModelSpec(detector=Detector(), alias="best")],
            output_dir=tmp_path / "canonical",
            params=RunParams(measure_shape=False),
            calibration=CalibrationPolicy(mode="manual", um_per_px=2.0),
        )
    )

    loaded = load_run(result.run_dir)
    output = generate_report(loaded, tmp_path / "report.html")

    assert output.is_file()
    assert "Informe de detección" in output.read_text(encoding="utf-8")
    assert "best" in output.read_text(encoding="utf-8")
