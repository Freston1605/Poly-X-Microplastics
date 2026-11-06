"""Contract tests for the core run data model and ``predict_auto``.

Ticket #57 moves the run parameters, the per-image result and the model
metadata into the Qt-free core, so the canonical run engine can be built on
them. The GUI keeps its loaded-model slots and its review verdicts, and keeps
importing the same names from ``polyx.gui.detector.state``.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from polyx.core.detect import (  # noqa: E402
    DetectionSource,
    NativeDetector,
    Sam2Segmenter,
    TwoStageDetector,
    YoloModel,
    YoloSegModel,
)
from polyx.core.detect.protocol import Instance  # noqa: E402
from polyx.core.detection_run import ImageResult, ModelInfo, RunParams  # noqa: E402

# ── Core shapes are Qt-free ─────────────────────────────────────────────


def test_core_run_data_imports_without_qt():
    """A clean interpreter imports the run shapes without loading PySide6."""
    script = (
        "import sys\n"
        "from polyx.core.detection_run import ImageResult, ModelInfo, RunParams\n"
        "from polyx.core.detection_run.result import ImageResult as Same\n"
        "assert Same is ImageResult\n"
        "bad = sorted(m for m in sys.modules if m == 'PySide6' or m.startswith('PySide6.'))\n"
        "print('QT_LOADED=' + repr(bad))\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(  # noqa: PLW1510
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        "importing the core run data dragged Qt or failed:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "QT_LOADED=[]" in result.stdout


def test_run_params_keep_the_inference_defaults():
    """The moved parameters keep every default the pages already rely on."""
    params = RunParams()
    assert params.conf == 0.25
    assert params.iou_nms == 0.45
    assert params.iou_tp == 0.50
    assert params.imgsz == 1280
    assert isinstance(params.device, str) and params.device
    assert params.um_per_px == 0.0
    assert params.size_min_um == 0.0
    assert params.size_max_um == 0.0
    assert params.conf_assignment == 0.0
    assert params.measure_plate is True
    assert params.diameter_plate_mm == 100.0
    assert params.height_plate_mm == 0.0
    assert params.distance_camera_mm == 0.0
    assert params.index_calibration == ""
    assert params.measure_shape is True
    assert params.tiling == "auto"
    assert params.tiling_threshold_px == 2000
    assert params.tiling_tile == 0
    assert params.tiling_overlap == 0.25


def test_core_image_result_has_no_review_verdict():
    """Review verdicts stay GUI state; the core shape cannot carry one."""
    names = {field.name for field in dataclasses.fields(ImageResult)}
    assert "verdict" not in names
    assert {
        "image_path",
        "model_idx",
        "predictions",
        "gt",
        "has_gt",
        "tp",
        "fp",
        "fn",
        "miscls",
        "annotated_path",
        "pred_path",
        "gt_path",
        "tiling",
    } <= names


def test_core_image_result_reads_views_from_disk_on_demand(tmp_path):
    """The PNG properties still return the bytes of the saved files."""
    png = tmp_path / "view.png"
    png.write_bytes(b"png-bytes")
    result = ImageResult(
        image_path=tmp_path / "photo.png",
        model_idx=0,
        annotated_path=png,
        pred_path=png,
    )
    assert result.annotated_png == b"png-bytes"
    assert result.pred_png == b"png-bytes"
    assert result.gt_png is None


def test_model_info_carries_metadata_without_a_backend():
    """Model metadata is plain, serialisable data: no loaded detector."""
    info = ModelInfo(
        alias="model_a",
        path=Path("best.pt"),
        model_type="yolo-seg",
        names={0: "PET", 1: "PP"},
        sha256="abc123",
    )
    assert info.alias == "model_a"
    assert info.path == Path("best.pt")
    assert info.model_type == "yolo-seg"
    assert info.names == {0: "PET", 1: "PP"}
    assert info.sha256 == "abc123"
    assert not hasattr(info, "loaded")


# ── GUI compatibility ──────────────────────────────────────────────────


def test_gui_state_reuses_the_core_shapes():
    """The pages keep importing their old names, now backed by core types."""
    pytest.importorskip("PySide6")
    from polyx.core.detection_run import ImageResult as CoreImageResult
    from polyx.core.detection_run import ModelInfo as CoreModelInfo
    from polyx.core.detection_run import RunParams as CoreRunParams
    from polyx.gui.detector.state import ImageResult as GuiImageResult
    from polyx.gui.detector.state import InferenceParams, ModelSlot
    from polyx.gui.detector.state import RunParams as GuiRunParams

    assert InferenceParams is CoreRunParams
    assert GuiRunParams is CoreRunParams
    # The GUI re-exports the core result unchanged: review verdicts are state,
    # not part of a run's measured data.
    assert GuiImageResult is CoreImageResult
    assert issubclass(ModelSlot, CoreModelInfo)

    result = GuiImageResult(image_path=Path("photo.png"), model_idx=0)
    assert not hasattr(result, "verdict")


# ── predict_auto contract ──────────────────────────────────────────────


def test_detection_source_declares_predict_auto():
    """The protocol exposes the single tiling entry point the engine calls."""
    method = getattr(DetectionSource, "predict_auto", None)
    assert callable(method)
    params = inspect.signature(method).parameters
    for name in (
        "image_path",
        "conf",
        "iou",
        "imgsz",
        "device",
        "tiling",
        "threshold_px",
        "tile",
        "overlap",
        "record",
    ):
        assert name in params, f"predict_auto does not accept {name}"


@pytest.mark.parametrize(
    "detector_class",
    [YoloModel, YoloSegModel, NativeDetector, Sam2Segmenter, TwoStageDetector],
    ids=lambda cls: cls.__name__,
)
def test_every_detector_implements_predict_auto(detector_class):
    """Every backend can be driven through the same tiling entry point."""
    assert callable(getattr(detector_class, "predict_auto", None))


def _sentinel_detections() -> list[Instance]:
    return [Instance(class_id=0, class_name="particle", conf=0.9, x1=1, y1=2, x2=3, y2=4)]


@pytest.mark.parametrize("kind", ["native", "sam2", "two_stage"])
def test_non_tiling_detectors_delegate_predict_auto_to_predict(monkeypatch, tmp_path, kind):
    """Non-tiling backends hand the single pass straight to ``predict``."""
    if kind == "native":
        detector = NativeDetector()
    elif kind == "sam2":
        detector = Sam2Segmenter(tmp_path / "sam2.pt")
    else:
        detector = TwoStageDetector(NativeDetector(), Sam2Segmenter(tmp_path / "sam2.pt"))

    expected = _sentinel_detections()
    calls: list[dict] = []

    def fake_predict(image_path, **kwargs):
        calls.append({"image_path": image_path, **kwargs})
        return expected

    monkeypatch.setattr(detector, "predict", fake_predict)
    result = detector.predict_auto(
        "photo.png",
        conf=0.3,
        iou=0.55,
        imgsz=1024,
        device="cpu",
        tiling="siempre",
        threshold_px=100,
        tile=512,
        overlap=0.1,
        record={},
    )

    assert result is expected
    assert calls == [
        {
            "image_path": "photo.png",
            "conf": 0.3,
            "iou": 0.55,
            "imgsz": 1024,
            "device": "cpu",
        }
    ]


def test_yolo_predict_auto_uses_predict_when_tiling_is_disabled(monkeypatch, tmp_path):
    """The existing single-pass path of a tiling detector stays untouched."""
    detector = YoloModel(tmp_path / "best.pt")
    expected = _sentinel_detections()
    calls: list[dict] = []

    def fake_predict(image_path, **kwargs):
        calls.append({"image_path": image_path, **kwargs})
        return expected

    monkeypatch.setattr(detector, "predict", fake_predict)
    record: dict = {}
    result = detector.predict_auto(
        tmp_path / "photo.png",
        conf=0.3,
        iou=0.55,
        imgsz=1024,
        device="cpu",
        tiling="nunca",
        record=record,
    )

    assert result is expected
    assert calls == [
        {
            "image_path": tmp_path / "photo.png",
            "conf": 0.3,
            "iou": 0.55,
            "imgsz": 1024,
            "device": "cpu",
        }
    ]
    assert record["troceado"] is False
