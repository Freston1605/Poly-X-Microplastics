"""End-to-end tests for the headless detection CLI on the canonical engine.

Ticket #59: ``polyx-detect`` runs ``run_detection`` (not the removed headless
batch writer), accepts repeatable models and model types, writes the exact
``--out`` folder, refuses to overwrite it without ``--force``, and keeps both
the compact and the full report paths working. The models are stubbed so the
suite never needs weights, a GPU or Ultralytics.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from polyx.cli import detect as cli_detect
from polyx.core.detect.protocol import Instance
from polyx.core.detection_run import load_run
from polyx.core.measure.calibration import ORIGIN_MANUAL, ORIGIN_NONE
from polyx.core.report_simple import generate as generate_compact

TOP_BOX = (20.0, 20.0, 60.0, 60.0)
BOTTOM_BOX = (90.0, 90.0, 130.0, 130.0)


def write_plate(path: Path, width: int = 160, height: int = 160) -> Path:
    """Write a plate with two bright squares so boxes have something to measure."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in (TOP_BOX, BOTTOM_BOX):
        image[int(y1) + 5 : int(y2) - 5, int(x1) + 5 : int(x2) - 5] = 255
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buffer))
    return path


class FakeDetector:
    """A loaded detection source: no weights, no tiling, fixed boxes."""

    def __init__(self, boxes: tuple[tuple[float, float, float, float], ...] = (TOP_BOX,)):
        self.boxes = boxes
        self.names = {0: "PET", 1: "PP"}
        self.load_calls = 0
        self.predict_calls = 0
        self.last_imgsz = 1280
        self.last_fallback = False
        self.call_kwargs: list[dict] = []

    def load(self):
        self.load_calls += 1
        return self

    def predict_auto(self, image_path, **kwargs) -> list[Instance]:
        self.predict_calls += 1
        self.call_kwargs.append(dict(kwargs))
        return [
            Instance(
                class_id=index % 2,
                class_name=self.names.get(index % 2, str(index % 2)),
                conf=0.9 - 0.1 * index,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
            for index, (x1, y1, x2, y2) in enumerate(self.boxes)
        ]


class ExplodingDetector(FakeDetector):
    def predict_auto(self, image_path, **kwargs):
        raise RuntimeError("inference exploded")


def install_factory(
    monkeypatch, detectors: FakeDetector | list[FakeDetector]
) -> list[tuple[str, Path]]:
    """Replace the real factory and record the (type, path) pairs it receives."""
    calls: list[tuple[str, Path]] = []
    pending = list(detectors) if isinstance(detectors, list) else [detectors]

    def factory(name, weights=None, **kwargs):
        calls.append((name, Path(weights)))
        return pending.pop(0)

    monkeypatch.setattr(cli_detect, "make_detector", factory)
    return calls


def weights_file(tmp_path: Path, stem: str) -> Path:
    path = tmp_path / f"{stem}.pt"
    path.write_bytes(b"weights")
    return path


def image_folder(tmp_path: Path, count: int = 2) -> Path:
    folder = tmp_path / "photos"
    for index in range(count):
        write_plate(folder / f"plate{index + 1}.png")
    return folder


# ── Models and aliases ──────────────────────────────────────────────────


def test_single_model_defaults_to_yolo_and_the_weights_stem(
    tmp_path, monkeypatch, capsys, selected_language
):
    selected_language("es")
    weights = weights_file(tmp_path, "best")
    detector = FakeDetector()
    calls = install_factory(monkeypatch, detector)

    code = cli_detect.main(
        [
            "--model",
            str(weights),
            "--images",
            str(image_folder(tmp_path)),
            "--out",
            str(tmp_path / "run"),
        ]
    )

    assert code == 0
    assert calls == [("yolo", weights)]
    assert detector.load_calls == 1
    run = load_run(tmp_path / "run")
    assert [model.alias for model in run.models] == ["best"]
    assert run.status == "completed"
    # The engine never loads a model: the CLI did it before the run.
    assert (tmp_path / "run" / "best" / "0000_plate1_annot.png").is_file()
    assert "INFORME COMPACTO:" in capsys.readouterr().out


def test_repeatable_models_and_aliases_come_from_the_stems(tmp_path, monkeypatch):
    first = weights_file(tmp_path, "detector_a")
    second = weights_file(tmp_path, "detector_b")
    install_factory(monkeypatch, [FakeDetector(), FakeDetector(boxes=(TOP_BOX, BOTTOM_BOX))])

    code = cli_detect.main(
        [
            "--model",
            str(first),
            "--model",
            str(second),
            "--images",
            str(image_folder(tmp_path)),
            "--out",
            str(tmp_path / "run"),
        ]
    )

    assert code == 0
    run = load_run(tmp_path / "run")
    assert [model.alias for model in run.models] == ["detector_a", "detector_b"]
    assert set(run.results) == {0, 1}
    assert (tmp_path / "run" / "detector_a").is_dir()
    assert (tmp_path / "run" / "detector_b").is_dir()


def test_a_single_model_type_applies_to_every_model(tmp_path, monkeypatch):
    first = weights_file(tmp_path, "detector_a")
    second = weights_file(tmp_path, "detector_b")
    calls = install_factory(monkeypatch, [FakeDetector(), FakeDetector()])

    code = cli_detect.main(
        [
            "--model",
            str(first),
            "--model",
            str(second),
            "--model-type",
            "yolo-seg",
            "--images",
            str(image_folder(tmp_path)),
            "--out",
            str(tmp_path / "run"),
        ]
    )

    assert code == 0
    assert calls == [("yolo-seg", first), ("yolo-seg", second)]
    run = load_run(tmp_path / "run")
    assert [model.model_type for model in run.models] == ["yolo-seg", "yolo-seg"]


def test_per_model_types_are_applied_in_order(tmp_path, monkeypatch):
    first = weights_file(tmp_path, "detector_a")
    second = weights_file(tmp_path, "detector_b")
    calls = install_factory(monkeypatch, [FakeDetector(), FakeDetector()])

    code = cli_detect.main(
        [
            "--model",
            str(first),
            "--model",
            str(second),
            "--model-type",
            "yolo",
            "--model-type",
            "yolo-seg",
            "--images",
            str(image_folder(tmp_path)),
            "--out",
            str(tmp_path / "run"),
        ]
    )

    assert code == 0
    assert calls == [("yolo", first), ("yolo-seg", second)]
    run = load_run(tmp_path / "run")
    assert [model.model_type for model in run.models] == ["yolo", "yolo-seg"]


def test_a_model_type_count_that_is_not_one_or_all_is_rejected(tmp_path, monkeypatch, capsys):
    first = weights_file(tmp_path, "detector_a")
    second = weights_file(tmp_path, "detector_b")
    calls = install_factory(monkeypatch, [FakeDetector(), FakeDetector()])

    code = cli_detect.main(
        [
            "--model",
            str(first),
            "--model",
            str(second),
            "--model-type",
            "yolo",
            "--model-type",
            "yolo-seg",
            "--model-type",
            "native",
            "--images",
            str(image_folder(tmp_path)),
            "--out",
            str(tmp_path / "run"),
        ]
    )

    assert code == cli_detect.EXIT_USAGE
    assert calls == []
    assert "--model-type" in capsys.readouterr().err


# ── Output folder and failure handling ─────────────────────────────────


def test_non_empty_output_is_refused_unless_forced(
    tmp_path, monkeypatch, capsys, selected_language
):
    selected_language("es")
    weights = weights_file(tmp_path, "best")
    images = image_folder(tmp_path)
    out = tmp_path / "run"
    detectors = [FakeDetector(), FakeDetector(), FakeDetector()]
    install_factory(monkeypatch, detectors)
    base = ["--model", str(weights), "--images", str(images), "--out", str(out)]

    assert cli_detect.main(base) == 0
    original = (out / "run.json").read_text(encoding="utf-8")

    assert cli_detect.main(base) == cli_detect.EXIT_OUTPUT_EXISTS
    assert (out / "run.json").read_text(encoding="utf-8") == original
    error = capsys.readouterr().err
    assert "Usa --force para sobrescribir los ficheros canónicos." in error
    assert "--force" in error

    assert cli_detect.main([*base, "--force"]) == 0
    assert load_run(out).status == "completed"


def test_an_output_path_that_is_a_file_is_rejected_with_its_own_message(
    tmp_path, monkeypatch, capsys, selected_language
):
    selected_language("es")
    weights = weights_file(tmp_path, "best")
    install_factory(monkeypatch, [FakeDetector()])
    out = tmp_path / "already-a-file.txt"
    out.write_text("keep me", encoding="utf-8")

    code = cli_detect.main(
        [
            "--model",
            str(weights),
            "--images",
            str(image_folder(tmp_path, count=1)),
            "--out",
            str(out),
        ]
    )

    assert code == cli_detect.EXIT_OUTPUT_EXISTS
    error = capsys.readouterr().err
    # The hint is the one that distinguishes this refusal from a non-empty
    # folder, and it follows the selected language.
    assert "Elige una ruta de carpeta para --out." in error
    assert out.read_text(encoding="utf-8") == "keep me"


def test_a_failed_run_reports_the_partial_folder(tmp_path, monkeypatch, capsys, selected_language):
    selected_language("es")
    weights = weights_file(tmp_path, "best")
    install_factory(monkeypatch, [ExplodingDetector()])

    code = cli_detect.main(
        [
            "--model",
            str(weights),
            "--images",
            str(image_folder(tmp_path)),
            "--out",
            str(tmp_path / "run"),
        ]
    )

    assert code == cli_detect.EXIT_ERROR
    error = capsys.readouterr().err
    assert "inference exploded" in error
    assert "EJECUCIÓN PARCIAL" in error
    partial = load_run(tmp_path / "run")
    assert partial.status == "failed"


# ── Existing flags keep their meaning ──────────────────────────────────


def test_existing_flags_keep_their_meaning(tmp_path, monkeypatch):
    weights = weights_file(tmp_path, "best")
    images = image_folder(tmp_path, count=1)
    labels = tmp_path / "labels"
    labels.mkdir()
    x1, y1, x2, y2 = TOP_BOX
    (labels / "plate1.txt").write_text(
        f"0 {(x1 + x2) / 2 / 160:.6f} {(y1 + y2) / 2 / 160:.6f} "
        f"{(x2 - x1) / 160:.6f} {(y2 - y1) / 160:.6f}\n",
        encoding="utf-8",
    )
    install_factory(monkeypatch, [FakeDetector(boxes=(TOP_BOX, BOTTOM_BOX))])

    code = cli_detect.main(
        [
            "--model",
            str(weights),
            "--images",
            str(images),
            "--out",
            str(tmp_path / "run"),
            "--conf",
            "0.31",
            "--iou",
            "0.22",
            "--imgsz",
            "640",
            "--device",
            "cpu",
            "--um-per-px",
            "2.5",
            "--gt-folder",
            str(labels),
            "--no-measure-shape",
        ]
    )

    assert code == 0
    run = load_run(tmp_path / "run")
    assert run.params.conf == pytest.approx(0.31)
    assert run.params.iou_nms == pytest.approx(0.22)
    assert run.params.imgsz == 640
    assert run.params.device == "cpu"
    # The manual scale is calibration policy, not an inference parameter.
    assert "um_per_px" not in json.loads((run.run_dir / "run.json").read_text("utf-8"))["params"]
    assert run.params.measure_shape is False
    assert run.has_gt
    assert run.metrics[0]["tp"] == 1
    assert run.calibrations[images / "plate1.png"].origin == ORIGIN_MANUAL
    assert run.calibrations[images / "plate1.png"].um_por_px == pytest.approx(2.5)
    # Without --um-per-px the run is explicitly uncalibrated, not "manual 0".
    assert all(det.size_source == "box" for det in run.results[0][0].predictions)


@pytest.mark.parametrize(
    ("choice", "backend"),
    [("auto", "auto"), ("always", "siempre"), ("never", "nunca")],
)
def test_tiling_cli_choices_are_english_and_reach_the_backend_values(
    tmp_path, monkeypatch, choice, backend
):
    """The developer-facing flag speaks English; the detector keeps its contract."""
    weights = weights_file(tmp_path, "best")
    detector = FakeDetector()
    install_factory(monkeypatch, detector)

    code = cli_detect.main(
        [
            "--model",
            str(weights),
            "--images",
            str(image_folder(tmp_path, count=1)),
            "--out",
            str(tmp_path / "run"),
            "--tiling",
            choice,
        ]
    )

    assert code == 0
    assert load_run(tmp_path / "run").params.tiling == backend
    assert detector.call_kwargs[0]["tiling"] == backend


def test_without_a_manual_scale_the_run_is_uncalibrated(tmp_path, monkeypatch):
    weights = weights_file(tmp_path, "best")
    images = image_folder(tmp_path, count=1)
    install_factory(monkeypatch, [FakeDetector()])

    code = cli_detect.main(
        [
            "--model",
            str(weights),
            "--images",
            str(images),
            "--out",
            str(tmp_path / "run"),
        ]
    )

    assert code == 0
    run = load_run(tmp_path / "run")
    calibration = run.calibrations[images / "plate1.png"]
    assert calibration.origin == ORIGIN_NONE
    assert calibration.um_por_px == 0.0
    assert all(det.diam_um is None for det in run.results[0][0].predictions)


# ── Reports ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("language", "title", "images", "detections"),
    [
        ("es", "Informe de ejecución de Poly-X", "Imágenes: 2", "Detecciones: 4"),
        ("en", "Poly-X run report", "Images: 2", "Detections: 4"),
    ],
)
def test_compact_report_reads_the_canonical_folder_first(
    tmp_path, monkeypatch, selected_language, language, title, images, detections
):
    selected_language(language)
    weights = weights_file(tmp_path, "best")
    install_factory(monkeypatch, [FakeDetector(boxes=(TOP_BOX, BOTTOM_BOX))])
    out = tmp_path / "run"

    assert (
        cli_detect.main(
            ["--model", str(weights), "--images", str(image_folder(tmp_path)), "--out", str(out)]
        )
        == 0
    )
    # Old-schema decoys must be ignored once run.json exists.
    (out / "summary.json").write_text(
        json.dumps({"images": 999, "detections": 999}), encoding="utf-8"
    )
    (out / "centroids.csv").write_text("image,class\nOLD-CLASS,OLD-CLASS\n", encoding="utf-8")

    report = generate_compact(out)
    text = report.read_text(encoding="utf-8")
    assert title in text
    assert "PET" in text
    assert "OLD-CLASS" not in text
    assert images in text
    assert detections in text


@pytest.mark.parametrize(
    ("language", "header", "title"),
    [
        ("es", "INFORME COMPLETO:", "Informe de detección"),
        ("en", "FULL REPORT:", "Detection report"),
    ],
)
def test_full_report_renders_the_engine_result_directly(
    tmp_path, monkeypatch, capsys, selected_language, language, header, title
):
    selected_language(language)
    weights = weights_file(tmp_path, "best")
    install_factory(monkeypatch, [FakeDetector(boxes=(TOP_BOX, BOTTOM_BOX))])
    out = tmp_path / "run"

    code = cli_detect.main(
        [
            "--model",
            str(weights),
            "--images",
            str(image_folder(tmp_path, count=1)),
            "--out",
            str(out),
            "--full-report",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert header in captured.out
    # The full report succeeded, so no compact fallback was announced.
    assert "COMPACT REPORT:" not in captured.out
    assert "INFORME COMPACTO:" not in captured.out
    report = (out / "report.html").read_text(encoding="utf-8")
    assert title in report
    # The cover names the canonical model alias, which proves the report
    # consumed the engine's RunResult directly (no DetectorState rebuild).
    assert "best" in report
