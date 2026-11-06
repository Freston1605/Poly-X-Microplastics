"""Contract tests for the per-photo calibration policy.

Ticket #60: the canonical run owns how every photo gets its micrometres per
pixel. ``auto`` tries an explicit index, the searched index, the plate on the
photo and finally the manual value; ``manual`` uses only the typed value,
``index`` only the CSV and ``none`` leaves the run uncalibrated. The origin,
validity and warning of every image are recorded in ``run.json``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from polyx.cli import detect as cli_detect
from polyx.core.detect.protocol import Instance
from polyx.core.detection_run import (
    CalibrationPolicy,
    ModelSpec,
    RunConfig,
    RunParams,
    load_run,
    run_detection,
)
from polyx.core.measure.calibration import (
    ORIGIN_INDEX,
    ORIGIN_MANUAL,
    ORIGIN_NONE,
    ORIGIN_PLATE,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

BOX = (20.0, 20.0, 60.0, 60.0)
PLATE_RADIUS = 300
PLATE_CANVAS = 900
PLATE_DIAMETER_MM = 100.0
PLATE_UM_PER_PX = PLATE_DIAMETER_MM * 1000.0 / (2 * PLATE_RADIUS)


def write_png(path: Path, image: np.ndarray) -> Path:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buffer))
    return path


def write_plate(path: Path, radius: int = PLATE_RADIUS) -> Path:
    """A synthetic Petri plate: bright ring over a dark field."""
    image = np.full((PLATE_CANVAS, PLATE_CANVAS, 3), 12, np.uint8)
    centre = (PLATE_CANVAS // 2, PLATE_CANVAS // 2)
    cv2.circle(image, centre, radius, (210, 205, 200), 9)
    cv2.circle(image, centre, radius - 40, (60, 55, 90), -1)
    return write_png(path, image)


def write_plain(path: Path, width: int = 160, height: int = 160) -> Path:
    """An image with one bright square and no plate ring."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    x1, y1, x2, y2 = BOX
    image[int(y1) + 5 : int(y2) - 5, int(x1) + 5 : int(x2) - 5] = 255
    return write_png(path, image)


def write_index(path: Path, values: dict[str, float], name_column: str = "recorte") -> Path:
    lines = [f"{name_column},um_por_px", *[f"{name},{value}" for name, value in values.items()]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class FakeDetector:
    """A loaded detection source with one fixed box and no backend."""

    names = {0: "PET"}
    last_imgsz = 640
    last_fallback = False

    def load(self) -> FakeDetector:
        return self

    def predict_auto(self, image_path, **kwargs) -> list[Instance]:
        x1, y1, x2, y2 = BOX
        return [Instance(class_id=0, class_name="PET", conf=0.9, x1=x1, y1=y1, x2=x2, y2=y2)]


def make_config(
    images: list[Path],
    output_dir: Path,
    calibration: CalibrationPolicy,
) -> RunConfig:
    return RunConfig(
        images=list(images),
        models=[ModelSpec(detector=FakeDetector(), alias="model")],
        output_dir=output_dir,
        params=RunParams(measure_shape=False),
        calibration=calibration,
    )


def calibration_payload(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


# ── Manual and none ─────────────────────────────────────────────────────


def test_manual_uses_only_the_typed_value(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    index = write_index(tmp_path / "photos" / "indice.csv", {"plate.png": 31.0})

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="manual", um_per_px=2.5, index_path=index),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_MANUAL
    assert calibration.um_por_px == pytest.approx(2.5)
    assert calibration.valid
    assert not calibration.warning
    assert all(det.diam_um is not None for det in result.results[0][0].predictions)


def test_manual_without_a_value_is_invalid_and_explains_why(tmp_path):
    image = write_plain(tmp_path / "photos" / "plate.png")
    index = write_index(tmp_path / "photos" / "indice.csv", {"plate.png": 31.0})

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="manual", um_per_px=0.0, index_path=index),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_NONE
    assert not calibration.valid
    assert calibration.warning
    assert all(det.diam_um is None for det in result.results[0][0].predictions)


def test_none_never_invents_micrometre_sizes(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    index = write_index(tmp_path / "photos" / "indice.csv", {"plate.png": 31.0})

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="none", um_per_px=9.0, index_path=index),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_NONE
    assert calibration.um_por_px == 0.0
    assert not calibration.valid
    assert all(det.diam_um is None for det in result.results[0][0].predictions)
    with (result.run_dir / "detections.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["um_per_px"] for row in rows} == {""}
    assert {row["calibration_origin"] for row in rows} == {ORIGIN_NONE}
    payload = calibration_payload(result.run_dir)
    assert payload["calibrations"][0]["valid"] is False
    assert payload["calibrations"][0]["origin"] == ORIGIN_NONE


# ── Index ───────────────────────────────────────────────────────────────


def test_index_uses_the_explicit_csv_and_ignores_the_manual_value(tmp_path):
    image = write_plain(tmp_path / "photos" / "plate.png")
    write_index(tmp_path / "photos" / "indice.csv", {"plate.png": 99.0})
    explicit = write_index(tmp_path / "calibration.csv", {"plate.png": 31.25})

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="index", um_per_px=2.5, index_path=explicit),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_INDEX
    assert calibration.um_por_px == pytest.approx(31.25)
    assert calibration.valid
    assert all(det.diam_um is not None for det in result.results[0][0].predictions)


def test_index_without_an_entry_is_invalid_and_warns(tmp_path):
    image = write_plain(tmp_path / "photos" / "plate.png")
    explicit = write_index(tmp_path / "calibration.csv", {"other.png": 31.25})

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="index", um_per_px=2.5, index_path=explicit),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_NONE
    assert not calibration.valid
    assert calibration.warning
    assert all(det.diam_um is None for det in result.results[0][0].predictions)


@pytest.mark.parametrize(
    ("filename", "name_column"),
    [("indice.csv", "recorte"), ("calibracion_placas.csv", "placa")],
)
def test_index_searches_the_known_csv_names_from_the_image_root(tmp_path, filename, name_column):
    image = write_plain(tmp_path / "photos" / "plate.png")
    write_index(tmp_path / "photos" / filename, {"plate.png": 32.5}, name_column)

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="index", index_path=None),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_INDEX
    assert calibration.um_por_px == pytest.approx(32.5)


# ── Auto fallback order ─────────────────────────────────────────────────


def test_auto_follows_searched_plate_manual_order(tmp_path):
    photos = tmp_path / "photos"
    searched_image = write_plain(photos / "searched.png")
    plate_image = write_plate(photos / "plate.png")
    manual_image = write_plain(photos / "manual.png")
    write_index(photos / "indice.csv", {"searched.png": 42.0})

    result = run_detection(
        make_config(
            [searched_image, plate_image, manual_image],
            tmp_path / "run",
            CalibrationPolicy(mode="auto", um_per_px=2.5),
        )
    )

    by_name = {path.name: calibration for path, calibration in result.calibrations.items()}
    assert by_name["searched.png"].origin == ORIGIN_INDEX
    assert by_name["searched.png"].um_por_px == pytest.approx(42.0)
    assert by_name["plate.png"].origin == ORIGIN_PLATE
    assert by_name["plate.png"].um_por_px == pytest.approx(PLATE_UM_PER_PX, rel=0.03)
    assert by_name["manual.png"].origin == ORIGIN_MANUAL
    assert by_name["manual.png"].um_por_px == pytest.approx(2.5)
    assert by_name["manual.png"].warning


def test_auto_prefers_the_explicit_index_over_the_searched_one_and_the_plate(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")
    write_index(tmp_path / "photos" / "indice.csv", {"plate.png": 99.0})
    explicit = write_index(tmp_path / "calibration.csv", {"plate.png": 31.0})

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="auto", um_per_px=2.5, index_path=explicit),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_INDEX
    assert calibration.um_por_px == pytest.approx(31.0)


def test_auto_without_any_source_is_invalid(tmp_path):
    image = write_plain(tmp_path / "photos" / "plate.png")

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="auto", um_per_px=0.0),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_NONE
    assert not calibration.valid
    assert calibration.warning
    assert all(det.diam_um is None for det in result.results[0][0].predictions)


# ── Run record ──────────────────────────────────────────────────────────


def test_run_json_records_every_calibration_and_its_summary(tmp_path):
    first = write_plain(tmp_path / "photos" / "first.png")
    second = write_plain(tmp_path / "photos" / "second.png")
    write_index(tmp_path / "photos" / "indice.csv", {"first.png": 32.0})

    result = run_detection(
        make_config(
            [first, second],
            tmp_path / "run",
            CalibrationPolicy(mode="auto", um_per_px=2.0),
        )
    )

    payload = calibration_payload(result.run_dir)
    entries = {Path(entry["image"]).name: entry for entry in payload["calibrations"]}
    assert set(entries) == {"first.png", "second.png"}
    assert entries["first.png"]["origin"] == ORIGIN_INDEX
    assert entries["first.png"]["valid"] is True
    assert entries["second.png"]["origin"] == ORIGIN_MANUAL
    assert entries["second.png"]["valid"] is True
    assert entries["second.png"]["warning"]
    for entry in entries.values():
        assert {"origin", "valid", "warning", "um_por_px"} <= set(entry)

    summary = payload["calibration_summary"]
    assert summary["n"] == 2
    assert summary["origenes"] == {ORIGIN_INDEX: 1, ORIGIN_MANUAL: 1}
    assert len(summary["avisos"]) == 1

    loaded = load_run(result.run_dir)
    assert loaded.calibrations[first].origin == ORIGIN_INDEX
    assert loaded.calibrations[second].origin == ORIGIN_MANUAL


# ── CLI ─────────────────────────────────────────────────────────────────


def install_factory(monkeypatch, detector: FakeDetector) -> list[tuple[str, Path]]:
    calls: list[tuple[str, Path]] = []

    def factory(name, weights=None, **kwargs):
        calls.append((name, Path(weights)))
        return detector

    monkeypatch.setattr(cli_detect, "make_detector", factory)
    return calls


def weights_file(tmp_path: Path) -> Path:
    path = tmp_path / "best.pt"
    path.write_bytes(b"weights")
    return path


def run_cli(tmp_path: Path, image: Path, *extra: str) -> int:
    return cli_detect.main(
        [
            "--model",
            str(weights_file(tmp_path)),
            "--images",
            str(image.parent),
            "--out",
            str(tmp_path / "run"),
            *extra,
        ]
    )


def test_cli_defaults_to_auto_and_measures_the_plate(tmp_path, monkeypatch):
    image = write_plate(tmp_path / "photos" / "plate.png")
    install_factory(monkeypatch, FakeDetector())

    assert run_cli(tmp_path, image) == 0

    run = load_run(tmp_path / "run")
    calibration = run.calibrations[image]
    assert calibration.origin == ORIGIN_PLATE
    assert calibration.um_por_px == pytest.approx(PLATE_UM_PER_PX, rel=0.03)
    assert calibration.valid


def test_cli_um_per_px_implies_manual(tmp_path, monkeypatch):
    image = write_plate(tmp_path / "photos" / "plate.png")
    write_index(tmp_path / "photos" / "indice.csv", {"plate.png": 31.0})
    install_factory(monkeypatch, FakeDetector())

    assert run_cli(tmp_path, image, "--um-per-px", "2.5") == 0

    calibration = load_run(tmp_path / "run").calibrations[image]
    assert calibration.origin == ORIGIN_MANUAL
    assert calibration.um_por_px == pytest.approx(2.5)


def test_cli_explicit_auto_keeps_um_per_px_as_the_last_fallback(tmp_path, monkeypatch):
    image = write_plain(tmp_path / "photos" / "plate.png")
    install_factory(monkeypatch, FakeDetector())

    assert run_cli(tmp_path, image, "--calibrate", "auto", "--um-per-px", "2.5") == 0

    calibration = load_run(tmp_path / "run").calibrations[image]
    assert calibration.origin == ORIGIN_MANUAL
    assert calibration.um_por_px == pytest.approx(2.5)
    assert calibration.warning


def test_cli_calibrate_none_overrides_the_manual_value(tmp_path, monkeypatch):
    image = write_plain(tmp_path / "photos" / "plate.png")
    install_factory(monkeypatch, FakeDetector())

    assert run_cli(tmp_path, image, "--calibrate", "none", "--um-per-px", "2.5") == 0

    run = load_run(tmp_path / "run")
    calibration = run.calibrations[image]
    assert calibration.origin == ORIGIN_NONE
    assert not calibration.valid
    assert all(det.diam_um is None for det in run.results[0][0].predictions)


def test_cli_index_calibration_uses_the_given_csv(tmp_path, monkeypatch):
    image = write_plain(tmp_path / "photos" / "plate.png")
    explicit = write_index(tmp_path / "calibration.csv", {"plate.png": 31.25})
    install_factory(monkeypatch, FakeDetector())

    assert (
        run_cli(
            tmp_path,
            image,
            "--calibrate",
            "index",
            "--index-calibration",
            str(explicit),
        )
        == 0
    )

    calibration = load_run(tmp_path / "run").calibrations[image]
    assert calibration.origin == ORIGIN_INDEX
    assert calibration.um_por_px == pytest.approx(31.25)


def test_cli_index_mode_searches_the_image_root(tmp_path, monkeypatch):
    image = write_plain(tmp_path / "photos" / "plate.png")
    write_index(tmp_path / "photos" / "calibracion_placas.csv", {"plate.png": 32.5}, "placa")
    install_factory(monkeypatch, FakeDetector())

    assert run_cli(tmp_path, image, "--calibrate", "index") == 0

    calibration = load_run(tmp_path / "run").calibrations[image]
    assert calibration.origin == ORIGIN_INDEX
    assert calibration.um_por_px == pytest.approx(32.5)


# ── Qt-free ─────────────────────────────────────────────────────────────


def test_auto_calibration_runs_and_loads_without_importing_qt(tmp_path):
    image = write_plain(tmp_path / "photos" / "plate.png")
    out = tmp_path / "run"
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from polyx.core.detect.protocol import Instance\n"
        "from polyx.core.detection_run import (\n"
        "    CalibrationPolicy, ModelSpec, RunConfig, RunParams, load_run, run_detection,\n"
        ")\n"
        "class Detector:\n"
        "    names = {0: 'PET'}\n"
        "    last_imgsz = 640\n"
        "    last_fallback = False\n"
        "    def predict_auto(self, image_path, **kwargs):\n"
        "        return [Instance(class_id=0, class_name='PET', conf=0.9,\n"
        "                         x1=20, y1=20, x2=60, y2=60)]\n"
        "image, out = Path(sys.argv[1]), Path(sys.argv[2])\n"
        "result = run_detection(RunConfig(\n"
        "    images=[image],\n"
        "    models=[ModelSpec(detector=Detector(), alias='model')],\n"
        "    output_dir=out,\n"
        "    params=RunParams(measure_shape=False),\n"
        "    calibration=CalibrationPolicy(mode='auto', um_per_px=1.0),\n"
        "))\n"
        "loaded = load_run(out)\n"
        "assert result.status == 'completed' and loaded.status == 'completed'\n"
        "assert loaded.calibrations[image].origin == 'manual'\n"
        "bad = sorted(m for m in sys.modules if m == 'PySide6' or m.startswith('PySide6.'))\n"
        "print('QT_LOADED=' + repr(bad))\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = "0"
    env["POLYX_LOG_DIR"] = str(tmp_path / "logs")
    result = subprocess.run(  # noqa: PLW1510
        [sys.executable, "-c", script, str(image), str(out)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        "executing/loading an auto-calibrated run dragged Qt or failed:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "QT_LOADED=[]" in result.stdout
