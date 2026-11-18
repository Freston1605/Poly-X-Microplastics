"""Contract tests for the per-photo calibration policy.

Ticket #60: the canonical run owns how every photo gets its micrometres per
pixel. ``auto`` measures the plate on the photo and then falls back to the
manual value; ``manual`` uses only the typed value and ``none`` leaves the run
uncalibrated. Pre-canonical index CSVs are not read by new code. The origin,
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

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="manual", um_per_px=2.5),
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

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="manual", um_per_px=0.0),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_NONE
    assert not calibration.valid
    assert calibration.warning
    assert all(det.diam_um is None for det in result.results[0][0].predictions)


def test_none_never_invents_micrometre_sizes(tmp_path):
    image = write_plate(tmp_path / "photos" / "plate.png")

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="none", um_per_px=9.0),
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


# ── Auto fallback order ─────────────────────────────────────────────────


def test_auto_follows_plate_manual_order(tmp_path):
    plate_image = write_plate(tmp_path / "photos" / "plate.png")
    manual_image = write_plain(tmp_path / "photos" / "manual.png")

    result = run_detection(
        make_config(
            [plate_image, manual_image],
            tmp_path / "run",
            CalibrationPolicy(mode="auto", um_per_px=2.5),
        )
    )

    by_name = {path.name: calibration for path, calibration in result.calibrations.items()}
    assert by_name["plate.png"].origin == ORIGIN_PLATE
    assert by_name["plate.png"].um_por_px == pytest.approx(PLATE_UM_PER_PX, rel=0.03)
    assert by_name["manual.png"].origin == ORIGIN_MANUAL
    assert by_name["manual.png"].um_por_px == pytest.approx(2.5)
    assert by_name["manual.png"].warning


def test_auto_ignores_a_precanonical_index_file_without_deleting_it(tmp_path):
    """An old index CSV stays on disk but no new code reads it."""
    image = write_plain(tmp_path / "photos" / "plate.png")
    index = write_index(tmp_path / "photos" / "indice.csv", {"plate.png": 42.0})

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="auto", um_per_px=2.5),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_MANUAL
    assert calibration.um_por_px == pytest.approx(2.5)
    assert index.read_text(encoding="utf-8") == "recorte,um_por_px\nplate.png,42.0\n"


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
    for entry in entries.values():
        assert entry["origin"] == ORIGIN_MANUAL
        assert entry["valid"] is True
        assert entry["warning"]
        assert {"origin", "valid", "warning", "um_per_px"} <= set(entry)

    summary = payload["calibration_summary"]
    assert summary["n"] == 2
    assert summary["origins"] == {ORIGIN_MANUAL: 2}
    # Persisted warnings are stable codes, never the sentence a person reads.
    warnings = summary["warnings"]
    assert len(warnings) == 2
    assert {item["code"] for item in warnings} == {
        entries["first.png"]["warning"]["code"],
        entries["second.png"]["warning"]["code"],
    }
    assert all(" " not in item["code"] for item in warnings)

    loaded = load_run(result.run_dir)
    assert loaded.calibrations[first].origin == ORIGIN_MANUAL
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
    # The unused manual value must not survive in the run parameters.
    assert "um_per_px" not in calibration_payload(run.run_dir)["params"]


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
