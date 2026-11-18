"""Per-image calibration index: English-only, strict, one scale per photo.

Crops whose Petri dish is not visible cannot be calibrated from the plate. One
batch-wide manual scale would give wrong sizes when photographs have different
scales. The index is the safe per-image source: these tests exercise the public
seams -- the strict reader, the engine that applies one scale per image, and the
CLI flag -- without restoring any legacy reader. Legacy files stay on disk
untouched and are rejected by name and by header.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from polyx.cli import detect as cli_detect
from polyx.core.detect.protocol import Instance
from polyx.core.detection_run import (
    CALIBRATION_MODES,
    CalibrationPolicy,
    ModelSpec,
    RunConfig,
    RunParams,
    load_run,
    run_detection,
)
from polyx.core.measure.calibration import (
    CALIBRATION_INDEX_NAME,
    ORIGIN_INDEX,
    ORIGIN_NONE,
    read_calibration_index,
)
from polyx.core.measure.warnings import (
    IMAGE_NOT_IN_INDEX,
    WARNING_CODES,
    Warning,
    warning_sentence,
)

CANONICAL_HEADER = "image,um_per_px"
CANONICAL_NAME = CALIBRATION_INDEX_NAME
BOX = (20.0, 20.0, 60.0, 60.0)


def write_index(path: Path, rows: list[str], header: str = CANONICAL_HEADER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def write_plain(path: Path, side: int = 160) -> Path:
    """A crop with one bright square and no plate ring."""
    image = np.zeros((side, side, 3), dtype=np.uint8)
    x1, y1, x2, y2 = (int(value) for value in BOX)
    image[y1 + 5 : y2 - 5, x1 + 5 : x2 - 5] = 255
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buffer))
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


def make_config(images: list[Path], output_dir: Path, calibration: CalibrationPolicy) -> RunConfig:
    return RunConfig(
        images=list(images),
        models=[ModelSpec(detector=FakeDetector(), alias="model")],
        output_dir=output_dir,
        params=RunParams(measure_shape=False),
        calibration=calibration,
    )


# ── Reader seam ──────────────────────────────────────────────────────────


def test_canonical_index_is_read_keyed_by_image_basename(tmp_path):
    index = write_index(
        tmp_path / CANONICAL_NAME,
        ["first.png,2.5", "second.png,3.75", "nested/third.png,4.0"],
    )

    mapping = read_calibration_index(index)

    assert mapping == {"first.png": 2.5, "second.png": 3.75, "third.png": 4.0}


@pytest.mark.parametrize("legacy_name", ["indice.csv", "calibracion_placas.csv"])
def test_legacy_filenames_are_rejected_by_name(tmp_path, legacy_name):
    """A legacy file with even canonical headers is still refused."""
    index = write_index(tmp_path / legacy_name, ["first.png,2.5"])
    before = index.read_bytes()

    with pytest.raises(ValueError, match=r"calibration_index\.csv") as excinfo:
        read_calibration_index(index)

    assert "image,um_per_px" in str(excinfo.value)
    assert index.read_bytes() == before


@pytest.mark.parametrize("bad_header", ["recorte,um_por_px", "placa,um_por_px"])
def test_legacy_spanish_headers_are_rejected(tmp_path, bad_header):
    index = write_index(tmp_path / CANONICAL_NAME, ["first.png,2.5"], header=bad_header)

    with pytest.raises(ValueError, match=r"calibration_index\.csv") as excinfo:
        read_calibration_index(index)

    assert "image,um_per_px" in str(excinfo.value)


def test_legacy_value_header_is_rejected(tmp_path):
    index = write_index(tmp_path / CANONICAL_NAME, ["first.png,2.5"], header="image,um_por_px")

    with pytest.raises(ValueError) as excinfo:
        read_calibration_index(index)

    message = str(excinfo.value)
    assert "calibration_index.csv" in message
    assert "image,um_per_px" in message


@pytest.mark.parametrize("bad_row", ["first.png,", "first.png,abc", "first.png,0", "first.png,-2"])
def test_invalid_values_are_rejected(tmp_path, bad_row):
    index = write_index(tmp_path / CANONICAL_NAME, [bad_row])

    with pytest.raises(ValueError, match=r"first\.png"):
        read_calibration_index(index)


@pytest.mark.parametrize("bad_row", ["first.png,nan", "first.png,inf"])
def test_non_finite_values_are_rejected(tmp_path, bad_row):
    index = write_index(tmp_path / CANONICAL_NAME, [bad_row])

    with pytest.raises(ValueError, match=r"first\.png"):
        read_calibration_index(index)


def test_duplicate_image_rows_are_rejected(tmp_path, selected_language):
    selected_language("en")
    index = write_index(tmp_path / CANONICAL_NAME, ["first.png,2.5", "first.png,3.5"])

    with pytest.raises(ValueError, match=r"first\.png") as excinfo:
        read_calibration_index(index)

    assert "duplicate" in str(excinfo.value)


def test_blank_image_name_is_rejected(tmp_path):
    index = write_index(tmp_path / CANONICAL_NAME, [",2.5"])

    with pytest.raises(ValueError):
        read_calibration_index(index)


def test_missing_file_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match=r"calibration_index\.csv"):
        read_calibration_index(tmp_path / CANONICAL_NAME)


def test_reader_never_rewrites_the_source(tmp_path):
    index = write_index(tmp_path / CANONICAL_NAME, ["first.png,2.5"])
    legacy = write_index(tmp_path / "indice.csv", ["first.png,2.5"], header="recorte,um_por_px")
    canonical_before, legacy_before = index.read_bytes(), legacy.read_bytes()

    read_calibration_index(index)
    with pytest.raises(ValueError):
        read_calibration_index(legacy)

    assert index.read_bytes() == canonical_before
    assert legacy.read_bytes() == legacy_before


# ── Engine seam ──────────────────────────────────────────────────────────


def test_index_is_an_explicit_calibration_mode():
    assert "index" in CALIBRATION_MODES


def test_two_images_get_their_own_scale_and_measured_size(tmp_path):
    """The two-images-different-scales regression: no batch-wide fallback."""
    first = write_plain(tmp_path / "photos" / "first.png")
    second = write_plain(tmp_path / "photos" / "second.png")
    index = write_index(
        tmp_path / CANONICAL_NAME,
        ["first.png,2.0", "second.png,4.0"],
    )

    result = run_detection(
        make_config(
            [first, second],
            tmp_path / "run",
            CalibrationPolicy(mode="index", index_path=index),
        )
    )

    by_name = {path.name: calibration for path, calibration in result.calibrations.items()}
    assert by_name["first.png"].origin == ORIGIN_INDEX
    assert by_name["first.png"].um_por_px == pytest.approx(2.0)
    assert by_name["second.png"].origin == ORIGIN_INDEX
    assert by_name["second.png"].um_por_px == pytest.approx(4.0)
    assert not any(calibration.warning for calibration in by_name.values())

    first_size = result.results[0][0].predictions[0].diam_um
    second_size = result.results[0][1].predictions[0].diam_um
    assert first_size is not None and second_size is not None
    assert second_size == pytest.approx(2.0 * first_size)

    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["calibration_summary"]["origins"] == {ORIGIN_INDEX: 2}
    loaded = load_run(result.run_dir)
    assert loaded.calibrations[first].um_por_px == pytest.approx(2.0)
    assert loaded.calibrations[second].um_por_px == pytest.approx(4.0)


def test_missing_entry_is_recorded_with_warning_and_no_scale(tmp_path):
    """All-calibrations regression: every image gets a record, even a miss."""
    listed = write_plain(tmp_path / "photos" / "listed.png")
    missing = write_plain(tmp_path / "photos" / "missing.png")
    index = write_index(tmp_path / CANONICAL_NAME, ["listed.png,3.0"])

    result = run_detection(
        make_config(
            [listed, missing],
            tmp_path / "run",
            CalibrationPolicy(mode="index", index_path=index),
        )
    )

    by_name = {path.name: calibration for path, calibration in result.calibrations.items()}
    assert set(by_name) == {"listed.png", "missing.png"}
    assert by_name["listed.png"].valid
    missed = by_name["missing.png"]
    assert not missed.valid
    assert missed.um_por_px == 0.0
    assert missed.origin == ORIGIN_NONE
    assert missed.warning is not None
    assert missed.warning.code == IMAGE_NOT_IN_INDEX
    assert all(det.diam_um is None for det in result.results[0][1].predictions)

    payload = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    entries = {Path(entry["image"]).name: entry for entry in payload["calibrations"]}
    assert set(entries) == {"listed.png", "missing.png"}
    assert entries["missing.png"]["warning"] == {"code": IMAGE_NOT_IN_INDEX}
    assert {item["code"] for item in payload["calibration_summary"]["warnings"]} == {
        IMAGE_NOT_IN_INDEX
    }


def test_index_mode_does_not_measure_the_plate_or_use_the_manual_value(tmp_path):
    """The index is the only source in its mode, even for a visible plate."""
    from tests.test_calibration_policy import write_plate

    image = write_plate(tmp_path / "photos" / "plate.png")
    index = write_index(tmp_path / CANONICAL_NAME, ["plate.png,7.5"])

    result = run_detection(
        make_config(
            [image],
            tmp_path / "run",
            CalibrationPolicy(mode="index", index_path=index, um_per_px=2.5),
        )
    )

    calibration = result.calibrations[image]
    assert calibration.origin == ORIGIN_INDEX
    assert calibration.um_por_px == pytest.approx(7.5)


def test_index_mode_without_a_path_fails_clearly_and_writes_nothing(tmp_path):
    image = write_plain(tmp_path / "photos" / "first.png")
    out = tmp_path / "run"

    with pytest.raises(ValueError, match="--calibration-index"):
        run_detection(make_config([image], out, CalibrationPolicy(mode="index")))

    assert not out.exists()


def test_same_basename_paths_collapse_to_one_index_key(tmp_path):
    """The collision the guard prevents, shown at the reader seam.

    The reader keys by basename and the engine looks up ``image_path.name``,
    so two photos in different folders with the same name resolve to the
    single ``sample.png`` row -- one scale silently applied to both.
    """
    index = write_index(tmp_path / CANONICAL_NAME, ["sample.png,5.0"])
    mapping = read_calibration_index(index)

    site_a = tmp_path / "site_A" / "sample.png"
    site_b = tmp_path / "site_B" / "sample.png"
    assert site_a.name == site_b.name == "sample.png"
    assert mapping[site_a.name] == mapping[site_b.name] == 5.0


def test_duplicate_image_basenames_are_refused_in_index_mode(tmp_path, selected_language):
    selected_language("en")
    site_a = write_plain(tmp_path / "site_A" / "sample.png")
    site_b = write_plain(tmp_path / "site_B" / "sample.png")
    index = write_index(tmp_path / CANONICAL_NAME, ["sample.png,5.0"])
    out = tmp_path / "run"

    with pytest.raises(ValueError, match=r"sample\.png") as excinfo:
        run_detection(
            make_config(
                [site_a, site_b],
                out,
                CalibrationPolicy(mode="index", index_path=index),
            )
        )

    message = str(excinfo.value)
    assert "--calibration-index" in message
    assert "unique" in message.lower()
    assert str(site_a) in message
    assert str(site_b) in message
    # The refusal is before inference and before the output folder is created.
    assert not out.exists()
    assert not any(tmp_path.rglob("run.json"))


def test_duplicate_image_basenames_are_allowed_in_manual_mode(tmp_path):
    """The refusal is index-only: other modes may reuse a basename."""
    site_a = write_plain(tmp_path / "site_A" / "sample.png")
    site_b = write_plain(tmp_path / "site_B" / "sample.png")

    result = run_detection(
        make_config(
            [site_a, site_b],
            tmp_path / "run",
            CalibrationPolicy(mode="manual", um_per_px=2.0),
        )
    )

    assert len(result.results[0]) == 2


# ── Warning-code seam ────────────────────────────────────────────────────


def test_the_missing_index_warning_has_a_code_and_a_sentence(selected_language):
    selected_language("en")
    assert IMAGE_NOT_IN_INDEX == "image_not_in_index"
    assert IMAGE_NOT_IN_INDEX in WARNING_CODES
    assert warning_sentence(Warning(IMAGE_NOT_IN_INDEX)) == (
        "The image is not listed in the calibration index."
    )


# ── CLI seam ─────────────────────────────────────────────────────────────


def install_factory(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_detect, "make_detector", lambda name, weights=None, **kw: FakeDetector()
    )


def run_cli(tmp_path: Path, image: Path, index: Path, *extra: str) -> int:
    return cli_detect.main(
        [
            "--model",
            str(tmp_path / "best.pt"),
            "--images",
            str(image.parent),
            "--out",
            str(tmp_path / "run"),
            *extra,
        ]
    )


def test_cli_calibration_index_is_applied_per_image(tmp_path, monkeypatch, capsys):
    first = write_plain(tmp_path / "photos" / "first.png")
    second = write_plain(tmp_path / "photos" / "second.png")
    index = write_index(tmp_path / CANONICAL_NAME, ["first.png,2.0", "second.png,4.0"])
    (tmp_path / "best.pt").write_bytes(b"weights")
    install_factory(monkeypatch)

    assert (
        run_cli(tmp_path, first, index, "--calibrate", "index", "--calibration-index", str(index))
        == 0
    )

    run = load_run(tmp_path / "run")
    assert run.calibrations[first].um_por_px == pytest.approx(2.0)
    assert run.calibrations[second].um_por_px == pytest.approx(4.0)
    assert capsys.readouterr().err == ""


def test_cli_index_flag_alone_selects_index_mode(tmp_path, monkeypatch):
    image = write_plain(tmp_path / "photos" / "first.png")
    index = write_index(tmp_path / CANONICAL_NAME, ["first.png,5.0"])
    (tmp_path / "best.pt").write_bytes(b"weights")
    install_factory(monkeypatch)

    assert run_cli(tmp_path, image, index, "--calibration-index", str(index)) == 0

    assert load_run(tmp_path / "run").calibrations[image].origin == ORIGIN_INDEX


def test_cli_rejects_a_legacy_index_csv_clearly(tmp_path, monkeypatch, capsys):
    image = write_plain(tmp_path / "photos" / "first.png")
    legacy = write_index(tmp_path / "indice.csv", ["first.png,2.5"])
    (tmp_path / "best.pt").write_bytes(b"weights")
    install_factory(monkeypatch)

    code = run_cli(tmp_path, image, legacy, "--calibration-index", str(legacy))

    assert code != 0
    error = capsys.readouterr().err
    assert "calibration_index.csv" in error
    assert "image,um_per_px" in error
