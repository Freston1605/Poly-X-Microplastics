"""The ``run.json`` contract at the artifact I/O seam.

These tests observe the bytes written to ``run.json`` and the error ``load_run``
raises. They do not assert private helpers or internal constant names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polyx.core.detection_run import RunResult
from polyx.core.detection_run.schema import build_run_payload, load_run


def _payload() -> dict:
    """A complete run record with the fewest moving parts the writer allows."""
    return build_run_payload(RunResult(run_dir=Path("run")))


def test_new_run_json_declares_schema_version_two(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps(_payload()), encoding="utf-8")

    document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    assert load_run(run_dir).run_dir == run_dir


def test_load_run_rejects_a_version_one_file_with_an_actionable_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "old_run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        load_run(run_dir)

    message = str(caught.value)
    assert "version 1" in message
    assert "old" in message.lower()
    assert "docs/specs/run-artifact-vocabulary.md" in message


def test_load_run_rejects_any_other_version(tmp_path: Path) -> None:
    run_dir = tmp_path / "future_run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"docs/specs/run-artifact-vocabulary\.md"):
        load_run(run_dir)


def test_provenance_in_run_json_uses_english_keys_and_a_boolean_dirty(tmp_path: Path) -> None:
    from polyx.core.provenance import provenance, versions

    block = provenance(tmp_path)
    assert set(block) == {"commit", "dirty", "versions"}
    assert isinstance(block["dirty"], bool)
    assert set(versions()) == {"polyx", "python", "torch", "ultralytics", "opencv", "numpy", "os"}

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(build_run_payload(RunResult(run_dir=run_dir), block)), encoding="utf-8"
    )
    written = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["provenance"]
    assert written == block
    assert "sucio" not in written and "versiones" not in written


def test_run_json_calibration_tokens_are_english(tmp_path: Path) -> None:
    from polyx.core.measure.calibration import (
        ORIGIN_INDEX,
        ORIGIN_MANUAL,
        ORIGIN_NONE,
        ORIGIN_PLATE,
        Calibration,
    )

    images = {
        name: tmp_path / name for name in ("plate.png", "index.png", "manual.png", "none.png")
    }
    result = RunResult(
        run_dir=tmp_path / "run",
        calibrations={
            images["plate.png"]: Calibration(um_por_px=30.0, origin=ORIGIN_PLATE),
            images["index.png"]: Calibration(um_por_px=31.0, origin=ORIGIN_INDEX),
            images["manual.png"]: Calibration(um_por_px=32.0, origin=ORIGIN_MANUAL),
            images["none.png"]: Calibration(origin=ORIGIN_NONE),
        },
    )
    payload = build_run_payload(result)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    raw = (run_dir / "run.json").read_text(encoding="utf-8")
    assert '"um_per_px"' in raw and '"um_por_px"' not in raw
    assert '"placa"' not in raw and '"indice"' not in raw and '"ninguna"' not in raw
    document = json.loads(raw)

    by_image = {entry["image"]: entry for entry in document["calibrations"]}
    assert by_image[str(images["plate.png"])]["origin"] == "plate"
    assert by_image[str(images["index.png"])]["origin"] == "index"
    assert by_image[str(images["manual.png"])]["origin"] == "manual"
    assert by_image[str(images["none.png"])]["origin"] == "none"
    assert by_image[str(images["plate.png"])]["um_per_px"] == 30.0

    summary = document["calibration_summary"]
    assert set(summary) == {
        "n",
        "min",
        "max",
        "std",
        "mean",
        "median",
        "variation",
        "origins",
        "warnings",
    }
    assert summary["origins"] == {"plate": 1, "index": 1, "manual": 1}
    assert summary["mean"] == pytest.approx(31.0)
    assert summary["median"] == pytest.approx(31.0)
    assert summary["n"] == 3

    # The canonical reader sees the same scale and origin it wrote.
    loaded = load_run(run_dir)
    assert loaded.calibrations[images["plate.png"]].origin == "plate"
    assert loaded.calibrations[images["plate.png"]].um_por_px == 30.0


def test_report_keeps_the_same_origin_meanings_with_english_tokens(
    tmp_path: Path, selected_language
) -> None:
    selected_language("en")
    from polyx.core.measure.calibration import (
        ORIGIN_INDEX,
        ORIGIN_MANUAL,
        ORIGIN_PLATE,
        Calibration,
    )
    from polyx.gui.report_html import generate_report

    images = {name: tmp_path / name for name in ("plate.png", "index.png", "manual.png")}
    run = RunResult(
        run_dir=tmp_path / "run",
        calibrations={
            images["plate.png"]: Calibration(um_por_px=30.0, origin=ORIGIN_PLATE),
            images["index.png"]: Calibration(um_por_px=31.0, origin=ORIGIN_INDEX),
            images["manual.png"]: Calibration(um_por_px=32.0, origin=ORIGIN_MANUAL),
        },
    )
    html = generate_report(run, tmp_path / "report.html").read_text(encoding="utf-8")

    assert "measured on this photograph's Petri dish" in html
    assert "inherited from the crop (calibration index)" in html
    assert "entered by hand in Parameters" in html
    # Each origin still gets only its own caveat, now keyed by the English token.
    assert "The length reference is the nominal outer diameter" in html
    assert "The scale was entered by hand in Parameters" in html
    assert "The scale is inherited from the calibration index" in html
