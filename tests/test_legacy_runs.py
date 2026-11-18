"""Pre-canonical formats are rejected: old runs need an older Poly-X version.

The hard cut removes every reader of a pre-canonical artifact. Old files stay
on disk untouched, but the compact report, the report CLI and the run reader
must refuse them with a clear pointer to an older version instead of emitting
a partial or misleading result. Canonical folders keep working unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "legacy_runs"
HEADLESS_RUN = FIXTURES / "headless" / "run"
HEADLESS_PHOTOS = FIXTURES / "headless" / "photos"

OLD_RUN_MESSAGE = "older version of Poly-X"


# ── The pre-canonical run reader is gone ────────────────────────────────


def test_the_precanonical_run_reader_is_removed():
    """No reader, no export and no per-result legacy flag survives."""
    import polyx.core.detection_run as detection_run
    from polyx.core.detection_run import RunResult

    source = Path(__file__).resolve().parents[1] / "src" / "polyx"

    assert not hasattr(detection_run, "load_legacy_run")
    assert "load_legacy_run" not in detection_run.__all__
    assert not (source / "core" / "detection_run" / "legacy_reader.py").exists()
    texts = {path: path.read_text(encoding="utf-8") for path in source.rglob("*.py")}
    assert [str(p.relative_to(source)) for p, t in texts.items() if "load_legacy_run" in t] == []
    assert "legacy" not in {field.name for field in RunResult.__dataclass_fields__.values()}


# ── Compact report: canonical only ─────────────────────────────────────


def test_compact_report_refuses_a_precanonical_folder(tmp_path, selected_language):
    """A summary.json + centroids.csv folder is not silently summarised."""
    selected_language("en")
    from polyx.core.report_simple import LegacyRunError, generate

    output = tmp_path / "report.html"
    with pytest.raises(LegacyRunError, match=OLD_RUN_MESSAGE):
        generate(HEADLESS_RUN, output)

    assert not output.exists()


def test_compact_report_still_reads_a_canonical_folder(tmp_path, selected_language):
    """The Qt-free report keeps its canonical seam after the fallback is cut."""
    selected_language("en")
    import json

    from polyx.core.report_simple import generate

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "completed",
                "images": ["plate.png"],
                "models": [{"alias": "best", "alias_generated": False}],
                "metrics": {"0": {"tp": 1, "fp": 0, "fn": 0, "miscls": 0, "detections": 1}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "detections.csv").write_text("class,image\nPET,plate.png\n", encoding="utf-8")

    output = generate(run_dir, tmp_path / "report.html")

    html = output.read_text(encoding="utf-8")
    assert "Poly-X run report" in html
    assert "Images:" in html
    assert "Detections:" in html


# ── Report CLI: both modes reject old runs ─────────────────────────────


def test_report_cli_refuses_a_precanonical_folder(tmp_path, capsys, selected_language):
    """`polyx-report` names an older version instead of writing a compact report."""
    selected_language("en")
    from polyx.cli.report import main

    output = tmp_path / "report.html"
    code = main(["--run", str(HEADLESS_RUN), "--output", str(output)])

    captured = capsys.readouterr()
    assert code != 0
    assert OLD_RUN_MESSAGE in captured.err
    assert not output.exists()


def test_report_cli_full_report_refuses_a_precanonical_folder(tmp_path, capsys, selected_language):
    """`--full-report` must not fall back to a compact report on rejection."""
    selected_language("en")
    from polyx.cli.report import main

    output = tmp_path / "report.html"
    code = main(["--run", str(HEADLESS_RUN), "--full-report", "--output", str(output)])

    captured = capsys.readouterr()
    assert code != 0
    assert OLD_RUN_MESSAGE in captured.err
    assert "COMPACT REPORT" not in captured.out
    assert not output.exists()


def test_report_cli_no_longer_accepts_the_images_flag(capsys):
    """The flag existed only to feed the deleted legacy reader."""
    from polyx.cli.report import main

    with pytest.raises(SystemExit) as exc:
        main(["--run", str(HEADLESS_RUN), "--images", str(HEADLESS_PHOTOS)])

    assert exc.value.code == 2
    assert "--images" in capsys.readouterr().err


# ── Calibration index CSV reader is gone ───────────────────────────────


def test_the_calibration_index_csv_reader_is_removed():
    """No CSV reader, index policy field or index mode survives the cut."""
    import inspect

    from polyx.core.detection_run import CALIBRATION_MODES, CalibrationPolicy
    from polyx.core.measure import __all__ as measure_exports
    from polyx.core.measure import calibration as cal

    for name in ("carry_index", "NAMES_INDEX", "search_index"):
        assert not hasattr(cal, name)
        assert name not in measure_exports

    assert "index" not in inspect.signature(cal.resolver).parameters
    assert not hasattr(CalibrationPolicy(), "index_path")
    assert "index" not in CALIBRATION_MODES


def test_cli_detect_refuses_the_index_calibration_flag(tmp_path, capsys):
    """`--index-calibration` belonged only to the deleted CSV reader."""
    from polyx.cli.detect import main

    index = tmp_path / "indice.csv"
    index.write_text("recorte,um_per_px\nplate1.png,31.0\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--model",
                "model.pt",
                "--images",
                str(HEADLESS_PHOTOS),
                "--out",
                str(tmp_path / "run"),
                "--index-calibration",
                str(index),
            ]
        )

    assert exc.value.code == 2
    assert "--index-calibration" in capsys.readouterr().err


# ── Canonical folders still work ───────────────────────────────────────


def test_core_exposes_the_canonical_engine_and_not_the_old_batch():
    import polyx.core as core
    import polyx.core.detection_run as detection_run

    assert not hasattr(core, "run_batch")
    assert "run_batch" not in core.__all__
    assert core.run_detection is detection_run.run_detection
    assert core.RunConfig is detection_run.RunConfig
    assert core.RunResult is detection_run.RunResult
    assert core.load_run is detection_run.load_run


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
