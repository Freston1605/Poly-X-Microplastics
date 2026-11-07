"""The compact report, the training summary and the CLI follow the language.

These are the three non-interactive surfaces that ticket #71 finishes: each
one renders text the user reads, and each one used to be hardcoded (English in
the compact report and the CLI, Spanish in the training summary). The seam is
the same for all three: select a language, render the surface, assert what the
user sees. No test here looks at catalogues or at how the string was wrapped.

The suite pins ``POLYX_LOCALE=es`` in ``conftest.py``; these tests switch the
active language explicitly and restore it afterwards, because the point is
that the *selected* language reaches the output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _canonical_run(tmp_path: Path) -> Path:
    """A minimal canonical run folder the compact report can read."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "schema_version": 1,
                "images": [{"path": "plate1.png"}],
                "models": [{"alias": "m1"}],
                "metrics": {
                    "m1": {"tp": 1, "fp": 0, "fn": 0, "miscls": 0, "detections": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "detections.csv").write_text("image,class,conf\nplate1.png,PET,0.90\n", encoding="utf-8")
    return run


def _training_run(tmp_path: Path) -> Path:
    """A minimal Ultralytics run folder the training summary can read."""
    run = tmp_path / "runs_train" / "basico"
    run.mkdir(parents=True)
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.50,0.30\n2,0.65,0.35\n",
        encoding="utf-8",
    )
    return run


# ── Compact report ──────────────────────────────────────────────────────


def test_compact_report_renders_the_selected_language(tmp_path, selected_language):
    from polyx.core.report_simple import generate

    run = _canonical_run(tmp_path)

    selected_language("es")
    spanish = generate(run).read_text(encoding="utf-8")

    selected_language("en")
    english = generate(run).read_text(encoding="utf-8")

    assert "<html lang='es'>" in spanish
    assert "Informe de ejecución de Poly-X" in spanish
    assert "<html lang='en'>" in english
    assert "Poly-X run report" in english


def test_compact_report_uses_the_catalog_plural_forms(tmp_path, selected_language):
    from polyx.core.report_simple import generate

    run = _canonical_run(tmp_path)
    payload = json.loads((run / "run.json").read_text(encoding="utf-8"))
    payload["skipped_images"] = ["a.png", "b.png"]
    (run / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    rows = ["image,class,conf"] + [f"plate{i}.png,PET,0.90" for i in range(502)]
    (run / "detections.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    selected_language("es")
    spanish = generate(run).read_text(encoding="utf-8")

    assert "2 imágenes omitidas" in spanish
    assert "se omiten 2 filas más" in spanish


def test_a_saved_compact_report_keeps_the_language_it_was_generated_in(tmp_path, selected_language):
    from polyx.core.report_simple import generate

    selected_language("es")
    output = generate(_canonical_run(tmp_path))

    selected_language("en")

    assert "<html lang='es'>" in output.read_text(encoding="utf-8")


# ── Training summary ────────────────────────────────────────────────────


def test_training_summary_renders_the_selected_language(tmp_path, selected_language):
    from polyx.gui.trainer.pages.summary import build_training_report

    run = _training_run(tmp_path)

    selected_language("es")
    spanish = build_training_report(run, tmp_path / "es.html").read_text(encoding="utf-8")

    selected_language("en")
    english = build_training_report(run, tmp_path / "en.html").read_text(encoding="utf-8")

    assert "<html lang='es'>" in spanish
    assert "Informe del entrenamiento" in spanish
    assert "<html lang='en'>" in english
    assert "Training report" in english


# ── Command line ────────────────────────────────────────────────────────


def test_command_line_help_follows_the_selected_language(selected_language, capsys):
    from polyx.cli import detect

    selected_language("es")
    with pytest.raises(SystemExit):
        detect.main(["--help"])
    assert "Detección por lotes de microplásticos (sin interfaz)" in capsys.readouterr().out

    selected_language("en")
    with pytest.raises(SystemExit):
        detect.main(["--help"])
    assert "Batch-detect microplastics (headless)" in capsys.readouterr().out


def test_command_line_error_follows_the_selected_language(tmp_path, selected_language, capsys):
    from polyx.cli import detect

    selected_language("es")

    code = detect.main(
        [
            "--model",
            str(tmp_path / "a.pt"),
            "--model",
            str(tmp_path / "b.pt"),
            "--model-type",
            "yolo",
            "--model-type",
            "yolo-seg",
            "--model-type",
            "native",
            "--images",
            str(tmp_path),
            "--out",
            str(tmp_path / "run"),
        ]
    )

    assert code == detect.EXIT_USAGE
    assert "toma un valor para todos los modelos" in capsys.readouterr().err


# ── Dynamically resolved definition-site strings ────────────────────────


def test_preset_tables_are_extracted_from_their_definition_site():
    """A table key resolved later still has to reach the committed template."""
    from tools.i18n import po

    template = po.parse_po(
        (Path(__file__).resolve().parents[1] / "src/polyx/locales/messages.pot").read_text(
            encoding="utf-8"
        )
    )
    message_ids = {entry.msgid for entry in template}

    for message in (
        "Balanced (recommended)",
        "Fast (little data)",
        "Long (paper-quality)",
        "Maximum imgsz (high resolution)",
        "None",
        "Light",
        "Medium",
        "Strong",
        "Fast",
        "Balanced",
        "Maximum detection",
        "Nile Red microplastics",
        "General (1 class)",
    ):
        assert message in message_ids, message


def test_a_dynamically_resolved_preset_label_renders_in_spanish(selected_language):
    from polyx.core.i18n import _
    from polyx.gui.trainer.state import PRESETS

    selected_language("es")
    label = next(iter(PRESETS))

    assert _(label) != label


# ── Text assembled into a local variable before it reaches a widget ─────


class _Label:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


def _viewer_with_particles():
    from pathlib import Path
    from types import SimpleNamespace

    from polyx.gui.viewer.window import ViewerWindow

    viewer = SimpleNamespace(
        state=SimpleNamespace(current_image=Path("plate.png"), um_per_px=0.0),
        lbl_morph=_Label(),
        _bgr_current=None,
    )
    dets = [
        SimpleNamespace(y1=10.0, x1=10.0, morphotype="fibra"),
        SimpleNamespace(y1=20.0, x1=20.0, morphotype="fragmento"),
        SimpleNamespace(y1=30.0, x1=30.0, morphotype="fragmento"),
    ]
    return ViewerWindow, viewer, dets


def test_viewer_morph_summary_is_translated(selected_language, monkeypatch):
    """The summary is built in a local variable; the language must reach it."""
    from polyx.core.measure import calibration, shape

    monkeypatch.setattr(calibration, "read_image", lambda _path: object())
    monkeypatch.setattr(shape, "measure_detections", lambda *args: None)
    viewer_class, viewer, dets = _viewer_with_particles()

    selected_language("es")
    viewer_class._measure_shapes(viewer, dets)
    assert "1 fibra" in viewer.lbl_morph.text
    assert "2 partículas" in viewer.lbl_morph.text
    assert "sin calibrar" in viewer.lbl_morph.text

    selected_language("en")
    viewer_class._measure_shapes(viewer, dets)
    assert "1 fibre" in viewer.lbl_morph.text
    assert "2 particles" in viewer.lbl_morph.text
    assert "uncalibrated" in viewer.lbl_morph.text
