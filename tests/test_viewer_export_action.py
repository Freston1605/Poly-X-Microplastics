"""Ticket #188: the Viewer export action writes English artifacts.

The seam is the public export action ``ViewerWindow._save()``: a user picks a
destination folder and the action creates a timestamped export folder beside the
canonical run artifacts. What lands on disk is the contract — the folder name,
the three file names and the JSON keys — not the private save helpers.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PySide6.QtWidgets import QFileDialog, QMessageBox  # noqa: E402

from polyx.core.detect.yolo import Detection  # noqa: E402
from polyx.gui.viewer.window import ViewerWindow  # noqa: E402
from tests.conftest import write_image  # noqa: E402


def _window_with_image(image: Path) -> ViewerWindow:
    window = ViewerWindow()
    window.state.images = [image]
    window.state.current_idx = 0
    window.state.um_per_px = 2.0
    det = Detection(class_id=0, class_name="PET", conf=0.9, x1=1, y1=1, x2=5, y2=5)
    window.state.detections = [det]
    return window


def _choose_folder(monkeypatch: pytest.MonkeyPatch, folder: Path, log: list) -> None:
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: log.append(("info", a, k)))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: log.append(("warning", a, k)))


def test_export_uses_english_folder_file_and_json_names(qapp, tmp_path, monkeypatch):
    image = write_image(tmp_path / "sample.png")
    messages: list = []
    _choose_folder(monkeypatch, tmp_path, messages)

    _window_with_image(image)._save()

    folders = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(folders) == 1, "one timestamped export folder"
    export = folders[0]
    assert re.fullmatch(r"viewer_sample_\d{8}_\d{6}", export.name), export.name
    assert {p.name for p in export.iterdir()} == {
        "sample_annotated.png",
        "detections.csv",
        "viewer_summary.json",
    }

    summary = json.loads((export / "viewer_summary.json").read_text(encoding="utf-8"))
    assert summary["image"] == "sample.png"
    assert summary["n_detections"] == 1
    assert summary["by_class"] == {"PET": {"n": 1}}
    assert summary["um_per_px"] == 2.0
    assert not {"imagen", "n_detecciones", "por_clase"} & summary.keys()


def test_export_refuses_a_canonical_run_folder(qapp, tmp_path, monkeypatch, selected_language):
    """A folder with run.json is a canonical run: never write a viewer export there."""
    selected_language("en")
    image = write_image(tmp_path / "sample.png")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text('{"schema_version": 2}', encoding="utf-8")
    original_csv = "model,image,size\n"
    (run_dir / "detections.csv").write_text(original_csv, encoding="utf-8")

    messages: list = []
    _choose_folder(monkeypatch, run_dir, messages)

    _window_with_image(image)._save()

    warnings = [m for m in messages if m[0] == "warning"]
    assert warnings, "the user is warned why the export was refused"
    text = " ".join(str(part) for part in warnings[0][1])
    assert "run.json" in text
    assert "different output folder" in text
    assert {p.name for p in run_dir.iterdir()} == {"run.json", "detections.csv"}
    assert (run_dir / "detections.csv").read_text(encoding="utf-8") == original_csv
