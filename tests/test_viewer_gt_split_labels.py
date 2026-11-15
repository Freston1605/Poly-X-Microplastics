"""The Viewer and the Detector GT page read and write split-layout labels.

Both surfaces open an image from ``images/<split>`` and must find its label
under ``labels/<split>``, use the dataset-root ``classes.txt`` and — in the
GT page — save back to the conventional label folder instead of next to the
photo. These tests pin the two consumers to the codec's folder contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _png(path: Path, width: int = 100, height: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(bytes(buf))
    return path


def _split_dataset(root: Path, names: tuple[str, ...] = ("HDPE", "PS")) -> tuple[Path, Path]:
    """A conventional dataset: images/train + labels/train + root classes.txt."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    image = _png(root / "images" / "train" / "plate.png")
    label = root / "labels" / "train" / "plate.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n", encoding="utf-8")
    return image, label


# ── Viewer ────────────────────────────────────────────────────────────────


def test_viewer_load_labels_reads_the_split_label_and_dataset_classes(qapp, tmp_path, monkeypatch):
    """Load labels finds labels/<split> and names classes from the dataset root."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QMessageBox

    from polyx.gui.viewer.window import ViewerWindow

    image, label = _split_dataset(tmp_path / "dataset")

    # A dialog would block the test and hide the failure.
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail("warning dialog"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: pytest.fail("info dialog"))

    window = ViewerWindow()
    try:
        window.state.images = [image]
        window.state.current_idx = 0
        window._load_labels()
    finally:
        window.close()

    assert label.exists()
    assert [det.class_name for det in window.state.detections] == ["HDPE", "PS"]
    assert len(window.state.detections) == 2
    assert "plate.txt" in window.lbl_status.text()


def test_viewer_run_load_reads_classes_from_the_model_folder(qapp, tmp_path, monkeypatch):
    """Run labels live in <alias>/labels/; their classes.txt lives in <alias>/."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from polyx.gui.viewer.window import ViewerWindow

    photo = _png(tmp_path / "photos" / "plate.png")
    run = tmp_path / "runs" / "detect_20260101"
    model = run / "alpha"
    labels = model / "labels"
    labels.mkdir(parents=True)
    (model / "classes.txt").write_text("HDPE\nPS\n", encoding="utf-8")
    (labels / "0000_plate.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n", encoding="utf-8"
    )

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(run))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail("warning dialog"))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: pytest.fail("info dialog"))

    window = ViewerWindow()
    try:
        window.state.images = [photo]
        window.state.current_idx = 0
        window._run_load()
    finally:
        window.close()

    assert [det.class_name for det in window.state.detections] == ["HDPE", "PS"]


# ── Detector GT page ──────────────────────────────────────────────────────


def _gt_page(state):
    from polyx.gui.detector.pages.gt_manual import GTManualPage

    return GTManualPage(state)


def test_gt_page_marks_loads_and_saves_the_split_label(qapp, tmp_path):
    """The GT page sees the label as present, loads it, and saves it back there."""
    pytest.importorskip("PySide6")
    from polyx.gui.detector.state import DetectorState

    root = tmp_path / "dataset"
    image, label = _split_dataset(root)

    state = DetectorState()
    state.images = [image]
    page = _gt_page(state)
    try:
        # Marked as present in the image list.
        assert page.list.count() == 1
        assert page.list.item(0).text().startswith("✓")

        # Loaded from labels/train, with the dataset-root class names.
        assert len(page.canvas.boxes) == 2
        assert page.canvas.class_names[:2] == ["HDPE", "PS"]
        assert [page.combo_class.itemText(i) for i in range(page.combo_class.count())] == [
            "HDPE",
            "PS",
        ]

        # Saved to the conventional location, never next to the photo.
        out = page._save_to_txt(image, page.canvas.detections())
        assert out == label
        assert not (root / "images" / "train" / "plate.txt").exists()

        # The dataset-root classes.txt was already there and stays untouched.
        assert (root / "classes.txt").read_text(encoding="utf-8") == "HDPE\nPS\n"
        assert not (root / "labels" / "train" / "classes.txt").exists()
    finally:
        page.close()


def test_gt_page_writes_classes_txt_once_at_the_dataset_root(qapp, tmp_path):
    """A conventional layout owns one classes.txt, at the dataset root."""
    pytest.importorskip("PySide6")
    from polyx.core.detect.yolo import Detection
    from polyx.gui.detector.state import DetectorState

    root = tmp_path / "dataset"
    image = _png(root / "images" / "train" / "plate.png")

    state = DetectorState()
    state.images = [image]
    page = _gt_page(state)
    try:
        dets = [Detection(class_id=0, class_name="PET", conf=1.0, x1=10, y1=10, x2=30, y2=30)]
        out = page._save_to_txt(image, dets)
        assert out == root / "labels" / "train" / "plate.txt"

        assert (root / "classes.txt").exists()
        assert not (root / "labels" / "train" / "classes.txt").exists()

        # A second save does not add another classes.txt anywhere.
        page._save_to_txt(image, dets)
        assert sorted(root.rglob("classes.txt")) == [root / "classes.txt"]
    finally:
        page.close()


def test_gt_page_auto_save_skips_an_empty_review(qapp, tmp_path):
    """Leaving an image with no boxes writes nothing: the rule is unchanged."""
    pytest.importorskip("PySide6")
    from polyx.gui.detector.state import DetectorState

    root = tmp_path / "dataset"
    image = _png(root / "images" / "train" / "plate.png")

    state = DetectorState()
    state.images = [image]
    page = _gt_page(state)
    try:
        assert page.canvas.boxes == []
        page._auto_save(image)

        assert not (root / "labels").exists()
        assert not (root / "images" / "train" / "plate.txt").exists()
    finally:
        page.close()
