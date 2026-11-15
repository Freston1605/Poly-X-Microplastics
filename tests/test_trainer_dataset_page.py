"""The Trainer dataset page looks labels up through the label codec.

Two surfaces are covered: the preview, which must read boxes through the
codec (so a YOLO-seg polygon is not painted as a box), and the
"labels found" check, which must resolve a conventional ``labels/<split>``
file with the codec's lookup.

The page's yaml/split policy is deliberately untouched by this ticket and is
not exercised here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtGui import QImage

from polyx.gui.trainer.pages import dataset as dataset_page
from polyx.gui.trainer.pages.dataset import DatasetPage
from polyx.gui.trainer.state import TrainerState

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PET_BOX = "0 0.5 0.5 0.10 0.10"
POLYGON = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"


def make_image(path: Path) -> Path:
    """A real, loadable image: the preview needs a non-null QPixmap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = QImage(40, 30, QImage.Format_RGB32)
    image.fill(0xFFFFFFFF)
    assert image.save(str(path))
    return path


def make_dataset(root: Path, *, label_text: str) -> Path:
    """A one-image conventional dataset: ``images/train`` + ``labels/train``."""
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    make_image(root / "images" / "train" / "particle.png")
    (root / "labels" / "train" / "particle.txt").write_text(label_text, encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: images/train\nval: images/train\nnames:\n  0: PET\n  1: PP\n",
        encoding="utf-8",
    )
    return yaml_path


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class _PainterRecorder:
    """Stands in for ``QPainter`` so a test can see which boxes were drawn."""

    Antialiasing = 0
    instances: list[_PainterRecorder] = []

    def __init__(self, image):
        self.rects: list[tuple] = []
        _PainterRecorder.instances.append(self)

    def setRenderHint(self, *args):
        pass

    def setPen(self, *args):
        pass

    def drawRect(self, *args):
        self.rects.append(args)

    def end(self):
        pass


def _page(qapp, yaml_path: Path) -> DatasetPage:
    state = TrainerState()
    state.dataset.yaml_path = yaml_path
    state.dataset.class_names = ["PET", "PP"]
    return DatasetPage(state)


def test_labels_present_check_finds_conventional_split_labels(qapp, tmp_path):
    """The check resolves ``labels/<split>/<stem>.txt`` through the codec."""
    yaml_path = make_dataset(tmp_path / "dataset", label_text=f"{PET_BOX}\n")
    page = _page(qapp, yaml_path)

    page.ed_yaml.setText(str(yaml_path))
    page._reload_yaml()

    icon, label = page._val_checks[3]
    assert icon.text() == "✓", label.toolTip()
    assert "1/1" in label.toolTip(), label.toolTip()


def test_preview_draws_boxes_read_by_the_codec(qapp, tmp_path, monkeypatch):
    yaml_path = make_dataset(tmp_path / "dataset", label_text=f"{PET_BOX}\n")
    page = _page(qapp, yaml_path)
    _PainterRecorder.instances = []
    monkeypatch.setattr(dataset_page, "QPainter", _PainterRecorder)

    page._draw_preview(page.tiles[0], yaml_path.parent / "images" / "train" / "particle.png")

    assert len(_PainterRecorder.instances) == 1
    assert len(_PainterRecorder.instances[0].rects) == 1


def test_preview_does_not_paint_a_polygon_as_a_box(qapp, tmp_path, monkeypatch):
    """A YOLO-seg polygon line is not a box, so nothing is drawn."""
    yaml_path = make_dataset(tmp_path / "dataset", label_text=f"{POLYGON}\n")
    page = _page(qapp, yaml_path)
    _PainterRecorder.instances = []
    monkeypatch.setattr(dataset_page, "QPainter", _PainterRecorder)

    page._draw_preview(page.tiles[0], yaml_path.parent / "images" / "train" / "particle.png")

    assert len(_PainterRecorder.instances) == 1
    assert _PainterRecorder.instances[0].rects == []
