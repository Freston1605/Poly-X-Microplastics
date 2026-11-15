"""Ticket #92: the Detector Images page marks split-layout ground truth.

The page shows the same GT check the run and the reload use. Its lookup is
the YOLO-label codec's conventional layout (``images/<split>`` ->
``labels/<split>``), with an explicit GT folder still taking precedence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.gui.detector.pages.images import ImagesPage  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402


def _plate(path: Path) -> Path:
    """Write a tiny decodable image at ``path``."""
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buffer))
    return path


def _gt_cell(page: ImagesPage, row: int) -> str:
    item = page.table.item(row, 2)
    assert item is not None
    return item.text()


def test_images_page_marks_split_layout_ground_truth(qapp, tmp_path):
    dataset = tmp_path / "dataset"
    labelled = _plate(dataset / "images" / "train" / "labelled.png")
    unlabelled = _plate(dataset / "images" / "train" / "unlabelled.png")
    label = dataset / "labels" / "train" / "labelled.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    state = DetectorState()
    state.images = [labelled, unlabelled]
    page = ImagesPage(state)
    page._refresh_table()

    assert _gt_cell(page, 0) == "✓"
    assert _gt_cell(page, 1) == "—"


def test_images_page_explicit_gt_folder_still_wins(qapp, tmp_path):
    dataset = tmp_path / "dataset"
    image = _plate(dataset / "images" / "train" / "plate.png")
    gt_folder = tmp_path / "gt"
    gt_folder.mkdir()
    (gt_folder / "plate.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    state = DetectorState()
    state.images = [image]
    state.gt_folder = gt_folder
    page = ImagesPage(state)
    page._refresh_table()

    assert _gt_cell(page, 0) == "✓"
