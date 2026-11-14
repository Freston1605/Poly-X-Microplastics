"""The Labeller reads and writes through the YOLO-label codec.

Ticket #91: opening ``images/train`` must show the labels that already live
in ``labels/train``, write new ones there, and restore reviewed state from
both the conventional and the legacy same-directory locations. The particle
count in the list has to read the same file the codec found, and
``classes.txt`` goes once at the dataset root for a split layout while a
plain folder keeps per-folder classes. The census rule (a ``.txt`` exists
only for images that were reviewed or drawn on) does not change.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from polyx.core.labels import BBox
from polyx.gui.labeler.state import LabelerState


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(bytes(buf))
    return path


def _txt(path: Path, lines: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join("0 0.500000 0.500000 0.100000 0.100000\n" for _ in range(lines)),
        encoding="utf-8",
    )
    return path


def _split(root: Path) -> tuple[Path, Path]:
    """A dataset root with ``images/train`` and ``labels/train``."""
    return root / "images" / "train", root / "labels" / "train"


def test_opening_images_train_reads_labels_train_and_reviewed_state(tmp_path):
    image_dir, label_dir = _split(tmp_path / "ds")
    a = _png(image_dir / "a.png")
    b = _png(image_dir / "b.png")
    _txt(label_dir / "a.txt", lines=2)

    state = LabelerState()
    state.load_images(image_dir)

    assert state.images == [a, b]
    assert state.is_reviewed(a) is True
    assert state.is_reviewed(b) is False
    assert state.n_reviewed() == 1

    state.goto(0)
    assert len(state.current_boxes) == 2


def test_saving_writes_labels_train_and_never_next_to_the_image(tmp_path):
    image_dir, label_dir = _split(tmp_path / "ds")
    _png(image_dir / "a.png")

    state = LabelerState()
    state.load_images(image_dir)
    state.set_current_boxes([BBox(class_id=1, cx=0.5, cy=0.5, w=0.2, h=0.2)])
    state.save_current()

    assert (label_dir / "a.txt").is_file()
    assert not (image_dir / "a.txt").exists()


def test_legacy_same_directory_label_is_read_and_counted_as_reviewed(tmp_path):
    image_dir, _label_dir = _split(tmp_path / "ds")
    c = _png(image_dir / "c.png")
    _txt(image_dir / "c.txt", lines=1)

    state = LabelerState()
    state.load_images(image_dir)

    assert state.is_reviewed(c) is True
    state.goto(0)
    assert len(state.current_boxes) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_unmark_reviewed_removes_the_label_file_that_was_found(tmp_path, legacy):
    image_dir, label_dir = _split(tmp_path / "ds")
    a = _png(image_dir / "a.png")
    found = image_dir / "a.txt" if legacy else label_dir / "a.txt"
    _txt(found, lines=2)

    state = LabelerState()
    state.load_images(image_dir)
    state.goto(0)
    assert state.is_reviewed(a) is True

    state.mark_reviewed(False)

    assert state.is_reviewed(a) is False
    assert not found.exists()


def test_unmarking_does_not_resurrect_the_file_when_navigating(tmp_path):
    image_dir, label_dir = _split(tmp_path / "ds")
    _png(image_dir / "a.png")
    _png(image_dir / "b.png")
    _txt(label_dir / "a.txt", lines=2)

    state = LabelerState()
    state.load_images(image_dir)
    state.goto(0)
    state.mark_reviewed(False)
    state.next_image()

    assert not (label_dir / "a.txt").exists()


def test_particle_count_reads_the_file_the_codec_found(tmp_path):
    image_dir, label_dir = _split(tmp_path / "ds")
    a = _png(image_dir / "a.png")
    c = _png(image_dir / "c.png")
    d = _png(image_dir / "d.png")
    _txt(label_dir / "a.txt", lines=2)
    _txt(image_dir / "c.txt", lines=1)

    state = LabelerState()
    state.load_images(image_dir)

    assert state.box_count_for(a) == 2  # conventional
    assert state.box_count_for(c) == 1  # legacy same-directory
    assert state.box_count_for(d) == 0  # no label at all


def test_list_display_uses_the_codec_lookup(tmp_path, qapp):
    from polyx.gui.labeler.window import LabelerWindow

    image_dir, label_dir = _split(tmp_path / "ds")
    _png(image_dir / "a.png")
    b = _png(image_dir / "b.png")
    _txt(label_dir / "a.txt", lines=2)
    _txt(label_dir / "b.txt", lines=1)

    window = LabelerWindow()
    try:
        window.state.load_images(image_dir)
        # The current image is already in memory; the count for any other row
        # has to come from the file the codec finds.
        assert "(1)" in window._text_item(b, count_2=True)
    finally:
        window.deleteLater()


def test_classes_txt_is_written_once_at_the_dataset_root(tmp_path):
    image_dir, _label_dir = _split(tmp_path / "ds")
    _png(image_dir / "a.png")
    _png(image_dir / "b.png")

    state = LabelerState()
    state.load_images(image_dir)
    state.set_class_names(["PET", "PP"])
    state.save_classes_txt()

    root = tmp_path / "ds" / "classes.txt"
    assert root.read_text(encoding="utf-8") == "PET\nPP\n"
    assert not (tmp_path / "ds" / "labels" / "train" / "classes.txt").exists()
    assert not (image_dir / "classes.txt").exists()


def test_plain_folder_keeps_per_folder_classes(tmp_path):
    plain = tmp_path / "plain"
    _png(plain / "a.png")
    _png(plain / "sub" / "b.png")

    state = LabelerState()
    state.load_images(plain)
    state.set_class_names(["PET"])
    state.save_classes_txt()

    assert (plain / "classes.txt").is_file()
    assert (plain / "sub" / "classes.txt").is_file()


def test_census_rule_is_unchanged(tmp_path):
    image_dir, label_dir = _split(tmp_path / "ds")
    _png(image_dir / "a.png")
    b = _png(image_dir / "b.png")

    state = LabelerState()
    state.load_images(image_dir)

    # Passing over an unreviewed, undrawn image writes nothing.
    state.goto(1)
    state.next_image()
    assert not (label_dir / "b.txt").exists()

    # Declaring it reviewed is a datum: an empty file is written.
    state.goto(1)
    state.mark_reviewed()
    assert b == state.current_image
    assert (label_dir / "b.txt").is_file()
    assert (label_dir / "b.txt").read_text(encoding="utf-8") == ""
