"""The Labeller window talks to its state through the public interface.

Ticket #118: the window owns widgets and progress; ``LabelerState`` owns
images, drawings, review policy and persistence. Opening a folder with
existing labels is the resume path, and its list marks and counts come from
the same codec lookup the editor uses.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from tests.conftest import split_dataset

BOX_LINE = "0 0.500000 0.500000 0.100000 0.100000\n"
WINDOW_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "polyx" / "gui" / "labeler" / "window.py"
)


def _boxes(lines: int) -> str:
    return BOX_LINE * lines


def test_the_labeller_window_never_reaches_into_private_state():
    """``state._`` is not part of the interface between the window and the state.

    The window holds widgets and progress; a private access there is the
    window making a second copy of a policy the state owns, or — as the
    resume crash showed — calling something that does not exist. The AST
    walk catches ``self.state._…`` anywhere, not just in a plain statement.
    """
    tree = ast.parse(WINDOW_SOURCE.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not node.attr.startswith("_") or node.attr.startswith("__"):
            continue
        target = node.value
        is_state = (isinstance(target, ast.Attribute) and target.attr == "state") or (
            isinstance(target, ast.Name) and target.id == "state"
        )
        if is_state:
            offenders.append(f"{WINDOW_SOURCE.name}:{node.lineno} state.{node.attr}")
    assert offenders == []


def test_opening_a_folder_with_existing_labels_shows_the_resume_marks(tmp_path, qapp):
    """Resume: the list recovers the marks and counts from the label files."""
    from polyx.gui.labeler.window import LabelerWindow

    root = split_dataset(
        tmp_path / "ds",
        {"train": {"counted": _boxes(2), "empty": "", "fresh": None}},
    )

    window = LabelerWindow()
    try:
        window.state.load_images(root / "images" / "train")
        marks = [window.img_list.item(i).text() for i in range(window.img_list.count())]
    finally:
        window.deleteLater()

    assert marks == ["✓  counted.png  (2)", "·  empty.png  (0)", "○  fresh.png"]


class _FakeModel:
    """One detection per call; records which images it was asked about."""

    def __init__(self):
        self.seen: list[Path] = []

    def predict_auto(self, image_path: str, **_kwargs):
        self.seen.append(Path(image_path))
        return [SimpleNamespace(x1=10, y1=10, x2=30, y2=30, class_id=1)]


def test_pre_annotation_asks_the_model_only_about_blank_images(tmp_path, qapp):
    """The batch fills the blanks; a labeled image is never re-inferred."""
    from polyx.gui.labeler.window import LabelerWindow

    root = split_dataset(tmp_path / "ds", {"train": {"a_blank": None, "z_drawn": _boxes(1)}})
    image_dir = root / "images" / "train"
    model = _FakeModel()

    window = LabelerWindow()
    try:
        window.state.load_images(image_dir)  # a_blank is current; z_drawn is on disk only
        window._pre_model = model
        window._pre_annotate_all()
    finally:
        window.deleteLater()

    assert model.seen == [image_dir / "a_blank.png"]
    assert (root / "labels" / "train" / "a_blank.txt").is_file()
    assert (root / "labels" / "train" / "z_drawn.txt").read_text(encoding="utf-8") == _boxes(1)
