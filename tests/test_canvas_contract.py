"""One editing path for the two annotation canvases — the card 07 contract (#147).

The Labeller's ``BboxCanvas`` and the Manual GT page's ``AnnotCanvas`` are two
thin shells over :mod:`polyx.core.editing`. Card 07 moved the editing domain
into that core; this module pins the part the shells still own and that must
stay identical:

* the same gesture sequence (draw, move, resize, delete, undo/redo, an empty
  click) leaves the same boxes and the same selection on both;
* the same sub-2 px box is refused by the same core invariant with the same
  visible message;
* a class is painted with the colour the one theme helper gives it, including
  classes outside the palette — a local colour table would paint another.

The Qt-free guarantee of the core is owned by
``tests/test_box_editing.py::test_the_editing_core_imports_without_qt``; the
guard here covers the static half (no Qt import in the core source) and the
shells (no private editing maths, no colour tables, no minimum-side alias).

The two rigs live in ``tests/canvas_rigs.py``: one gesture script, either
shell, so this module and the per-canvas modules drive the widgets the same
way. The Viewer canvas also carries a ``_CLASS_COLORS`` table with the same
literal palette. It is out of card 07's scope (the two annotation editors) and
its index-ordered colours are a different rule from the theme's name lookup,
so it is left alone and reported rather than silently re-coloured.
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from polyx.core.detect.protocol import size_image
from polyx.core.editing import MIN_SIDE_PX, rejection_message
from polyx.core.i18n import _
from polyx.core.labels import STUDY_CLASSES, bbox_from, find_label_file, read_boxes
from polyx.gui import theme as T
from polyx.gui.detector.pages import gt_manual
from polyx.gui.labeler import canvas as labeler_canvas
from polyx.gui.labeler import window as labeler_window
from polyx.gui.labeler.canvas import BboxCanvas
from tests.canvas_rigs import CanvasRig, GtRig, LabellerRig
from tests.conftest import write_image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

IMAGE_W, IMAGE_H = 400, 300
CLASSES = [*STUDY_CLASSES, "HDPE"]


# ── The two rigs: the same gestures through each shell's real events ──────


@pytest.fixture
def image(tmp_path) -> Path:
    return write_image(tmp_path / "imgs" / "plate.png", IMAGE_W, IMAGE_H, value=40)


@pytest.fixture
def rigs(qapp, image):
    labeller = LabellerRig.over(qapp, image.parent, CLASSES)
    gt = GtRig.over(qapp, image, CLASSES)
    yield labeller, gt
    labeller.canvas.deleteLater()
    gt.canvas.close()
    gt.canvas.deleteLater()
    qapp.processEvents()


# ── The contract: one gesture sequence, one result ─────────────────────────


def _assert_same(labeller: CanvasRig, gt: CanvasRig) -> None:
    left, right = labeller.boxes(), gt.boxes()
    assert len(left) == len(right), (
        f"{len(left)} boxes in the Labeller, {len(right)} in the GT page"
    )
    for labeller_box, gt_box in zip(left, right, strict=True):
        assert labeller_box[:4] == pytest.approx(gt_box[:4], abs=1.0), (
            f"Labeller {labeller_box} != GT {gt_box}"
        )
        assert labeller_box[4] == gt_box[4]
    assert labeller.selection() == gt.selection()


def _move_by(rig: CanvasRig, index: int, dx: float, dy: float) -> None:
    x1, y1, x2, y2, _cls = rig.boxes()[index]
    centre = ((x1 + x2) / 2, (y1 + y2) / 2)
    rig.drag(centre, (centre[0] + dx, centre[1] + dy))


def _resize_bottom_right_by(rig: CanvasRig, index: int, dx: float, dy: float) -> None:
    _x1, _y1, x2, y2, _cls = rig.boxes()[index]
    rig.drag((x2, y2), (x2 + dx, y2 + dy))


def test_the_same_draw_move_resize_delete_and_undo_leaves_both_editors_identical(rigs):
    labeller, gt = rigs

    # Draw two boxes; each drawing is selected as it lands.
    for rig in (labeller, gt):
        rig.drag((50, 40), (150, 140))
    assert len(labeller.boxes()) == len(gt.boxes()) == 1
    _assert_same(labeller, gt)
    assert labeller.selection() == gt.selection() == 0

    for rig in (labeller, gt):
        rig.drag((200, 60), (280, 160))
    assert len(labeller.boxes()) == len(gt.boxes()) == 2
    _assert_same(labeller, gt)
    assert labeller.selection() == gt.selection() == 1

    # Move the newest box by grabbing its centre: the press selects it and
    # the drag translates it by the same delta in both editors.
    _move_by(labeller, 1, 20, 10)
    _move_by(gt, 1, 20, 10)
    _assert_same(labeller, gt)
    assert labeller.selection() == gt.selection() == 1

    # Resize it from its bottom-right handle: the box only grows there.
    _resize_bottom_right_by(labeller, 1, 15, 15)
    _resize_bottom_right_by(gt, 1, 15, 15)
    _assert_same(labeller, gt)
    assert labeller.selection() == gt.selection() == 1

    # Draw a third box, then delete it: the selection goes with it.
    for rig in (labeller, gt):
        rig.drag((60, 200), (120, 260))
    assert len(labeller.boxes()) == len(gt.boxes()) == 3
    _assert_same(labeller, gt)
    for rig in (labeller, gt):
        rig.key(Qt.Key_Delete)
    assert labeller.boxes() and len(labeller.boxes()) == len(gt.boxes()) == 2
    assert labeller.selection() == gt.selection() == -1
    _assert_same(labeller, gt)

    # Undo the delete (box and selection come back), redo it, undo it again.
    for rig in (labeller, gt):
        rig.key(Qt.Key_Z, Qt.ControlModifier)
    assert len(labeller.boxes()) == len(gt.boxes()) == 3
    assert labeller.selection() == gt.selection() == 2
    _assert_same(labeller, gt)

    for rig in (labeller, gt):
        rig.key(Qt.Key_Y, Qt.ControlModifier)
    assert len(labeller.boxes()) == len(gt.boxes()) == 2
    assert labeller.selection() == gt.selection() == -1
    _assert_same(labeller, gt)

    # Walk the rest of the history back: the delete again, the draw, the
    # resize and the move, asserting both editors at every step.
    for rig in (labeller, gt):
        rig.key(Qt.Key_Z, Qt.ControlModifier)
    assert len(labeller.boxes()) == len(gt.boxes()) == 3
    assert labeller.selection() == gt.selection() == 2
    _assert_same(labeller, gt)

    for rig in (labeller, gt):
        rig.key(Qt.Key_Z, Qt.ControlModifier)
    assert len(labeller.boxes()) == len(gt.boxes()) == 2
    _assert_same(labeller, gt)
    resized_height = labeller.boxes()[1][3] - labeller.boxes()[1][1]

    for rig in (labeller, gt):
        rig.key(Qt.Key_Z, Qt.ControlModifier)
    _assert_same(labeller, gt)
    restored_height = labeller.boxes()[1][3] - labeller.boxes()[1][1]
    assert restored_height == pytest.approx(resized_height - 15, abs=1.0)

    for rig in (labeller, gt):
        rig.key(Qt.Key_Z, Qt.ControlModifier)
    _assert_same(labeller, gt)
    assert labeller.boxes()[1][0] == pytest.approx(200, abs=1.0)


def test_an_empty_click_is_a_click_in_both_editors(rigs):
    """No box is drawn and the selection is cleared, as the Labeller does."""
    labeller, gt = rigs

    for rig in (labeller, gt):
        rig.click((350, 280))
    assert labeller.boxes() == gt.boxes() == []
    assert labeller.selection() == gt.selection() == -1
    assert labeller.can_undo() is False
    assert gt.can_undo() is False

    for rig in (labeller, gt):
        rig.drag((50, 40), (150, 140))
    assert labeller.selection() == gt.selection() == 0

    for rig in (labeller, gt):
        rig.click((350, 280))
    assert len(labeller.boxes()) == len(gt.boxes()) == 1
    assert labeller.selection() == gt.selection() == -1


# ── The annotation frame: the pixels, the editor and the .txt agree ───────


def _write_exif_rotated_jpeg(path: Path) -> tuple[int, int]:
    """A JPEG stored 60×40 but tagged to display as 40×60.

    ``write_image`` writes the stored pixels; Pillow then tags orientation 6
    (rotate 90° clockwise), so every reader that honours EXIF — cv2,
    ``size_image`` — sees a 40×60 frame. Returns the displayed size.
    """
    Image = pytest.importorskip("PIL.Image")
    write_image(path, 60, 40, suffix=".jpg")
    with Image.open(path) as im:
        exif = im.getexif()
        exif[274] = 6
        im.save(path, exif=exif)
    return 40, 60


@pytest.mark.parametrize("shell", ["labeller", "gt"])
def test_an_exif_rotated_photo_is_annotated_in_the_frame_it_is_painted(qapp, tmp_path, shell):
    """One frame per canvas: measured, painted and edited sizes must agree.

    A photo tagged orientation 6 is stored 60×40 and displayed 40×60. Qt's
    ``QPixmap(path)`` ignores the tag; cv2 — and therefore ``size_image`` —
    honours it. If a canvas paints the unrotated pixels while the editor and
    the .txt work in the rotated frame, boxes are drawn, hit-tested and saved
    in two different geometries.
    """
    folder = tmp_path / "imgs"
    path = folder / "rotated.jpg"
    displayed = _write_exif_rotated_jpeg(path)
    assert displayed != (60, 40), "the fixture must change the frame, not just tag it"

    measured = size_image(path)
    try:
        if shell == "labeller":
            rig = LabellerRig.over(qapp, folder)
            editor = rig.state.current_editor
            painted = rig.canvas._pixmap
        else:
            rig = GtRig.over(qapp, path)
            editor = rig.canvas.editor
            painted = rig.canvas._pixmap_item.pixmap()

        painted_size = (painted.width(), painted.height())
        editor_size = (editor.img_w, editor.img_h)
        assert measured == displayed, f"size_image measured {measured}, not the displayed frame"
        assert painted_size == measured, (
            f"{shell} paints {painted_size} while size_image measures {measured}"
        )
        assert editor_size == measured, (
            f"{shell} edits {editor_size} while size_image measures {measured}"
        )

        # A box drawn in the displayed frame saves normalized against it and
        # reads back as the same absolute box through the same frame.
        rig.drag((2, 2), (18, 26))
        assert len(rig.boxes()) == 1
        assert rig.boxes()[0][:4] == pytest.approx((2, 2, 18, 26), abs=1.0)
        w, h = measured
        if shell == "labeller":
            rig.state.save_current()
            boxes, _malformed = read_boxes(find_label_file(path))
        else:
            # What the page's save does for the loaded image: normalize the
            # detections against size_image, the frame the canvas painted.
            boxes = [bbox_from(det, w, h) for det in rig.canvas.detections()]
        assert (boxes[0].cx * w, boxes[0].cy * h) == pytest.approx((10, 14), abs=0.5)
    finally:
        rig.canvas.deleteLater()
        qapp.processEvents()


# ── The 2 px rule: one core invariant, one silence, one message ───────────


def test_a_click_a_one_px_drag_and_a_two_px_box_are_announced_alike(qapp, image):
    """A 0 px click is silent, a 1 px drag speaks, a 2 px box commits."""
    from polyx.gui.labeler.window import LabelerWindow

    window = LabelerWindow()
    gt = GtRig.over(qapp, image, CLASSES)
    gt_messages: list[str] = []
    gt.canvas.status_changed.connect(gt_messages.append)
    rejected: list[float] = []
    window.canvas.box_rejected.connect(rejected.append)
    try:
        window.show()
        window.state.load_images(image.parent)
        qapp.processEvents()
        labeller = LabellerRig(window.state, window.canvas)

        # 1. A click on empty space: a 0 px draw, refused without a message.
        labeller.click((350, 280))
        gt.click((350, 280))
        assert rejected == []
        assert _rejections(gt_messages) == []
        assert window.lbl_status.text() == _("● Ready")
        assert labeller.boxes() == []
        assert gt.boxes() == []

        # 2. A 1 px drag: refused with the translated message in both.
        labeller.drag((20, 20), (21, 80), flush=False)
        labeller_message = window.lbl_status.text()
        gt.drag((20, 20), (21, 80))
        gt_message = _rejections(gt_messages)[-1]
        assert len(rejected) == 1
        assert rejected[0] == pytest.approx(1.0, abs=0.3)
        assert _side_of(gt_message) < MIN_SIDE_PX
        # Each shell names the side the core refused; the template and the
        # minimum are the same one, only widget rounding moves the number.
        assert labeller_message == rejection_message(rejected[0])
        assert gt_message == rejection_message(_side_of(gt_message))
        assert re.sub(r"\d+(?:[.,]\d+)?\s*px", "<side>", labeller_message) == re.sub(
            r"\d+(?:[.,]\d+)?\s*px", "<side>", gt_message
        )
        assert labeller.boxes() == []
        assert gt.boxes() == []

        # 3. A box at the 2 px minimum: a real mark, and no new message.
        labeller.drag((20, 80), (22, 110), flush=False)
        gt.drag((20, 80), (22, 110))
        assert len(rejected) == 1  # unchanged: nothing was refused
        assert len(_rejections(gt_messages)) == 1
        assert len(labeller.boxes()) == len(gt.boxes()) == 1
        assert min(labeller.boxes()[0][2] - labeller.boxes()[0][0], 2.0) >= MIN_SIDE_PX
    finally:
        gt.canvas.close()
        gt.canvas.deleteLater()
        window.deleteLater()
        qapp.processEvents()


def _side_of(message: str) -> float:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*px", message)
    assert match, f"the message does not name the side: {message!r}"
    return float(match.group(1).replace(",", "."))


def _rejections(messages: list[str]) -> list[str]:
    """The messages that are a rejection, not a drawing-size readout.

    The GT canvas reports both through ``status_changed``; the core owns the
    rejection template, so a message is a rejection when it is exactly what
    the helper renders for the side it names.
    """
    return [message for message in messages if message == rejection_message(_side_of(message))]


def test_both_canvases_refuse_a_sub_two_pixel_box_with_the_same_visible_message(qapp, image):
    from polyx.gui.labeler.window import LabelerWindow

    window = LabelerWindow()
    gt = GtRig.over(qapp, image, CLASSES)
    messages: list[str] = []
    gt.canvas.status_changed.connect(messages.append)
    try:
        window.show()
        window.state.load_images(image.parent)
        qapp.processEvents()
        labeller = LabellerRig(window.state, window.canvas)

        # A language change can rebuild the window on the next event flush;
        # read the status straight after the release, as the labeller sees it.
        labeller.drag((20, 20), (21, 80), flush=False)
        labeller_message = window.lbl_status.text()

        gt.drag((20, 20), (21, 80))
        gt_message = messages[-1]
    finally:
        gt.canvas.close()
        gt.canvas.deleteLater()
        window.deleteLater()
        qapp.processEvents()

    assert window.state.current_boxes == []
    assert labeller.can_undo() is False
    assert gt.boxes() == []
    assert gt.can_undo() is False

    for message in (labeller_message, gt_message):
        assert _side_of(message) < MIN_SIDE_PX
        assert f"{MIN_SIDE_PX:.0f} px" in message

    # The same template and the same minimum, only the measured side differs.
    assert re.sub(r"\d+(?:[.,]\d+)?\s*px", "<side>", labeller_message) == re.sub(
        r"\d+(?:[.,]\d+)?\s*px", "<side>", gt_message
    )


# ── One class-colour path ─────────────────────────────────────────────────


def test_class_qcolor_at_resolves_names_and_falls_back():
    assert T.class_qcolor_at(CLASSES, 0).name() == T.CLASS_COLOR_HEX["PET"]
    assert T.class_qcolor_at(CLASSES, 3).name() == T.FALLBACK_COLOR_HEX
    # Out of range and unknown both fall back instead of raising.
    assert T.class_qcolor_at(CLASSES, 99).name() == T.FALLBACK_COLOR_HEX
    assert T.class_qcolor_at([], 0).name() == T.FALLBACK_COLOR_HEX


@pytest.mark.parametrize("name", CLASSES)
def test_both_canvases_paint_the_class_colour_the_theme_gives(rigs, name):
    labeller, gt = rigs
    class_id = CLASSES.index(name)

    for rig in (labeller, gt):
        rig.set_class(class_id)
        rig.drag((50, 40), (150, 140))
        assert rig.boxes()[0][4] == class_id

    expected = T.class_qcolor(name).name()
    assert labeller.tag_color(name).name() == expected
    assert gt.tag_color(name).name() == expected


# ── Guards: no second implementation may grow back ────────────────────────


def test_the_editing_core_stays_qt_free():
    """The static half of the core's Qt-free guarantee.

    The dynamic half (a clean interpreter imports it and PySide6 never
    loads) is owned by ``tests/test_box_editing.py``; this one catches an
    import that a conditional would hide from that run.
    """
    source = (SRC / "polyx" / "core" / "editing.py").read_text(encoding="utf-8")
    assert "PySide6" not in source
    assert not re.search(
        r"^\s*(from|import)\s+(PySide6|PyQt|QtCore|QtGui|QtWidgets)", source, re.MULTILINE
    )


def test_neither_canvas_reintroduces_private_editing_maths(rigs):
    private = (
        "_hit_box",
        "_hit_handle",
        "_handle_points",
        "_handles_for",
        "_push_undo",
        "_snapshot",
        "_undo_stack",
        "_redo_stack",
        "_undo",
        "_redo",
        "_clamp_box",
        "_normalize_box",
        "_normalized_box",
    )
    for rig in rigs:
        for name in private:
            assert not hasattr(rig.canvas, name), f"{type(rig.canvas).__name__} regrew {name}"

    # A geometry helper cannot come back as a module-level function either.
    regrown = re.compile(r"^_(hit|handle_points|snapshot|push_undo|clamp|normalis|normalize)")
    for module in (labeler_canvas, gt_manual):
        functions = [
            name
            for name, value in vars(module).items()
            if inspect.isfunction(value)
            and value.__module__ == module.__name__
            and regrown.match(name)
        ]
        assert functions == [], f"{module.__name__} defines {functions}"


def test_the_canvases_and_the_labeller_panel_hold_no_colour_table():
    def tables(module) -> list[str]:
        return [
            name
            for name, value in vars(module).items()
            if isinstance(value, (list, tuple, dict)) and re.search(r"colou?r", name, re.IGNORECASE)
        ]

    for module in (labeler_canvas, labeler_window, gt_manual):
        assert tables(module) == [], f"{module.__name__} keeps {tables(module)}"


def test_the_minimum_side_has_one_name():
    assert not hasattr(BboxCanvas, "SIDE_MINIMUM_PX")
    offenders = [
        path
        for path in (SRC / "polyx").rglob("*.py")
        if "SIDE_MINIMUM_PX" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
