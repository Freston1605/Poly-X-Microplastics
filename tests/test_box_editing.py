"""Headless tests for the shared box-editing core (ticket #144).

The Labeller canvas and the Manual GT canvas are two complete box editors
today, with their own geometry, handle math, selection and undo stacks. This
module tests the one Qt-free editor they are meant to share: absolute-pixel
boxes with normalized accessors, draw/move/resize-by-handle/delete/clear/
class/select operations, bounded undo/redo, the 2 px minimum-side invariant,
and conversion through the canonical label codec. Nothing here touches a
widget, and a clean interpreter must be able to import the core without Qt.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from polyx.core import labels  # noqa: E402
from polyx.core.detect.protocol import Instance  # noqa: E402
from polyx.core.editing import (  # noqa: E402
    HANDLES,
    MIN_SIDE_PX,
    BoxEditor,
    CommitResult,
    EditBox,
    EditMode,
    Handle,
    RejectionReason,
)

# ── The core is Qt-free ───────────────────────────────────────────────────


def test_the_editing_core_imports_without_qt():
    """A clean interpreter imports the editor without loading PySide6."""
    script = (
        "import sys\n"
        "from polyx.core.editing import BoxEditor, EditBox, Handle\n"
        "bad = sorted(m for m in sys.modules if m == 'PySide6' or m.startswith('PySide6.'))\n"
        "print('QT_LOADED=' + repr(bad))\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(  # noqa: PLW1510
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        "importing the editing core dragged Qt or failed:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "QT_LOADED=[]" in result.stdout


# ── The value type ────────────────────────────────────────────────────────


def test_edit_box_keeps_absolute_pixels_and_exposes_normalized_geometry():
    box = EditBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0, class_id=1)

    assert (box.x1, box.y1, box.x2, box.y2) == (10.0, 20.0, 30.0, 60.0)
    assert box.class_id == 1
    assert box.width == 20.0
    assert box.height == 40.0
    assert box.center == (20.0, 40.0)

    bbox = box.to_bbox(img_w=100, img_h=200)

    assert bbox == labels.BBox(class_id=1, cx=0.2, cy=0.2, w=0.2, h=0.2)


def test_edit_box_round_trips_a_normalized_box_through_the_codec():
    source = labels.BBox(class_id=2, cx=0.5, cy=0.25, w=0.4, h=0.2)

    box = EditBox.from_bbox(source, img_w=200, img_h=100)

    assert (box.x1, box.y1, box.x2, box.y2) == (60.0, 15.0, 140.0, 35.0)
    assert box.class_id == 2
    assert box.to_bbox(200, 100).cx == source.cx
    assert box.to_bbox(200, 100).cy == source.cy
    assert box.to_bbox(200, 100).w == source.w
    assert box.to_bbox(200, 100).h == source.h


def test_edit_box_converts_to_the_canonical_detection_type():
    box = EditBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0, class_id=1)

    det = box.to_detection({1: "PP"})

    assert det == Instance(
        class_id=1,
        class_name="PP",
        conf=1.0,
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=60.0,
    )
    assert EditBox.from_detection(det) == box


# ── Drawing ───────────────────────────────────────────────────────────────


def test_a_draw_commits_one_box_and_selects_it():
    editor = BoxEditor(100, 80, active_class=2)

    index = editor.begin_draw(10.0, 20.0)
    editor.update_draw(40.0, 60.0)
    result = editor.finish_draw()

    assert index == 0
    assert result == CommitResult(committed=True, side_px=30.0)
    assert editor.boxes == (EditBox(10.0, 20.0, 40.0, 60.0, class_id=2),)
    assert editor.selection == 0
    assert editor.mode is EditMode.IDLE
    assert editor.can_undo is True


def test_a_draw_ignores_the_corner_order_and_clips_to_the_image():
    editor = BoxEditor(100, 80)
    editor.begin_draw(90.0, 70.0)
    editor.update_draw(120.0, -10.0)

    result = editor.finish_draw()

    assert result.committed is True
    assert editor.boxes == (EditBox(90.0, 0.0, 100.0, 70.0, class_id=0),)


def test_an_explicit_draw_class_wins_over_the_active_one():
    editor = BoxEditor(100, 80, active_class=0)
    editor.begin_draw(10.0, 10.0, class_id=3)
    editor.update_draw(30.0, 30.0)
    editor.finish_draw()

    assert editor.boxes[0].class_id == 3


def test_a_box_under_the_minimum_side_is_refused_with_a_named_reason():
    editor = BoxEditor(100, 80)

    editor.begin_draw(10.0, 10.0)
    editor.update_draw(11.5, 40.0)  # 1.5 px wide
    result = editor.finish_draw()

    assert result.committed is False
    assert result.reason is RejectionReason.TOO_SMALL
    assert result.side_px == 1.5
    assert editor.boxes == ()
    assert editor.selection == -1
    assert editor.mode is EditMode.IDLE


def test_a_refused_commit_records_no_undo_entry():
    editor = BoxEditor(100, 80)
    editor.begin_draw(10.0, 10.0)
    editor.update_draw(40.0, 40.0)
    editor.finish_draw()

    editor.begin_draw(60.0, 60.0)
    editor.update_draw(61.0, 70.0)
    editor.finish_draw()

    assert editor.undo() is True
    assert editor.boxes == ()
    assert editor.undo() is False  # the refused draw left no entry behind


def test_a_side_of_exactly_two_pixels_is_accepted():
    editor = BoxEditor(100, 80)
    editor.begin_draw(10.0, 10.0)
    editor.update_draw(10.0 + MIN_SIDE_PX, 10.0 + MIN_SIDE_PX)

    result = editor.finish_draw()

    assert result.committed is True
    assert result.side_px == MIN_SIDE_PX
    assert editor.boxes == (EditBox(10.0, 10.0, 12.0, 12.0),)


# ── Moving ────────────────────────────────────────────────────────────────


def test_a_move_translates_the_box_and_commits_it():
    editor = BoxEditor(100, 80, boxes=[EditBox(10.0, 10.0, 30.0, 30.0)])

    assert editor.begin_move(20.0, 20.0) is True
    editor.update_move(30.0, 40.0)
    result = editor.finish_move()

    assert result.committed is True
    assert editor.boxes == (EditBox(20.0, 30.0, 40.0, 50.0),)
    assert editor.selection == 0


def test_a_move_clips_to_the_image_on_commit():
    editor = BoxEditor(100, 80, boxes=[EditBox(10.0, 10.0, 30.0, 30.0)])
    editor.begin_move(20.0, 20.0)
    editor.update_move(5.0, 5.0)  # dx = -15: x1 goes to -5, y1 to -5

    result = editor.finish_move()

    assert result.committed is True
    assert editor.boxes == (EditBox(0.0, 0.0, 15.0, 15.0),)


def test_a_move_that_leaves_the_image_is_refused_and_restores_the_box():
    editor = BoxEditor(100, 80, boxes=[EditBox(10.0, 10.0, 30.0, 30.0)])
    editor.begin_move(20.0, 20.0)
    editor.update_move(-50.0, -50.0)  # entirely outside the image

    result = editor.finish_move()

    assert result.committed is False
    assert result.reason is RejectionReason.TOO_SMALL
    assert editor.boxes == (EditBox(10.0, 10.0, 30.0, 30.0),)
    assert editor.can_undo is False


def test_begin_move_misses_return_false_and_change_nothing():
    editor = BoxEditor(100, 80, boxes=[EditBox(10.0, 10.0, 30.0, 30.0)])

    assert editor.begin_move(80.0, 70.0) is False
    assert editor.mode is EditMode.IDLE


# ── Resizing from every handle ────────────────────────────────────────────

_BASE = EditBox(20.0, 20.0, 80.0, 60.0)

_RESIZE_CASES = [
    (Handle.TOP_LEFT, (10.0, 10.0), EditBox(10.0, 10.0, 80.0, 60.0)),
    (Handle.TOP_RIGHT, (90.0, 10.0), EditBox(20.0, 10.0, 90.0, 60.0)),
    (Handle.BOTTOM_RIGHT, (90.0, 70.0), EditBox(20.0, 20.0, 90.0, 70.0)),
    (Handle.BOTTOM_LEFT, (10.0, 70.0), EditBox(10.0, 20.0, 80.0, 70.0)),
    (Handle.TOP, (50.0, 10.0), EditBox(20.0, 10.0, 80.0, 60.0)),
    (Handle.RIGHT, (90.0, 40.0), EditBox(20.0, 20.0, 90.0, 60.0)),
    (Handle.BOTTOM, (50.0, 70.0), EditBox(20.0, 20.0, 80.0, 70.0)),
    (Handle.LEFT, (10.0, 40.0), EditBox(10.0, 20.0, 80.0, 60.0)),
]


@pytest.mark.parametrize(("handle", "drag_to", "expected"), _RESIZE_CASES)
def test_resize_moves_only_the_edges_of_its_handle(handle, drag_to, expected):
    editor = BoxEditor(100, 80, boxes=[_BASE])
    editor.select(0)

    assert editor.begin_resize(handle, *editor.handle_points()[int(handle)]) is True
    editor.update_resize(*drag_to)
    result = editor.finish_resize()

    assert result.committed is True
    assert editor.boxes == (expected,)


def test_a_resize_across_the_opposite_edge_normalizes_the_box():
    editor = BoxEditor(100, 80, boxes=[_BASE])
    editor.select(0)
    editor.begin_resize(Handle.TOP_LEFT, 20.0, 20.0)
    editor.update_resize(90.0, 70.0)  # past the bottom-right corner

    result = editor.finish_resize()

    assert result.committed is True
    assert editor.boxes == (EditBox(80.0, 60.0, 90.0, 70.0),)


def test_a_resize_that_collapses_the_box_is_refused_and_restores_it():
    editor = BoxEditor(100, 80, boxes=[_BASE])
    editor.select(0)
    editor.begin_resize(Handle.RIGHT, 80.0, 40.0)
    editor.update_resize(20.5, 40.0)  # 0.5 px wide

    result = editor.finish_resize()

    assert result.committed is False
    assert result.reason is RejectionReason.TOO_SMALL
    assert result.side_px == 0.5
    assert editor.boxes == (_BASE,)
    assert editor.can_undo is False


# ── Handle hit-testing ───────────────────────────────────────────────────


def test_every_handle_is_found_at_its_own_point():
    editor = BoxEditor(100, 80, boxes=[_BASE])
    editor.select(0)

    for handle, (hx, hy) in zip(HANDLES, editor.handle_points(), strict=True):
        assert editor.handle_at(hx, hy, tolerance=2.0) is handle


def test_handle_hit_testing_uses_the_tolerance_in_image_pixels():
    editor = BoxEditor(100, 80, boxes=[_BASE])
    editor.select(0)
    x, y = editor.handle_points()[int(Handle.TOP_LEFT)]

    assert editor.handle_at(x + 5.0, y, tolerance=2.0) is Handle.NONE
    assert editor.handle_at(x + 5.0, y, tolerance=6.0) is Handle.TOP_LEFT
    assert editor.handle_at(50.0, 40.0, tolerance=5.0) is Handle.NONE  # centre


def test_handle_points_follow_the_selection():
    editor = BoxEditor(100, 80, boxes=[EditBox(0.0, 0.0, 10.0, 10.0), _BASE])

    assert editor.handle_points() == ()
    editor.select(1)
    assert editor.handle_points()[int(Handle.BOTTOM_RIGHT)] == (80.0, 60.0)


# ── Selection, delete, clear and class assignment ────────────────────────

_A = EditBox(0.0, 0.0, 10.0, 10.0, class_id=0)
_B = EditBox(20.0, 20.0, 30.0, 30.0, class_id=1)
_C = EditBox(40.0, 40.0, 50.0, 50.0, class_id=2)


def test_select_at_prefers_the_selection_then_the_newest_box():
    editor = BoxEditor(100, 80, boxes=[EditBox(0, 0, 50, 50), EditBox(10, 10, 60, 60)])

    assert editor.select_at(20.0, 20.0) == 1  # no selection: newest wins
    editor.select(0)
    assert editor.select_at(20.0, 20.0) == 0  # the selection keeps priority
    assert editor.select_at(90.0, 70.0) == -1


def test_select_clears_and_rejects_out_of_range_indices():
    editor = BoxEditor(100, 80, boxes=[_A, _B, _C])

    assert editor.select(1) is True
    assert editor.select(1) is False  # already selected
    assert editor.select(99) is False
    assert editor.selection == 1
    assert editor.select(-1) is True
    assert editor.selection == -1
    assert editor.select(-1) is False


def test_delete_removes_the_selection_and_undo_restores_it_selected():
    editor = BoxEditor(100, 80, boxes=[_A, _B, _C])
    editor.select(1)

    assert editor.delete_selected() is True
    assert editor.boxes == (_A, _C)
    assert editor.selection == -1

    assert editor.undo() is True
    assert editor.boxes == (_A, _B, _C)
    assert editor.selection == 1

    assert editor.redo() is True
    assert editor.boxes == (_A, _C)
    assert editor.selection == -1


def test_delete_without_a_selection_does_nothing():
    editor = BoxEditor(100, 80, boxes=[_A])

    assert editor.delete_selected() is False
    assert editor.boxes == (_A,)
    assert editor.can_undo is False


def test_clear_removes_every_box_and_undo_brings_them_back():
    editor = BoxEditor(100, 80, boxes=[_A, _B, _C])
    editor.select(2)

    assert editor.clear() is True
    assert editor.boxes == ()
    assert editor.selection == -1

    assert editor.undo() is True
    assert editor.boxes == (_A, _B, _C)
    assert editor.selection == 2


def test_clear_on_an_empty_editor_records_nothing():
    editor = BoxEditor(100, 80)

    assert editor.clear() is False
    assert editor.can_undo is False


def test_assign_class_changes_the_box_and_undo_restores_it():
    editor = BoxEditor(100, 80, boxes=[_A, _B])
    editor.select(0)

    assert editor.assign_class(2) is True
    assert editor.boxes[0].class_id == 2

    assert editor.undo() is True
    assert editor.boxes[0].class_id == 0

    assert editor.redo() is True
    assert editor.boxes[0].class_id == 2


def test_assign_class_can_target_an_index_without_moving_the_selection():
    editor = BoxEditor(100, 80, boxes=[_A, _B])
    editor.select(0)

    assert editor.assign_class(2, index=1) is True
    assert editor.boxes[1].class_id == 2
    assert editor.selection == 0


def test_assigning_the_same_class_records_nothing():
    editor = BoxEditor(100, 80, boxes=[_A])
    editor.select(0)

    assert editor.assign_class(0) is False
    assert editor.can_undo is False


def test_assign_class_without_a_target_does_nothing():
    editor = BoxEditor(100, 80, boxes=[_A])

    assert editor.assign_class(2) is False
    assert editor.assign_class(2, index=9) is False
    assert editor.can_undo is False


def test_replacing_every_box_records_one_undo_entry_and_clears_the_selection():
    """Pre-annotation replaces the drawing through the same history."""
    editor = BoxEditor(100, 80, boxes=[_A])
    editor.select(0)

    editor.replace([_B, _C])

    assert editor.boxes == (_B, _C)
    assert editor.selection == -1
    assert editor.undo() is True
    assert editor.boxes == (_A,)


def test_replacing_without_recording_leaves_history_and_undo_alone():
    """A batch pre-annotation is not an operator gesture to undo."""
    editor = BoxEditor(100, 80, boxes=[_A])

    editor.replace([_B], record=False)

    assert editor.boxes == (_B,)
    assert editor.can_undo is False
    assert editor.undo() is False


# ── Undo history ─────────────────────────────────────────────────────────


def _state(editor: BoxEditor) -> tuple[tuple[EditBox, ...], int]:
    return editor.boxes, editor.selection


def test_undo_and_redo_are_symmetric_across_every_gesture():
    editor = BoxEditor(100, 80, boxes=[EditBox(10.0, 10.0, 30.0, 30.0)])
    states = [_state(editor)]

    editor.begin_draw(40.0, 40.0)
    editor.update_draw(60.0, 60.0)
    editor.finish_draw()
    states.append(_state(editor))

    editor.begin_move(50.0, 50.0)
    editor.update_move(55.0, 55.0)
    editor.finish_move()
    states.append(_state(editor))

    editor.begin_resize(Handle.RIGHT, 65.0, 55.0)
    editor.update_resize(70.0, 55.0)
    editor.finish_resize()
    states.append(_state(editor))

    editor.assign_class(2)
    states.append(_state(editor))

    editor.delete_selected()
    states.append(_state(editor))

    editor.clear()
    states.append(_state(editor))

    for expected in reversed(states[:-1]):
        assert editor.undo() is True
        assert _state(editor) == expected
    assert editor.undo() is False

    for expected in states[1:]:
        assert editor.redo() is True
        assert _state(editor) == expected
    assert editor.redo() is False


def test_the_undo_history_is_bounded():
    editor = BoxEditor(
        100.0,
        80.0,
        boxes=[EditBox(0.0, 0.0, 10.0, 10.0)],
        history_limit=2,
    )
    for i in range(3):
        editor.begin_draw(20.0 + i * 20, 20.0)
        editor.update_draw(30.0 + i * 20, 30.0)
        editor.finish_draw()

    assert len(editor.boxes) == 4
    assert editor.undo() is True
    assert len(editor.boxes) == 3
    assert editor.undo() is True
    assert len(editor.boxes) == 2  # the oldest states were dropped
    assert editor.undo() is False

    assert editor.redo() is True
    assert len(editor.boxes) == 3
    assert editor.redo() is True
    assert len(editor.boxes) == 4
    assert editor.redo() is False


def test_loading_another_image_resets_history_and_selection_but_not_the_class():
    editor = BoxEditor(100, 80, boxes=[_A], active_class=2)
    editor.begin_draw(40.0, 40.0)
    editor.update_draw(60.0, 60.0)
    editor.finish_draw()

    editor.load(50, 40, [_B])

    assert (editor.img_w, editor.img_h) == (50, 40)
    assert editor.boxes == (_B,)
    assert editor.selection == -1
    assert editor.can_undo is False
    assert editor.can_redo is False
    assert editor.active_class == 2


# ── YOLO and detection conversion ────────────────────────────────────────


def test_normalized_yolo_conversion_round_trips_through_the_label_codec(tmp_path):
    editor = BoxEditor(200, 100, boxes=[EditBox(60.0, 15.0, 140.0, 35.0, class_id=2)])

    bbox = editor.to_bboxes()[0]
    assert bbox == labels.BBox(class_id=2, cx=0.5, cy=0.25, w=0.4, h=0.2)

    path = labels.write_lines(tmp_path / "labels" / "a.txt", [bbox])
    parsed, malformed = labels.read_boxes(path)

    assert malformed == 0
    assert parsed == [labels.BBox(class_id=2, cx=0.5, cy=0.25, w=0.4, h=0.2)]
    back = EditBox.from_bbox(parsed[0], 200.0, 100.0)
    assert (back.x1, back.y1, back.x2, back.y2) == pytest.approx((60.0, 15.0, 140.0, 35.0))


def test_detections_use_the_canonical_type_and_the_codec_class_lookup():
    editor = BoxEditor(100, 80, boxes=[EditBox(10.0, 20.0, 30.0, 60.0, class_id=1)])

    assert editor.detections({1: "PP"}) == [
        Instance(class_id=1, class_name="PP", conf=1.0, x1=10.0, y1=20.0, x2=30.0, y2=60.0)
    ]
    assert editor.detections({})[0].class_name == "1"  # the codec's fallback


# ── The rejection message: one threshold, one wording ────────────────────


def test_a_refusal_is_silent_below_the_click_threshold_and_named_above_it():
    """A 0 px click says nothing; a mark the operator meant is named."""
    from polyx.core.editing import SILENT_SIDE_PX, rejection_message
    from polyx.core.i18n import _

    assert rejection_message(0.0) is None
    assert rejection_message(SILENT_SIDE_PX - 0.01) is None

    named = rejection_message(SILENT_SIDE_PX)
    assert named is not None
    assert named == _("⚠ Box of %(side).1f px discarded (minimum %(minimum).0f px) — zoom in") % {
        "side": SILENT_SIDE_PX,
        "minimum": MIN_SIDE_PX,
    }
    # A refused 1.2 px mark names the side it measured and the shared minimum.
    assert rejection_message(1.24) == _(
        "⚠ Box of %(side).1f px discarded (minimum %(minimum).0f px) — zoom in"
    ) % {"side": 1.24, "minimum": MIN_SIDE_PX}
