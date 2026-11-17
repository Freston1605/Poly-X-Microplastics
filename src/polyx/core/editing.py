"""The shared box-editing core — geometry, selection and undo, without Qt.

The Labeller and the Manual GT page are two complete box editors. This
module is the one implementation of the editing domain they share: a box
under edit in absolute image coordinates (:class:`EditBox`), and a per-image
editor (:class:`BoxEditor`) that owns the boxes, the selection, a bounded
undo/redo history, and every gesture — draw, move, resize by handle, delete,
clear, class assignment and select.

Two invariants live here, not in a widget:

* **The 2 px minimum side.** A box whose shorter side is under
  :data:`MIN_SIDE_PX` (a census-honesty rule) cannot be committed. The
  commit returns a :class:`CommitResult` naming the rejection, and a refused
  gesture records no undo entry.
* **One coordinate convention.** Callers work in image pixels and pass a
  hit-testing tolerance in image pixels, so the shell converts its roughly
  constant widget distance through the current scale and the core never
  needs to know about zoom, pan or Qt.

Normalized YOLO coordinates and the canonical detection type are reached
through :mod:`polyx.core.labels` — this module never re-derives width/height
normalization or class-name lookup, and it never touches the label file
format.

Host policy stays outside, on purpose: the Labeller keeps the image list,
review marks and census; the GT page keeps saving and auto-save; each shell
keeps zoom and pan.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum, IntEnum

from polyx.core import labels
from polyx.core.detect.protocol import Instance
from polyx.core.i18n import _

# A box shorter than this on either side is not a real particle mark. The
# smallest particles in the study measure ~8 px; a 5 px threshold discarded
# legitimate marks, so the census rule settled on 2.
MIN_SIDE_PX = 2.0

# A refused gesture whose side is under this was an accidental click, not a
# mark the operator believed they made: it is discarded without a message.
# Above it the message is shown, because the mark vanished without warning.
SILENT_SIDE_PX = 0.3

# How close, in widget pixels, a press must land to a handle's centre to grab
# it. The shells scale it by their current zoom and pass image pixels to
# :meth:`BoxEditor.handle_at`; the value is one constant so the two
# annotation canvases cannot disagree about how forgiving a handle is.
HANDLE_TOLERANCE_PX = 6.0

# The GT page's current undo depth; the bounded history keeps one editor's
# memory predictable while the Labeller keeps one editor per loaded image.
HISTORY_LIMIT = 100


class EditMode(Enum):
    """What the editor is currently in the middle of."""

    IDLE = "idle"
    DRAW = "draw"
    MOVE = "move"
    RESIZE = "resize"


class Handle(IntEnum):
    """The eight resize handles, in the order the canvases paint them."""

    NONE = -1
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 2
    BOTTOM_LEFT = 3
    TOP = 4
    RIGHT = 5
    BOTTOM = 6
    LEFT = 7


# Iteration order for the tuple returned by ``handle_points``. ``Handle``
# itself includes NONE, which has no position and must stay out of it.
HANDLES = (
    Handle.TOP_LEFT,
    Handle.TOP_RIGHT,
    Handle.BOTTOM_RIGHT,
    Handle.BOTTOM_LEFT,
    Handle.TOP,
    Handle.RIGHT,
    Handle.BOTTOM,
    Handle.LEFT,
)


class RejectionReason(Enum):
    """Why a finished gesture was not committed."""

    TOO_SMALL = "too-small"


def rejection_message(side_px: float) -> str | None:
    """The translated message a refused gesture shows, or ``None`` if silent.

    One rule for both annotation shells: a side under :data:`SILENT_SIDE_PX`
    is an accidental click and says nothing; anything above it, still under
    :data:`MIN_SIDE_PX`, is a mark the operator believed they made — it is
    refused and named, so it does not vanish without warning.
    """
    if side_px < SILENT_SIDE_PX:
        return None
    return _("⚠ Box of %(side).1f px discarded (minimum %(minimum).0f px) — zoom in") % {
        "side": side_px,
        "minimum": MIN_SIDE_PX,
    }


@dataclass(frozen=True, slots=True)
class CommitResult:
    """The outcome of finishing a draw, move or resize.

    ``side_px`` is the shorter side of the box that was offered to the
    commit, after clamping: the shells show it in the translated rejection
    message ("the box was 1.2 px"). It is ``0.0`` when the gesture was
    committed or when there was nothing to commit.
    """

    committed: bool
    reason: RejectionReason | None = None
    side_px: float = 0.0


@dataclass(frozen=True, slots=True)
class EditBox:
    """A box under edit, in absolute image pixels.

    ``x1, y1`` and ``x2, y2`` are opposite corners; a live drag may carry
    them crossed until the commit normalizes the box. :attr:`width`,
    :attr:`height` and :attr:`center` are pixel-space conveniences; the
    normalized YOLO currency is reached through :meth:`to_bbox`.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def to_bbox(self, img_w: int, img_h: int) -> labels.BBox:
        """Normalized YOLO box, normalized by the canonical label codec.

        A box under edit is manual ground truth: it carries no confidence
        column, so the prediction-only ``conf`` the codec copies from the
        detection shape is cleared.
        """
        return replace(labels.bbox_from(self._as_instance(), img_w, img_h), conf=None)

    @classmethod
    def from_bbox(cls, box: labels.BBox, img_w: int, img_h: int) -> EditBox:
        """Absolute box from a normalized YOLO box, via the label codec."""
        det = labels.to_instance(box, {}, img_w, img_h)
        return cls(det.x1, det.y1, det.x2, det.y2, det.class_id)

    def to_detection(self, class_names: dict[int, str]) -> Instance:
        """The canonical detection shape, with the codec's class-name lookup.

        The geometry is already absolute, so it is copied exactly. The name
        comes from :func:`polyx.core.labels.class_name_for`; a manual box is
        total certainty, the same 1.0 the codec's ``to_instance`` uses for a
        box with no confidence column.
        """
        return Instance(
            class_id=self.class_id,
            class_name=labels.class_name_for(self.class_id, class_names),
            conf=1.0,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

    @classmethod
    def from_detection(cls, det: Instance) -> EditBox:
        """A box from the canonical detection type (already in pixels)."""
        return cls(det.x1, det.y1, det.x2, det.y2, det.class_id)

    def _as_instance(self) -> Instance:
        return Instance(
            class_id=self.class_id,
            class_name=str(self.class_id),
            conf=1.0,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )


class BoxEditor:
    """The boxes, selection and undo history of one image under edit.

    Image dimensions are fixed for the editor's lifetime; call :meth:`load`
    when the shell loads another image. Every mutating gesture ends in the
    same commit path, so the minimum-side invariant and the undo entry are
    decided in one place.
    """

    def __init__(
        self,
        img_w: int,
        img_h: int,
        boxes: Iterable[EditBox] = (),
        active_class: int = 0,
        history_limit: int = HISTORY_LIMIT,
    ) -> None:
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"image size must be positive, got {img_w}x{img_h}")
        self._history_limit = max(1, history_limit)
        self.load(img_w, img_h, boxes)
        self.active_class = active_class

    # ── Read-only state ─────────────────────────────────────
    @property
    def img_w(self) -> int:
        return self._img_w

    @property
    def img_h(self) -> int:
        return self._img_h

    @property
    def boxes(self) -> tuple[EditBox, ...]:
        return tuple(self._boxes)

    @property
    def selection(self) -> int:
        return self._selected

    @property
    def selected_box(self) -> EditBox | None:
        if 0 <= self._selected < len(self._boxes):
            return self._boxes[self._selected]
        return None

    @property
    def mode(self) -> EditMode:
        return self._mode

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # ── Loading ─────────────────────────────────────────────
    def load(
        self,
        img_w: int,
        img_h: int,
        boxes: Iterable[EditBox] = (),
    ) -> None:
        """Replace the image and its boxes; history and selection reset."""
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"image size must be positive, got {img_w}x{img_h}")
        self._img_w = img_w
        self._img_h = img_h
        self._boxes: list[EditBox] = list(boxes)
        self._selected = -1
        self._undo: list[tuple[tuple[EditBox, ...], int]] = []
        self._redo: list[tuple[tuple[EditBox, ...], int]] = []
        self._mode = EditMode.IDLE
        self._pending: tuple[tuple[EditBox, ...], int] | None = None
        self._base_box: EditBox | None = None
        self._press = (0.0, 0.0)
        self._handle = Handle.NONE

    # ── Selection ───────────────────────────────────────────
    def select(self, index: int) -> bool:
        """Select a box by index, or ``-1`` to clear the selection."""
        if index == -1:
            changed = self._selected != -1
            self._selected = -1
            return changed
        if not 0 <= index < len(self._boxes):
            return False
        changed = self._selected != index
        self._selected = index
        return changed

    def box_at(self, x: float, y: float) -> int:
        """Index of the box under ``(x, y)``, or ``-1``.

        The selected box wins when it is under the cursor, so a drag on an
        overlapping selection keeps it; otherwise the newest box wins.
        """
        if 0 <= self._selected < len(self._boxes) and self._contains(
            self._boxes[self._selected], x, y
        ):
            return self._selected
        for index in range(len(self._boxes) - 1, -1, -1):
            if self._contains(self._boxes[index], x, y):
                return index
        return -1

    def select_at(self, x: float, y: float) -> int:
        """Select the box under ``(x, y)``; returns the index or ``-1``."""
        index = self.box_at(x, y)
        self._selected = index
        return index

    @staticmethod
    def _contains(box: EditBox, x: float, y: float) -> bool:
        left, right = sorted((box.x1, box.x2))
        top, bottom = sorted((box.y1, box.y2))
        return left <= x <= right and top <= y <= bottom

    # ── Undo / redo ─────────────────────────────────────────
    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    def _snapshot(self) -> tuple[tuple[EditBox, ...], int]:
        return tuple(self._boxes), self._selected

    def _restore(self, state: tuple[tuple[EditBox, ...], int]) -> None:
        boxes, selected = state
        self._boxes = list(boxes)
        self._selected = selected
        self._mode = EditMode.IDLE
        self._pending = None
        self._base_box = None
        self._handle = Handle.NONE

    def _record(self, state: tuple[tuple[EditBox, ...], int] | None = None) -> None:
        """Push a pre-edit snapshot and drop the redo branch."""
        self._undo.append(state if state is not None else self._snapshot())
        if len(self._undo) > self._history_limit:
            self._undo.pop(0)
        self._redo.clear()

    # ── Drawing ─────────────────────────────────────────────
    def begin_draw(self, x: float, y: float, class_id: int | None = None) -> int:
        """Start a draw at ``(x, y)``; the provisional box is selected."""
        self._cancel_gesture()
        self._pending = self._snapshot()
        cls = self.active_class if class_id is None else class_id
        self._boxes.append(EditBox(x, y, x, y, cls))
        self._selected = len(self._boxes) - 1
        self._mode = EditMode.DRAW
        return self._selected

    def update_draw(self, x: float, y: float) -> None:
        if self._mode is not EditMode.DRAW or not self._boxes:
            return
        box = self._boxes[-1]
        self._boxes[-1] = EditBox(box.x1, box.y1, x, y, box.class_id)

    def finish_draw(self) -> CommitResult:
        """Commit the provisional box, or refuse it and record nothing."""
        if self._mode is not EditMode.DRAW or not self._boxes:
            return CommitResult(False)
        return self._commit(len(self._boxes) - 1)

    # ── Moving ──────────────────────────────────────────────
    def begin_move(self, x: float, y: float) -> bool:
        """Start moving the box under ``(x, y)``; selects it first."""
        self._cancel_gesture()
        index = self.box_at(x, y)
        if index < 0:
            return False
        self._selected = index
        self._pending = self._snapshot()
        self._base_box = self._boxes[index]
        self._press = (x, y)
        self._mode = EditMode.MOVE
        return True

    def update_move(self, x: float, y: float) -> None:
        if self._mode is not EditMode.MOVE or self._base_box is None:
            return
        dx = x - self._press[0]
        dy = y - self._press[1]
        base = self._base_box
        self._boxes[self._selected] = EditBox(
            base.x1 + dx,
            base.y1 + dy,
            base.x2 + dx,
            base.y2 + dy,
            base.class_id,
        )

    def finish_move(self) -> CommitResult:
        return self._finish_transform(EditMode.MOVE)

    # ── Resizing ────────────────────────────────────────────
    def handle_at(self, x: float, y: float, tolerance: float) -> Handle:
        """The selected box's handle under ``(x, y)``, within ``tolerance``.

        ``tolerance`` is in image pixels: the shell converts its fixed
        widget-side handle target through the current scale.
        """
        box = self.selected_box
        if box is None:
            return Handle.NONE
        for handle, (hx, hy) in zip(HANDLES, self.handle_points(box), strict=True):
            if abs(x - hx) <= tolerance and abs(y - hy) <= tolerance:
                return handle
        return Handle.NONE

    def handle_points(self, box: EditBox | None = None) -> tuple[tuple[float, float], ...]:
        """The eight handle positions of ``box`` (the selection by default)."""
        box = box if box is not None else self.selected_box
        if box is None:
            return ()
        cx, cy = box.center
        return (
            (box.x1, box.y1),  # TOP_LEFT
            (box.x2, box.y1),  # TOP_RIGHT
            (box.x2, box.y2),  # BOTTOM_RIGHT
            (box.x1, box.y2),  # BOTTOM_LEFT
            (cx, box.y1),  # TOP
            (box.x2, cy),  # RIGHT
            (cx, box.y2),  # BOTTOM
            (box.x1, cy),  # LEFT
        )

    def begin_resize(self, handle: Handle, x: float, y: float) -> bool:
        """Start resizing the selected box from ``handle`` at ``(x, y)``."""
        self._cancel_gesture()
        if handle is Handle.NONE or self.selected_box is None:
            return False
        self._pending = self._snapshot()
        self._base_box = self.selected_box
        self._press = (x, y)
        self._handle = handle
        self._mode = EditMode.RESIZE
        return True

    def update_resize(self, x: float, y: float) -> None:
        if self._mode is not EditMode.RESIZE or self._base_box is None:
            return
        base = self._base_box
        x1, y1, x2, y2 = base.x1, base.y1, base.x2, base.y2
        handle = self._handle
        if handle in (Handle.TOP_LEFT, Handle.TOP, Handle.TOP_RIGHT):
            y1 = y
        if handle in (Handle.TOP_RIGHT, Handle.RIGHT, Handle.BOTTOM_RIGHT):
            x2 = x
        if handle in (Handle.BOTTOM_RIGHT, Handle.BOTTOM, Handle.BOTTOM_LEFT):
            y2 = y
        if handle in (Handle.BOTTOM_LEFT, Handle.LEFT, Handle.TOP_LEFT):
            x1 = x
        self._boxes[self._selected] = EditBox(x1, y1, x2, y2, base.class_id)

    def finish_resize(self) -> CommitResult:
        return self._finish_transform(EditMode.RESIZE)

    # ── Committing a gesture ────────────────────────────────
    def _finish_transform(self, mode: EditMode) -> CommitResult:
        if self._mode is not mode or self.selected_box is None:
            return CommitResult(False)
        return self._commit(self._selected)

    def _commit(self, index: int) -> CommitResult:
        """Clamp, check the minimum side and either record or refuse.

        The pending snapshot is the pre-gesture state: a successful commit
        pushes it once, a refusal restores it without touching either
        history stack, so undo never sees a box the census refused.
        """
        box = _clamp(self._boxes[index], self._img_w, self._img_h)
        side = min(box.width, box.height)
        if side < MIN_SIDE_PX:
            self._cancel_gesture()
            return CommitResult(False, RejectionReason.TOO_SMALL, side)
        self._boxes[index] = box
        self._selected = index
        self._record(self._pending)
        self._mode = EditMode.IDLE
        self._pending = None
        self._base_box = None
        self._handle = Handle.NONE
        return CommitResult(True, side_px=side)

    def _cancel_gesture(self) -> None:
        """Drop an in-flight gesture and restore the pre-gesture state."""
        if self._pending is None:
            return
        state = self._pending
        self._pending = None
        self._restore(state)

    # ── Whole-box operations ────────────────────────────────
    def delete_selected(self) -> bool:
        if not 0 <= self._selected < len(self._boxes):
            return False
        self._record()
        del self._boxes[self._selected]
        self._selected = -1
        return True

    def clear(self) -> bool:
        if not self._boxes:
            return False
        self._record()
        self._boxes.clear()
        self._selected = -1
        return True

    def set_active_class(self, class_id: int) -> int:
        """Change the class new boxes get; returns the previous class."""
        previous = self.active_class
        self.active_class = class_id
        return previous

    def assign_class(self, class_id: int, index: int | None = None) -> bool:
        """Assign ``class_id`` to a box (the selection by default)."""
        target = self._selected if index is None else index
        if not 0 <= target < len(self._boxes):
            return False
        box = self._boxes[target]
        if box.class_id == class_id:
            return False
        self._record()
        self._boxes[target] = EditBox(box.x1, box.y1, box.x2, box.y2, class_id)
        return True

    def replace(self, boxes: Iterable[EditBox], record: bool = True) -> None:
        """Swap the whole drawing, as a host policy does after pre-annotation.

        ``record`` decides whether the previous drawing becomes one undo
        entry: replacing by an operator gesture is undoable, a batch that
        fills the blanks is not. The selection is cleared because the new
        boxes have no history with it.
        """
        if record:
            self._record()
        self._boxes = list(boxes)
        self._selected = -1
        self._mode = EditMode.IDLE
        self._pending = None
        self._base_box = None
        self._handle = Handle.NONE

    # ── Conversion ──────────────────────────────────────────
    def to_bboxes(self) -> list[labels.BBox]:
        """The boxes as normalized YOLO boxes, through the label codec."""
        return [box.to_bbox(self._img_w, self._img_h) for box in self._boxes]

    def detections(self, class_names: dict[int, str]) -> list[Instance]:
        """The boxes as the canonical detection type."""
        return [box.to_detection(class_names) for box in self._boxes]


def _clamp(box: EditBox, img_w: int, img_h: int) -> EditBox:
    """Normalize the corners and clip the box to the image."""
    x1, x2 = sorted((box.x1, box.x2))
    y1, y2 = sorted((box.y1, box.y2))
    return EditBox(
        min(max(x1, 0.0), img_w),
        min(max(y1, 0.0), img_h),
        min(max(x2, 0.0), img_w),
        min(max(y2, 0.0), img_h),
        box.class_id,
    )


__all__ = [
    "HANDLES",
    "HANDLE_TOLERANCE_PX",
    "HISTORY_LIMIT",
    "MIN_SIDE_PX",
    "SILENT_SIDE_PX",
    "BoxEditor",
    "CommitResult",
    "EditBox",
    "EditMode",
    "Handle",
    "RejectionReason",
    "rejection_message",
]
