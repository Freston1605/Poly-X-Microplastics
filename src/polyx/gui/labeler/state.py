"""Labeller state — images, annotations, review policy and persistence.

Editing itself — geometry, selection and undo — belongs to one
:class:`polyx.core.editing.BoxEditor` per loaded image, kept here so the
history survives navigating away and back. This object stays the host: the
image list, review marks, census rules and what gets written to disk.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from polyx.core.detect.protocol import size_image
from polyx.core.editing import BoxEditor, EditBox

# The normalized box lives in the YOLO-label codec: one currency for every
# module that reads or writes label files. Re-exported so the canvas and
# the pre-annotation intake keep importing it from here.
from polyx.core.labels import (
    STUDY_CLASSES,
    BBox,
    classes_dir_for,
    find_label_file,
    label_path_for,
    read_boxes,
    write_classes,
    write_lines,
)
from polyx.core.paths import IMAGE_EXTS


class LabelerState(QObject):
    images_loaded = Signal(list)  # List[Path]
    image_changed = Signal(int)  # current_idx
    annotations_changed = Signal()
    classes_changed = Signal(list)  # List[str]
    active_class_changed = Signal(int)
    progress_changed = Signal()  # the set of reviewed images changed

    def __init__(self):
        super().__init__()
        self.images: list[Path] = []
        self.current_idx: int = -1
        self.class_names: list[str] = list(STUDY_CLASSES)
        self.active_class: int = 0
        self.root_folder: Path | None = None

        # One editor per image visited in this session. Keeping it is what
        # makes ``Ctrl+Z`` work after going to another image and back.
        self._editors: dict[str, BoxEditor] = {}
        # Drawings cached for images not visited yet (batch pre-annotation);
        # they seed the editor when the image is opened.
        self._drawings: dict[str, list[BBox]] = {}
        # Images the operator declared reviewed. A census count needs
        # to distinguish "reviewed, zero particles" from "not looked at yet": both
        # have zero boxes, but only the first one is a datum.
        self._reviewed: set[str] = set()

    # ── Properties ───────────────────────────────────────────
    @property
    def current_image(self) -> Path | None:
        if 0 <= self.current_idx < len(self.images):
            return self.images[self.current_idx]
        return None

    @property
    def current_boxes(self) -> list[BBox]:
        editor = self.current_editor
        return editor.to_bboxes() if editor is not None else []

    @property
    def current_editor(self) -> BoxEditor | None:
        """The editor of the current image, created on first use."""
        img = self.current_image
        return self._editor_for(img) if img is not None else None

    # ── Annotation editing ──────────────────────────────────
    def set_current_boxes(self, boxes: list[BBox], push_undo: bool = True):
        editor = self.current_editor
        if editor is None:
            return
        editor.replace(
            (EditBox.from_bbox(b, editor.img_w, editor.img_h) for b in boxes),
            record=push_undo,
        )
        self.annotations_changed.emit()

    def notify_edited(self):
        """Announce an edit the canvas made directly on the current editor."""
        self.annotations_changed.emit()

    def undo(self):
        editor = self.current_editor
        if editor is not None and editor.undo():
            self.annotations_changed.emit()

    def redo(self):
        editor = self.current_editor
        if editor is not None and editor.redo():
            self.annotations_changed.emit()

    # ── Navigation ───────────────────────────────────────────
    def load_images(self, folder: Path):
        imgs = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        self.images = imgs
        self.root_folder = folder
        self._editors.clear()
        self._drawings.clear()
        # Restores progress from previous sessions: a .txt on disk means
        # that the image was already reviewed, whether or not it has boxes.
        # The codec decides which file that is: the conventional
        # labels/<split>/ one or a legacy same-directory file.
        self._reviewed = {str(p) for p in imgs if find_label_file(p) is not None}
        self.images_loaded.emit(imgs)
        if imgs:
            self.current_idx = -1  # force goto to emit image_changed
            self.goto(0)
        else:
            self.current_idx = -1

    def goto(self, idx: int):
        if not self.images:
            return
        idx = max(0, min(idx, len(self.images) - 1))
        if idx == self.current_idx:
            return
        self.save_current()
        self.current_idx = idx
        self._editor_for(self.images[idx])
        self.image_changed.emit(idx)
        self.annotations_changed.emit()

    def next_image(self):
        self.goto(self.current_idx + 1)

    def prev_image(self):
        self.goto(self.current_idx - 1)

    # ── Review ───────────────────────────────────────────────
    def is_reviewed(self, img: Path) -> bool:
        return str(img) in self._reviewed

    def mark_reviewed(self, reviewed: bool = True):
        """Declares the current image as reviewed, even without particles."""
        img = self.current_image
        if img is None:
            return
        key = str(img)
        if reviewed:
            self._reviewed.add(key)
            self.save_labels(img, self.current_boxes, force=True)
        else:
            self._reviewed.discard(key)
            # Withdraw the datum: remove the file the codec found, wherever
            # it lives (conventional or legacy). The drawing stays in memory:
            # it is the operator's work, so Ctrl+Z keeps working, and a
            # drawn-on image is data — moving on may save it again.
            found = find_label_file(img)
            if found is not None:
                found.unlink(missing_ok=True)
            self.annotations_changed.emit()
        self.progress_changed.emit()

    def n_reviewed(self) -> int:
        return len(self._reviewed)

    def next_unreviewed(self) -> int:
        """Index of the next unreviewed image; -1 if none is left."""
        n = len(self.images)
        for break_value in range(1, n + 1):
            i = (self.current_idx + break_value) % n
            if str(self.images[i]) not in self._reviewed:
                return i
        return -1

    # ── Persistence ──────────────────────────────────────────
    def save_current(self):
        img = self.current_image
        if img is None:
            return
        editor = self.current_editor
        if editor is None:
            # The image could not be measured, so there is no frame to read
            # the drawing in and ``current_boxes`` is empty for lack of an
            # editor, not because the plate is empty. Writing that empty
            # drawing would destroy a reviewed GT; leave the file alone.
            return
        self.save_labels(img, editor.to_bboxes())

    def save_labels(self, img: Path, boxes: list[BBox], force: bool = False):
        """Writes the .txt. With no boxes and not reviewed, it does NOT create the file.

        Creating empty .txt files just from passing over an image would
        falsify the census: an image barely glanced at would be recorded as
        reviewed with zero particles, which is a different datum. That
        policy lives HERE, not in the codec: the codec writes exactly what
        it is told.
        """
        if not boxes and not force and str(img) not in self._reviewed:
            return
        write_lines(label_path_for(img), boxes)
        if boxes:
            self._reviewed.add(str(img))

    def _editor_for(self, img: Path) -> BoxEditor | None:
        """The image's editor, created and cached on first use.

        The drawing comes from the in-memory cache a batch pre-annotation may
        have left, or from the label file the codec finds. The image size is
        read header-only, with the EXIF orientation the canvas also applies.
        """
        key = str(img)
        editor = self._editors.get(key)
        if editor is None:
            size = size_image(img)
            if size is None:
                return None
            img_w, img_h = size
            boxes = self._drawings.pop(key, None)
            if boxes is None:
                txt = find_label_file(img)
                boxes = read_boxes(txt)[0] if txt is not None else []
            editor = BoxEditor(
                img_w,
                img_h,
                (EditBox.from_bbox(b, img_w, img_h) for b in boxes),
            )
            self._editors[key] = editor
        editor.set_active_class(self.active_class)
        return editor

    def box_count_for(self, img: Path) -> int:
        """How many boxes the image has: the live editor, the cache or the file.

        A drawing in memory wins, because it is what the operator currently
        sees; otherwise the count comes from the file the codec found.
        """
        key = str(img)
        editor = self._editors.get(key)
        if editor is not None:
            return len(editor.boxes)
        if key in self._drawings:
            return len(self._drawings[key])
        txt = find_label_file(img)
        if txt is None:
            return 0
        boxes, _malformed = read_boxes(txt)
        return len(boxes)

    def has_boxes_for(self, img: Path) -> bool:
        """Whether the image already has a box, in memory or in its label file."""
        return self.box_count_for(img) > 0

    def set_boxes_for(self, img: Path, boxes: list[BBox]) -> None:
        """Caches a drawing for an image that is not the current one.

        Used by batch pre-annotation, which walks the whole list without
        navigating; the current image's edits go through
        :meth:`set_current_boxes` so they push undo. A batch is not an
        operator gesture, so replacing the drawing records no history.
        """
        key = str(img)
        editor = self._editors.get(key)
        if editor is not None:
            editor.replace(
                (EditBox.from_bbox(b, editor.img_w, editor.img_h) for b in boxes),
                record=False,
            )
        else:
            self._drawings[key] = list(boxes)

    def preannotate(self, boxes_by_image: dict[Path, list[BBox]]) -> int:
        """Applies and saves a batch of model drawings; returns how many it kept.

        Skips images that already have a box, in memory or in the file the
        codec finds: the batch fills the blanks, it does not overwrite the
        operator's work. Saving goes through :meth:`save_labels`, so the
        empty-file census rule is the same one the editor uses, and
        ``annotations_changed`` is emitted once for the whole batch. The
        progress feedback stays in the window: the model runs there.
        """
        annotated = 0
        for img, boxes in boxes_by_image.items():
            if self.has_boxes_for(img):
                continue
            self.set_boxes_for(img, boxes)
            self.save_labels(img, boxes)
            annotated += 1
        if annotated:
            self.annotations_changed.emit()
        return annotated

    def save_classes_txt(self):
        """Writes classes.txt where the labels of the loaded images live.

        The codec resolves the destination: for a split layout every label
        folder resolves to the one dataset root, so the file is written
        once; a plain folder keeps one ``classes.txt`` per folder that holds
        labels (the YOLO tools look for it next to them).
        """
        if not self.images:
            return
        destinations = {classes_dir_for(label_path_for(p).parent) for p in self.images}
        for d in destinations:
            with contextlib.suppress(OSError):
                write_classes(d, self.class_names)

    # ── Classes ──────────────────────────────────────────────
    def set_active_class(self, cls_id: int):
        """The class new boxes get, and the class the selection takes.

        One rule for both annotation canvases: the key (or the class button)
        sets the active class and, when a box is selected, reclassifies it
        through the core — which records one undo entry and nothing when the
        class is already the box's.
        """
        if not 0 <= cls_id < len(self.class_names):
            return
        self.active_class = cls_id
        editor = self.current_editor
        if editor is not None:
            editor.set_active_class(cls_id)
            if editor.assign_class(cls_id):
                self.annotations_changed.emit()
        self.active_class_changed.emit(cls_id)

    def set_class_names(self, names: list[str]):
        self.class_names = names
        self.classes_changed.emit(names)
