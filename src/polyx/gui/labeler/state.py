"""Labeller state — images, annotations, undo/redo."""

from __future__ import annotations

import contextlib
import copy
from pathlib import Path

from PySide6.QtCore import QObject, Signal

# The normalized box lives in the YOLO-label codec: one currency for every
# module that reads or writes label files. Re-exported so the canvas and
# the pre-annotation intake keep importing it from here.
from polyx.core.labels import (
    STUDY_CLASSES,
    BBox,
    label_path_for,
    read_boxes,
    write_classes,
    write_lines,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


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

        self._annotations: dict[str, list[BBox]] = {}
        self._undo_stacks: dict[str, list[list[BBox]]] = {}
        self._redo_stacks: dict[str, list[list[BBox]]] = {}
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
        img = self.current_image
        return self._annotations.get(str(img), []) if img else []

    # ── Annotation editing ──────────────────────────────────
    def set_current_boxes(self, boxes: list[BBox], push_undo: bool = True):
        img = self.current_image
        if img is None:
            return
        key = str(img)
        if push_undo:
            self._push_undo(key)
        self._annotations[key] = list(boxes)
        self.annotations_changed.emit()

    def _push_undo(self, key: str):
        self._undo_stacks.setdefault(key, []).append(copy.deepcopy(self._annotations.get(key, [])))
        self._redo_stacks.setdefault(key, []).clear()

    def undo(self):
        img = self.current_image
        if not img:
            return
        key = str(img)
        stack = self._undo_stacks.get(key, [])
        if stack:
            self._redo_stacks.setdefault(key, []).append(
                copy.deepcopy(self._annotations.get(key, []))
            )
            self._annotations[key] = stack.pop()
            self.annotations_changed.emit()

    def redo(self):
        img = self.current_image
        if not img:
            return
        key = str(img)
        stack = self._redo_stacks.get(key, [])
        if stack:
            self._undo_stacks.setdefault(key, []).append(
                copy.deepcopy(self._annotations.get(key, []))
            )
            self._annotations[key] = stack.pop()
            self.annotations_changed.emit()

    # ── Navigation ───────────────────────────────────────────
    def load_images(self, folder: Path):
        imgs = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        self.images = imgs
        self.root_folder = folder
        self._annotations.clear()
        self._undo_stacks.clear()
        self._redo_stacks.clear()
        # Restores progress from previous sessions: a .txt on disk means
        # that the image was already reviewed, whether or not it has boxes.
        self._reviewed = {str(p) for p in imgs if label_path_for(p).exists()}
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
        self._load_labels_for(self.images[idx])
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
            self._save_labels_for(img, self.current_boxes, force=True)
        else:
            self._reviewed.discard(key)
            txt = label_path_for(img)
            if txt.exists() and not self.current_boxes:
                txt.unlink()
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
        if img:
            self._save_labels_for(img, self.current_boxes)

    def _save_labels_for(self, img: Path, boxes: list[BBox], force: bool = False):
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

    def _load_labels_for(self, img: Path):
        key = str(img)
        if key in self._annotations:
            return
        txt = label_path_for(img)
        boxes: list[BBox] = []
        if txt.exists():
            boxes, _malformed = read_boxes(txt)
        self._annotations[key] = boxes

    def save_classes_txt(self):
        """Writes classes.txt at the root and in every subfolder with images.

        With nested folders (one directory per control, `testigo`) the root
        is not enough: the YOLO tools look for classes.txt next to the
        labels.
        """
        if not self.images:
            return
        destinations = {label_path_for(p).parent for p in self.images}
        if self.root_folder is not None:
            destinations.add(self.root_folder)
        for d in destinations:
            with contextlib.suppress(OSError):
                write_classes(d, self.class_names)

    # ── Classes ──────────────────────────────────────────────
    def set_active_class(self, cls_id: int):
        self.active_class = cls_id
        self.active_class_changed.emit(cls_id)

    def set_class_names(self, names: list[str]):
        self.class_names = names
        self.classes_changed.emit(names)
