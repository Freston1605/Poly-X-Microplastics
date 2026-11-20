"""The YOLO-label codec — one owner for the YOLO ``.txt`` format.

Every module that reads or writes a YOLO annotation file goes through here:
parse (box lines and YOLO-seg polygons), serialize (with or without the
confidence column), locate (the ``images/`` <-> ``labels/`` folder
convention, both directions), ``classes.txt`` handling, and — at the file
level — the EXIF-safe image size that normalized coordinates depend on.

Folder contract
---------------

The conventional Ultralytics layout is a dataset root holding ``images/``
and ``labels/`` side by side, each with the same relative subpath per image
(``images/train/a.jpg`` -> ``labels/train/a.txt``). The codec resolves the
*nearest ancestor named* ``images`` (case-insensitive) to its sibling
``labels/`` and preserves the relative subpath; no split name is ever
hard-coded, so ``train``, ``val``, ``test`` and any custom folder work
alike. A folder with no ``images`` ancestor is a *plain* folder: its labels
live next to the images, with a nested ``labels/`` file accepted as a
legacy fallback.

Reads prefer the conventional location, then fall back to a legacy
same-directory file and — in plain folders — to a nested ``labels/`` file,
logging a warning whenever a legacy file is used. Writes always go to the
conventional location for a split layout (never next to the image) and to
the image's own folder for a plain one. An explicit GT folder is a flat
override: flat read first, mirrored read second, flat write.

``classes.txt`` is folder-local: for a split layout the dataset root (the
parent of ``labels/``) owns one file, and any path under ``labels/``
resolves to it; a plain folder keeps its own ``classes.txt``.

``data.yaml`` is deliberately outside this module: the dataset module owns
YAML class names and passes them in as explicit ``class_names``. The codec
never parses YAML; ``classes.txt`` is its folder-local fallback.

Two currencies, on purpose:

- :class:`BBox` — the normalized box (``cx, cy, w, h`` in ``[0, 1]``), the
  format's native currency and the Labeller's editing currency. ``conf`` is
  optional: prediction records carry it as a sixth column, manual GT does
  not.
- :class:`~polyx.core.detect.protocol.Instance` — the pipeline's absolute
  pixel box (plus polygon masks when the file carries YOLO-seg polygons).

Dialects: 5 columns is the GT/manual dialect, 6 columns (``conf`` last) is
the prediction-record dialect — the extra column is what lets a run's
labels reopen in the Labeller as pre-annotation while keeping the record
of each box's confidence. YOLO-seg polygon lines are written by
:func:`write_polygon_lines` when the headless pipeline converts the canonical
COCO mask dataset into a family's expected format at job time; that conversion
is an ephemeral intermediate, so the codec stays the only owner of the format
without becoming a second source of truth.

Policies that stay OUTSIDE the codec: when a file may be written at all
(the Labeller's census honesty — a ``.txt`` exists only for images that
were reviewed or drawn on), and where predictions go (never next to the
photo — that would destroy the manual GT; the run folder decides).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from polyx.core.detect.protocol import Instance, size_image
from polyx.core.paths import IMAGE_EXTS
from polyx.core.provenance.observability import logger_for

# The study's classes. One definition: every reader falls back to this when
# there is no classes.txt, and every writer that has no better information
# writes it. Interface colours per class live in gui/theme.py — domain data,
# not the same thing.
STUDY_CLASSES = ["PET", "PP", "LDPE"]


@dataclass
class BBox:
    """A box in the format's native, normalized currency.

    ``cx, cy, w, h`` are fractions of the image side in ``[0, 1]``. ``conf``
    is ``None`` for manual annotations and set for prediction records.
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float | None = None


# ── Lines (pure) ──────────────────────────────────────────────────────────


def parse_box_line(line: str) -> BBox | None:
    """Parse one YOLO detection line into a :class:`BBox`, or ``None``.

    Accepts the 5-column GT dialect and the 6-column prediction dialect
    (``conf`` last). Returns ``None`` for anything that is not a box line:
    garbage, too few fields, or non-numeric values. YOLO-seg polygon lines
    are NOT box lines; :func:`read_boxes` counts them instead of mangling
    them into boxes.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else None
    except ValueError:
        return None
    return BBox(class_id=class_id, cx=cx, cy=cy, w=w, h=h, conf=conf)


def format_line(box: BBox, include_conf: bool = False) -> str:
    """Serialize a :class:`BBox` as one YOLO line.

    The confidence column is written only when both requested AND present:
    the dialect is the caller's decision, and a missing confidence must not
    be invented.
    """
    line = f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}"
    if include_conf and box.conf is not None:
        line += f" {box.conf:.4f}"
    return line


def bbox_from(det: Instance, img_w: int, img_h: int) -> BBox:
    """Normalize an absolute-pixel :class:`Instance` into a :class:`BBox`."""
    w = det.x2 - det.x1
    h = det.y2 - det.y1
    return BBox(
        class_id=det.class_id,
        cx=((det.x1 + det.x2) / 2) / img_w,
        cy=((det.y1 + det.y2) / 2) / img_h,
        w=w / img_w,
        h=h / img_h,
        conf=det.conf,
    )


def class_name_for(class_id: int, class_names: dict[int, str]) -> str:
    """The name for a class id: the dataset's, or the id itself.

    One lookup for every module that converts a box to the canonical
    detection type, so an unknown class is rendered the same everywhere.
    """
    return class_names.get(class_id, str(class_id))


def to_instance(box: BBox, class_names: dict[int, str], img_w: int, img_h: int) -> Instance:
    """Denormalize a :class:`BBox` into an absolute-pixel :class:`Instance`.

    A box without ``conf`` is manual annotation: total certainty (1.0).
    """
    return Instance(
        class_id=box.class_id,
        class_name=class_name_for(box.class_id, class_names),
        conf=box.conf if box.conf is not None else 1.0,
        x1=(box.cx - box.w / 2) * img_w,
        y1=(box.cy - box.h / 2) * img_h,
        x2=(box.cx + box.w / 2) * img_w,
        y2=(box.cy + box.h / 2) * img_h,
    )


# ── Files at explicit paths (low level) ───────────────────────────────────


def read_boxes(txt_path: str | Path) -> tuple[list[BBox], int]:
    """Read a label file as box lines. Returns ``(boxes, malformed_count)``.

    The count covers every line that is not a readable box: garbage, short
    lines, non-numeric fields, and YOLO-seg polygon lines (valid YOLO, but
    not boxes — counted and skipped rather than mangled into a wrong box
    from their first four numbers).
    """
    boxes: list[BBox] = []
    malformed = 0
    try:
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                # A polygon has an even number >= 6 of coordinates: valid
                # YOLO-seg, but this currency is boxes.
                if len(parts) - 1 >= 6 and (len(parts) - 1) % 2 == 0:
                    malformed += 1
                    continue
                box = parse_box_line(stripped)
                if box is None:
                    malformed += 1
                else:
                    boxes.append(box)
    except Exception as exc:  # noqa: BLE001
        logger_for("labels").warning("read_boxes failed: %s", exc)
    return boxes, malformed


def write_lines(txt_path: Path, boxes: list[BBox], include_conf: bool = False) -> Path:
    """Write boxes to an explicit path, creating parent folders.

    Writes EXACTLY what it is told: an empty list produces an empty file.
    Whether an empty file may be written at all (the census-honesty rule)
    is the caller's decision, not this function's.
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(format_line(b, include_conf) for b in boxes)
    txt_path.write_text(content + "\n" if content else "", encoding="utf-8")
    return txt_path


def format_polygon_line(
    class_id: int,
    points: Sequence[tuple[float, float]],
    img_w: int,
    img_h: int,
) -> str:
    """Serialize one YOLO-seg polygon line (``class x1 y1 x2 y2 ...``).

    Coordinates arrive in absolute pixels, the currency of a COCO polygon,
    and are normalized against the image size the file belongs to. Values are
    clamped to ``[0, 1]``: the segmentation dataset audit already rejects
    out-of-bounds polygons, so clamping only absorbs raster rounding.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"polygon needs a positive image size, got {img_w}x{img_h}")
    if len(points) < 3:
        raise ValueError("a YOLO-seg polygon needs at least three points")
    coordinates: list[str] = []
    for x, y in points:
        coordinates.append(f"{min(1.0, max(0.0, float(x) / img_w)):.6f}")
        coordinates.append(f"{min(1.0, max(0.0, float(y) / img_h)):.6f}")
    return " ".join([str(int(class_id)), *coordinates])


def write_polygon_lines(
    txt_path: Path,
    polygons: Iterable[tuple[int, Sequence[tuple[float, float]]]],
    img_w: int,
    img_h: int,
) -> Path:
    """Write YOLO-seg polygon lines for one image, one line per instance.

    The segmentation adapter calls this when it converts the canonical COCO
    dataset into the family's expected label files at job time. An empty
    iterable writes an empty file, exactly like :func:`write_lines`.
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        format_polygon_line(class_id, points, img_w, img_h) for class_id, points in polygons
    )
    txt_path.write_text(content + "\n" if content else "", encoding="utf-8")
    return txt_path


# ── The folder contract, both directions ──────────────────────────────────


def _nearest_named(start: Path, name: str) -> Path | None:
    """The nearest ancestor of ``start`` (inclusive) named ``name``."""
    for candidate in (start, *start.parents):
        if candidate.name.lower() == name:
            return candidate
    return None


@dataclass(frozen=True)
class _LabelLayout:
    """The folder convention resolved for one image path.

    ``images_root`` is the nearest ``images`` ancestor and ``labels_root`` its
    sibling ``labels/``; ``rel_dir`` is the subpath below ``images_root`` that
    the label mirrors. A plain folder has neither root and an empty
    ``rel_dir``.
    """

    image_path: Path
    images_root: Path | None
    labels_root: Path | None
    rel_dir: Path

    @property
    def has_split(self) -> bool:
        return self.images_root is not None and self.labels_root is not None

    @property
    def conventional(self) -> Path | None:
        """The conventional label path, or ``None`` for a plain folder."""
        if not self.has_split:
            return None
        assert self.labels_root is not None
        return self.labels_root / self.rel_dir / f"{self.image_path.stem}.txt"

    @property
    def same_dir(self) -> Path:
        return self.image_path.parent / f"{self.image_path.stem}.txt"

    @property
    def nested(self) -> Path:
        return self.image_path.parent / "labels" / f"{self.image_path.stem}.txt"


def _layout_for(image_path: Path) -> _LabelLayout:
    """Resolve the split/plain convention for ``image_path``."""
    images_root = _nearest_named(image_path.parent, "images")
    if images_root is None:
        return _LabelLayout(image_path, None, None, Path())
    return _LabelLayout(
        image_path,
        images_root,
        images_root.parent / "labels",
        image_path.parent.relative_to(images_root),
    )


def _warn_legacy(path: Path, kind: Literal["same-directory", "nested"]) -> None:
    logger_for("labels").warning(
        "legacy %s label used: %s (conventional layout mirrors images/ to labels/)",
        kind,
        path,
    )


def find_label_file(image_path: Path, gt_folder: Path | None = None) -> Path | None:
    """Find the label file that belongs to an image, or ``None``.

    Precedence: the explicit GT folder (flat read first, then the mirrored
    ``<gt>/<relpath>/<stem>.txt``), the conventional sibling ``labels/`` of
    the nearest ``images`` ancestor, a legacy same-directory file, and — for
    a plain folder — a nested ``labels/`` file. Legacy hits are read but
    logged as warnings; nothing is moved or deleted.
    """
    layout = _layout_for(image_path)
    stem = image_path.stem
    if gt_folder is not None:
        flat = gt_folder / f"{stem}.txt"
        if flat.exists():
            return flat
        if layout.has_split:
            mirrored = gt_folder / layout.rel_dir / f"{stem}.txt"
            if mirrored.exists():
                return mirrored

    conventional = layout.conventional
    if conventional is not None and conventional.exists():
        return conventional

    if layout.same_dir.exists():
        if layout.has_split:
            _warn_legacy(layout.same_dir, "same-directory")
        return layout.same_dir

    if not layout.has_split and layout.nested.exists():
        _warn_legacy(layout.nested, "nested")
        return layout.nested
    return None


def label_dir_for(image_path: Path, gt_folder: Path | None = None) -> Path:
    """The folder :func:`label_path_for` writes to, without creating it.

    An explicit GT folder wins (flat write), then the conventional
    ``labels/<relpath>/`` of an ``images`` ancestor, then the image's own
    folder for a plain one. Pure: it resolves the destination only, so a
    caller can read the ``classes.txt`` that governs it before writing.
    """
    layout = _layout_for(image_path)
    if gt_folder is not None:
        return gt_folder
    if layout.has_split:
        assert layout.labels_root is not None
        return layout.labels_root / layout.rel_dir
    return image_path.parent


def label_path_for(image_path: Path, gt_folder: Path | None = None) -> Path:
    """The write-side mirror of :func:`find_label_file` (folders created).

    An explicit GT folder wins (flat write), then the conventional
    ``labels/<relpath>/`` of an ``images`` ancestor, then the image's own
    folder for a plain one. A split layout never writes next to the image.
    """
    out_dir = label_dir_for(image_path, gt_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{image_path.stem}.txt"


def classes_dir_for(labels_dir: Path) -> Path:
    """Where the ``classes.txt`` that governs ``labels_dir`` lives.

    Any path under a ``labels/`` ancestor resolves to the dataset root (the
    parent of ``labels/``); everything else keeps its own folder.
    """
    labels_root = _nearest_named(labels_dir, "labels")
    if labels_root is not None:
        return labels_root.parent
    return labels_dir


def image_path_for(label_path: Path, extensions: tuple[str, ...] = IMAGE_EXTS) -> Path | None:
    """The image that belongs to a label file, or ``None``.

    Mirrors the label folder back to ``images/`` and returns the first
    existing image with the same stem and a known extension. A nested legacy
    label (``sueltas/labels/plate.txt``) is read by
    :func:`find_label_file` for the plain folder's image, so the mirrored
    search falls back to the label's own folder and to that plain folder.
    """
    folders: list[Path] = []
    labels_root = _nearest_named(label_path.parent, "labels")
    if labels_root is not None:
        rel_dir = label_path.parent.relative_to(labels_root)
        folders.append(labels_root.parent / "images" / rel_dir)
        folders.append(label_path.parent)
        folders.append(labels_root.parent / rel_dir)
    else:
        folders.append(label_path.parent)
    for folder in folders:
        for extension in extensions:
            candidate = folder / f"{label_path.stem}{extension}"
            if candidate.exists():
                return candidate
    return None


# ── Full reader (polygon-aware, Instance currency) ────────────────────────


def read_label_file(
    txt_path: str | Path,
    img_w: int,
    img_h: int,
    class_names: dict[int, str],
) -> tuple[list[Instance], int]:
    """Read a label file into absolute-pixel :class:`Instance` objects.

    Handles both dialects of the format: detection lines (``class cx cy w
    h``, optional ``conf`` — extra columns ignored) and YOLO-seg polygon
    lines, which become a filled boolean mask on the instance. Returns
    ``(instances, malformed_count)``; a malformed line is skipped, never
    guessed into existence.
    """
    instances: list[Instance] = []
    malformed = 0
    try:
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    if parts:
                        malformed += 1
                    continue
                try:
                    class_id = int(float(parts[0]))
                    nums = [float(x) for x in parts[1:]]
                except ValueError:
                    malformed += 1
                    continue
                if len(nums) >= 6 and len(nums) % 2 == 0:
                    # YOLO-seg polygon: class x1 y1 x2 y2 ... (normalized).
                    pts = [(nums[k] * img_w, nums[k + 1] * img_h) for k in range(0, len(nums), 2)]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    # fillPoly mutates in place: fill a named uint8 array and
                    # only then cast to bool. Passing mask.astype(uint8) as
                    # a throwaway expression silently filled a copy.
                    mask_u8 = np.zeros((img_h, img_w), dtype=np.uint8)
                    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(mask_u8, [arr], (1,))
                    instances.append(
                        Instance(
                            class_id=class_id,
                            class_name=class_names.get(class_id, str(class_id)),
                            conf=1.0,
                            x1=min(xs),
                            y1=min(ys),
                            x2=max(xs),
                            y2=max(ys),
                            mask=mask_u8.astype(bool),
                            mask_source="gt-polygon",
                        )
                    )
                else:
                    # YOLO detection: class cx cy w h (normalized).
                    cx_n, cy_n, w_n, h_n = nums[:4]
                    cx = cx_n * img_w
                    cy = cy_n * img_h
                    w = w_n * img_w
                    h = h_n * img_h
                    instances.append(
                        Instance(
                            class_id=class_id,
                            class_name=class_names.get(class_id, str(class_id)),
                            conf=1.0,
                            x1=cx - w / 2,
                            y1=cy - h / 2,
                            x2=cx + w / 2,
                            y2=cy + h / 2,
                        )
                    )
    except Exception as exc:  # noqa: BLE001
        logger_for("labels").warning("read_label_file failed: %s", exc)
    return instances, malformed


# ── High level: locate + EXIF-safe size in one call ───────────────────────


def read_labels(
    image_path: Path,
    gt_folder: Path | None = None,
    class_names: dict[int, str] | None = None,
) -> tuple[list[Instance], int]:
    """Find, size and read the labels that belong to an image.

    Locates the file (:func:`find_label_file`), resolves the image size the
    way the whole program agrees on (:func:`size_image` — EXIF-aware,
    header-only), and falls back to :data:`STUDY_CLASSES` when no
    ``classes.txt`` governs the file. The governing folder is
    :func:`classes_dir_for` of the label's folder, so a split layout reads
    the dataset-root ``classes.txt``. Raises ``ValueError`` when the image
    size cannot be read, because normalized coordinates without a real
    size are silently wrong — that error must be loud.
    """
    txt_path = find_label_file(image_path, gt_folder)
    if txt_path is None:
        return [], 0
    wh = size_image(image_path)
    if wh is None or wh[0] <= 0 or wh[1] <= 0:
        raise ValueError(f"could not read image size: {image_path.name}")
    if class_names is None:
        class_names = dict(enumerate(read_classes(classes_dir_for(txt_path.parent))))
    return read_label_file(txt_path, wh[0], wh[1], class_names)


def write_labels(
    image_path: Path,
    dets: list[Instance],
    gt_folder: Path | None = None,
    include_conf: bool = False,
) -> Path:
    """Write an image's annotations to its conventional label file.

    Normalizes against the EXIF-safe image size and writes through
    :func:`label_path_for`. Raises ``ValueError`` when the size cannot be
    read. Whether this image may be written at all stays with the caller.
    """
    wh = size_image(image_path)
    if wh is None or wh[0] <= 0 or wh[1] <= 0:
        raise ValueError(f"could not read image size: {image_path.name}")
    boxes = [bbox_from(d, wh[0], wh[1]) for d in dets]
    return write_lines(label_path_for(image_path, gt_folder), boxes, include_conf)


# ── classes.txt ───────────────────────────────────────────────────────────


def read_classes(folder: Path) -> list[str]:
    """Read ``classes.txt`` from a folder; fall back to the study's classes.

    A blank or missing file is treated the same way: no real information,
    so the study's classes apply.
    """
    path = folder / "classes.txt"
    if path.exists():
        names = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if names:
            return names
    return list(STUDY_CLASSES)


def write_classes(folder: Path, names: list[str] | None = None) -> Path:
    """Write ``classes.txt`` INTO ``folder`` (order as given).

    Where the folder should be is the caller's business — usually
    :func:`classes_dir_for` of the labels folder, and for the Labeller
    every subfolder that holds labels.
    """
    if names is None:
        names = list(STUDY_CLASSES)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return folder / "classes.txt"


__all__ = [
    "STUDY_CLASSES",
    "BBox",
    "bbox_from",
    "class_name_for",
    "classes_dir_for",
    "find_label_file",
    "format_line",
    "image_path_for",
    "label_dir_for",
    "label_path_for",
    "parse_box_line",
    "read_boxes",
    "read_classes",
    "read_label_file",
    "read_labels",
    "to_instance",
    "write_classes",
    "write_labels",
    "write_lines",
]
