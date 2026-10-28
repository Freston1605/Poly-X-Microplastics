"""Detection protocol — shared types for all detection backends.

This module defines the public interface that all detection backends must implement.
It contains no implementation, only the contracts (Protocol classes and data types).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass
class Instance:
    """A single detected instance with its geometry and classification.

    This is the canonical representation used throughout the pipeline.
    """

    class_id: int
    class_name: str
    conf: float
    # Box in absolute pixel coordinates (x1, y1, x2, y2)
    x1: float
    y1: float
    x2: float
    y2: float
    # Optional calibrated measurements (filled by measurement layer)
    diam_um: float | None = None
    area_um2: float | None = None
    # True shape measurements (from mask-based measurement)
    length_um: float | None = None
    width_um: float | None = None
    aspect: float | None = None
    curvature: float | None = None
    morphotype: str | None = None  # "fibre" | "fragment"
    feret_um: float | None = None
    geodesic_um: float | None = None
    equivalent_rect_length_um: float | None = None
    method_length: str | None = None
    review: bool = False
    warning_shape: str | None = None
    number: int | None = None
    assignable: bool = True
    # ── Segmentation mask (optional) ──────────────────────────────────────
    # Full-image boolean mask when a segmentation backend produced one.
    # Stored as a cropped array + offset when memory matters; None means
    # "no mask available, fall back to box-derived segmentation".
    mask: np.ndarray | None = None
    mask_rle: str | None = None  # COCO RLE for persistence
    mask_source: str | None = None  # "yolo-seg"|"sam2"|"otsu-box"|...
    mask_offset: tuple[int, int] | None = None  # (x0, y0) of cropped mask

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)


class DetectionSource(Protocol):
    """Protocol for any detection backend.

    Implementations must provide a `predict` method that takes an image path
    and returns a list of Instance objects.
    """

    def load(self) -> DetectionSource:
        """Load the model weights. Called lazily before first prediction."""
        ...

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "0",
        **kwargs,
    ) -> list[Instance]:
        """Run inference on a single image."""
        ...

    def predict_sliced(
        self,
        image_path: str | Path,
        conf: float = 0.10,
        iou: float = 0.45,
        tile: int = 1024,
        overlap: float = 0.25,
        device: str = "0",
        **kwargs,
    ) -> list[Instance]:
        """Run sliced inference on a large image."""
        ...

    @property
    def names(self) -> dict[int, str]:
        """Class id → name mapping."""
        ...

    @property
    def last_imgsz(self) -> int:
        """Image size actually used in the last inference."""
        ...


# Convenience type alias
Detections = list[Instance]


# Utility functions (kept here for now; could move to a utils module)
def size_image(image_path: str | Path) -> tuple[int, int] | None:
    """(width, height) reading only the header, without decoding the whole image.

    It is queried before every inference, so decoding a 3260 px photo just
    to measure it would cost more than the decision itself.

    Returns the size ALREADY ROTATED per EXIF, because that is the frame the
    rest of the program works in: ``cv2.imdecode`` applies the orientation
    and Pillow does not. Without the fix, a photo tagged orientation 6
    measured 4096x3072 but decoded 3072x4096, so boxes normalized against
    one size were denormalized against the other and ended up displaced.
    """
    try:
        from PIL import Image

        with Image.open(str(image_path)) as im:
            width, height = int(im.size[0]), int(im.size[1])
            # 5, 6, 7, 8 are the orientations that swap the axes.
            if im.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
            return width, height
    except Exception:  # noqa: BLE001
        # Paths with accents or formats Pillow cannot open: it falls back to the cv2 path,
        # which is already used throughout the rest of the project for that same reason.
        try:
            from polyx.core.image_io import read_image

            img = read_image(image_path)
            if img is None:
                return None
            return int(img.shape[1]), int(img.shape[0])
        except Exception:  # noqa: BLE001
            return None


def policy_tiling(
    width: int,
    height: int,
    imgsz: int,
    threshold_px: int = 2000,
    tile: int = 0,
    overlap: float = 0.25,
) -> dict | None:
    """Decides whether the photo gets tiled, and with which geometry, from its
    size alone.

        ``None`` means a single pass over the full image.

        The threshold is not compared against the area but against the **longer
        side**, because what makes a particle disappear is the rescale to
        ``imgsz``: a 3260 px photo entering at 2080 shrinks every particle to
        0.64x, and below the network's stride it stops existing. Tiling avoids
        that rescale.

        With ``tile=0`` the tile side comes from ``min(umbral_px, imgsz)``: this
        way the crop enters the network without being shrunk, which is exactly
        the goal.

        WHETHER to tile and AT WHICH SIZE are two different questions, and
        mixing them up caused a failure: to force tiling, ``predict_auto``
        passes ``umbral_px=0`` -- its way of saying "do not decide, tile" --,
        and that yielded ``min(0, imgsz) = 0``, which the clamp raised to 256.
        On a 4096x3072 photo that is 336 tiles of 256 px: extremely slow and
        with ZERO detections, because each tile gets rescaled 8x up to imgsz and
        the model sees nothing resembling what it was trained on. With
        umbral_px at 0 the size is governed by imgsz, the only thing that truly
        determines it.
    """
    side = max(int(width), int(height))
    if side <= int(threshold_px):
        return None
    if int(tile) > 0:
        t = int(tile)
    else:
        candidates_2 = [v for v in (int(threshold_px), int(imgsz)) if v > 0]
        t = min(candidates_2) if candidates_2 else 1280
    t = max(256, min(t, side))
    overlap = min(max(float(overlap), 0.0), 0.9)
    step = max(1, int(round(t * (1.0 - overlap))))  # noqa: RUF046
    n = len(_tile_starts(int(width), t, step)) * len(_tile_starts(int(height), t, step))
    return {
        "tile": t,
        "overlap": overlap,
        "n_tiles": n,
        "lado": side,
        "umbral_px": int(threshold_px),
    }


def read_yolo_txt(
    txt_path: str | Path,
    img_w: int,
    img_h: int,
    class_names: dict[int, str],
) -> list[Instance]:
    """Read YOLO-format label file and return Instance list."""
    instances = []
    try:
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                nums = [float(x) for x in parts[1:]]
                # YOLO-seg polygon: class x1 y1 x2 y2 ... (normalized)
                if len(nums) >= 6 and len(nums) % 2 == 0:
                    pts = [(nums[k] * img_w, nums[k + 1] * img_h) for k in range(0, len(nums), 2)]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    # fillPoly mutates in place: fill a named uint8 array and
                    # only then cast to bool. Passing mask.astype(uint8) as
                    # a throwaway expression silently filled a copy.
                    mask_u8 = np.zeros((img_h, img_w), dtype=np.uint8)
                    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(mask_u8, [arr], (1,))
                    mask = mask_u8.astype(bool)
                    instances.append(
                        Instance(
                            class_id=cid,
                            class_name=class_names.get(cid, str(cid)),
                            conf=1.0,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            mask=mask,
                            mask_source="gt-polygon",
                        )
                    )
                else:
                    # YOLO detection: class cx cy w h (normalized)
                    if len(nums) < 4:
                        continue
                    cx_n, cy_n, w_n, h_n = nums[:4]
                    cx = cx_n * img_w
                    cy = cy_n * img_h
                    w = w_n * img_w
                    h = h_n * img_h
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    instances.append(
                        Instance(
                            class_id=cid,
                            class_name=class_names.get(cid, str(cid)),
                            conf=1.0,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    )
    except Exception as exc:  # noqa: BLE001
        try:
            from polyx.core.provenance.observability import logger_for

            logger_for("detector").warning("read_yolo_txt failed: %s", exc)
        except Exception:  # noqa: BLE001, S110
            pass
    return instances


def find_gt_for_image(image_path: Path, gt_folder: Path | None) -> Path | None:
    """Finds the GT .txt next to the image, in the sibling /labels, or in gt_folder."""
    stem = image_path.stem
    candidates = []
    if gt_folder is not None:
        candidates.append(gt_folder / f"{stem}.txt")
    # same directory
    candidates.append(image_path.parent / f"{stem}.txt")
    # sibling labels/
    if image_path.parent.name.lower() == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{stem}.txt")
    else:
        candidates.append(image_path.parent / "labels" / f"{stem}.txt")
    for c in candidates:
        if c.exists():
            return c
    return None


def compute_box_size_um(det: Instance, um_per_px: float | None) -> None:
    """Approximate size from the BOX, not the particle.

    It is a fallback: it overestimates the area 1.87x at the median of this
    material because the box of an elongated particle is nearly empty, and
    it depends on orientation. Used only when segmentation could not isolate
    the particle. The good version is ``morfologia.medir_deteccion``.
    """
    if um_per_px is None or um_per_px <= 0:
        return
    area_px2 = det.w * det.h
    area_um2 = area_px2 * (um_per_px**2)
    # equivalent diameter (circle with the same area)
    diam_um = 2.0 * (area_um2 / np.pi) ** 0.5
    det.diam_um = diam_um
    det.area_um2 = area_um2


@dataclass
class SegmentationPrompts:
    """Prompts fed to a segmentation backend."""

    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    points: list[tuple[float, float]] | None = None
    point_labels: list[int] | None = None


class SegmentationSource(Protocol):
    """Protocol for a mask-generating backend (SAM, U-Net, ...).

    Unlike DetectionSource, this does not detect; it segments given prompts
    (typically boxes from a detector) and returns Instance objects carrying masks.
    """

    def load(self) -> SegmentationSource: ...

    def segment(
        self,
        image_path: str | Path,
        prompts: SegmentationPrompts,
        **kwargs,
    ) -> list[Instance]: ...

    def segment_from_boxes(
        self,
        image_path: str | Path,
        boxes: list[tuple[float, float, float, float]],
        **kwargs,
    ) -> list[Instance]: ...

    @property
    def names(self) -> dict[int, str]: ...

    @property
    def last_imgsz(self) -> int: ...


def _tile_starts(total: int, tile: int, step: int) -> list[int]:
    """Start positions of the tiles along one axis (the last one flush with the edge)."""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, step))
    if not starts or starts[-1] != total - tile:
        starts.append(total - tile)
    return starts
