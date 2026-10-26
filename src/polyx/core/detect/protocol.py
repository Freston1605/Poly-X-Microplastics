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
    """Read image dimensions without loading full image."""
    try:
        import cv2

        h, w = cv2.imread(str(image_path)).shape[:2]
        return w, h
    except Exception:  # noqa: BLE001
        return None


def policy_tiling(
    width: int,
    height: int,
    imgsz: int,
    threshold_px: int = 2000,
    tile: int = 0,
    overlap: float = 0.25,
) -> tuple[int, int, int, float]:
    """Compute tiling parameters.

    Returns (tile_size, overlap_px, n_tiles_x, n_tiles_y)
    """
    long_side = max(width, height)
    if long_side <= threshold_px:
        return 0, 0, 1, 1
    if tile == 0:
        tile = min(threshold_px, imgsz)
    tile = max(256, tile)
    step = int(tile * (1 - overlap))
    n_x = max(1, (width - tile) // step + 1)
    n_y = max(1, (height - tile) // step + 1)
    return tile, int(tile * overlap), n_x, n_y


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
                    cv2.fillPoly(mask_u8, [arr], 1)
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
    except Exception:  # noqa: BLE001, S110
        pass
    return instances


def find_gt_for_image(image_path: str | Path, gt_folder: str | Path) -> Path | None:
    """Find the ground-truth .txt file for an image."""
    img_path = Path(image_path)
    for ext in (".txt",):
        candidate = Path(gt_folder) / img_path.with_suffix(ext).name
        if candidate.exists():
            return candidate
    return None


def compute_box_size_um(det: Instance, um_per_px: float) -> None:
    """Fill diam_um and area_um2 from box dimensions and calibration."""
    if det.diam_um is not None:
        return
    w = det.w
    h = det.h
    det.diam_um = ((w + h) / 2) * um_per_px
    det.area_um2 = (w * h) * (um_per_px**2)


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
