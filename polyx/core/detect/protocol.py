"""Detection protocol — shared types for all detection backends.

This module defines the public interface that all detection backends must implement.
It contains no implementation, only the contracts (Protocol classes and data types).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Protocol
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
    diam_um: Optional[float] = None
    area_um2: Optional[float] = None
    # True shape measurements (from mask-based measurement)
    length_um: Optional[float] = None
    width_um: Optional[float] = None
    aspect: Optional[float] = None
    curvature: Optional[float] = None
    morphotype: Optional[str] = None  # "fibre" | "fragment"
    feret_um: Optional[float] = None
    geodesic_um: Optional[float] = None
    equivalent_rect_length_um: Optional[float] = None
    method_length: Optional[str] = None
    review: bool = False
    warning_shape: Optional[str] = None
    number: Optional[int] = None
    assignable: bool = True

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
    ) -> List[Instance]:
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
    ) -> List[Instance]:
        """Run sliced inference on a large image."""
        ...

    @property
    def names(self) -> Dict[int, str]:
        """Class id → name mapping."""
        ...

    @property
    def last_imgsz(self) -> int:
        """Image size actually used in the last inference."""
        ...


# Convenience type alias
Detections = List[Instance]


# Utility functions (kept here for now; could move to a utils module)
def size_image(image_path: str | Path) -> Optional[Tuple[int, int]]:
    """Read image dimensions without loading full image."""
    try:
        import cv2
        h, w = cv2.imread(str(image_path)).shape[:2]
        return w, h
    except Exception:
        return None


def policy_tiling(
    width: int,
    height: int,
    imgsz: int,
    threshold_px: int = 2000,
    tile: int = 0,
    overlap: float = 0.25,
) -> Tuple[int, int, int, float]:
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
    class_names: Dict[int, str],
) -> List[Instance]:
    """Read YOLO-format label file and return Instance list."""
    instances = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid, cx_n, cy_n, w_n, h_n = map(float, parts)
                cid = int(cid)
                cx = cx_n * img_w
                cy = cy_n * img_h
                w = w_n * img_w
                h = h_n * img_h
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                instances.append(Instance(
                    class_id=cid,
                    class_name=class_names.get(cid, str(cid)),
                    conf=1.0,  # GT has no confidence
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
    except Exception:
        pass
    return instances


def find_gt_for_image(image_path: str | Path, gt_folder: str | Path) -> Optional[Path]:
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
    det.area_um2 = (w * h) * (um_per_px ** 2)
