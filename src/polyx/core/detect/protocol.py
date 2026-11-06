"""Detection protocol — shared types for all detection backends.

This module defines the public interface that all detection backends must implement.
It contains no implementation, only the contracts (Protocol classes and data types).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    # Legacy names, resolved lazily below (see ``__getattr__``). Declared
    # here (redundant-alias form = re-export) so type checkers see them as
    # module attributes.
    from polyx.core.labels import find_gt_for_image as find_gt_for_image
    from polyx.core.labels import read_yolo_txt as read_yolo_txt


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
    # How the size fields were obtained: "mask" (measured over the real
    # particle) or "box" (the declared fallback). None = never sized.
    size_source: Literal["mask", "box"] | None = None
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

    Implementations must provide ``predict_auto`` —the single entry point the
    run engine calls— and a ``predict`` method that takes an image path and
    returns a list of Instance objects. Backends that do not tile delegate
    ``predict_auto`` to ``predict``.
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

    def predict_auto(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1280,
        device: str = "0",
        tiling: str = "auto",
        threshold_px: int = 2000,
        tile: int = 0,
        overlap: float = 0.25,
        record: dict | None = None,
        **kwargs,
    ) -> list[Instance]:
        """Run inference, tiling the image when the policy decides to.

        This is the single entry point the run engine calls. Backends that
        cannot tile delegate to their existing single-pass ``predict``.
        ``record``, when given, is filled with the tiling decision taken.
        """
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


def __getattr__(name: str):
    """Legacy re-exports from the YOLO-label codec.

    ``read_yolo_txt`` and ``find_gt_for_image`` moved to
    :mod:`polyx.core.labels` — this module is contracts, not implementation.
    Resolved lazily so the import stays one-directional (labels imports
    ``Instance`` from here).
    """
    if name == "read_yolo_txt":
        from polyx.core.labels import read_yolo_txt

        return read_yolo_txt
    if name == "find_gt_for_image":
        from polyx.core.labels import find_gt_for_image

        return find_gt_for_image
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def equal_area_diameter(area: float) -> float:
    """The diameter of the circle with the given area — THE size convention.

    One definition everywhere: the box fallback (``compute_box_size_um``),
    the mask-measured size (``measure.shape``) and the Viewer's box
    descriptor all take their diameters from this one function, so a
    number written anywhere speaks the same language.
    """
    return 2.0 * (area / np.pi) ** 0.5


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
    det.diam_um = equal_area_diameter(area_um2)
    det.area_um2 = area_um2
    det.size_source = "box"


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
