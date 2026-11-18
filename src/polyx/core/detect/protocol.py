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
    from polyx.core.measure.warnings import Warning

# ``policy_tiling`` lives in ``tiling`` with the decision and the sentence;
# it is re-exported here because this module was its historical home and other
# modules import it from the protocol.
from .tiling import TilingMode as TilingMode
from .tiling import policy_tiling as policy_tiling


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
    # Stable machine code plus structured parameters; prose is chosen at display time.
    warning_shape: Warning | None = None
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
        tiling: TilingMode | str = TilingMode.AUTO,
        threshold_px: int = 2000,
        tile: int = 0,
        overlap: float = 0.25,
        record: dict | None = None,
        **kwargs,
    ) -> list[Instance]:
        """Run inference, tiling the image when the policy decides to.

        This is the single entry point the run engine calls. Backends that
        cannot tile delegate to their existing single-pass ``predict``.

        ``tiling`` accepts a :class:`TilingMode` or its English string; the
        deprecated Spanish values still arrive from old callers and are
        normalised once. ``record``, when given, is filled with the decision
        under English keys: ``width``, ``height``, ``tiled`` and ``plan`` (a
        :class:`~polyx.core.detect.tiling.TilingPlan` or ``None``).
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


class SinglePassDetectionSource:
    """Mixin for backends that only run a single pass over the full image.

    The engine always calls ``predict_auto``; a backend without a tiling
    policy must not copy the same delegation three times. The shared
    implementation drops the tiling arguments and forwards the inference
    parameters to ``predict``.
    """

    def predict_auto(
        self: DetectionSource,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1280,
        device: str = "0",
        tiling: TilingMode | str = TilingMode.AUTO,
        threshold_px: int = 2000,
        tile: int = 0,
        overlap: float = 0.25,
        record: dict | None = None,
        **kwargs,
    ) -> list[Instance]:
        return self.predict(image_path, conf=conf, iou=iou, imgsz=imgsz, device=device)


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
