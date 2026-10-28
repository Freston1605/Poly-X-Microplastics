"""Detector factory — create detection backends by name."""

from __future__ import annotations

from pathlib import Path

from .native import NativeDetector
from .protocol import DetectionSource
from .sam import Sam2Segmenter
from .yolo import YoloModel, YoloSegModel

_DETECTOR_REGISTRY: dict[str, type] = {
    "yolo": YoloModel,
    "yolo-seg": YoloSegModel,
    "native": NativeDetector,
    "sam2": Sam2Segmenter,
}


def make_detector(
    name: str,
    weights: str | Path | None = None,
    **kwargs,
) -> DetectionSource:
    """Create a detector instance by name.

    Args:
        name: Detector name ("yolo", "yolo-seg", "native", "sam2").
        weights: Path to model weights (required for "yolo" and "sam2").
        **kwargs: Additional arguments passed to the detector constructor.

    Returns:
        A DetectionSource instance ready to use (call .load() first).

    Raises:
        ValueError: If name is unknown or required weights are missing.
    """
    if name not in _DETECTOR_REGISTRY:
        available = ", ".join(_DETECTOR_REGISTRY.keys())
        raise ValueError(f"Unknown detector: {name!r}. Available: {available}")

    cls = _DETECTOR_REGISTRY[name]

    if name in ("yolo", "yolo-seg", "sam2"):
        if weights is None:
            raise ValueError(f"Detector {name!r} requires a weights path")
        return cls(weights, **kwargs)
    else:
        return cls(**kwargs)


def register_detector(name: str, cls: type) -> None:
    """Register a custom detector class."""
    if not (
        isinstance(cls, type)
        and all(hasattr(cls, attr) for attr in ("load", "predict", "names"))
        and callable(getattr(cls, "load", None))
        and callable(getattr(cls, "predict", None))
    ):
        raise TypeError("Detector class must implement DetectionSource protocol")
    _DETECTOR_REGISTRY[name] = cls


def list_detectors() -> list[str]:
    """Return list of available detector names."""
    return list(_DETECTOR_REGISTRY.keys())
