"""Detection backends and shared protocol."""

from .protocol import Instance, DetectionSource, Detections, size_image, policy_tiling
from .yolo import YoloModel
from .native import NativeDetector
from .sam import Sam2Segmenter
from .factory import make_detector, list_detectors, register_detector

__all__ = [
    "Instance",
    "DetectionSource",
    "Detections", 
    "size_image",
    "policy_tiling",
    "YoloModel",
    "NativeDetector",
    "Sam2Segmenter",
    "make_detector",
    "list_detectors",
    "register_detector",
]
