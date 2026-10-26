"""Detection backends and shared protocol."""

from .protocol import (Instance, DetectionSource, SegmentationSource,
                       SegmentationPrompts, Detections, size_image, policy_tiling)
from .yolo import YoloModel, YoloSegModel, Detection
from .native import NativeDetector
from .sam import Sam2Segmenter
from .factory import make_detector, list_detectors, register_detector
from .two_stage import TwoStageDetector, build_two_stage

__all__ = [
    "Instance",
    "Detection",
    "DetectionSource",
    "SegmentationSource",
    "SegmentationPrompts",
    "Detections",
    "size_image",
    "policy_tiling",
    "YoloModel",
    "YoloSegModel",
    "NativeDetector",
    "Sam2Segmenter",
    "TwoStageDetector",
    "build_two_stage",
    "make_detector",
    "list_detectors",
    "register_detector",
]
