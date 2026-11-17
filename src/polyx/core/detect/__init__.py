"""Detection backends and shared protocol."""

from .factory import list_detectors, make_detector, register_detector
from .native import NativeDetector
from .protocol import (
    Detections,
    DetectionSource,
    Instance,
    SegmentationPrompts,
    SegmentationSource,
    size_image,
)
from .sam import Sam2Segmenter
from .tiling import (
    TilingDecision,
    TilingMode,
    TilingPlan,
    decide_tiling,
    describe_tiling,
    policy_tiling,
)
from .two_stage import TwoStageDetector, build_two_stage
from .yolo import Detection, YoloModel, YoloSegModel

__all__ = [
    "Detection",
    "DetectionSource",
    "Detections",
    "Instance",
    "NativeDetector",
    "Sam2Segmenter",
    "SegmentationPrompts",
    "SegmentationSource",
    "TilingDecision",
    "TilingMode",
    "TilingPlan",
    "TwoStageDetector",
    "YoloModel",
    "YoloSegModel",
    "build_two_stage",
    "decide_tiling",
    "describe_tiling",
    "list_detectors",
    "make_detector",
    "policy_tiling",
    "register_detector",
    "size_image",
]
