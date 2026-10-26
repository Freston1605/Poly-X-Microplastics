"""Two-stage detection: a detector proposes boxes, a segmenter refines them into masks.

The composer keeps the detector's classification/confidence and attaches the
segmenter's mask to each Instance, falling back to the detector's box-only
instance when segmentation fails for a given particle.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Any

from .protocol import (
    DetectionSource,
    SegmentationSource,
    Instance,
    SegmentationPrompts,
)


class TwoStageDetector(DetectionSource):
    """Compose a detector (stage 1) with a segmenter (stage 2)."""

    def __init__(
        self,
        detector: DetectionSource,
        segmenter: SegmentationSource,
        segmenter_conf: float = 0.0,
        fallback_on_failure: bool = True,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.segmenter_conf = segmenter_conf
        self.fallback_on_failure = fallback_on_failure
        self._names: dict = {}
        self._last_imgsz: int = 0

    def load(self) -> "TwoStageDetector":
        self.detector.load()
        self.segmenter.load()
        try:
            self._names = dict(self.detector.names)
        except Exception:
            self._names = {}
        return self

    def predict(self, image_path, conf=0.25, iou=0.45, imgsz=640,
                device="0", **kwargs) -> List[Instance]:
        boxes_instances = self.detector.predict(
            image_path, conf=conf, iou=iou, imgsz=imgsz, device=device, **kwargs)
        if not boxes_instances:
            return boxes_instances

        boxes = [(d.x1, d.y1, d.x2, d.y2) for d in boxes_instances]
        seg_instances: List[Instance] = []
        try:
            if hasattr(self.segmenter, "segment_from_boxes"):
                seg_instances = self.segmenter.segment_from_boxes(
                    image_path, boxes, conf=self.segmenter_conf or conf)
            else:
                prompts = SegmentationPrompts(boxes=boxes)
                seg_instances = self.segmenter.segment(image_path, prompts)
        except Exception:
            seg_instances = []

        # Merge: keep detector class/conf, attach segmenter mask.
        out: List[Instance] = []
        for i, det in enumerate(boxes_instances):
            seg = seg_instances[i] if i < len(seg_instances) else None
            if seg is not None and getattr(seg, "mask", None) is not None:
                det.mask = seg.mask
                det.mask_rle = getattr(seg, "mask_rle", None)
                det.mask_source = getattr(seg, "mask_source", None) or "segmenter"
                det.mask_offset = getattr(seg, "mask_offset", None)
                # refine box from mask if segmenter provided one
                if seg.x2 > seg.x1 and seg.y2 > seg.y1:
                    det.x1, det.y1, det.x2, det.y2 = seg.x1, seg.y1, seg.x2, seg.y2
                out.append(det)
            else:
                if self.fallback_on_failure:
                    det.mask_source = det.mask_source or "detector-box"
                    out.append(det)
        self._last_imgsz = getattr(self.detector, "last_imgsz", 0)
        return out

    def predict_sliced(self, image_path, conf=0.10, iou=0.45, tile=1024,
                       overlap=0.25, device="0", **kwargs):
        # Fall back to detector-only sliced prediction; segmentation on tiles
        # is left to callers that need it.
        return self.detector.predict_sliced(
            image_path, conf=conf, iou=iou, tile=tile, overlap=overlap,
            device=device, **kwargs)

    @property
    def names(self):
        return self._names or getattr(self.detector, "names", {})

    @property
    def last_imgsz(self):
        return self._last_imgsz


def build_two_stage(detector: DetectionSource,
                    segmenter: SegmentationSource,
                    **kwargs) -> TwoStageDetector:
    return TwoStageDetector(detector, segmenter, **kwargs)
