"""Native threshold/connected-component detector (no ML dependencies)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2

from .protocol import (
    DetectionSource,
    Instance,
)


class NativeDetector(DetectionSource):
    """Threshold + connected-component detector.

    This is a fallback detector that uses classical computer vision
    (thresholding + connected components) instead of a learned model.
    It requires no GPU and no ML dependencies.
    """

    def __init__(self, threshold: int = 127, min_area: int = 10, max_area: int = 100000):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self._names: Dict[int, str] = {0: "particle"}
        self._last_imgsz: int = 0

    def load(self) -> "NativeDetector":
        return self

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "0",
        **kwargs,
    ) -> List[Instance]:
        """Run threshold + connected components on a single image."""
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(  # noqa: RUF059
            binary, connectivity=8
        )

        instances = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area or area > self.max_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]  # noqa: RUF059

            instances.append(
                Instance(
                    class_id=0,
                    class_name="particle",
                    conf=conf,
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                )
            )

        return instances

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
        # For now, just call the regular predict
        return self.predict(image_path, conf=conf, iou=iou, **kwargs)

    @property
    def names(self) -> Dict[int, str]:
        return self._names

    @property
    def last_imgsz(self) -> int:
        return self._last_imgsz
