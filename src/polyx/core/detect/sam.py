"""SAM2 segmenter — runtime backend for mask generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .protocol import DetectionSource, Instance


class Sam2Segmenter(DetectionSource):
    """SAM2-based segmenter.

    This wraps the SAM2 model for generating masks from boxes or points.
    It requires the SAM2 runtime to be available (see provision_sam2_runtime.py).
    """

    def __init__(
        self,
        checkpoint: str | Path,
        model_cfg: str = "sam2_hiera_l.yaml",
        device: str = "cuda",
    ):
        self.checkpoint = str(checkpoint)
        self.model_cfg = model_cfg
        self.device = device
        self._predictor = None
        self._names: dict[int, str] = {}
        self._last_imgsz: int = 0

    def load(self) -> Sam2Segmenter:
        if self._predictor is not None:
            return self
        try:
            from sam2.build_sam import build_sam2  # ty: ignore[unresolved-import]
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise RuntimeError(
                "SAM2 not installed. Install with: pip install git+https://github.com/facebookresearch/sam2.git"
            ) from e

        self._predictor = SAM2ImagePredictor(
            build_sam2(self.model_cfg, self.checkpoint, device=self.device)
        )
        return self

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "0",
        **kwargs,
    ) -> list[Instance]:
        """SAM2 requires prompts (boxes/points). This method is not directly usable
        without a detector to provide prompts. Use `segment_from_boxes` instead.
        """
        raise NotImplementedError(
            "SAM2 requires box/point prompts. Use `segment_from_boxes` with a detector's output."
        )

    def segment_from_boxes(
        self,
        image_path: str | Path,
        boxes: list[tuple[float, float, float, float]],
        conf: float = 0.25,
    ) -> list[Instance]:
        """Generate masks for the given boxes using SAM2."""
        if self._predictor is None:
            self.load()
        assert self._predictor is not None

        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self._predictor.set_image(img_rgb)
        instances = []

        for i, (x1, y1, x2, y2) in enumerate(boxes):  # noqa: B007
            input_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            masks, scores, _ = self._predictor.predict(
                box=input_box,
                multimask_output=False,
            )
            if len(masks) == 0:
                continue
            mask = masks[0].astype(bool)
            # Compute box from mask
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            mx1, mx2 = int(xs.min()), int(xs.max())
            my1, my2 = int(ys.min()), int(ys.max())
            instances.append(
                Instance(
                    class_id=0,
                    class_name="particle",
                    conf=float(scores[0]) if len(scores) > 0 else 1.0,
                    x1=float(mx1),
                    y1=float(my1),
                    x2=float(mx2),
                    y2=float(my2),
                    mask=mask,
                    mask_source="sam2",
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
    ) -> list[Instance]:
        raise NotImplementedError("SAM2 does not support sliced prediction directly")

    @property
    def names(self) -> dict[int, str]:
        return self._names

    @property
    def last_imgsz(self) -> int:
        return self._last_imgsz
