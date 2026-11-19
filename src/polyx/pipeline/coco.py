"""Per-adapter conversion of framework results to the shared COCO currency.

Only conversion happens here. Intermediate prediction files are never written
to the canonical artifact layout: they stay in memory for the evaluator.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def coco_predictions_from_results(results: Any) -> list[dict]:
    """Turn Ultralytics detection results into COCO instance predictions.

    Result order matches the sorted validation image order used by the dataset
    reader, so the zero-based position is the COCO ``image_id``.
    """
    predictions: list[dict] = []
    for image_id, result in enumerate(results or []):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = np.asarray(getattr(boxes, "xyxy", []), dtype=float).reshape(-1, 4)
        conf = np.asarray(getattr(boxes, "conf", []), dtype=float).reshape(-1)
        classes = np.asarray(getattr(boxes, "cls", []), dtype=int).reshape(-1)
        for index, (x1, y1, x2, y2) in enumerate(xyxy):
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(classes[index]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(conf[index]),
                }
            )
    return predictions


def coco_segmentation_predictions_from_results(results: Any) -> list[dict]:
    """Turn Ultralytics segmentation results into COCO instance predictions.

    Ultralytics reports masks as pixel-coordinate polygons in ``masks.xy``,
    aligned one-to-one with ``boxes``; the polygon is emitted as a COCO
    ``segmentation`` so the shared evaluator measures mask IoU. A result
    without masks stays out: a mask prediction is never scored as a box.
    """
    predictions: list[dict] = []
    for image_id, result in enumerate(results or []):
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None:
            continue
        xyxy = np.asarray(getattr(boxes, "xyxy", []), dtype=float).reshape(-1, 4)
        conf = np.asarray(getattr(boxes, "conf", []), dtype=float).reshape(-1)
        classes = np.asarray(getattr(boxes, "cls", []), dtype=int).reshape(-1)
        polygons = list(getattr(masks, "xy", []) or [])
        for index, polygon in enumerate(polygons):
            points = np.asarray(polygon, dtype=float).reshape(-1, 2)
            if points.shape[0] < 3 or index >= len(xyxy):
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy[index])
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(classes[index]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "segmentation": [points.reshape(-1).tolist()],
                    "score": float(conf[index]),
                }
            )
    return predictions
