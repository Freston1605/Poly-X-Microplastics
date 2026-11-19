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
        # Ultralytics results may hold CUDA tensors. This helper detaches and
        # moves them to CPU before NumPy sees them, exactly as the torchvision
        # path does, so a GPU validation pass cannot raise here.
        xyxy = _numpy(getattr(boxes, "xyxy", [])).reshape(-1, 4)
        conf = _numpy(getattr(boxes, "conf", [])).reshape(-1)
        classes = _numpy(getattr(boxes, "cls", [])).astype(int).reshape(-1)
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


def coco_segmentation_predictions_from_torchvision(
    outputs: Any, image_ids: list[int]
) -> list[dict]:
    """Turn torchvision Mask R-CNN outputs into COCO instance predictions.

    torchvision returns soft masks at the input image's resolution. They are
    thresholded at 0.5 and polygonized so the shared evaluator measures mask
    IoU exactly as it does for YOLO-seg. Label 0 is torchvision's background
    and is never emitted; the remaining labels are shifted back to the dense,
    zero-based class indices the evaluator's ground truth uses.
    """
    predictions: list[dict] = []
    for output, image_id in zip(outputs or [], image_ids, strict=False):
        boxes = _numpy(output.get("boxes")).reshape(-1, 4)
        labels = _numpy(output.get("labels")).reshape(-1)
        scores = _numpy(output.get("scores")).reshape(-1)
        masks = output.get("masks")
        for index in range(boxes.shape[0]):
            label = int(labels[index]) if index < labels.shape[0] else 0
            if label <= 0:
                continue
            if masks is None or index >= len(masks):
                continue
            segmentation = _polygons_from_mask(masks[index])
            if not segmentation:
                continue
            x1, y1, x2, y2 = (float(value) for value in boxes[index])
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": label - 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "segmentation": segmentation,
                    "score": float(scores[index]) if index < scores.shape[0] else 1.0,
                }
            )
    return predictions


def _numpy(value: Any) -> np.ndarray:
    """A numpy view of a tensor or array without importing torch here."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _polygons_from_mask(mask: Any) -> list[list[float]]:
    """Exterior contours of one thresholded instance mask, as COCO polygons."""
    import cv2

    array = _numpy(mask)
    binary = (np.squeeze(array) >= 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        polygons.append(contour.reshape(-1).astype(float).tolist())
    return polygons


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
        xyxy = _numpy(getattr(boxes, "xyxy", [])).reshape(-1, 4)
        conf = _numpy(getattr(boxes, "conf", [])).reshape(-1)
        classes = _numpy(getattr(boxes, "cls", [])).astype(int).reshape(-1)
        polygons = list(getattr(masks, "xy", []) or [])
        for index, polygon in enumerate(polygons):
            points = _numpy(polygon).reshape(-1, 2)
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
