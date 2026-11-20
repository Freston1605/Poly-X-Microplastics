"""Per-adapter conversion of framework results to the shared COCO currency.

Only conversion happens here. Intermediate prediction files are never written
to the canonical artifact layout: they stay in memory for the evaluator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from polyx.pipeline.masks import PackedMask


def coco_predictions_from_results(results: Any, image_ids: list[int] | None = None) -> list[dict]:
    """Turn Ultralytics detection results into COCO instance predictions.

    ``image_ids`` names the split position of each result. Validation runs in
    batches, so a batch cannot infer its offset from the result order alone;
    the loader carries the id beside the path. Without ``image_ids`` the
    zero-based position is used, which is what a whole-split call means.
    """
    predictions: list[dict] = []
    for position, result in enumerate(results or []):
        image_id = image_ids[position] if image_ids is not None else position
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

    torchvision returns soft masks at the input image's resolution; they are
    thresholded and carried as a packed mask, the exact instance geometry at
    one bit per pixel. The shared evaluator unpacks it, so a perfect prediction
    scores IoU 1.0 instead of losing half a pixel per edge to a polygon
    round-trip, and a whole validation split stays inside memory. Outputs that
    already arrive packed (the chunked postprocess in the adapter) are carried
    through unchanged. Label 0 is torchvision's background and is never
    emitted; the remaining labels are shifted back to the dense, zero-based
    class indices the evaluator's ground truth uses.
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
            raw = masks[index]
            packed = raw if isinstance(raw, PackedMask) else PackedMask.pack(_numpy(raw) >= 0.5)
            if not packed.any():
                continue
            x1, y1, x2, y2 = (float(value) for value in boxes[index])
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": label - 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "packed_mask": packed,
                    "score": float(scores[index]) if index < scores.shape[0] else 1.0,
                }
            )
    return predictions


def _numpy(value: Any) -> np.ndarray:
    """A numpy view of a tensor or array without importing torch here."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def coco_segmentation_predictions_from_results(results: Any, image_ids: list[int]) -> list[dict]:
    """Turn Ultralytics segmentation results into COCO instance predictions.

    Ultralytics reports masks as pixel-coordinate polygons in ``masks.xy``,
    aligned one-to-one with ``boxes``; the polygon is emitted as a COCO
    ``segmentation`` so the shared evaluator measures mask IoU. Asking for
    retina masks instead would upsample every candidate to the original image
    -- hundreds of full-resolution masks per plate -- so the framework's own
    boundary representation is what travels. A result without masks stays out:
    a mask prediction is never scored as a box. ``image_ids`` names each
    result's place in the split, as in the detection converter.
    """
    predictions: list[dict] = []
    for position, result in enumerate(results or []):
        image_id = image_ids[position]
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
