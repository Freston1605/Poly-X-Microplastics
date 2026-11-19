"""The one shared evaluator for every model family.

It consumes COCO-format predictions and the COCO instance view of the
validation split, so a detector and a segmenter are scored by the same
arithmetic. Every figure it emits carries the criterion that produced it:
localisation asks only whether the object was found, while the with-class
score also requires the polymer identity. Selection always uses the with-class
score; localisation is reported next to it, never instead of it.

A segmentation family adds a second currency: overlap is measured as mask IoU
between rasterized polygons, and the criterion names that explicitly. A
detection family keeps box IoU. Overlap is only computed between a prediction
and a ground-truth instance of the same image and the same category, so
identical geometry in another image or under another polymer is a false
positive rather than a match.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_IOU_THRESHOLDS = [0.50 + 0.05 * index for index in range(10)]


def _box_iou(first: list[float], second: list[float]) -> float:
    """IoU of two COCO boxes ``[x, y, width, height]``."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2]) * max(0.0, first[3])
    area_second = max(0.0, second[2]) * max(0.0, second[3])
    union = area_first + area_second - overlap
    return overlap / union if union > 0 else 0.0


def rasterize_segmentation(segmentation: Any, width: int, height: int) -> np.ndarray | None:
    """Fill a COCO polygon list into a boolean mask, or ``None`` if unusable.

    The canonical dataset audit already rejected invalid or degenerate
    polygons; this only guards the shape so a malformed prediction scores zero
    instead of raising mid-evaluation.
    """
    if not isinstance(segmentation, (list, tuple)) or not segmentation:
        return None
    if width <= 0 or height <= 0:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    filled = False
    for ring in segmentation:
        if not isinstance(ring, (list, tuple)) or len(ring) < 6 or len(ring) % 2 != 0:
            continue
        points = np.asarray(list(ring), dtype=float).reshape(-1, 2)
        if not np.all(np.isfinite(points)):
            continue
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], (1,))
        filled = True
    return mask.astype(bool) if filled else None


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return 0.0
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 0.0
    return int(np.logical_and(first, second).sum()) / union


def _prepare(items: list[dict], sizes: dict[int, tuple[int, int]], task: str) -> list[dict]:
    """Copy the items and attach the rasterized mask a segmenter compares."""
    prepared: list[dict] = []
    for item in items:
        copy = dict(item)
        if task == "segment":
            size = sizes.get(int(copy.get("image_id", -1)))
            copy["_mask"] = (
                rasterize_segmentation(copy.get("segmentation"), *size) if size else None
            )
        prepared.append(copy)
    return prepared


def _overlap(prediction: dict, truth: dict, task: str) -> float:
    """Overlap of one prediction and one truth, in the task's currency."""
    if task == "segment":
        first, second = prediction.get("_mask"), truth.get("_mask")
        if first is None or second is None:
            return 0.0
        return _mask_iou(first, second)
    return _box_iou(prediction["bbox"], truth["bbox"])


def _same_image(prediction: dict, truth: dict) -> bool:
    return int(prediction.get("image_id", -1)) == int(truth.get("image_id", -2))


def _ap_at_iou(
    predictions: list[dict], ground_truth: list[dict], threshold: float, task: str
) -> float:
    """Average precision for one category at one IoU threshold."""
    if not ground_truth:
        return 0.0
    used: set[int] = set()
    precision_points: list[float] = []
    ordered = sorted(predictions, key=lambda item: -float(item.get("score", 0.0)))
    for rank, prediction in enumerate(ordered, start=1):
        best_iou = 0.0
        best_index = -1
        for index, truth in enumerate(ground_truth):
            if index in used or not _same_image(prediction, truth):
                continue
            value = _overlap(prediction, truth, task)
            if value > best_iou:
                best_iou = value
                best_index = index
        if best_index >= 0 and best_iou >= threshold:
            used.add(best_index)
            precision_points.append(len(used) / rank)
        else:
            precision_points.append(0.0)
    hits = sum(1 for point in precision_points if point > 0)
    if hits == 0:
        return 0.0
    return sum(point for point in precision_points if point > 0) / len(ground_truth)


def _map(
    predictions: list[dict], ground_truth: list[dict], thresholds: list[float], task: str
) -> float:
    return sum(
        _ap_at_iou(predictions, ground_truth, threshold, task) for threshold in thresholds
    ) / len(thresholds)


def _best_iou(predictions: list[dict], truth: dict, task: str) -> float:
    """Best overlap found for one truth, from a prediction of the same class.

    The figure is class-aware: geometry is only an overlap when the predicted
    polymer is the truth's polymer, the same rule the selection score uses.
    """
    category = int(truth.get("category_id", -1))
    best = 0.0
    for prediction in predictions:
        if not _same_image(prediction, truth):
            continue
        if int(prediction.get("category_id", -2)) != category:
            continue
        best = max(best, _overlap(prediction, truth, task))
    return best


def _mean_best_iou(predictions: list[dict], truths: list[dict], task: str) -> float:
    """Mean best overlap per ground-truth instance, the reported IoU figure."""
    if not truths:
        return 0.0
    return sum(_best_iou(predictions, truth, task) for truth in truths) / len(truths)


class SharedEvaluator:
    """Scores COCO predictions against one frozen COCO ground truth."""

    def __init__(self, ground_truth: dict, split: str = "val", task: str = "detect") -> None:
        if task not in ("detect", "segment"):
            raise ValueError(f"unknown evaluation task: {task}")
        self.ground_truth = ground_truth
        self.split = split
        self.task = task
        overlap = "mask IoU" if task == "segment" else "box IoU"
        metric = "mask " if task == "segment" else ""
        self.class_criterion = (
            f"COCO {metric}mAP@[0.50:0.95] ({overlap}), class-aware, real-domain "
            f"validation split '{split}' (category id must match)"
        )
        self.localisation_criterion = (
            f"COCO {metric}AP@0.50 ({overlap}), class-agnostic, real-domain "
            f"validation split '{split}' (localisation only)"
        )
        self._sizes = {
            int(image["id"]): (int(image["width"]), int(image["height"]))
            for image in ground_truth.get("images", [])
        }
        self._truths = self._prepare(ground_truth.get("annotations", []))

    def _prepare(self, items: list[dict]) -> list[dict]:
        return _prepare(items, self._sizes, self.task)

    def _grouped(self, predictions: list[dict], *, class_aware: bool) -> dict[int, list[dict]]:
        grouped: dict[int, list[dict]] = {}
        if class_aware:
            for prediction in predictions:
                grouped.setdefault(int(prediction["category_id"]), []).append(prediction)
        elif predictions:
            grouped[0] = list(predictions)
        return grouped

    def _truth_groups(self, *, class_aware: bool) -> dict[int, list[dict]]:
        grouped: dict[int, list[dict]] = {}
        for annotation in self._truths:
            key = int(annotation["category_id"]) if class_aware else 0
            grouped.setdefault(key, []).append(annotation)
        return grouped

    def _score(
        self, predictions: list[dict], *, class_aware: bool, thresholds: list[float]
    ) -> float:
        predicted = self._grouped(predictions, class_aware=class_aware)
        truths = self._truth_groups(class_aware=class_aware)
        keys = sorted(truths)
        if not keys:
            return 0.0
        return sum(
            _map(predicted.get(key, []), truths[key], thresholds, self.task) for key in keys
        ) / len(keys)

    def evaluate(
        self,
        predictions: list[dict],
        *,
        adapter_figures: dict | None = None,
    ) -> dict:
        """Return named figures, per-class figures and the selection criterion."""
        prepared = self._prepare(predictions)
        with_class = self._score(prepared, class_aware=True, thresholds=_IOU_THRESHOLDS)
        localisation = self._score(prepared, class_aware=False, thresholds=[0.50])
        map50 = self._score(prepared, class_aware=True, thresholds=[0.50])
        figures: dict[str, dict[str, Any]] = {
            "with_class_score": {"value": with_class, "criterion": self.class_criterion},
            "localisation_score": {
                "value": localisation,
                "criterion": self.localisation_criterion,
            },
            "mAP50": {
                "value": map50,
                "criterion": (
                    f"COCO {'mask ' if self.task == 'segment' else ''}AP@0.50, "
                    f"class-aware, real-domain validation split '{self.split}'"
                ),
            },
            "mAP50-95": {"value": with_class, "criterion": self.class_criterion},
        }

        predicted = self._grouped(prepared, class_aware=True)
        truths = self._truth_groups(class_aware=True)
        all_truths = self._truths
        if self.task == "segment":
            mask_criterion = (
                f"mean best mask IoU per ground-truth instance, class-aware, "
                f"real-domain validation split '{self.split}'"
            )
            figures["mask_iou"] = {
                "value": _mean_best_iou(prepared, all_truths, self.task),
                "criterion": mask_criterion,
            }
            figures["mask_mAP50"] = {
                "value": map50,
                "criterion": (
                    f"COCO mask AP@0.50 (mask IoU), class-aware, real-domain "
                    f"validation split '{self.split}'"
                ),
            }
            figures["mask_mAP50-95"] = {
                "value": with_class,
                "criterion": self.class_criterion,
            }

        names = {
            int(item["id"]): str(item["name"]) for item in self.ground_truth.get("categories", [])
        }
        per_class: dict[str, dict[str, dict]] = {}
        for category in sorted(truths):
            name = names.get(category, str(category))
            class_figures: dict[str, dict[str, Any]] = {
                "AP50": {
                    "value": _ap_at_iou(
                        predicted.get(category, []), truths[category], 0.50, self.task
                    ),
                    "criterion": f"COCO AP@0.50, class '{name}', real-domain validation split",
                },
                "with_class_score": {
                    "value": _map(
                        predicted.get(category, []),
                        truths[category],
                        _IOU_THRESHOLDS,
                        self.task,
                    ),
                    "criterion": f"{self.class_criterion}; class '{name}'",
                },
            }
            if self.task == "segment":
                class_truths = [item for item in all_truths if item["category_id"] == category]
                class_figures["mask_iou"] = {
                    "value": _mean_best_iou(prepared, class_truths, self.task),
                    "criterion": (
                        f"mean best mask IoU per ground-truth instance, class '{name}', "
                        f"real-domain validation split '{self.split}'"
                    ),
                }
                class_figures["mask_AP50"] = {
                    "value": _ap_at_iou(
                        predicted.get(category, []), truths[category], 0.50, self.task
                    ),
                    "criterion": (
                        f"COCO mask AP@0.50 (mask IoU), class '{name}', real-domain "
                        f"validation split '{self.split}'"
                    ),
                }
            per_class[name] = class_figures

        result: dict[str, Any] = {
            "split": self.split,
            "task": self.task,
            "figures": figures,
            "per_class": per_class,
            "selection": {
                "metric": "with_class_score",
                "criterion": self.class_criterion,
                "value": with_class,
            },
        }
        if adapter_figures is not None:
            # The existing evaluator's per-class extraction is kept as a
            # cross-check of the adapter's own Ultralytics validation figures.
            result["adapter_cross_check"] = adapter_figures
        return result
