"""The one shared evaluator for every model family.

It consumes COCO-format predictions and the COCO instance view of the
validation split, so a detector and a segmenter are scored by the same
arithmetic. Every figure it emits carries the criterion that produced it:
localisation asks only whether the object was found, while the with-class
score also requires the polymer identity. Selection always uses the with-class
score; localisation is reported next to it, never instead of it.
"""

from __future__ import annotations

from typing import Any

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


def _ap_at_iou(predictions: list[dict], ground_truth: list[dict], threshold: float) -> float:
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
            if index in used:
                continue
            value = _box_iou(prediction["bbox"], truth["bbox"])
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


def _map(predictions: list[dict], ground_truth: list[dict], thresholds: list[float]) -> float:
    return sum(_ap_at_iou(predictions, ground_truth, threshold) for threshold in thresholds) / len(
        thresholds
    )


class SharedEvaluator:
    """Scores COCO predictions against one frozen COCO ground truth."""

    def __init__(self, ground_truth: dict, split: str = "val") -> None:
        self.ground_truth = ground_truth
        self.split = split
        self.class_criterion = (
            f"COCO mAP@[0.50:0.95], class-aware, real-domain validation split '{split}' "
            "(category id must match)"
        )
        self.localisation_criterion = (
            f"COCO AP@0.50, class-agnostic, real-domain validation split '{split}' "
            "(localisation only)"
        )

    def _grouped(self, predictions: list[dict], *, class_aware: bool) -> dict[int, list[dict]]:
        grouped: dict[int, list[dict]] = {}
        if class_aware:
            for prediction in predictions:
                grouped.setdefault(int(prediction["category_id"]), []).append(prediction)
        elif predictions:
            grouped[0] = list(predictions)
        return grouped

    def _truths(self, *, class_aware: bool) -> dict[int, list[dict]]:
        grouped: dict[int, list[dict]] = {}
        for annotation in self.ground_truth.get("annotations", []):
            key = int(annotation["category_id"]) if class_aware else 0
            grouped.setdefault(key, []).append(annotation)
        return grouped

    def _score(
        self, predictions: list[dict], *, class_aware: bool, thresholds: list[float]
    ) -> float:
        predicted = self._grouped(predictions, class_aware=class_aware)
        truths = self._truths(class_aware=class_aware)
        keys = sorted(truths)
        if not keys:
            return 0.0
        return sum(_map(predicted.get(key, []), truths[key], thresholds) for key in keys) / len(
            keys
        )

    def evaluate(
        self,
        predictions: list[dict],
        *,
        adapter_figures: dict | None = None,
    ) -> dict:
        """Return named figures, per-class figures and the selection criterion."""
        with_class = self._score(predictions, class_aware=True, thresholds=_IOU_THRESHOLDS)
        localisation = self._score(predictions, class_aware=False, thresholds=[0.50])
        map50 = self._score(predictions, class_aware=True, thresholds=[0.50])
        figures = {
            "with_class_score": {"value": with_class, "criterion": self.class_criterion},
            "localisation_score": {
                "value": localisation,
                "criterion": self.localisation_criterion,
            },
            "mAP50": {
                "value": map50,
                "criterion": f"COCO AP@0.50, class-aware, real-domain validation split '{self.split}'",
            },
            "mAP50-95": {"value": with_class, "criterion": self.class_criterion},
        }

        predicted = self._grouped(predictions, class_aware=True)
        truths = self._truths(class_aware=True)
        names = {
            int(item["id"]): str(item["name"]) for item in self.ground_truth.get("categories", [])
        }
        per_class: dict[str, dict[str, dict]] = {}
        for category in sorted(truths):
            name = names.get(category, str(category))
            per_class[name] = {
                "AP50": {
                    "value": _ap_at_iou(predicted.get(category, []), truths[category], 0.50),
                    "criterion": f"COCO AP@0.50, class '{name}', real-domain validation split",
                },
                "with_class_score": {
                    "value": _map(predicted.get(category, []), truths[category], _IOU_THRESHOLDS),
                    "criterion": f"{self.class_criterion}; class '{name}'",
                },
            }

        result: dict[str, Any] = {
            "split": self.split,
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
