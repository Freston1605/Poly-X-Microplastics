"""The one shared evaluator for every model family.

It consumes COCO-format predictions and the COCO instance view of the
validation split, so a detector and a segmenter are scored by the same
arithmetic. Two different questions are kept apart:

* average precision (COCO AP) rewards the ranking of all predictions;
* the operating-point F1 at one confidence threshold is the quantity used to
  select checkpoints and to drive Optuna.

The second one reuses the project's ``Criterion``/``Score`` definition through
``Score.from_counts``: a well-placed box of the wrong polymer is a false
positive of the predicted class and a false negative of the true one. Every
figure names the overlap, the criterion and the split it was computed on.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from polyx.core.detect.protocol import Instance
from polyx.core.detection_run.operating_point import Criterion, Score
from polyx.core.quality.metrics import match_image

#: Ten IoU points, the threshold range used by COCO's mAP@[0.50:0.95].
_IOU_THRESHOLDS = [0.50 + 0.05 * index for index in range(10)]
#: One match rule for every family. Recorded beside every score.
_OPERATING_IOU = 0.50
#: COCO's per-image, per-category detection cap. Predictions below a very low
#: floor are still collected so AP can rank them; the operating point filters
#: at the protocol's confidence instead.
PREDICTION_FLOOR = 0.001
_MAX_DETECTIONS = 100


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
        copied = dict(item)
        if task == "segment":
            size = sizes.get(int(copied.get("image_id", -1)))
            copied["_mask"] = (
                rasterize_segmentation(copied.get("segmentation"), *size) if size else None
            )
        prepared.append(copied)
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


def _ranked_matches(
    predictions: list[dict], ground_truth: list[dict], threshold: float, task: str
) -> list[bool]:
    """Greedy, score-ordered matches: whether each prediction found a truth."""
    used: set[int] = set()
    matches: list[bool] = []
    for prediction in sorted(predictions, key=lambda item: -float(item.get("score", 0.0))):
        best_iou = 0.0
        best_index = -1
        for index, truth in enumerate(ground_truth):
            if index in used or not _same_image(prediction, truth):
                continue
            value = _overlap(prediction, truth, task)
            if value > best_iou:
                best_iou = value
                best_index = index
        found = best_index >= 0 and best_iou >= threshold
        if found:
            used.add(best_index)
        matches.append(found)
    return matches


def _apply_detection_cap(predictions: list[dict]) -> list[dict]:
    """Keep COCO's top 100 predictions per image and category.

    The cap is part of the reference protocol: a category with hundreds of
    low-score boxes per plate must not be scored differently from a detector
    that stops at 100. Ground truth is not capped.
    """
    by_image: dict[int, list[dict]] = defaultdict(list)
    for prediction in predictions:
        by_image[int(prediction.get("image_id", -1))].append(prediction)
    capped: list[dict] = []
    for items in by_image.values():
        capped.extend(
            sorted(items, key=lambda item: -float(item.get("score", 0.0)))[:_MAX_DETECTIONS]
        )
    return capped


def _ap_at_iou(
    predictions: list[dict], ground_truth: list[dict], threshold: float, task: str
) -> float:
    """COCO-style 101-point interpolated AP for one category at one IoU.

    Interpolation matters: a late low-score false positive must not erase the
    precision already achieved at a lower recall. Summing precision only at
    the ranks where a match happened, as an earlier version did, produced a
    number that was neither COCO AP nor thresholded F1. The per-image,
    per-category 100-detection cap is applied before ranking.
    """
    total = len(ground_truth)
    if not total:
        return 0.0
    matches = _ranked_matches(_apply_detection_cap(predictions), ground_truth, threshold, task)
    recalls: list[float] = []
    precisions: list[float] = []
    hits = 0
    for rank, found in enumerate(matches, start=1):
        if found:
            hits += 1
        recalls.append(hits / total)
        precisions.append(hits / rank)
    # Step-wise interpolation: precision at recall r is the best precision
    # achieved at any recall >= r.
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    return sum(_precision_at(recalls, precisions, point / 100) for point in range(101)) / 101


def _precision_at(recalls: list[float], precisions: list[float], target: float) -> float:
    """Interpolated precision at one recall point."""
    best = 0.0
    for recall, precision in zip(recalls, precisions, strict=True):
        if recall >= target:
            best = max(best, precision)
    return best


def _map(
    predictions: list[dict], ground_truth: list[dict], thresholds: list[float], task: str
) -> float:
    return sum(
        _ap_at_iou(predictions, ground_truth, threshold, task) for threshold in thresholds
    ) / len(thresholds)


def _instance(item: dict) -> Instance:
    """The canonical instance type the core matcher and scorer consume."""
    x, y, width, height = (float(value) for value in item["bbox"])
    return Instance(
        class_id=int(item["category_id"]),
        class_name=str(item.get("category_id", item["category_id"])),
        conf=float(item.get("score", 1.0)),
        x1=x,
        y1=y,
        x2=x + width,
        y2=y + height,
        mask=item.get("_mask"),
    )


def _operating_score(
    predictions: list[dict],
    ground_truth: list[dict],
    task: str,
    criterion: Criterion,
    confidence_threshold: float,
    *,
    category: int | None = None,
) -> Score:
    """Score one fixed operating point with the project's one F1 definition."""
    selected_predictions = [
        item for item in predictions if float(item.get("score", 0.0)) >= confidence_threshold
    ]
    selected_truths = ground_truth
    if category is not None:
        selected_predictions = [
            item for item in selected_predictions if int(item["category_id"]) == category
        ]
        selected_truths = [item for item in ground_truth if int(item["category_id"]) == category]
    by_image: dict[int, tuple[list[Instance], list[Instance]]] = defaultdict(lambda: ([], []))
    for item in selected_predictions:
        by_image[int(item.get("image_id", -1))][0].append(_instance(item))
    for item in selected_truths:
        by_image[int(item.get("image_id", -1))][1].append(_instance(item))

    tp = fp = fn = miscls = 0
    for predicted, truths in by_image.values():
        match = match_image(
            predicted,
            truths,
            iou_thr=_OPERATING_IOU,
            use_mask_iou=task == "segment",
        )
        tp += match.tp
        fp += match.fp
        fn += match.fn
        miscls += match.miscls
    return Score.from_counts(
        criterion,
        confidence_threshold,
        tp=tp,
        fp=fp,
        fn=fn,
        miscls=miscls,
    )


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

    def __init__(
        self,
        ground_truth: dict,
        split: str = "val",
        task: str = "detect",
        *,
        confidence_threshold: float = 0.25,
    ) -> None:
        if task not in ("detect", "segment"):
            raise ValueError(f"unknown evaluation task: {task}")
        self.ground_truth = ground_truth
        self.split = split
        self.task = task
        self.confidence_threshold = float(confidence_threshold)
        overlap = "mask IoU" if task == "segment" else "box IoU"
        metric = "mask " if task == "segment" else ""
        operating = (
            f"with-class operating-point F1 (confidence >= {self.confidence_threshold:.2f}, "
            f"{overlap} >= {_OPERATING_IOU:.2f})"
        )
        self.class_criterion = (
            f"{operating}, class-aware, real-domain validation split '{split}' "
            "(category id must match)"
        )
        self.localisation_criterion = (
            f"localisation operating-point F1 (confidence >= {self.confidence_threshold:.2f}, "
            f"{overlap} >= {_OPERATING_IOU:.2f}), class-agnostic, real-domain "
            f"validation split '{split}' (localisation only)"
        )
        self.map_criterion = (
            f"COCO {metric}mAP@[0.50:0.95] ({overlap}), class-aware, "
            f"real-domain validation split '{split}'"
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
        with_class = _operating_score(
            prepared,
            self._truths,
            self.task,
            Criterion.WITH_CLASS,
            self.confidence_threshold,
        ).f1
        localisation = _operating_score(
            prepared,
            self._truths,
            self.task,
            Criterion.LOCALISATION,
            self.confidence_threshold,
        ).f1
        map50 = self._score(prepared, class_aware=True, thresholds=[0.50])
        map5095 = self._score(prepared, class_aware=True, thresholds=_IOU_THRESHOLDS)
        metric = "mask " if self.task == "segment" else ""
        figures: dict[str, dict[str, Any]] = {
            "with_class_score": {"value": with_class, "criterion": self.class_criterion},
            "localisation_score": {
                "value": localisation,
                "criterion": self.localisation_criterion,
            },
            "mAP50": {
                "value": map50,
                "criterion": (
                    f"COCO {metric}AP@0.50, class-aware, real-domain validation split "
                    f"'{self.split}'"
                ),
            },
            "mAP50-95": {"value": map5095, "criterion": self.map_criterion},
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
                "value": map5095,
                "criterion": (
                    f"COCO mask mAP@[0.50:0.95] (mask IoU), class-aware, "
                    f"real-domain validation split '{self.split}'"
                ),
            }

        names = {
            int(item["id"]): str(item["name"]) for item in self.ground_truth.get("categories", [])
        }
        per_class: dict[str, dict[str, dict]] = {}
        for category in sorted(truths):
            name = names.get(category, str(category))
            class_score = _operating_score(
                prepared,
                self._truths,
                self.task,
                Criterion.WITH_CLASS,
                self.confidence_threshold,
                category=category,
            )
            class_figures: dict[str, dict[str, Any]] = {
                "AP50": {
                    "value": _ap_at_iou(
                        predicted.get(category, []), truths[category], 0.50, self.task
                    ),
                    "criterion": f"COCO AP@0.50, class '{name}', real-domain validation split",
                },
                "with_class_score": {
                    "value": class_score.f1,
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
                "criterion_name": Criterion.WITH_CLASS.value,
                "value": with_class,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": _OPERATING_IOU,
            },
        }
        if adapter_figures:
            # Only real adapter figures are attached. An empty mapping is never
            # presented as a cross-check: the shared evaluator is the authority.
            result["adapter_cross_check"] = adapter_figures
        return result
