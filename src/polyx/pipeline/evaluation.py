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

import numpy as np

from polyx.core.detect.protocol import Instance
from polyx.core.detection_run.operating_point import Criterion, Score
from polyx.core.quality.metrics import match_image
from polyx.pipeline.masks import PackedMask, as_mask

#: Ten IoU points, the threshold range used by COCO's mAP@[0.50:0.95].
_IOU_THRESHOLDS = [0.50 + 0.05 * index for index in range(10)]
#: One match rule for every family. Recorded beside every score.
_OPERATING_IOU = 0.50
#: COCO's per-image, per-category detection cap. Predictions below a very low
#: floor are still collected so AP can rank them; the operating point filters
#: at the protocol's confidence instead.
PREDICTION_FLOOR = 0.001
#: COCO's per-image, per-category detection cap. The cap applies to each
#: category separately, so a plate crowded with one polymer cannot hide the
#: highest-scoring detection of another.
_MAX_DETECTIONS = 100
#: Candidates one family may emit per image before shared scoring. Both
#: frameworks cap their output with one global top-k, which cannot express a
#: per-category cap; the budget is therefore 100 slots per class and never
#: below this ceiling, so a busy polymer cannot consume another one's share
#: before the evaluator applies the reference cap per category.
_CANDIDATE_CEILING = 1000


def prediction_budget(class_count: int) -> int:
    """How many detections a family may emit per image before shared scoring.

    The reference protocol keeps the top 100 predictions **per category** per
    image. A detector returns one ranked list where every class competes for
    the same slots, so the family's own limit must not bind: a plate with more
    than 100 detections of one polymer would otherwise drop every detection of
    another class before the evaluator ever sees it. Both families use this
    one number, so neither scores a different tail.
    """
    return max(_CANDIDATE_CEILING, _MAX_DETECTIONS * max(1, int(class_count)))


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


# ──────────────────────────────────────────────────────────────────
# Polygon rasterization.
#
# ``_polygon_to_mask`` below is a Python port of ``rleFrPoly``/``rleDecode``
# from pycocotools' ``maskApi.c``, the rasterizer COCO's own evaluation uses.
# The upstream implementation is redistributed under the Simplified BSD
# licence, reproduced here because this file is a derivative work. The same
# notice ships in the repository's NOTICE file.
#
#   Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
# ──────────────────────────────────────────────────────────────────

#: pycocotools traces polygon boundaries on a grid this many times finer.
_POLYGON_SCALE = 5


def rasterize_segmentation(segmentation: Any, width: int, height: int) -> np.ndarray | None:
    """Fill a COCO polygon list into a boolean mask, or ``None`` if unusable.

    This is a Python port of ``rleFrPoly``/``rleDecode`` from pycocotools
    (``maskApi.c``), the rasterizer COCO's own evaluation uses. COCO does not
    fill integer-rounded polygons: it traces the boundary at 5x, keeps the
    crossings that land on a whole pixel column, and encodes column runs. A
    single subpixel-rounded ``fillPoly`` therefore made a polygon COCO counts
    as a hit come out empty. Rings are merged as a union, as ``rleMerge`` does
    for a COCO instance made of several polygons.
    """
    if not isinstance(segmentation, (list, tuple)) or not segmentation:
        return None
    if width <= 0 or height <= 0:
        return None
    mask: np.ndarray | None = None
    for ring in segmentation:
        if not isinstance(ring, (list, tuple)) or len(ring) < 6 or len(ring) % 2 != 0:
            continue
        points = np.asarray(list(ring), dtype=float).reshape(-1, 2)
        if not np.all(np.isfinite(points)):
            continue
        ring_mask = _polygon_to_mask(points, height, width)
        if ring_mask is None:
            continue
        mask = ring_mask if mask is None else (mask | ring_mask)
    return mask


def _polygon_to_mask(points: np.ndarray, height: int, width: int) -> np.ndarray | None:
    """One polygon as a ``(width, height)`` boolean mask, as pycocotools does.

    The boundary is walked at 5x resolution, reduced to the pixel columns where
    it crosses a column boundary, and decoded as alternating column runs. The
    result is the same mask ``maskUtils.frPyObjects`` produces, not an
    approximation of it.
    """
    scale = float(_POLYGON_SCALE)
    count = len(points)
    # ``(int)`` in C truncates toward zero; a polygon outside the image
    # must round the same way pycocotools rounds it.
    scaled_x = np.trunc(scale * points[:, 0] + 0.5).astype(np.int64)
    scaled_y = np.trunc(scale * points[:, 1] + 0.5).astype(np.int64)
    boundary_x: list[int] = []
    boundary_y: list[int] = []
    for index in range(count):
        x_start = int(scaled_x[index])
        y_start = int(scaled_y[index])
        x_end = int(scaled_x[(index + 1) % count])
        y_end = int(scaled_y[(index + 1) % count])
        dx = abs(x_end - x_start)
        dy = abs(y_end - y_start)
        if dx == 0 and dy == 0:
            # A repeated vertex (a closed ring, or two vertices that land on the
            # same 5x grid point) contributes no crossing. ``rleFrPoly`` skips
            # it; without this guard the slope below divides by zero.
            continue
        flip = (dx >= dy and x_start > x_end) or (dx < dy and y_start > y_end)
        if flip:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        slope = (y_end - y_start) / dx if dx >= dy else (x_end - x_start) / dy
        if dx >= dy:
            for step in range(dx + 1):
                position = dx - step if flip else step
                boundary_x.append(position + x_start)
                boundary_y.append(int(y_start + slope * position + 0.5))
        else:
            for step in range(dy + 1):
                position = dy - step if flip else step
                boundary_y.append(position + y_start)
                boundary_x.append(int(x_start + slope * position + 0.5))
    transitions: list[int] = []
    for index in range(1, len(boundary_x)):
        if boundary_x[index] == boundary_x[index - 1]:
            continue
        column = float(
            boundary_x[index]
            if boundary_x[index] < boundary_x[index - 1]
            else boundary_x[index] - 1
        )
        column = (column + 0.5) / scale - 0.5
        if np.floor(column) != column or column < 0 or column > width - 1:
            continue
        row = float(min(boundary_y[index - 1], boundary_y[index]))
        row = (row + 0.5) / scale - 0.5
        if row < 0:
            row = 0.0
        elif row > height:
            row = float(height)
        transitions.append(int(column) * height + int(np.ceil(row)))
    if not transitions:
        return None
    # pycocotools sorts the crossings, appends the image end as a sentinel and
    # merges a zero-length run into the run before it: two crossings that land
    # on the same pixel cancel out instead of flipping the on/off parity.
    ordered = sorted(transitions)
    ordered.append(width * height)
    deltas = [ordered[0]]
    deltas.extend(ordered[index] - ordered[index - 1] for index in range(1, len(ordered)))
    runs = [deltas[0]]
    index = 1
    while index < len(deltas):
        if deltas[index] > 0:
            runs.append(deltas[index])
            index += 1
        else:
            index += 1
            if index < len(deltas):
                runs[-1] += deltas[index]
                index += 1
    flat = np.zeros(width * height, dtype=bool)
    cursor = 0
    for run_index, length in enumerate(runs):
        if run_index % 2 == 1 and length > 0:
            flat[cursor : cursor + length] = True
        cursor += length
    return flat.reshape(width, height).T


def _mask_iou(first: PackedMask | np.ndarray, second: PackedMask | np.ndarray) -> float:
    first = as_mask(first)
    second = as_mask(second)
    if first is None or second is None or first.shape != second.shape:
        return 0.0
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 0.0
    return int(np.logical_and(first, second).sum()) / union


def _prepare(items: list[dict], sizes: dict[int, tuple[int, int]], task: str) -> list[dict]:
    """Copy the items and attach the mask a segmenter compares.

    A prediction may carry its own mask (``packed_mask`` at one bit per pixel,
    or ``mask`` as an array), taken straight from the framework at the original
    resolution. That path is exact: converting a predicted mask into a polygon
    and back loses up to half a pixel per edge, enough to turn a perfect
    detection into a miss at IoU 0.50. Keeping the packed form is what lets a
    whole validation split of 4K plates stay in memory. A prediction that only
    carries COCO polygons is rasterized with the reference port and packed the
    same way.
    """
    prepared: list[dict] = []
    for item in items:
        copied = dict(item)
        if task == "segment":
            copied["_mask"] = _segment_mask(copied, sizes)
        prepared.append(copied)
    return prepared


def _segment_mask(item: dict, sizes: dict[int, tuple[int, int]]) -> PackedMask | np.ndarray | None:
    """The boolean mask of one item, from its own mask or its polygons."""
    size = sizes.get(int(item.get("image_id", -1)))
    packed = item.get("packed_mask")
    if isinstance(packed, PackedMask) and (size is None or packed.shape == (size[1], size[0])):
        return packed
    # A mask the caller did not produce at the image's resolution lives on
    # another grid; fall back to the polygons it also carried, if any.
    explicit = item.get("mask")
    if explicit is not None:
        array = np.asarray(explicit)
        if array.dtype != np.bool_:
            array = array >= 0.5
        array = array.reshape(array.shape[-2:])
        if size is None or array.shape == (size[1], size[0]):
            return PackedMask.pack(array)
        # Same fallback as above: an off-grid array is not this image's mask.
    if size is None:
        return None
    rasterized = rasterize_segmentation(item.get("segmentation"), *size)
    return PackedMask.pack(rasterized) if rasterized is not None else None


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
    """Keep COCO's top 100 predictions per image **and category**.

    The cap is part of the reference protocol: a category with hundreds of
    low-score boxes per plate must not be scored differently from a detector
    that stops at 100. Applying it per image alone would let one crowded class
    push another class's confident predictions out of the ranking, so the
    grouping key includes the category. Ground truth is not capped.
    """
    by_group: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for prediction in predictions:
        key = (int(prediction.get("image_id", -1)), int(prediction["category_id"]))
        by_group[key].append(prediction)
    capped: list[dict] = []
    for items in by_group.values():
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
        # The core matcher scores one managed operating point, so only the
        # handful of masks above the confidence threshold are unpacked here.
        mask=as_mask(item.get("_mask")),
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
