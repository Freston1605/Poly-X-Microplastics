"""TP/FP/FN computation, confusion matrix and per-class P/R/F1 (COCO standard)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from polyx.core.detect.protocol import Instance as Detection

# ────────────────────────────────────────────────────────────────────
# Labels for display (Spanish).
#
# IMPORTANT: the internal logic still uses the short attributes
# tp / fp / fn / miscls (and the precision/recall/f1 properties). These
# constants ONLY change how they are shown to the end user in the
# interface, the console and the reports; the computation is unaffected.
# ────────────────────────────────────────────────────────────────────
LABEL_TP = "Verdaderos Positivos"
LABEL_FP = "Falsos Positivos"
LABEL_FN = "Falsos Negativos"
LABEL_MISCLS = "Mal Clasificados"

# Code -> readable label map (useful for tables/filters)
LABELS_BY_CODE = {
    "TP": LABEL_TP,
    "FP": LABEL_FP,
    "FN": LABEL_FN,
    "MISCLS": LABEL_MISCLS,
}


def iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (a.w * a.h) + (b.w * b.h) - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a: Detection, b: Detection) -> float:
    """IoU on masks when both have them; falls back to box IoU otherwise."""
    ma = getattr(a, "mask", None)
    mb = getattr(b, "mask", None)
    if ma is None or mb is None:
        return iou(a, b)
    ma = np.asarray(ma).astype(bool)
    mb = np.asarray(mb).astype(bool)
    if ma.shape != mb.shape:
        return iou(a, b)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


@dataclass
class MatchResult:
    """Result of pred vs GT matching for one image."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    miscls: int = 0  # located correctly but wrong class
    # (pred_idx, gt_idx) lists for each category
    tp_pairs: List[Tuple[int, int]] = field(default_factory=list)
    fp_idx: List[int] = field(default_factory=list)  # indices into preds
    fn_idx: List[int] = field(default_factory=list)  # indices into gts
    miscls_pairs: List[Tuple[int, int]] = field(default_factory=list)


def match_image(
    preds: List[Detection], gts: List[Detection], iou_thr: float = 0.5, use_mask_iou: bool = False
) -> MatchResult:
    """Matches predictions to GTs by IoU. Greedy, descending by confidence."""
    res = MatchResult()
    if not preds and not gts:
        return res
    if not preds:
        res.fn = len(gts)
        res.fn_idx = list(range(len(gts)))
        return res
    if not gts:
        res.fp = len(preds)
        res.fp_idx = list(range(len(preds)))
        return res

    pred_order = sorted(range(len(preds)), key=lambda i: -preds[i].conf)
    gt_used = [False] * len(gts)
    pred_matched = [False] * len(preds)

    for pi in pred_order:
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gts):
            if gt_used[gi]:
                continue
            if use_mask_iou:
                v = mask_iou(preds[pi], gt)
            else:
                v = iou(preds[pi], gt)
            if v > best_iou:
                best_iou = v
                best_gi = gi
        if best_gi >= 0 and best_iou >= iou_thr:
            gt_used[best_gi] = True
            pred_matched[pi] = True
            if preds[pi].class_id == gts[best_gi].class_id:
                res.tp += 1
                res.tp_pairs.append((pi, best_gi))
            else:
                res.miscls += 1
                res.miscls_pairs.append((pi, best_gi))
    for pi, m in enumerate(pred_matched):
        if not m:
            res.fp += 1
            res.fp_idx.append(pi)
    for gi, u in enumerate(gt_used):
        if not u:
            res.fn += 1
            res.fn_idx.append(gi)
    return res


@dataclass
class ClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def aggregate_per_class(
    all_preds: List[List[Detection]],
    all_gts: List[List[Detection]],
    class_ids: List[int],
    iou_thr: float = 0.5,
    use_mask_iou: bool = False,
) -> Dict[int, ClassMetrics]:
    """Returns {class_id: ClassMetrics} summed over all images."""
    out = {cid: ClassMetrics() for cid in class_ids}
    for preds, gts in zip(all_preds, all_gts):
        m = match_image(preds, gts, iou_thr, use_mask_iou=use_mask_iou)
        # TP per class: use the TP pairs
        for pi, gi in m.tp_pairs:
            cid = preds[pi].class_id
            if cid in out:
                out[cid].tp += 1
        # FP per predicted class
        for pi in m.fp_idx:
            cid = preds[pi].class_id
            if cid in out:
                out[cid].fp += 1
        # FN per GT class
        for gi in m.fn_idx:
            cid = gts[gi].class_id
            if cid in out:
                out[cid].fn += 1
        # miscls: counts as FP of the prediction + FN of the GT
        for pi, gi in m.miscls_pairs:
            cid_p = preds[pi].class_id
            cid_g = gts[gi].class_id
            if cid_p in out:
                out[cid_p].fp += 1
            if cid_g in out:
                out[cid_g].fn += 1
    return out


def confusion_matrix(
    all_preds: List[List[Detection]],
    all_gts: List[List[Detection]],
    class_ids: List[int],
    iou_thr: float = 0.5,
    use_mask_iou: bool = False,
) -> np.ndarray:
    """Matrix (N+1)x(N+1). Last row/col = background (FN/FP)."""
    n = len(class_ids)
    cm = np.zeros((n + 1, n + 1), dtype=int)
    idx_of = {c: i for i, c in enumerate(class_ids)}
    for preds, gts in zip(all_preds, all_gts):
        m = match_image(preds, gts, iou_thr, use_mask_iou=use_mask_iou)
        for pi, gi in m.tp_pairs:
            i = idx_of.get(gts[gi].class_id)
            if i is not None:
                cm[i, i] += 1
        for pi, gi in m.miscls_pairs:
            ig = idx_of.get(gts[gi].class_id)
            ip = idx_of.get(preds[pi].class_id)
            if ig is not None and ip is not None:
                cm[ig, ip] += 1
        for pi in m.fp_idx:
            ip = idx_of.get(preds[pi].class_id)
            if ip is not None:
                cm[n, ip] += 1  # GT background -> predicted as a class
        for gi in m.fn_idx:
            ig = idx_of.get(gts[gi].class_id)
            if ig is not None:
                cm[ig, n] += 1  # GT class -> not detected
    return cm
