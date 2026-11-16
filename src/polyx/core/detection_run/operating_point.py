"""The operating point: one definition of precision, recall and F1.

A detection run can be scored in two ways, and they are not the same number:

- **Localisation** asks only whether the particle was found. A box on the
  particle with the wrong polymer is reported apart (``miscls``) and stays out
  of both denominators.
- **With class** asks for the polymer too: that same box is a false positive of
  the predicted class and a false negative of the true one.

The report has always named both. This module owns them so the page, the
comparison table and the report cannot compute a third definition by hand.

There are also two *situations*, and they must not be confused:

- :meth:`Score.from_decision` reads the match the run actually performed
  (``ImageResult.tp/fp/fn/miscls``, the serialised projection of
  ``ImageResult.match``). It never matches again, so a reader cannot disagree
  with the run it is describing. This is what the KPIs and the report show.
- :func:`score` is the *hypothetical* threshold sweep: "what if only the boxes
  above this confidence had counted". It re-matches each image with the rule
  the run recorded (IoU threshold, box or mask IoU), so it differs from the run
  only in the threshold.

The module is Qt-free: the CLI and the report use it too.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from polyx.core.detection_run.result import ImageResult
from polyx.core.i18n import LazyString, lazy
from polyx.core.quality.metrics import match_image


class Criterion(StrEnum):
    """Which errors count against a detection.

    A ``str`` enum because the value is written into summaries and read by
    scripts; ``Criterion("with_class")`` keeps that stable.
    """

    LOCALISATION = "localisation"
    WITH_CLASS = "with_class"


#: Display names, resolved when they are shown (the report's own words).
CRITERION_LABELS: dict[Criterion, LazyString] = {
    Criterion.LOCALISATION: lazy("localisation"),
    Criterion.WITH_CLASS: lazy("with class"),
}


def criterion_label(criterion: Criterion) -> str:
    """The criterion's name in the active language."""
    return str(CRITERION_LABELS[criterion])


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall and F1 from three counters.

    The counters are passed already chosen: under the with-class criterion the
    caller folds ``miscls`` into both ``fp`` and ``fn`` before calling this.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


@dataclass(frozen=True)
class Score:
    """Precision, recall and F1 at one threshold, under one named criterion."""

    criterion: Criterion
    threshold: float
    tp: int
    fp: int
    fn: int
    miscls: int
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_counts(
        cls,
        criterion: Criterion,
        threshold: float,
        *,
        tp: int,
        fp: int,
        fn: int,
        miscls: int,
    ) -> Score:
        """Score counters that were already decided, under ``criterion``.

        ``tp/fp/fn`` are the plain match counters and ``miscls`` the
        well-placed-but-wrong-class boxes; only the denominators change with
        the criterion. Every caller that has counts goes through here, so the
        two criteria are defined once.
        """
        if criterion is Criterion.WITH_CLASS:
            precision, recall, f1 = prf1(tp, fp + miscls, fn + miscls)
        else:
            precision, recall, f1 = prf1(tp, fp, fn)
        return cls(
            criterion=criterion,
            threshold=threshold,
            tp=tp,
            fp=fp,
            fn=fn,
            miscls=miscls,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    @classmethod
    def from_decision(
        cls,
        results: Iterable[ImageResult],
        criterion: Criterion,
        threshold: float,
    ) -> Score:
        """Score the match the run decided, summed over ``results``.

        ``ImageResult.tp/fp/fn/miscls`` are the serialised projection of the
        match the run performed; nothing is matched again here. The criterion
        only decides which counters enter the denominators, so a caller cannot
        silently change the decision, only the question asked of it.

        The ``threshold`` is the one the run used (``RunParams.conf``). It is
        recorded, not applied: the predictions were already filtered when the
        run stored them.
        """
        rows = list(results)
        return cls.from_counts(
            criterion,
            threshold,
            tp=sum(row.tp for row in rows),
            fp=sum(row.fp for row in rows),
            fn=sum(row.fn for row in rows),
            miscls=sum(row.miscls for row in rows),
        )

    @property
    def label(self) -> str:
        """The criterion's display name, resolved in the active language."""
        return criterion_label(self.criterion)


def score(results: Iterable[ImageResult], criterion: Criterion, threshold: float) -> Score:
    """Score a hypothetical confidence threshold.

    This is the legitimate sweep: it answers "what if the run had kept only the
    boxes at or above ``threshold``". Each image is matched once, with the rule
    the run recorded for it, so the sweep can differ from the run in its
    threshold but never in its mask rule. Images without ground truth are
    skipped: they have nothing to be right or wrong about.
    """
    tp = fp = fn = miscls = 0
    for result in results:
        if not result.has_gt:
            continue
        predictions = [
            prediction for prediction in result.predictions if prediction.conf >= threshold
        ]
        match = match_image(
            predictions,
            result.gt,
            iou_thr=result.match_iou_thr,
            use_mask_iou=result.match_use_mask_iou,
        )
        tp += match.tp
        fp += match.fp
        fn += match.fn
        miscls += match.miscls
    return Score.from_counts(
        criterion,
        threshold,
        tp=tp,
        fp=fp,
        fn=fn,
        miscls=miscls,
    )


def recommend(
    results: Iterable[ImageResult],
    criterion: Criterion = Criterion.WITH_CLASS,
) -> Score:
    """The best observed confidence threshold for ``criterion``.

    The candidates are the confidences the run already produced, so the sweep
    only moves upward: lowering the threshold below the one used cannot recover
    boxes that were never computed. Ties keep the lowest threshold, and the
    returned :class:`Score` carries its criterion, so the caller can label a
    recommendation instead of assuming which one it is.
    """
    rows = [row for row in results if row.has_gt]
    candidates = sorted(
        {round(prediction.conf, 2) for row in rows for prediction in row.predictions}
    )
    if not candidates:
        return Score.from_counts(criterion, 0.0, tp=0, fp=0, fn=0, miscls=0)
    # At most ~25 points: the page draws a line, not a dense curve.
    step = max(1, len(candidates) // 25)
    best: Score | None = None
    for threshold in candidates[::step]:
        current = score(rows, criterion, threshold)
        if best is None or current.f1 > best.f1:
            best = current
    assert best is not None  # `candidates` is non-empty, so the loop ran
    return best


__all__ = [
    "CRITERION_LABELS",
    "Criterion",
    "Score",
    "criterion_label",
    "prf1",
    "recommend",
    "score",
]
