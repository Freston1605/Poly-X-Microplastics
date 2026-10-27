"""Report-specific metric helpers (moved out of the GUI)."""

from __future__ import annotations

from collections import Counter

from polyx.core.quality.assignment import NO_ASSIGNABLE, class_id_reported

_ORDER_CLASSES = ["PET", "PP", "LDPE"]


def _count_per_class(detections) -> Counter:
    """Particles by reported polymer.

    Uses the *reported* class, not plain ``class_name``: under active
    abstention, a particle whose confidence does not support the polymer
    counts under `«no asignable»`. It is still a detected particle — just no
    polymer is attributed to it.
    """
    c: Counter[str] = Counter()
    for d in detections:
        c[class_id_reported(d)] += 1
    return c


def _classes_views(results_planes) -> list[str]:
    """Classes present in the batch, in presentation order.

    Taken from the detections **and** from the Ground Truth: a polymer the
    model never detects would still be a table row, with a zero, which is
    exactly the figure worth looking at.
    """
    views = {class_id_reported(d) for r in results_planes for d in list(r.predictions) + list(r.gt)}
    order = [c for c in _ORDER_CLASSES if c in views]
    rest = sorted(views - set(order) - {NO_ASSIGNABLE})
    return order + rest + ([NO_ASSIGNABLE] if NO_ASSIGNABLE in views else [])


def _pr_f1(tp: int, fp: int, fn: int):
    """Precision, recall and F1 from the tp/fp/fn counters."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)
