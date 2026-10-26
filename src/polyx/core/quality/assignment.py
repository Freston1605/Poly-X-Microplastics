"""When a detected particle can be assigned a polymer, and when it cannot.

Detecting a particle and saying which polymer it is are two different questions,
and the second is far more fragile than the first. Nile Red is solvatochromic:
its emission responds to the **polarity** of the environment, not to the chemical
identity of the polymer. That is why PET —polar polyester— separates cleanly,
while PP and LDPE, the two apolar polyolefins, share a tone and differ only in
brightness; and brightness depends on exposure, focus, particle thickness and
how much dye was taken up. None of that is the polymer.

The practical consequence is that the model gets the box right far more often
than the class. This module allows the report to say so instead of hiding it:
below a confidence threshold the particle **still counts** as a detected
particle, but it is reported as `«no asignable»` instead of inventing a
polymer.

What this does NOT do, and why
──────────────────────────────
Ideally the model would abstain on *margin*: if the top class scores 0.42 and
the runner-up 0.39, the choice is a coin toss even though 0.42 clears the
threshold. That cannot be done with what is available: after NMS the YOLO
output carries a single confidence and a single class per box, and the scores
of the remaining classes are lost inside it. Recovering them means intercepting
the raw detection-head tensor, which is fragile and breaks with every
ultralytics release. The confidence band is what can honestly be sustained
today.

It is not a new model class either. A trained `«desconocido»` class is defined
by the annotator's doubt, not by what the image shows: the same particle would
end up labelled LDPE one day and unknown the next, and that noise would land
right on PP and LDPE, already the classes with the worst recall."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List

# Label used to report a particle whose polymer claim does not hold.
# It is left in Spanish and lowercase because it travels into the report tables as
# is; the English translation is applied by i18n to this string.
NO_ASSIGNABLE = "no asignable"


def mark_assignables(detections: Iterable, threshold_2: float) -> int:
    """Marks each detection as assignable or not, according to confidence.

    Args:
        detecciones: Detection objects already produced by the model.
        umbral: minimum confidence to sustain the polymer. With 0 (or less)
            abstention is disabled and everything is reported with its class,
            which is the historical behaviour.

    Returns:
        How many detections were left marked as unassignable.
    """
    if threshold_2 <= 0:
        for d in detections:
            d.assignable = True
        return 0
    n = 0
    for d in detections:
        d.assignable = bool(d.conf >= threshold_2)
        if not d.assignable:
            n += 1
    return n


def class_id_reported(detection) -> str:
    """Name under which the particle must appear in tables and figures.

    It is read with ``getattr`` with a fallback because detections parsed from
    a ground-truth .txt, and those from runs saved before abstention existed,
    lack the field: there the correct answer is the usual class.
    """
    if getattr(detection, "assignable", True):
        return detection.class_name
    return NO_ASSIGNABLE


@dataclass
class SummaryAssignment:
    """How much of the batch could be assigned a polymer."""

    total: int = 0
    assigned: int = 0
    no_assignables: int = 0
    threshold_2: float = 0.0

    @property
    def active(self) -> bool:
        """Abstention was enabled in this analysis."""
        return self.threshold_2 > 0

    @property
    def percentage_no_assignable(self) -> float:
        return 100.0 * self.no_assignables / self.total if self.total else 0.0

    def __str__(self) -> str:
        if not self.active:
            return "abstención desactivada"
        return (
            f"{self.no_assignables} de {self.total} partículas sin polímero "
            f"asignable ({self.percentage_no_assignable:.1f} %), "
            f"umbral {self.threshold_2:.2f}"
        )


def resume(detections: Iterable, threshold_2: float) -> SummaryAssignment:
    """Counts assigned and unassignable over an already-marked batch."""
    dets: List = list(detections)
    no_assignment = sum(1 for d in dets if not getattr(d, "assignable", True))
    return SummaryAssignment(
        total=len(dets),
        assigned=len(dets) - no_assignment,
        no_assignables=no_assignment,
        threshold_2=threshold_2,
    )


def count_reported(detections: Iterable) -> Counter:
    """Particles per reported class, with `«no asignable»` as one more category.

    The total is still the number of detected particles: abstaining on the
    polymer does not drop the particle, which is exactly what is wanted when
    the result is reported by particle count.
    """
    c: Counter = Counter()
    for d in detections:
        c[class_id_reported(d)] += 1
    return c
