"""Named, immutable batch precomputes for the HTML report (#84).

``BatchMetrics`` is the one batch-wide input every section reads, and
``LotForms`` is the one shape input the shape, depth and card sections read.
Both are frozen dataclasses: a misspelled field fails loudly instead of
silently reading the wrong tuple slot.

The module is pure and Qt-free. The calibration import stays lazy inside the
lot-shape function, because the report must stay importable in the headless
path without pulling the scientific stack.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from polyx.core.report_metrics import _classes_views

if TYPE_CHECKING:
    from polyx.core.detection_run import ImageResult, RunResult


@dataclass(frozen=True)
class BatchMetrics:
    """Everything the report needs to know about the batch before it renders.

    ``results_by_model`` keeps the run's per-model shape (dense engine index ->
    image results) after the ``only_images`` filter, so the sections that read
    one model (counts, errors) and the sections that read all of them (models,
    methods, gallery) see the same subset.
    """

    results_by_model: dict[int, list[ImageResult]]
    all_results: list[ImageResult]
    image_count: int
    detection_count: int
    confidences: list[float]
    sizes_um: list[float]
    mean_confidence: float
    mean_size_um: float
    has_gt: bool
    classes: list[str]
    per_class: Counter[str]


@dataclass(frozen=True)
class LotForms:
    """The shape precompute shared by the shape, depth and card sections.

    ``shapes`` holds ``(Detection, image path)`` pairs whose shape was
    measured and that did not fall on the plate ring. ``ring_count`` says how
    many measured detections were set apart for falling on the ring, and
    ``formless_count`` how many predictions could not be segmented at all.
    """

    shapes: list[tuple[object, Path]]
    ring_count: int
    formless_count: int


@dataclass(frozen=True)
class ShapeBodies:
    """The two section bodies the shape pass produces in one go (#85).

    ``forma`` is the particle-shape section and ``by_folder`` its companion
    size-by-folder section. They are separate sections in the catalogue, but
    share the per-class size grouping computed by the shape pass, so the
    result is named instead of being split from a tuple by convention.
    """

    forma: str
    by_folder: str


def compute_batch_metrics(
    run: RunResult, only_images: Iterable[Path | str] | None = None
) -> BatchMetrics:
    """Compute the batch-wide inputs every section builder consumes.

    ``only_images`` narrows the batch to those image paths before anything is
    counted, averaged or plotted. ``None`` keeps the whole run. The filter is
    applied here once, so the totals, the figures and the tables cannot end up
    describing different subsets.
    """
    if only_images is None:
        results_by_model = dict(run.results)
    else:
        chosen = {Path(path) for path in only_images}
        results_by_model = {
            model_idx: [r for r in rows if Path(r.image_path) in chosen]
            for model_idx, rows in run.results.items()
        }

    all_results = [r for rows in results_by_model.values() for r in rows]
    image_count = len({r.image_path for r in all_results})
    detection_count = sum(len(r.predictions) for r in all_results)
    confidences = [p.conf for r in all_results for p in r.predictions]
    sizes_um = [p.diam_um for r in all_results for p in r.predictions if p.diam_um]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0
    mean_size_um = sum(sizes_um) / len(sizes_um) if sizes_um else 0
    has_gt = any(r.has_gt for r in all_results)

    classes = _classes_views(all_results)

    # ── Distribution by class ──
    per_class: Counter[str] = Counter()
    for r in all_results:
        for p in r.predictions:
            per_class[p.class_name] += 1

    return BatchMetrics(
        results_by_model=results_by_model,
        all_results=all_results,
        image_count=image_count,
        detection_count=detection_count,
        confidences=confidences,
        sizes_um=sizes_um,
        mean_confidence=mean_confidence,
        mean_size_um=mean_size_um,
        has_gt=has_gt,
        classes=classes,
        per_class=per_class,
    )


def compute_lot_forms(run: RunResult, batch: BatchMetrics) -> LotForms:
    """Detections with a measured shape, minus the ones over the plate ring.

    Reads ``batch.all_results`` so the shape sections see the same subset as
    every other section. The ``polyx.core.measure.calibration`` import stays
    lazy inside :func:`_lot_forms_from_results`: it must not load at module
    import time (S01 laziness gate).
    """
    return _lot_forms_from_results(run, batch.all_results)


def _lot_forms_from_results(run: RunResult, all_results: list[ImageResult]) -> LotForms:
    """Build the lot-shape precompute from an already-filtered result list."""
    # ── What falls on the plate ring is not sample ──
    # The ring is a bright, continuous band, and the detector confuses it with
    # material. Since it also comes out huge, it entered among the largest: in one
    # report it occupied an example card as `PET fibra` of 5978 um, which is
    # more than the 5 mm limit by which a microplastic is defined.
    #
    # The fitted circle was already known -- the scale comes from it -- and it only needed
    # to be asked. It is removed from the SIZE and from the cards, not from the count:
    # removing detections from the count is a study decision and not the
    # report generator's. What is done is to declare how many were set apart.
    from polyx.core.measure.calibration import about_the_ring

    def _on_the_ring(detection, image_path) -> bool:
        return about_the_ring(
            run.calibration_for(Path(image_path)),
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
        )

    # Each particle with its source image, so it can be found later.
    measured = [
        (prediction, result.image_path)
        for result in all_results
        for prediction in result.predictions
        if getattr(prediction, "aspect", None)
    ]
    shapes = [
        (detection, path) for detection, path in measured if not _on_the_ring(detection, path)
    ]
    ring_count = len(measured) - len(shapes)
    formless_count = sum(
        1
        for result in all_results
        for p in result.predictions
        if getattr(p, "aspect", None) is None
    )
    return LotForms(shapes=shapes, ring_count=ring_count, formless_count=formless_count)
