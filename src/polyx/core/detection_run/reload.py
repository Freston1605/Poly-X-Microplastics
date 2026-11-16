"""Re-read the ground truth of a completed run through the engine's rules.

Inference is the expensive part; the ground truth is not. When a badly placed
annotation is corrected there is no reason to infer again over the whole
batch: the predictions are already in the run, and what is needed is to pair
them again with the labels on disk and redraw the affected views.

The operation is a second reading of the run's own sequence, not a second
implementation of it. It reuses the engine's measurement (``measure_particles``), its
matching rule (``decide_match``) and its drawing (``triptych``), takes each
photo's calibration from the run record, and republishes the affected rows
through :class:`RunWriter`. A reloaded image therefore cannot disagree with a
fresh run over the same labels, and the run folder says what the images show.

The function mutates the :class:`RunResult` in place, because the GUI pages
share those ``ImageResult`` objects, and returns a :class:`ReloadSummary`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from polyx.core.annotate import triptych
from polyx.core.detect.protocol import Instance, size_image
from polyx.core.image_io import read_image
from polyx.core.labels import find_label_file, read_label_file

from .engine import decide_match, measure_particles
from .result import ImageResult, RunResult
from .schema import aggregate_metrics
from .writer import RunWriter


@dataclass(frozen=True)
class ReloadSummary:
    """What one ground-truth reload did, in names a caller can use directly.

    ``changed`` and ``missing_gt`` list the photos, not their names: two
    folders in one batch can hold photos with the same basename.
    ``recalculated`` counts the images whose views and counts were actually
    rewritten; it can be smaller than ``len(changed)`` when a photo cannot be
    decoded or the pass was cancelled.
    """

    revisited: int = 0
    changed: list[Path] = field(default_factory=list)
    missing_gt: list[Path] = field(default_factory=list)
    recalculated: int = 0


def _same_boxes(a: list[Instance], b: list[Instance], tol: float = 0.5) -> bool:
    """True when two lists of boxes match in class and coordinates."""
    if len(a) != len(b):
        return False
    for one, other in zip(a, b, strict=True):
        if one.class_id != other.class_id:
            return False
        if (
            abs(one.x1 - other.x1) > tol
            or abs(one.y1 - other.y1) > tol
            or abs(one.x2 - other.x2) > tol
            or abs(one.y2 - other.y2) > tol
        ):
            return False
    return True


def _per_image(run: RunResult) -> dict[Path, list[tuple[int, ImageResult]]]:
    """Group the run's results by photo, keeping the run's own image order."""
    grouped: dict[Path, list[tuple[int, ImageResult]]] = {}
    for model_idx in sorted(run.results):
        for row in run.results[model_idx]:
            grouped.setdefault(row.image_path, []).append((model_idx, row))
    ordered: dict[Path, list[tuple[int, ImageResult]]] = {}
    for image_path in run.images:
        if image_path in grouped:
            ordered[image_path] = grouped.pop(image_path)
    ordered.update(grouped)
    return ordered


def reload_ground_truth(
    run: RunResult,
    *,
    gt_folder: Path | None = None,
    progress: Callable[[int, int, str], bool | None] | None = None,
) -> ReloadSummary:
    """Re-read the labels of ``run`` and recompute metrics and annotated views.

    Only the photos whose ``.txt`` actually differs from the GT the run froze
    are redrawn: comparing is much cheaper than redrawing a 3260 px plate, and
    in the normal case -- one plate of sixty-nine was corrected -- the work
    reduces to that one.

    ``gt_folder`` is the folder to look the labels up in, as the run's own
    configuration had it; ``None`` means the conventional layout next to the
    images. ``progress(done, total, name)`` is called once per photo before it
    is examined; returning ``False`` aborts the pass and keeps what was already
    recomputed and written. The run is mutated in place and published to its
    folder as each changed photo is finished, so an abort never leaves
    ``run.json`` describing views the folder does not have.

    The matching rule is re-derived from the inputs the reload actually has,
    exactly as a fresh run would derive it. Mask IoU can only decide a pair
    when both sides carry a mask -- ``match_image`` falls back to box IoU pair
    by pair -- so a masked run reloaded from the GUI still keeps its prediction
    masks and matches by mask IoU, while the same run reopened from its folder
    has box-only predictions and matches the way a fresh run of those would.
    The resulting flag is recorded with the new counts, so the run record and
    the report's Methods section describe the reloaded numbers instead of
    repeating the old decision.
    """
    revisited = 0
    changed: list[Path] = []
    missing_gt: list[Path] = []
    recalculated = 0
    by_image = _per_image(run)
    if not by_image:
        return ReloadSummary()

    # Class names come from the run's first model: the .txt stores only the
    # index, and without the map the boxes would have no readable class.
    names = run.models[0].names if run.models else {}
    image_indices: dict[Path, int] = {}
    for index, image_path in enumerate(run.images):
        image_indices.setdefault(image_path, index)

    writer: RunWriter | None = None
    total = len(by_image)
    for made, (image_path, cluster) in enumerate(by_image.items(), start=1):
        if progress is not None and progress(made, total, image_path.name) is False:
            break
        revisited += 1

        first = cluster[0][1]
        gt_path = find_label_file(image_path, gt_folder)
        has_gt = gt_path is not None
        if not has_gt:
            missing_gt.append(image_path)

        # The size is read from the header, without decoding the whole photo: it
        # is needed to denormalize the .txt, and in most images it is the only
        # thing the pass will need.
        size = size_image(image_path)
        if size is None:
            continue
        width, height = size

        gts = read_label_file(gt_path, width, height, names)[0] if has_gt else []
        if has_gt == first.has_gt and _same_boxes(gts, first.gt):
            continue
        changed.append(image_path)

        bgr = read_image(image_path)
        if bgr is None:
            continue

        calibration = run.calibration_for(image_path)
        um_per_px = calibration.um_por_px if calibration is not None and calibration.valid else None
        if has_gt:
            measure_particles(gts, bgr, um_per_px, run.params)

        image_index = image_indices.get(image_path, made - 1)
        for _model_idx, row in cluster:
            row.gt = list(gts)
            row.has_gt = has_gt
            row.match_iou_thr = run.params.iou_tp
            if has_gt:
                # The reload is the run deciding again over new labels, so the
                # match it just made replaces the old one: keeping the previous
                # match next to new counters would leave every reader of the
                # identity (Errors page, report) describing the labels that
                # were thrown away.
                match, row.match_use_mask_iou = decide_match(
                    row.predictions, gts, run.params.iou_tp
                )
                row.match = match
                row.tp, row.fp, row.fn, row.miscls = match.tp, match.fp, match.fn, match.miscls
            else:
                row.match = None
                row.match_use_mask_iou = False
                row.tp = row.fp = row.fn = row.miscls = 0

            # The predictions view does not change -- the predictions are the
            # same -- but the control and GT ones do. They are drawn by the
            # same composer the original run used, so the reload stays
            # identical in style to what it is replacing.
            views = triptych(
                bgr,
                row.predictions,
                gts,
                pred_label_for=lambda det: f"{det.class_name} {det.conf:.2f}",
                has_gt=has_gt,
            )
            if writer is None:
                writer = RunWriter(run.run_dir, run, update=True)
            writer.update_image(row, views, image_index)
            recalculated += 1

    # ``run.json`` recomputes its aggregates from the rows every time it is
    # written; the in-memory record must not lag behind the folder it describes.
    run.metrics = aggregate_metrics(run)
    return ReloadSummary(
        revisited=revisited,
        changed=changed,
        missing_gt=missing_gt,
        recalculated=recalculated,
    )


__all__ = ["ReloadSummary", "reload_ground_truth"]
