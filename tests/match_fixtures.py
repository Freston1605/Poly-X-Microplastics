"""One run whose mask match disagrees with a box match on purpose (#120).

The prediction covers the label's mask exactly, but its box is twice as tall,
so mask IoU finds the true positive (1.0) where box IoU does not (0.125). A
second prediction is a plain false positive. Any reader that re-matches with
box IoU instead of reading ``ImageResult.match`` reports two false positives
and a false negative the run never counted.

Shared by the errors page, the results page and the report tests: they must
all describe the same run, not three matches of their own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from polyx.core.detect.protocol import Instance
from polyx.core.detection_run import ImageResult, ModelInfo, RunParams, RunResult
from polyx.core.quality.metrics import match_image

MASK_BOX = (20, 20, 60, 60)
LOOSE_BOX = (0, 0, 160, 80)
STRAY_BOX = (90, 90, 130, 130)


def _mask(box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((160, 160), dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def matched_result(image: Path, model_idx: int = 0, *, use_mask: bool = True) -> ImageResult:
    """The image's result as the engine would have left it, match included.

    ``use_mask=False`` reproduces a box-matched run over the same boxes: no
    true positive, two false positives and one false negative.
    """
    label_mask = _mask(MASK_BOX)
    found = Instance(
        class_id=0,
        class_name="PET",
        conf=0.9,
        x1=float(LOOSE_BOX[0]),
        y1=float(LOOSE_BOX[1]),
        x2=float(LOOSE_BOX[2]),
        y2=float(LOOSE_BOX[3]),
        mask=label_mask,
    )
    stray = Instance(
        class_id=1,
        class_name="PP",
        conf=0.4,
        x1=float(STRAY_BOX[0]),
        y1=float(STRAY_BOX[1]),
        x2=float(STRAY_BOX[2]),
        y2=float(STRAY_BOX[3]),
    )
    gt = Instance(
        class_id=0,
        class_name="PET",
        conf=1.0,
        x1=float(MASK_BOX[0]),
        y1=float(MASK_BOX[1]),
        x2=float(MASK_BOX[2]),
        y2=float(MASK_BOX[3]),
        mask=label_mask.copy(),
    )
    match = match_image([found, stray], [gt], iou_thr=0.5, use_mask_iou=use_mask)
    if use_mask:
        assert (match.tp, match.fp, match.fn) == (1, 1, 0)
    else:
        assert (match.tp, match.fp, match.fn) == (0, 2, 1)

    return ImageResult(
        image_path=image,
        model_idx=model_idx,
        predictions=[found, stray],
        gt=[gt],
        has_gt=True,
        tp=match.tp,
        fp=match.fp,
        fn=match.fn,
        miscls=match.miscls,
        match=match,
        match_iou_thr=0.5,
        match_use_mask_iou=use_mask,
    )


def matched_run(root: Path, *, use_mask: bool = True) -> RunResult:
    """A one-model, one-image run carrying ``matched_result``."""
    image = root / "plate.png"
    return RunResult(
        run_dir=root / "run",
        images=[image],
        models=[ModelInfo(alias="m1", path=root / "m1.pt")],
        params=RunParams(),
        results={0: [matched_result(image, use_mask=use_mask)]},
    )
