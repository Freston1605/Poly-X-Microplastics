"""Re-reads the ground truth from disk and recomputes without running the model again.

Inference is the expensive part; the ground truth is not. When a badly placed
annotation is corrected there is no reason to run inference again on the
69 plates: the predictions are already in memory and all that is needed is to
re-pair and redraw the control image.

Until now the only way out was to re-run the whole batch, and whoever did not
ended up with a report carrying the old annotation with no warning at all,
because the run freezes the GT it read. This module closes that gap.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import cv2

from polyx.ml.yolo_wrap import (
    find_gt_for_image, read_yolo_txt, compute_box_size_um, size_image,
)
from polyx.ml.metrics import match_image
from .runner import _draw_annotated, _color_bgr_for_class


def _same_boxes(a, b, tol: float = 0.5) -> bool:
    """True when two lists of boxes match in class and coordinates."""
    if len(a) != len(b):
        return False
    for d1, d2 in zip(a, b):
        if d1.class_id != d2.class_id:
            return False
        if (abs(d1.x1 - d2.x1) > tol or abs(d1.y1 - d2.y1) > tol
                or abs(d1.x2 - d2.x2) > tol or abs(d1.y2 - d2.y2) > tol):
            return False
    return True


def _um_px_de(state, path_2, params):
    """Scale of that image as used by the run, with the manual fallback value.

    It is read from ``state.calibraciones`` and not from ``params.um_per_px``
    because the scale is per photo: re-reading the GT must not recompute the
    sizes with a factor different from the one the run used, or the report
    would say one thing and the boxes another.
    """
    cal = (getattr(state, "calibrations", None) or {}).get(Path(path_2).name)
    if cal is not None and getattr(cal, "valid", False):
        return cal.um_por_px
    return params.um_per_px if params.um_per_px > 0 else None


def reload_gt(state, progress=None) -> dict:
    """Re-reads the GT of each image and recomputes metrics and annotated images.

    It only touches the images whose .txt actually changed: comparing is much
    cheaper than redrawing a 3260 px plate, and in the normal case -- one
    plate out of sixty-nine was corrected -- the work reduces to that one.

    ``progreso`` is called as ``progreso(hechas, total, nombre)`` if passed;
    if it returns ``False`` the pass aborts and what was already recomputed is
    kept. Returns a summary with the images revisited, the ones that changed
    and the ones still without a .txt.
    """
    summary = {"revisadas": 0, "cambiadas": [], "sin_gt": [], "recalculadas": 0}
    if not state.results or state.run_dir is None:
        return summary

    params = state.params
    # Class names come from the first loaded model: the .txt stores only
    # the index, and without the map the boxes would have no readable class.
    names_2 = {}
    for slot in state.active_models():
        if slot.loaded is not None and slot.loaded.names:
            names_2 = slot.loaded.names
            break

    per_image: dict = {}
    for mi in sorted(state.results):
        for r in state.results[mi]:
            per_image.setdefault(r.image_path, []).append((mi, r))

    total = len(per_image)
    for made, (path_2, cluster) in enumerate(per_image.items(), start=1):
        if progress is not None and progress(made, total, path_2.name) is False:
            break
        summary["revisadas"] += 1

        gt_txt = find_gt_for_image(path_2, state.gt_folder)
        has_gt = gt_txt is not None
        if not has_gt:
            summary["sin_gt"].append(path_2.name)

        # The size is read from the header, without decoding the whole photo: it is
        # needed to denormalize the .txt, and in most images it is the
        # only thing that will be needed.
        wh = size_image(path_2)
        if wh is None:
            continue
        W, H = wh

        gts = []
        if has_gt:
            gts = read_yolo_txt(gt_txt, W, H, names_2)
            for d in gts:
                compute_box_size_um(
                    d, _um_px_de(state, r.image_path, params))

        first = cluster[0][1]
        if has_gt == first.has_gt and _same_boxes(gts, first.gt):
            continue
        summary["cambiadas"].append(path_2.name)

        img_bgr = cv2.imdecode(np.fromfile(str(path_2), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        try:
            idx_img = state.images.index(path_2)
        except ValueError:
            idx_img = made - 1

        for mi, r in cluster:
            r.gt = list(gts)
            r.has_gt = has_gt
            if has_gt:
                m = match_image(r.predictions, gts, iou_thr=params.iou_tp)
                r.tp, r.fp, r.fn, r.miscls = m.tp, m.fp, m.fn, m.miscls
            else:
                r.tp = r.fp = r.fn = r.miscls = 0

            # The predictions image does not change -- the predictions are the
            # same -- but the control and combined ones do.
            sub = state.run_dir / state.model_slots[mi].alias
            sub.mkdir(parents=True, exist_ok=True)
            base = f"{idx_img:04d}_{path_2.stem}"

            pred_img = _draw_annotated(
                img_bgr, r.predictions,
                color_for=lambda d: _color_bgr_for_class(d.class_name),
                label_for=lambda d: f"{d.class_name} {d.conf:.2f}",
            )
            if has_gt:
                combined = _draw_annotated(
                    pred_img, gts,
                    color_for=lambda d: (255, 200, 80),
                    label_for=lambda d: f"GT {d.class_name}",
                )
                gt_img = _draw_annotated(
                    img_bgr, gts,
                    color_for=lambda d: _color_bgr_for_class(d.class_name),
                    label_for=lambda d: f"{d.class_name}",
                )
            else:
                combined, gt_img = pred_img, None

            r.annotated_path = _save(combined, sub, base, "_annot")
            r.gt_path = _save(gt_img, sub, base, "_gt")
            summary["recalculadas"] += 1

    return summary


def _save(image, sub: Path, base: str, suffix_2: str):
    """Writes the PNG into the run folder and returns its path."""
    if image is None:
        return None
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return None
    destination = sub / f"{base}{suffix_2}.png"
    destination.write_bytes(bytes(buf))
    return destination
