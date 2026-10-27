"""Background inference thread. Processes all the images with all the models."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
from polyx.core.image_io import read_image
import numpy as np
from PySide6.QtCore import QThread, Signal

from polyx.core import observability as obs
from polyx.core.calibration import carry_index, resolver, search_index, summary_batch
from polyx.core.detect import make_detector
from polyx.core.detect.yolo import (
    Detection,
    compute_box_size_um,
    find_gt_for_image,
    read_yolo_txt,
)
from polyx.core.morphology import apply_to_detection
from polyx.core.provenance import provenance, sha256_file
from polyx.core.quality.assignment import mark_assignables
from polyx.core.quality.metrics import match_image
from polyx.gui import theme as T

from .state import DetectorState, ImageResult


def _draw_annotated(img_bgr: np.ndarray, dets, color_for, label_for) -> np.ndarray:
    """Draws the boxes with a line width proportional to the image size.

    With a fixed 2 px width the annotations became invisible on the full
    plates: a 4096 px photo is rescaled to 1100 to embed it in the report, so
    that line ended up measuring half a pixel. Scaling with the longer side, a
    plate is drawn with ~6 px (the same as paper/recomponer_placas.py) and a
    1630 px crop keeps its 2.
    """
    out = img_bgr.copy()
    side = max(out.shape[:2])
    thickness = max(2, int(round(side / 700)))  # noqa: RUF046
    scale_txt = max(0.5, side / 3000)
    thickness_txt = max(1, int(round(scale_txt * 2)))  # noqa: RUF046

    for d in dets:
        c = color_for(d)
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)
        label = label_for(d)
        # The number comes before everything else: it is what one looks for when coming
        # from the report table.
        if getattr(d, "number", None):
            label = f"{d.number}. {label}" if label else str(d.number)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale_txt, thickness_txt)
            margin = max(4, thickness * 2)
            cv2.rectangle(out, (x1, y1 - th - margin), (x1 + tw + margin, y1), c, -1)
            cv2.putText(
                out,
                label,
                (x1 + margin // 2, y1 - margin // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale_txt,
                (255, 255, 255),
                thickness_txt,
                cv2.LINE_AA,
            )
    return out


def _save_labels(dets, path_txt: Path, W: int, H: int) -> None:
    """Writes the predictions in YOLO format inside the run folder.

    Never next to the photo: that is where the manual-count .txt lives, and
    dumping the predictions on top would destroy the ground truth
    irrecoverably.

    The confidence goes as the sixth column. ``read_yolo_txt`` ignores the
    extra columns, so the file can be reopened in the Annotator as
    pre-annotation -- counting 500 particles by hand starting from the model's
    boxes is another task than doing it from zero -- and at the same time the
    record remains of with what confidence each box came out.
    """
    lines = []
    for d in dets:
        cx = ((d.x1 + d.x2) / 2) / W
        cy = ((d.y1 + d.y2) / 2) / H
        w = (d.x2 - d.x1) / W
        h = (d.y2 - d.y1) / H
        lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {d.conf:.4f}")
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    path_txt.write_text("\n".join(lines), encoding="utf-8")


def _color_bgr_for_class(name: str):
    hex_ = T.CLASS_COLOR_HEX.get(name, "#33aaff").lstrip("#")
    r, g, b = int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)
    return (b, g, r)


class DetectorRunner(QThread):
    progress = Signal(int, int, str)
    image_done = Signal(int, object)  # model_idx, ImageResult
    finished_ok = Signal()
    aborted = Signal()
    failed = Signal(str)

    def __init__(self, state: DetectorState, parent=None):
        super().__init__(parent)
        self.state = state

    def run(self):
        # Observability logic: run_id starts as None and is set when the
        # run folder exists; evento_error leaves the traceback that previously
        # was lost under pythonw.
        _log = obs.logger_for("detector")
        obs.set_run_id(None)
        try:
            state = self.state
            params = state.params
            slots = state.active_models()
            start = datetime.now()
            obs.event(
                _log,
                "run.config",
                placas=len(state.images),
                modelos=[s.alias for s in slots],
                conf=params.conf,
                iou=params.iou_nms,
                imgsz=params.imgsz,
                device=params.device,
                medir_forma=params.measure_shape,
                um_por_px=params.um_per_px,
            )
            # What was actually used, image by image, versus what was
            # requested on the `Parametros` page: the auto-fallback may have lowered imgsz
            # for lack of VRAM, and tiling depends on each photo's size.
            # Without recording it, the report declares a resolution that may not
            # be the one that produced those boxes.
            execution: dict[str, dict] = {}

            # The scale is resolved per image, not once for the batch. The
            # index is read only once: it has 552 rows, and re-reading it per photo would
            # waste time.
            index_2 = carry_index(params.index_calibration) if params.index_calibration else {}
            if not index_2 and state.images:
                index_2 = search_index(state.images[0])
            state.calibrations = {}
            # Particles that could not be segmented, per model. It goes into the
            # summary: a size measured on the box is not equivalent to
            # one measured on the mask, and it must be possible to say how many there are.
            shapeless: dict[str, int] = {}
            if not slots:
                self.failed.emit("No hay modelos cargados.")
                return
            if not state.images:
                self.failed.emit("No hay imágenes seleccionadas.")
                return

            # Create run folder
            run_dir = Path(__file__).resolve().parents[4] / "runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state.run_dir = run_dir / f"detect_{stamp}"
            state.run_dir.mkdir(parents=True, exist_ok=True)
            obs.set_run_id(stamp)

            # Load models
            for s in slots:
                if s.loaded is None:
                    s.loaded = make_detector(s.model_type or "yolo", str(s.path), alias=s.alias)
                try:
                    s.loaded.load()
                    obs.event(_log, "model.loaded", alias=s.alias, file=str(s.path))
                except Exception as exc:
                    obs.event_error(_log, "model.error", exc, alias=s.alias, file=str(s.path))
                    raise

            # Reset results
            state.results = {i: [] for i in range(len(state.model_slots))}
            total = len(state.images) * len(slots)
            done = 0

            for idx_img, img_path in enumerate(state.images):
                if state.consume_abort():
                    self.aborted.emit()
                    return

                # Load the image once for GT/annotation
                img_bgr = read_image(img_path)
                if img_bgr is None:
                    obs.event(_log, "image.undecodable", level=logging.WARNING, file=str(img_path))
                    done += len(slots)
                    self.progress.emit(done, total, img_path.name)
                    continue
                H, W = img_bgr.shape[:2]

                # Scale of THIS photo. The already-read image is passed in so as not to
                # decode 12 MP again just to measure the ring.
                cal = resolver(
                    img_path,
                    index=index_2,
                    um_por_px_manual=params.um_per_px,
                    diameter_mm=params.diameter_plate_mm,
                    measure_plate=params.measure_plate,
                    bgr=img_bgr,
                    height_plate_mm=params.height_plate_mm,
                    distance_camera_mm=params.distance_camera_mm,
                )
                state.calibrations[img_path.name] = cal
                um_px = cal.um_por_px if cal.valid else None
                if not cal.valid:
                    obs.event(
                        _log,
                        "calibration.invalid",
                        level=logging.WARNING,
                        file=str(img_path),
                        reason=getattr(cal, "reason", None),
                    )

                # Load GT if it exists (uses first-model names when available)
                names_first = slots[0].loaded.names if slots[0].loaded else {}
                gt_txt = find_gt_for_image(img_path, state.gt_folder)
                gts: list[Detection] = []
                if gt_txt is not None:
                    gts = read_yolo_txt(gt_txt, W, H, names_first)
                    # The GT size does not depend on the model → it is computed once
                    for d in gts:
                        compute_box_size_um(d, um_px)
                        if params.measure_shape:
                            apply_to_detection(d, img_bgr, um_px)

                for mi_idx, slot in enumerate(slots):  # noqa: B007
                    # mi_idx is the order within active_models()
                    # we need the real slot index within model_slots
                    real_idx = state.model_slots.index(slot)
                    plan_tiling: dict = {}
                    preds = slot.loaded.predict_auto(
                        str(img_path),
                        conf=params.conf,
                        iou=params.iou_nms,
                        imgsz=params.imgsz,
                        device=params.device,
                        tiling=params.tiling,
                        threshold_px=params.tiling_threshold_px,
                        tile=params.tiling_tile,
                        overlap=params.tiling_overlap,
                        record=plan_tiling,
                    )
                    # The decision is saved, not logged in passing: the report
                    # must be able to declare whether that photo was counted as tiled.
                    if plan_tiling.get("troceado"):
                        p = plan_tiling["plan"]
                        tiling_desc = (
                            f"{plan_tiling['ancho']}×{plan_tiling['alto']} px → "
                            f"{p['n_tiles']} tiles de {p['tile']} px, "
                            f"{int(p['overlap'] * 100)}% solape"
                        )
                    else:
                        tiling_desc = ""

                    reg = execution.setdefault(
                        slot.alias,
                        {"imagenes": 0, "troceadas": 0, "imgsz_usados": set(), "fallback_en": []},
                    )
                    reg["imagenes"] += 1
                    if plan_tiling.get("troceado"):
                        reg["troceadas"] += 1
                        # The tile enters the network at its native resolution; the
                        # requested imgsz is not applied to the full photo.
                        reg["imgsz_usados"].add(int(plan_tiling["plan"]["tile"]))
                    else:
                        # last_imgsz/last_fallback are only written by predict(),
                        # which is precisely the untiled path.
                        used_2 = getattr(slot.loaded, "last_imgsz", 0)
                        if used_2:
                            reg["imgsz_usados"].add(int(used_2))
                        if getattr(slot.loaded, "last_fallback", False):
                            reg["fallback_en"].append(img_path.name)

                    # Compute the prediction sizes (GT was already computed above)
                    for d in preds:
                        # The box first, as a fallback: if segmentation does not
                        # manage to isolate the particle, an approximate size -- noted
                        # as such -- is preferable to losing it.
                        compute_box_size_um(d, um_px)
                        if params.measure_shape and not apply_to_detection(d, img_bgr, um_px):
                            shapeless[slot.alias] = shapeless.get(slot.alias, 0) + 1

                    # Size filter
                    if um_px and (params.size_min_um > 0 or params.size_max_um > 0):
                        keep = []
                        for d in preds:
                            if d.diam_um is None:
                                keep.append(d)
                                continue
                            if params.size_min_um > 0 and d.diam_um < params.size_min_um:
                                continue
                            if params.size_max_um > 0 and d.diam_um > params.size_max_um:
                                continue
                            keep.append(d)
                        preds = keep

                    # Abstention: mark which ones support their polymer. It goes AFTER
                    # the size filter and BEFORE matching against GT, because it does not
                    # discard particles -- it only decides whether to assign them a
                    # polymer or report them as «not assignable».
                    mark_assignables(preds, params.conf_assignment)

                    # Number from top to bottom and left to right, not in
                    # the order in which the model returned them: this way number 1
                    # is at the very top, and finding number 37 in the photo is viable.
                    for _i, _d in enumerate(
                        sorted(preds, key=lambda z: (round(z.y1 / 40), z.x1)), 1
                    ):
                        _d.number = _i

                    # Match against GT
                    # The .txt exists = there is ground truth, even with no
                    # boxes: a reviewed plate with zero particles is a
                    # datum, not an unreviewed plate. With bool(gts) those plates
                    # were left out of the metrics, and the detections the
                    # model made on them did not count as false positives.
                    has_gt = gt_txt is not None
                    if has_gt:
                        m = match_image(preds, gts, iou_thr=params.iou_tp)
                        tp, fp, fn, mc = m.tp, m.fp, m.fn, m.miscls
                    else:
                        tp = fp = fn = mc = 0

                    # ── Annotated images ──────────────────────────────
                    # 1) Model predictions (YOLO boxes in class color).
                    pred_img = _draw_annotated(
                        img_bgr,
                        preds,
                        color_for=lambda d: _color_bgr_for_class(d.class_name),
                        label_for=lambda d: f"{d.class_name} {d.conf:.2f}",
                    )
                    # 2) Combined (prediction + GT in cyan) for preview and
                    #    error inspection. With no GT, it matches pred.
                    combined = pred_img
                    gt_img = None
                    if has_gt:
                        combined = _draw_annotated(
                            pred_img,
                            gts,
                            color_for=lambda d: (255, 200, 80),
                            label_for=lambda d: f"GT {d.class_name}",
                        )
                        # 3) Ground Truth only (real boxes in class color)
                        #    over the original image → side-by-side view.
                        gt_img = _draw_annotated(
                            img_bgr,
                            gts,
                            color_for=lambda d: _color_bgr_for_class(d.class_name),
                            label_for=lambda d: f"{d.class_name}",
                        )

                    # ── Save PNGs in run_dir ────────────────────────
                    sub = state.run_dir / slot.alias
                    sub.mkdir(parents=True, exist_ok=True)

                    def _encode_and_save(image, suffix: str):
                        """Encodes to PNG, saves it to disk and returns its path.

                        Returns the path and not the bytes on purpose: in a
                        large batch holding every PNG in memory reached several
                        GB, and the file is already written right here anyway.
                        """
                        if image is None:
                            return None
                        is_ok, buf = cv2.imencode(".png", image)
                        if not is_ok:
                            return None
                        # The name carries the image index because two photos
                        # from different folders can share a name (1.1.jpg
                        # exists in several controls) and would overwrite each other.
                        destination = sub / f"{idx_img:04d}_{img_path.stem}{suffix}.png"  # noqa: B023
                        destination.write_bytes(bytes(buf))
                        return destination

                    annotated_path = _encode_and_save(combined, "_annot")
                    pred_path = _encode_and_save(pred_img, "_pred")
                    gt_path = _encode_and_save(gt_img, "_gt") if has_gt else None

                    # The boxes in text, not only drawn: a PNG cannot be
                    # re-analyzed or reopened for correction.
                    _save_labels(preds, sub / "labels" / f"{idx_img:04d}_{img_path.stem}.txt", W, H)

                    res = ImageResult(
                        image_path=img_path,
                        model_idx=real_idx,
                        predictions=preds,
                        gt=gts,
                        has_gt=has_gt,
                        tp=tp,
                        fp=fp,
                        fn=fn,
                        miscls=mc,
                        annotated_path=annotated_path,
                        pred_path=pred_path,
                        gt_path=gt_path,
                        tiling=tiling_desc,
                    )
                    state.results.setdefault(real_idx, []).append(res)
                    self.image_done.emit(real_idx, res)

                    done += 1
                    self.progress.emit(done, total, img_path.name)

            ms = int((datetime.now() - start).total_seconds() * 1000)
            obs.event(
                _log,
                "run.finish",
                ms=ms,
                placas=len(state.images),
                imgsz_usados=sorted(
                    {int(s) for reg in execution.values() for s in reg.get("imgsz_usados", [])}
                ),
                sin_forma=dict(shapeless),
            )
            # A batch with many shapeless particles deserves a look: it may be
            # the model, the threshold, or out-of-focus photos, and it is not shown in the UI.
            worst = max(shapeless.items(), key=lambda kv: kv[1], default=())
            if worst and worst[1] > 0:
                obs.event(
                    _log,
                    "run.shapeless_high",
                    level=logging.WARNING,
                    modelo=worst[0],
                    sin_forma=worst[1],
                )
            self.finished_ok.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
            obs.event_error(_log, "run.error", e)

    def _write_summary(self, state, slots, start=None, execution=None, shapeless=None) -> None:
        """Leaves the run self-describing: provenance, parameters and totals per model.

        Without this a run folder is loose PNGs and there is no way to know
        with what confidence or with which weights they came out. The run is
        not failed if the dump cannot be written: the results are already in
        memory and on disk.
        """
        p = state.params
        execution = execution or {}
        fin = datetime.now()
        data = {
            # Who produced this: version, commit and libraries. It goes first because
            # it is what one checks when trying to reproduce a table from the paper.
            "procedencia": provenance(),
            "cuando": {
                "inicio": start.isoformat(timespec="seconds") if start else "",
                "fin": fin.isoformat(timespec="seconds"),
                "duracion_s": round((fin - start).total_seconds(), 1) if start else 0.0,
            },
            "parametros": {
                "confianza": p.conf,
                "iou_nms": p.iou_nms,
                "iou_tp": p.iou_tp,
                "imgsz": p.imgsz,
                "device": p.device,
                "um_por_px": p.um_per_px,
                "troceo": p.tiling,
                "troceo_umbral_px": p.tiling_threshold_px,
            },
            # What standard the sizes were measured against. A value in um without this
            # cannot be verified or reproduced.
            "calibracion": summary_batch(list(getattr(state, "calibrations", {}).values())),
            "imagenes": len(state.images),
            "modelos": [],
        }
        for slot in slots:
            mi = state.model_slots.index(slot)
            rs = state.results.get(mi, [])
            con_gt = [r for r in rs if r.has_gt]
            tp = sum(r.tp for r in con_gt)
            fp = sum(r.fp for r in con_gt)
            fn = sum(r.fn for r in con_gt)
            mc = sum(r.miscls for r in con_gt)
            reg = execution.get(slot.alias, {})
            data["modelos"].append(
                {
                    "alias": slot.alias,
                    "peso": str(slot.path) if slot.path else "",
                    # Two best.pt from two different trainings are named the same;
                    # the fingerprint is the only thing that truly distinguishes them.
                    "sha256": sha256_file(slot.path) if slot.path else "",
                    "detecciones": sum(len(r.predictions) for r in rs),
                    "imagenes_con_gt": len(con_gt),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "mal_clasificados": mc,
                    "imgsz_usados": sorted(reg.get("imgsz_usados", ())),
                    "imagenes_troceadas": reg.get("troceadas", 0),
                    # If this list is not empty, those photos were inferred at lower
                    # resolution than requested because the GPU ran out of memory.
                    "fallback_por_memoria": reg.get("fallback_en", []),
                    "particulas_sin_forma": (shapeless or {}).get(slot.alias, 0),
                }
            )
            # classes.txt next to the labels, so they can be reopened as
            # pre-annotation in the Etiquetador without guessing the class order.
            names_2 = getattr(slot.loaded, "names", None) or {}
            if names_2:
                destination = state.run_dir / slot.alias / "classes.txt"
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(
                        "\n".join(names_2[k] for k in sorted(names_2)), encoding="utf-8"
                    )
                except OSError:
                    pass
        try:  # noqa: SIM105
            (state.run_dir / "resumen.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass
