"""Background inference thread. Processes all the images with all the models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from polyx.core import observability as obs
from polyx.core.detect import make_detector
from polyx.core.detect.yolo import (
    Detection,
    compute_box_size_um,
    find_gt_for_image,
    read_yolo_txt,
)
from polyx.core.image_io import read_image
from polyx.core.measure.calibration import carry_index, resolver, search_index, summary_batch
from polyx.core.measure.morphology import apply_to_detection
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


@dataclass
class _ImageContext:
    """One decoded image plus what is derived once per image, not per model."""

    img_path: Path
    idx_img: int
    bgr: np.ndarray
    height: int
    width: int
    um_px: float | None
    gt_txt: Path | None
    gts: list


class DetectorRunner(QThread):
    progress = Signal(int, int, str)
    image_done = Signal(int, object)  # model_idx, ImageResult
    finished_ok = Signal()
    aborted = Signal()
    failed = Signal(str)

    def __init__(self, state: DetectorState, parent=None):
        super().__init__(parent)
        self.state = state

    def _load_models(self, slots, _log) -> None:
        """Lazily resolve and load every active model. Raises on the first failure."""
        for s in slots:
            if s.loaded is None:
                s.loaded = make_detector(s.model_type or "yolo", str(s.path), alias=s.alias)
            try:
                s.loaded.load()
                obs.event(_log, "model.loaded", alias=s.alias, file=str(s.path))
            except Exception as exc:
                obs.event_error(_log, "model.error", exc, alias=s.alias, file=str(s.path))
                raise

    def _tiling_description(self, plan_tiling: dict) -> str:
        """Human-readable tiling plan for one image, or '' when not tiled."""
        if not plan_tiling.get("troceado"):
            return ""
        p = plan_tiling["plan"]
        return (
            f"{plan_tiling['ancho']}×{plan_tiling['alto']} px → "
            f"{p['n_tiles']} tiles de {p['tile']} px, "
            f"{int(p['overlap'] * 100)}% solape"
        )

    def _record_execution(self, reg: dict, plan_tiling: dict, slot, img_path) -> None:
        """Update the per-model execution bookkeeping for one image.

        The decision is recorded, not logged in passing: the report must be
        able to declare whether that photo was counted as tiled.
        """
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

    def _annotate_images(self, img_bgr, preds, gts, has_gt):
        """Render the three annotated views: predictions, predictions+GT, GT only."""
        # 1) Model predictions (YOLO boxes in class color).
        pred_img = _draw_annotated(
            img_bgr,
            preds,
            color_for=lambda d: _color_bgr_for_class(d.class_name),
            label_for=lambda d: f"{d.class_name} {d.conf:.2f}",
        )
        # 2) Combined (prediction + GT in cyan) for preview and error inspection.
        #    With no GT, it matches pred.
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
        return pred_img, combined, gt_img

    def _save_outputs(
        self, state, slot, idx_img, img_path, W, H, preds, combined, pred_img, gt_img, has_gt
    ):
        """Write the PNGs and the YOLO labels of one image, and return their paths.

        Returns the path and not the bytes on purpose: in a large batch holding
        every PNG in memory reached several GB, and the file is already written
        right here anyway. The name carries the image index because two photos
        from different folders can share a name (1.1.jpg exists in several
        controls) and would overwrite each other.
        """
        sub = state.run_dir / slot.alias
        sub.mkdir(parents=True, exist_ok=True)

        def _encode_and_save(image, suffix: str):
            if image is None:
                return None
            is_ok, buf = cv2.imencode(".png", image)
            if not is_ok:
                return None
            destination = sub / f"{idx_img:04d}_{img_path.stem}{suffix}.png"
            destination.write_bytes(bytes(buf))
            return destination

        annotated_path = _encode_and_save(combined, "_annot")
        pred_path = _encode_and_save(pred_img, "_pred")
        gt_path = _encode_and_save(gt_img, "_gt") if has_gt else None

        # The boxes in text, not only drawn: a PNG cannot be
        # re-analyzed or reopened for correction.
        _save_labels(preds, sub / "labels" / f"{idx_img:04d}_{img_path.stem}.txt", W, H)
        return annotated_path, pred_path, gt_path

    def _filter_by_size(self, preds, um_px, params):
        """Drop particles outside the configured μm range (no filter when both bounds are 0)."""
        if not um_px or (params.size_min_um <= 0 and params.size_max_um <= 0):
            return preds
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
        return keep

    def _measure_detections(self, dets, img_bgr, um_px, params) -> int:
        """Measure every detection: box size first, then real shape if enabled.

        The box size comes first as a fallback: if segmentation does not manage
        to isolate the particle, an approximate size -- noted as such -- is
        preferable to losing it.

        Returns how many particles had no measurable shape. Ground truth goes
        through here too (it does not depend on the model), where the count is
        simply not used.
        """
        failures = 0
        for d in dets:
            compute_box_size_um(d, um_px)
            if params.measure_shape and not apply_to_detection(d, img_bgr, um_px):
                failures += 1
        return failures

    def _score_predictions(self, preds, img_bgr, um_px, params, slot, shapeless):
        """Size, filter, abstain and number one model's predictions for one image.

        Order matters: abstention goes AFTER the size filter and BEFORE matching
        against GT, because it does not discard particles -- it only decides
        whether to assign them a polymer or report them as «not assignable».
        """
        # Compute the prediction sizes (GT was already computed above)
        failures = self._measure_detections(preds, img_bgr, um_px, params)
        if failures:
            shapeless[slot.alias] = shapeless.get(slot.alias, 0) + failures

        # Size filter
        preds = self._filter_by_size(preds, um_px, params)

        mark_assignables(preds, params.conf_assignment)

        # Number from top to bottom and left to right, not in
        # the order in which the model returned them: this way number 1
        # is at the very top, and finding number 37 in the photo is viable.
        for _i, _d in enumerate(sorted(preds, key=lambda z: (round(z.y1 / 40), z.x1)), 1):
            _d.number = _i
        return preds

    def _match_gt(self, preds, gts, has_gt, params) -> tuple[int, int, int, int]:
        """Match predictions against ground truth: (tp, fp, fn, misclassified).

        The .txt existing = there is ground truth, even with no boxes: a
        reviewed plate with zero particles is a datum, not an unreviewed
        plate. With ``bool(gts)`` those plates were left out of the metrics,
        and the detections the model made on them did not count as false
        positives.
        """
        if not has_gt:
            return 0, 0, 0, 0
        m = match_image(preds, gts, iou_thr=params.iou_tp)
        return m.tp, m.fp, m.fn, m.miscls

    def _prepare_image(self, state, img_path, idx_img, slots, params, index_2, _log):
        """Decode, calibrate and load the ground truth of ONE image.

        Returns ``None`` when the image cannot be decoded, otherwise an
        :class:`_ImageContext` with everything every model of this image needs.
        """
        # Load the image once for GT/annotation
        img_bgr = read_image(img_path)
        if img_bgr is None:
            obs.event(_log, "image.undecodable", level=logging.WARNING, file=str(img_path))
            return None
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
            self._measure_detections(gts, img_bgr, um_px, params)

        return _ImageContext(
            img_path=img_path,
            idx_img=idx_img,
            bgr=img_bgr,
            height=H,
            width=W,
            um_px=um_px,
            gt_txt=gt_txt,
            gts=gts,
        )

    def _create_run_dir(self, state) -> str:
        """Create this run's folder under ``<project>/runs`` and return its stamp.

        The stamp is what ``obs.set_run_id`` correlates every event of the
        batch with.
        """
        run_dir = Path(__file__).resolve().parents[4] / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state.run_dir = run_dir / f"detect_{stamp}"
        state.run_dir.mkdir(parents=True, exist_ok=True)
        return stamp

    def _process_image(
        self,
        state,
        img_path,
        idx_img,
        slots,
        params,
        index_2,
        execution,
        shapeless,
        total,
        done_start,
        _log,
    ) -> int:
        """Decode, calibrate and run every model on ONE image.

        Emits ``image_done``/``progress`` per model and returns how many
        models were processed, so the caller can advance its batch counter.
        """
        # ``done_start`` is the caller's cumulative count for the batch.
        done = done_start

        ctx = self._prepare_image(state, img_path, idx_img, slots, params, index_2, _log)
        if ctx is None:
            done += len(slots)
            self.progress.emit(done, total, img_path.name)
            return len(slots)

        for slot in slots:
            res = self._process_model(state, ctx, slot, params, execution, shapeless)
            state.results.setdefault(res.model_idx, []).append(res)
            self.image_done.emit(res.model_idx, res)

            done += 1
            self.progress.emit(done, total, img_path.name)

        return len(slots)

    def _process_model(self, state, ctx, slot, params, execution, shapeless) -> ImageResult:
        """Run ONE model on one already-decoded image and build its result."""
        # ``slot`` is the order within active_models(); the result needs the
        # real slot index within model_slots.
        real_idx = state.model_slots.index(slot)
        plan_tiling: dict = {}
        preds = slot.loaded.predict_auto(
            str(ctx.img_path),
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
        tiling_desc = self._tiling_description(plan_tiling)

        reg = execution.setdefault(
            slot.alias,
            {"imagenes": 0, "troceadas": 0, "imgsz_usados": set(), "fallback_en": []},
        )
        self._record_execution(reg, plan_tiling, slot, ctx.img_path)

        # Size, filter, abstention and numbering (GT was measured above)
        preds = self._score_predictions(preds, ctx.bgr, ctx.um_px, params, slot, shapeless)

        # Match against GT
        has_gt = ctx.gt_txt is not None
        tp, fp, fn, mc = self._match_gt(preds, ctx.gts, has_gt, params)

        # ── Annotated images ──────────────────────────────
        pred_img, combined, gt_img = self._annotate_images(ctx.bgr, preds, ctx.gts, has_gt)

        # ── Save PNGs in run_dir ────────────────────────
        annotated_path, pred_path, gt_path = self._save_outputs(
            state,
            slot,
            ctx.idx_img,
            ctx.img_path,
            ctx.width,
            ctx.height,
            preds,
            combined,
            pred_img,
            gt_img,
            has_gt,
        )

        return ImageResult(
            image_path=ctx.img_path,
            model_idx=real_idx,
            predictions=preds,
            gt=ctx.gts,
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

    def _finalize_run(self, state, slots, start, execution, shapeless, _log) -> None:
        """Emit the closing observability events and signal success."""
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
            stamp = self._create_run_dir(state)
            obs.set_run_id(stamp)

            # Load models
            self._load_models(slots, _log)

            # Reset results
            state.results = {i: [] for i in range(len(state.model_slots))}
            total = len(state.images) * len(slots)
            done = 0

            for idx_img, img_path in enumerate(state.images):
                if state.consume_abort():
                    self.aborted.emit()
                    return

                done += self._process_image(
                    state,
                    img_path,
                    idx_img,
                    slots,
                    params,
                    index_2,
                    execution,
                    shapeless,
                    total,
                    done,
                    _log,
                )

            self._finalize_run(state, slots, start, execution, shapeless, _log)
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
