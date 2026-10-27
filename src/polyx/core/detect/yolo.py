"""Thin wrapper over Ultralytics YOLO for inference + YOLO-txt utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from polyx.core.detect.protocol import (
    Instance,
    _tile_starts,
    policy_tiling,
    size_image,
)
from polyx.core.image_io import read_image
from polyx.core.provenance.observability import event, logger_for

# Canonical alias: `Detection` is `Instance` from the protocol. Kept for
# backwards compatibility with GUI/report code that imports `Detection`.
Detection = Instance


def _is_oom(e: Exception) -> bool:
    """Is the exception caused by GPU memory exhaustion?"""
    try:
        import torch

        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001, S110
        pass
    return "out of memory" in str(e).lower()


def _cuda_empty_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001, S110
        pass


def _log():
    """Module logger, resolved late so importing does not require write access."""
    return logger_for("detector")


class YoloModel:
    """Lazy loading of Ultralytics YOLO. Allows several models in parallel."""

    # Descending ladder for auto-fallback when the GPU runs out of
    # memory: instead of failing, it retries with the next smaller size.
    FALLBACK_SIZES = (8192, 7168, 6144, 5120, 4096, 3008, 2048, 1536, 1280, 1024, 640)

    # Detection cap per pass. Ultralytics brings 300, which here truncates
    # silently: an estuary plate exceeds 500 particles, and at low
    # confidence the model proposes quite a few more. Trimming the surplus would lose
    # precisely the lowest-confidence boxes without warning, so a threshold
    # sweep would no longer match a run made at that threshold.
    MAX_DET = 5000

    def __init__(self, weights_path: str | Path, alias: str = ""):
        self.weights_path = str(weights_path)
        self.alias = alias or Path(self.weights_path).stem
        self._model = None
        self.names: dict[int, str] = {}
        # imgsz actually used in the last inference (may be smaller than
        # requested if auto-fallback ran due to GPU memory exhaustion)
        self.last_imgsz: int = 0
        self.last_fallback: bool = False

    def load(self):
        if self._model is not None:
            return self
        from ultralytics import YOLO

        self._model = YOLO(self.weights_path)
        nm = getattr(self._model, "names", None) or {}
        self.names = dict(nm)
        return self

    def probe_max_imgsz(
        self,
        image_path: str | Path,
        device: str = "0",
        candidates: tuple[int, ...] = (2048, 3008, 4096, 5120, 6144, 7168, 8192),
        progress=None,
    ) -> tuple[int, dict[int, str]]:
        """Finds the largest imgsz the GPU can take for THIS model.

        Runs real inference at increasing sizes until it fails on memory.
        Returns (max_ok, detalle) where detalle is {imgsz: "ok"/"oom"/"err"}.
        On CPU there is no VRAM cap → returns the largest candidate untested.
        """
        self.load()
        assert self._model is not None
        detail: dict[int, str] = {}
        dev = str(device).strip().lower()
        if dev in ("cpu", "-1"):
            return candidates[-1], dict.fromkeys(candidates, "cpu")
        try:
            import torch
        except Exception:  # noqa: BLE001
            return 1920, {}
        max_ok = 0
        for sz in candidates:
            if progress is not None:
                progress(sz)
            try:
                self._model.predict(
                    source=str(image_path),
                    imgsz=sz,
                    conf=0.25,
                    device=device,
                    verbose=False,
                    save=False,
                )
                detail[sz] = "ok"
                max_ok = sz
            except torch.cuda.OutOfMemoryError:
                detail[sz] = "oom"
                break
            except Exception as e:  # noqa: BLE001
                detail[sz] = f"err: {type(e).__name__}"
                break
            finally:
                try:  # noqa: SIM105
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001, S110
                    pass
        return (max_ok or 1920), detail

    def _predict_with_fallback(
        self,
        image_path,
        conf,
        iou,
        imgsz,
        device,
        auto_fallback,
        task=None,
    ):
        """Run inference with VRAM auto-fallback. Returns the raw Ultralytics result list."""
        self.load()
        assert self._model is not None
        sizes = [int(imgsz)]
        if auto_fallback:
            sizes += [s for s in self.FALLBACK_SIZES if s < int(imgsz)]
        res = None
        for i, sz in enumerate(sizes):
            try:
                kwargs: dict[str, Any] = {
                    "source": str(image_path),
                    "conf": conf,
                    "iou": iou,
                    "imgsz": sz,
                    "device": device,
                    "verbose": False,
                    "save": False,
                    "max_det": self.MAX_DET,
                }
                if task is not None:
                    kwargs["task"] = task
                res = self._model.predict(**kwargs)
                self.last_imgsz = sz
                self.last_fallback = i > 0
                break
            except Exception as e:
                if not auto_fallback or i == len(sizes) - 1 or not _is_oom(e):
                    raise
                event(
                    _log(),
                    "imgsz.fallback",
                    level=logging.WARNING,
                    alias=self.alias,
                    file=str(image_path),
                    tried_previous=sz,
                    imgsz_next=(sizes[i + 1] if i + 1 < len(sizes) else None),
                    reason="gpu_oom",
                )
                _cuda_empty_cache()
        return res

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "0",
        auto_fallback: bool = True,
        **kwargs,
    ) -> list[Detection]:
        res = self._predict_with_fallback(image_path, conf, iou, imgsz, device, auto_fallback)
        out: list[Detection] = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):  # noqa: B905
            out.append(
                Detection(
                    class_id=int(cls),
                    class_name=self.names.get(int(cls), str(int(cls))),
                    conf=float(c),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        return out

    def predict_sliced(
        self,
        image_path: str | Path,
        conf: float = 0.10,
        iou: float = 0.45,
        tile: int = 1024,
        overlap: float = 0.25,
        device: str = "0",
        agnostic_nms: bool = True,
        imgsz: int | None = None,
        batch: int = 8,
        **kwargs,
    ) -> list[Detection]:
        """Tile-based inference (SAHI / Slicing Aided Hyper Inference).

        Splits the image into overlapping ``tile``×``tile`` px crops, infers
        each one at the tile's native resolution and reprojects the boxes
        onto the full image. Meant for tiny particles on high-resolution
        photos, where a single pass at a low imgsz would make them disappear.

        The boxes of all tiles are merged with a global NMS
        (``agnostic_nms`` suppresses duplicates even when their classes
        differ, useful when the detection matters more than the class).
        """
        self.load()
        assert self._model is not None
        img = read_image(image_path)
        if img is None:
            return []
        H, W = img.shape[:2]
        overlap = min(max(overlap, 0.0), 0.9)
        step = max(1, int(round(tile * (1.0 - overlap))))  # noqa: RUF046
        xs = _tile_starts(W, tile, step)
        ys = _tile_starts(H, tile, step)

        crops: list[np.ndarray] = []
        origins: list[tuple[int, int]] = []
        for y0 in ys:
            for x0 in xs:
                crops.append(img[y0 : y0 + tile, x0 : x0 + tile])
                origins.append((x0, y0))
        if not crops:
            return []

        # imgsz controls the tile input resolution. If it is larger than the
        # tile, the crop is upscaled -> tiny particles are enlarged towards
        # the scale at which the model was trained (key for small objects).
        infer_sz = int(imgsz) if imgsz else tile

        dets: list[Detection] = []
        bs = max(1, int(batch))
        # Run in sub-batches to avoid exhausting VRAM (many tiles at high imgsz).
        for start in range(0, len(crops), bs):
            chunk = crops[start : start + bs]
            chunk_org = origins[start : start + bs]
            results = self._model.predict(
                source=chunk,
                conf=conf,
                iou=iou,
                imgsz=infer_sz,
                device=device,
                verbose=False,
                save=False,
                max_det=self.MAX_DET,
            )
            for (x0, y0), r in zip(chunk_org, results):  # noqa: B905
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                for (a, b, c, d), cc, k in zip(xyxy, cf, cl):  # noqa: B905
                    dets.append(
                        Detection(
                            class_id=int(k),
                            class_name=self.names.get(int(k), str(int(k))),
                            conf=float(cc),
                            x1=float(a) + x0,
                            y1=float(b) + y0,
                            x2=float(c) + x0,
                            y2=float(d) + y0,
                        )
                    )
        return _nms(dets, iou_thr=iou, agnostic=agnostic_nms)

    def predict_auto(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1280,
        device: str = "0",
        tiling: str = "auto",
        threshold_px: int = 2000,
        tile: int = 0,
        overlap: float = 0.25,
        batch: int = 8,
        agnostic_nms: bool = True,
        record: dict | None = None,
    ) -> list[Detection]:
        """Infers with or without tiling according to the photo's size, hands-free.

        ``troceo``: ``"auto"`` decides on the longer side against
        ``umbral_px``; ``"siempre"`` forces tiling; ``"nunca"`` forces the
        single pass.

        If ``registro`` is passed, it is filled with the decision taken so it
        can be declared in the report: counting 500 particles with tiling or
        without it is not the same method, and the number does not mean the
        same thing.
        """
        wh = size_image(image_path)
        plan = None
        if tiling == "nunca" or wh is None:
            plan = None
        elif tiling == "siempre":
            plan = policy_tiling(wh[0], wh[1], imgsz, threshold_px=0, tile=tile, overlap=overlap)
        else:
            plan = policy_tiling(
                wh[0], wh[1], imgsz, threshold_px=threshold_px, tile=tile, overlap=overlap
            )

        if record is not None:
            record.clear()
            record.update(
                {
                    "ancho": wh[0] if wh else 0,
                    "alto": wh[1] if wh else 0,
                    "troceado": plan is not None,
                    "plan": plan,
                }
            )

        if plan is None:
            return self.predict(image_path, conf=conf, iou=iou, imgsz=imgsz, device=device)
        return self.predict_sliced(
            image_path,
            conf=conf,
            iou=iou,
            tile=plan["tile"],
            overlap=plan["overlap"],
            device=device,
            agnostic_nms=agnostic_nms,
            imgsz=imgsz,
            batch=batch,
        )


def _box_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: list[Detection], iou_thr: float = 0.45, agnostic: bool = True) -> list[Detection]:
    """Greedy NMS over the detections of all tiles. Keeps the one with the highest conf."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i].conf)
    keep: list[int] = []
    suppressed = [False] * len(dets)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        for jdx in order:
            if jdx == idx or suppressed[jdx]:
                continue
            if not agnostic and dets[jdx].class_id != dets[idx].class_id:
                continue
            if _box_iou(dets[idx], dets[jdx]) >= iou_thr:
                suppressed[jdx] = True
    return [dets[i] for i in keep]


class YoloSegModel(YoloModel):
    """YOLO segmentation backend (task='segment').

    Wraps an Ultralytics YOLO-seg model and returns Instance objects that
    carry a boolean mask in addition to the box and class.
    """

    def __init__(self, weights_path: str | Path, alias: str = ""):
        super().__init__(weights_path, alias=alias)
        self._seg_task = "segment"

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "0",
        auto_fallback: bool = True,
        **kwargs,
    ) -> list[Detection]:
        res = self._predict_with_fallback(
            image_path, conf, iou, imgsz, device, auto_fallback, task="segment"
        )
        out: list[Detection] = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        masks = None
        if getattr(r, "masks", None) is not None and r.masks is not None:
            try:
                masks = r.masks.data.cpu().numpy()
            except Exception:  # noqa: BLE001
                masks = None
        for idx, ((x1, y1, x2, y2), c, cls) in enumerate(zip(xyxy, confs, clss)):  # noqa: B905
            mask = None
            mask_source = "yolo-seg"
            if masks is not None and idx < len(masks):
                m = masks[idx]
                mask = m > 0.5
            out.append(
                Detection(
                    class_id=int(cls),
                    class_name=self.names.get(int(cls), str(int(cls))),
                    conf=float(c),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    mask=mask,
                    mask_source=mask_source if mask is not None else None,
                )
            )
        return out
