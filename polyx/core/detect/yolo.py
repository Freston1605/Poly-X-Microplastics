"""Thin wrapper over Ultralytics YOLO for inference + YOLO-txt utilities."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import os

import logging

import numpy as np
import cv2

from polyx.core.provenance.observability import logger_for, event
from polyx.core.detect.protocol import Instance, DetectionSource, Detections, size_image, policy_tiling, read_yolo_txt, find_gt_for_image, compute_box_size_um

@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    # box in absolute pixel coordinates (x1,y1,x2,y2)
    x1: float
    y1: float
    x2: float
    y2: float
    # equivalent diameter (μm) if calibrated
    diam_um: Optional[float] = None
    area_um2: Optional[float] = None
    # ── True shape, measured on the particle mask ──
    # The box does not describe the shape: on this material it overestimates area 1.87x and
    # length by 14%, and a diagonal fibre has a square box. These fields
    # are filled by `morfologia.medir_deteccion()`; they stay None if not measured.
    length_um: Optional[float] = None      # major dimension, following the curve
    width_um: Optional[float] = None
    aspect: Optional[float] = None       # true length/width
    curvature: Optional[float] = None     # 1.0 = straight; >1.15 = curved
    morphotype: Optional[str] = None       # "fibre" | "fragment"
    # The two length bounds separately, plus the equivalent rectangle. They are
    # stored because the pair reports how irregular the particle is.
    feret_um: Optional[float] = None      # chord: greatest distance across the boundary
    geodesic_um: Optional[float] = None  # longest path inside
    equivalent_rect_length_um: Optional[float] = None
    method_length: Optional[str] = None
    # The measurement came out, but it warrants a look: typically several particles in
    # contact measured as one, which INFLATES the size. It travels in the Detection
    # -- not only in the Morphology that produced it -- because otherwise the report
    # has no way to flag it without re-measuring the whole batch.
    review: bool = False
    warning_shape: Optional[str] = None
    # Number of the particle within ITS image, starting at 1. It is what
    # allows going from a table row to the specific particle in the photo: without
    # it, `«la mayor mide 5878 µm»` cannot be checked.
    number: Optional[int] = None
    # Does this particle's polymer claim hold, or only its detection?
    # It is decided by `core.asignacion` after inference, checking conf against the
    # assignment threshold. When False the particle still counts, but it is
    # reported as `«no asignable»` instead of being assigned a polymer.
    # True by default: with no threshold configured, everything is assigned as before.
    assignable: bool = True

    @property
    def w(self) -> float: return self.x2 - self.x1
    @property
    def h(self) -> float: return self.y2 - self.y1
    @property
    def cx(self) -> float: return 0.5 * (self.x1 + self.x2)
    @property
    def cy(self) -> float: return 0.5 * (self.y1 + self.y2)


def _is_oom(e: Exception) -> bool:
    """Is the exception caused by GPU memory exhaustion?"""
    try:
        import torch
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(e).lower()


def _cuda_empty_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
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
        self.names: Dict[int, str] = {}
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

    def probe_max_imgsz(self, image_path: str | Path, device: str = "0",
                        candidates: Tuple[int, ...] = (2048, 3008, 4096, 5120, 6144, 7168, 8192),
                        progress=None) -> Tuple[int, Dict[int, str]]:
        """Finds the largest imgsz the GPU can take for THIS model.

        Runs real inference at increasing sizes until it fails on memory.
        Returns (max_ok, detalle) where detalle is {imgsz: "ok"/"oom"/"err"}.
        On CPU there is no VRAM cap → returns the largest candidate untested.
        """
        self.load()
        detail: Dict[int, str] = {}
        dev = str(device).strip().lower()
        if dev in ("cpu", "-1"):
            return candidates[-1], {c: "cpu" for c in candidates}
        try:
            import torch
        except Exception:
            return 1920, {}
        max_ok = 0
        for sz in candidates:
            if progress is not None:
                progress(sz)
            try:
                self._model.predict(source=str(image_path), imgsz=sz, conf=0.25,
                                    device=device, verbose=False, save=False)
                detail[sz] = "ok"
                max_ok = sz
            except torch.cuda.OutOfMemoryError:
                detail[sz] = "oom"
                break
            except Exception as e:
                detail[sz] = f"err: {type(e).__name__}"
                break
            finally:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return (max_ok or 1920), detail

    def predict(self, image_path: str | Path, conf: float = 0.25, iou: float = 0.45,
                imgsz: int = 640, device: str = "0",
                auto_fallback: bool = True) -> List[Detection]:
        self.load()
        # Auto-fallback: if the requested imgsz exhausts VRAM, it retries with each
        # smaller size in the ladder instead of aborting the whole analysis.
        sizes = [int(imgsz)]
        if auto_fallback:
            sizes += [s for s in self.FALLBACK_SIZES if s < int(imgsz)]
        res = None
        for i, sz in enumerate(sizes):
            try:
                res = self._model.predict(
                    source=str(image_path), conf=conf, iou=iou, imgsz=sz,
                    device=device, verbose=False, save=False,
                    max_det=self.MAX_DET,
                )
                self.last_imgsz = sz
                self.last_fallback = (i > 0)
                break
            except Exception as e:
                if not auto_fallback or i == len(sizes) - 1 or not _is_oom(e):
                    raise
                event(_log(), "imgsz.fallback", level=logging.WARNING,
                       alias=self.alias, file=str(image_path),
                       tried_previous=sz,
                       imgsz_next=(sizes[i + 1]
                                        if i + 1 < len(sizes) else None),
                       reason="gpu_oom")
                _cuda_empty_cache()
        out: List[Detection] = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
            out.append(Detection(
                class_id=int(cls),
                class_name=self.names.get(int(cls), str(int(cls))),
                conf=float(c),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            ))
        return out

    def predict_sliced(self, image_path: str | Path, conf: float = 0.10,
                       iou: float = 0.45, tile: int = 1024, overlap: float = 0.25,
                       device: str = "0", agnostic_nms: bool = True,
                       imgsz: Optional[int] = None, batch: int = 8) -> List[Detection]:
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
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return []
        H, W = img.shape[:2]
        overlap = min(max(overlap, 0.0), 0.9)
        step = max(1, int(round(tile * (1.0 - overlap))))
        xs = _tile_starts(W, tile, step)
        ys = _tile_starts(H, tile, step)

        crops: List[np.ndarray] = []
        origins: List[Tuple[int, int]] = []
        for y0 in ys:
            for x0 in xs:
                crops.append(img[y0:y0 + tile, x0:x0 + tile])
                origins.append((x0, y0))
        if not crops:
            return []

        # imgsz controls the tile input resolution. If it is larger than the
        # tile, the crop is upscaled -> tiny particles are enlarged towards
        # the scale at which the model was trained (key for small objects).
        infer_sz = int(imgsz) if imgsz else tile

        dets: List[Detection] = []
        bs = max(1, int(batch))
        # Run in sub-batches to avoid exhausting VRAM (many tiles at high imgsz).
        for start in range(0, len(crops), bs):
            chunk = crops[start:start + bs]
            chunk_org = origins[start:start + bs]
            results = self._model.predict(
                source=chunk, conf=conf, iou=iou, imgsz=infer_sz,
                device=device, verbose=False, save=False,
                max_det=self.MAX_DET,
            )
            for (x0, y0), r in zip(chunk_org, results):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                for (a, b, c, d), cc, k in zip(xyxy, cf, cl):
                    dets.append(Detection(
                        class_id=int(k),
                        class_name=self.names.get(int(k), str(int(k))),
                        conf=float(cc),
                        x1=float(a) + x0, y1=float(b) + y0,
                        x2=float(c) + x0, y2=float(d) + y0,
                    ))
        return _nms(dets, iou_thr=iou, agnostic=agnostic_nms)

    def predict_auto(self, image_path: str | Path, conf: float = 0.25,
                     iou: float = 0.45, imgsz: int = 1280, device: str = "0",
                     tiling: str = "auto", threshold_px: int = 2000, tile: int = 0,
                     overlap: float = 0.25, batch: int = 8,
                     agnostic_nms: bool = True,
                     record: Optional[Dict] = None) -> List[Detection]:
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
            plan = policy_tiling(wh[0], wh[1], imgsz,
                                     threshold_px=0, tile=tile, overlap=overlap)
        else:
            plan = policy_tiling(wh[0], wh[1], imgsz,
                                     threshold_px=threshold_px, tile=tile, overlap=overlap)

        if record is not None:
            record.clear()
            record.update({"ancho": wh[0] if wh else 0, "alto": wh[1] if wh else 0,
                             "troceado": plan is not None, "plan": plan})

        if plan is None:
            return self.predict(image_path, conf=conf, iou=iou, imgsz=imgsz,
                                device=device)
        return self.predict_sliced(image_path, conf=conf, iou=iou,
                                   tile=plan["tile"], overlap=plan["overlap"],
                                   device=device, agnostic_nms=agnostic_nms,
                                   imgsz=imgsz, batch=batch)


def size_image(image_path: str | Path) -> Optional[Tuple[int, int]]:
    """(width, height) reading only the header, without decoding the whole image.

    It is queried before every inference, so decoding a 3260 px photo just
    to measure it would cost more than the decision itself.

    Returns the size ALREADY ROTATED per EXIF, because that is the frame the
    rest of the program works in: ``cv2.imdecode`` applies the orientation
    and Pillow does not. Without the fix, a photo tagged orientation 6
    measured 4096x3072 but decoded 3072x4096, so boxes normalized against
    one size were denormalized against the other and ended up displaced.
    """
    try:
        from PIL import Image
        with Image.open(str(image_path)) as im:
            width, height = int(im.size[0]), int(im.size[1])
            # 5, 6, 7, 8 are the orientations that swap the axes.
            if im.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
            return width, height
    except Exception:
        # Paths with accents or formats Pillow cannot open: it falls back to the cv2 path,
        # which is already used throughout the rest of the project for that same reason.
        try:
            img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                return None
            return int(img.shape[1]), int(img.shape[0])
        except Exception:
            return None


def policy_tiling(width: int, height: int, imgsz: int, threshold_px: int = 2000,
                      tile: int = 0, overlap: float = 0.25) -> Optional[Dict]:
    """Decides whether the photo gets tiled, and with which geometry, from its
size alone.

    ``None`` means a single pass over the full image.

    The threshold is not compared against the area but against the **longer
    side**, because what makes a particle disappear is the rescale to
    ``imgsz``: a 3260 px photo entering at 2080 shrinks every particle to
    0.64x, and below the network's stride it stops existing. Tiling avoids
    that rescale.

    With ``tile=0`` the tile side comes from ``min(umbral_px, imgsz)``: this
    way the crop enters the network without being shrunk, which is exactly
    the goal.

    WHETHER to tile and AT WHICH SIZE are two different questions, and
    mixing them up caused a failure: to force tiling, ``predict_auto``
    passes ``umbral_px=0`` -- its way of saying "do not decide, tile" --,
    and that yielded ``min(0, imgsz) = 0``, which the clamp raised to 256.
    On a 4096x3072 photo that is 336 tiles of 256 px: extremely slow and
    with ZERO detections, because each tile gets rescaled 8x up to imgsz and
    the model sees nothing resembling what it was trained on. With
    umbral_px at 0 the size is governed by imgsz, the only thing that truly
    determines it.
    """
    side = max(int(width), int(height))
    if side <= int(threshold_px):
        return None
    if int(tile) > 0:
        t = int(tile)
    else:
        candidates_2 = [v for v in (int(threshold_px), int(imgsz)) if v > 0]
        t = min(candidates_2) if candidates_2 else 1280
    t = max(256, min(t, side))
    overlap = min(max(float(overlap), 0.0), 0.9)
    step = max(1, int(round(t * (1.0 - overlap))))
    n = len(_tile_starts(int(width), t, step)) * len(_tile_starts(int(height), t, step))
    return {"tile": t, "overlap": overlap, "n_tiles": n,
            "lado": side, "umbral_px": int(threshold_px)}


def _tile_starts(total: int, tile: int, step: int) -> List[int]:
    """Start positions of the tiles along one axis (the last one flush with the edge)."""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, step))
    if not starts or starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def _box_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1); y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2); y2 = min(a.y2, b.y2)
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: List[Detection], iou_thr: float = 0.45,
         agnostic: bool = True) -> List[Detection]:
    """Greedy NMS over the detections of all tiles. Keeps the one with the highest conf."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i].conf)
    keep: List[int] = []
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


def read_yolo_txt(txt_path: str | Path, img_w: int, img_h: int,
                  class_names: Dict[int, str]) -> List[Detection]:
    """Reads a YOLO .txt (normalized class cx cy w h) and returns a Detection with conf=1."""
    out: List[Detection] = []
    p = Path(txt_path)
    if not p.exists():
        return out
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = [float(x) for x in parts[1:5]]
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            out.append(Detection(
                class_id=cls,
                class_name=class_names.get(cls, str(cls)),
                conf=1.0,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
    except Exception:
        pass
    return out


def find_gt_for_image(image_path: Path, gt_folder: Optional[Path]) -> Optional[Path]:
    """Finds the GT .txt next to the image, in the sibling /labels, or in gt_folder."""
    stem = image_path.stem
    candidates = []
    if gt_folder is not None:
        candidates.append(gt_folder / f"{stem}.txt")
    # same directory
    candidates.append(image_path.parent / f"{stem}.txt")
    # sibling labels/
    if image_path.parent.name.lower() == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{stem}.txt")
    else:
        candidates.append(image_path.parent / "labels" / f"{stem}.txt")
    for c in candidates:
        if c.exists():
            return c
    return None


def compute_box_size_um(det: Detection, um_per_px: Optional[float]) -> None:
    """Approximate size from the BOX, not the particle.

    It is a fallback: it overestimates the area 1.87x at the median of this
    material because the box of an elongated particle is nearly empty, and
    it depends on orientation. Used only when segmentation could not isolate
    the particle. The good version is ``morfologia.medir_deteccion``.
    """
    if um_per_px is None or um_per_px <= 0:
        return
    area_px2 = det.w * det.h
    area_um2 = area_px2 * (um_per_px ** 2)
    # equivalent diameter (circle with the same area)
    diam_um = 2.0 * (area_um2 / np.pi) ** 0.5
    det.diam_um = diam_um
    det.area_um2 = area_um2
