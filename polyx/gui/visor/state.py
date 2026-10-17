"""Estado del Visor — imagen, calibración μm/px, detecciones."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QPointF, Signal

from polyx.model.yolo_wrap import Detection

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class VisorState(QObject):
    image_loaded        = Signal(Path)
    detections_changed  = Signal(list)          # List[Detection]
    calib_changed       = Signal(float, str)    # um_per_px, nombre_modo
    calib_pts_changed   = Signal(list)          # List[QPointF] en coords imagen
    calib_ready         = Signal()              # suficientes puntos recogidos
    status_changed      = Signal(str, str)      # texto, color_hex

    def __init__(self):
        super().__init__()
        self.images:      list[Path]      = []
        self.current_idx: int             = -1
        self.um_per_px:   float           = 0.0
        self.calib_mode:  str             = "none"   # "none" | "linea" | "circulo"
        self._calib_pts:  list[QPointF]   = []
        self.detections:  list[Detection] = []
        self.model        = None           # YoloModel, cargado bajo demanda

    # ── Imagen ────────────────────────────────────────────────
    @property
    def current_image(self) -> Optional[Path]:
        if 0 <= self.current_idx < len(self.images):
            return self.images[self.current_idx]
        return None

    def load_single(self, path: Path):
        self.images = [path]
        self.current_idx = 0
        self.detections = []
        self.image_loaded.emit(path)

    def load_folder(self, folder: Path):
        imgs = sorted(p for p in folder.rglob("*")
                      if p.suffix.lower() in IMAGE_EXTS)
        self.images = imgs
        self.current_idx = 0 if imgs else -1
        self.detections = []
        if imgs:
            self.image_loaded.emit(imgs[0])

    def goto(self, idx: int):
        if not self.images:
            return
        idx = max(0, min(idx, len(self.images) - 1))
        if idx == self.current_idx:
            return
        self.current_idx = idx
        self.detections = []
        self.detections_changed.emit([])
        self.image_loaded.emit(self.images[idx])

    def next_image(self): self.goto(self.current_idx + 1)
    def prev_image(self): self.goto(self.current_idx - 1)

    # ── Calibración ───────────────────────────────────────────
    def start_calib(self, mode: str):
        """Inicia el modo calibración ('linea' o 'circulo')."""
        self.calib_mode = mode
        self._calib_pts = []
        self.calib_pts_changed.emit([])

    def cancel_calib(self):
        self.calib_mode = "none"
        self._calib_pts = []
        self.calib_pts_changed.emit([])

    def add_calib_point(self, pt: QPointF):
        """Agrega un punto de calibración. Emite calib_ready si hay suficientes."""
        self._calib_pts.append(pt)
        self.calib_pts_changed.emit(list(self._calib_pts))
        needed = 2 if self.calib_mode == "linea" else 3
        if len(self._calib_pts) >= needed:
            self.calib_ready.emit()

    @property
    def calib_pts(self) -> list[QPointF]:
        return list(self._calib_pts)

    def finish_calib_linea(self, real_um: float):
        """Calcula μm/px con los 2 puntos de línea y el tamaño real ingresado."""
        if len(self._calib_pts) < 2 or real_um <= 0:
            return
        p1, p2 = self._calib_pts[0], self._calib_pts[1]
        px_dist = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
        if px_dist == 0:
            return
        um_per_px = real_um / px_dist
        self._apply_calib(um_per_px, "línea")

    def finish_calib_circulo(self, real_diam_um: float):
        """Calcula μm/px con los 3 puntos de círculo y el diámetro real."""
        if len(self._calib_pts) < 3 or real_diam_um <= 0:
            return
        radius_px = _circumscribed_radius(self._calib_pts)
        if radius_px <= 0:
            return
        diam_px = 2 * radius_px
        um_per_px = real_diam_um / diam_px
        self._apply_calib(um_per_px, "círculo")

    def _apply_calib(self, um_per_px: float, mode_name: str):
        self.um_per_px = um_per_px
        self.calib_mode = "none"
        self._calib_pts = []
        self.calib_pts_changed.emit([])
        self.calib_changed.emit(um_per_px, mode_name)

    # ── Utilidades ────────────────────────────────────────────
    def px_to_um(self, px: float) -> Optional[float]:
        return px * self.um_per_px if self.um_per_px > 0 else None


def _circumscribed_radius(pts: list[QPointF]) -> float:
    """Radio del círculo circunscrito a 3 puntos (fórmula analítica)."""
    x1, y1 = pts[0].x(), pts[0].y()
    x2, y2 = pts[1].x(), pts[1].y()
    x3, y3 = pts[2].x(), pts[2].y()
    ax, ay = x2 - x1, y2 - y1
    bx, by = x3 - x1, y3 - y1
    D = 2 * (ax * by - ay * bx)
    if abs(D) < 1e-10:
        return 0.0
    ux = (by * (ax * ax + ay * ay) - ay * (bx * bx + by * by)) / D
    uy = (ax * (bx * bx + by * by) - bx * (ax * ax + ay * ay)) / D
    return math.hypot(ux, uy)


def _circumscribed_center(pts: list[QPointF]) -> QPointF:
    """Centro del círculo circunscrito a 3 puntos."""
    x1, y1 = pts[0].x(), pts[0].y()
    x2, y2 = pts[1].x(), pts[1].y()
    x3, y3 = pts[2].x(), pts[2].y()
    ax, ay = x2 - x1, y2 - y1
    bx, by = x3 - x1, y3 - y1
    D = 2 * (ax * by - ay * bx)
    if abs(D) < 1e-10:
        return QPointF(x1, y1)
    ux = (by * (ax * ax + ay * ay) - ay * (bx * bx + by * by)) / D
    uy = (ax * (bx * bx + by * by) - bx * (ax * ax + ay * ay)) / D
    return QPointF(x1 + ux, y1 + uy)
