"""Viewer state — image, μm/px calibration, detections."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QPointF, Signal

from polyx.core.detect.yolo import Detection
from polyx.core.measure.calibration import circle_calibration, line_calibration
from polyx.core.paths import IMAGE_EXTS

# Named refusal that belongs to the collection, not to the geometry: the core
# names every degenerate drawing, and this one says there is no drawing yet.
FAILURE_NOT_ENOUGH_POINTS = "not_enough_points"

# Calibration mode tokens. They are machine values -- they travel on
# ``calib_changed`` and the canvas reads them -- so they stay English; the
# words the user reads are chosen with ``_()`` where they are displayed.
CALIB_MODE_NONE = "none"
CALIB_MODE_LINE = "line"
CALIB_MODE_CIRCLE = "circle"


class ViewerState(QObject):
    image_loaded = Signal(Path)
    detections_changed = Signal(list)  # List[Detection]
    calib_changed = Signal(float, str)  # um_per_px, mode_name
    calib_pts_changed = Signal(list)  # List[QPointF] in image coords
    calib_ready = Signal()  # enough points collected
    status_changed = Signal(str, str)  # text, color_hex

    def __init__(self):
        super().__init__()
        self.images: list[Path] = []
        self.current_idx: int = -1
        self.um_per_px: float = 0.0
        self.calib_mode: str = CALIB_MODE_NONE  # CALIB_MODE_NONE | _LINE | _CIRCLE
        self._calib_pts: list[QPointF] = []
        self.detections: list[Detection] = []
        self.model = None  # YoloModel, loaded on demand

    # ── Image ────────────────────────────────────────────────
    @property
    def current_image(self) -> Path | None:
        if 0 <= self.current_idx < len(self.images):
            return self.images[self.current_idx]
        return None

    def load_single(self, path: Path):
        self.images = [path]
        self.current_idx = 0
        self.detections = []
        self.image_loaded.emit(path)

    def load_folder(self, folder: Path):
        imgs = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
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

    def next_image(self):
        self.goto(self.current_idx + 1)

    def prev_image(self):
        self.goto(self.current_idx - 1)

    # ── Calibration ──────────────────────────────────────────
    def start_calib(self, mode: str):
        """Starts the calibration mode (``"line"`` or ``"circle"``)."""
        self.calib_mode = mode
        self._calib_pts = []
        self.calib_pts_changed.emit([])

    def cancel_calib(self):
        self.calib_mode = CALIB_MODE_NONE
        self._calib_pts = []
        self.calib_pts_changed.emit([])

    def add_calib_point(self, pt: QPointF):
        """Adds a calibration point. Emits calib_ready when there are enough."""
        self._calib_pts.append(pt)
        self.calib_pts_changed.emit(list(self._calib_pts))
        needed = 2 if self.calib_mode == CALIB_MODE_LINE else 3
        if len(self._calib_pts) >= needed:
            self.calib_ready.emit()

    @property
    def calib_pts(self) -> list[QPointF]:
        return list(self._calib_pts)

    def finish_calib_line(self, real_um: float) -> str:
        """Applies a line calibration; returns "" or the named failure.

        The geometry is the calibration core's: this method only hands it the
        clicked points. A refused drawing emits nothing and keeps whatever
        scale the Viewer already had.
        """
        if len(self._calib_pts) < 2:
            return FAILURE_NOT_ENOUGH_POINTS
        p1, p2 = self._calib_pts[:2]
        result = line_calibration((p1.x(), p1.y()), (p2.x(), p2.y()), real_um)
        if result.um_per_px is None:
            return result.failure
        self._apply_calib(result.um_per_px, CALIB_MODE_LINE)
        return ""

    def finish_calib_circle(self, real_diam_um: float) -> str:
        """Applies a circle calibration; returns "" or the named failure.

        Same contract as :meth:`finish_calib_line`: the circumscribed circle
        and the µm/px come from the core, and a refusal leaves the scale alone.
        """
        if len(self._calib_pts) < 3:
            return FAILURE_NOT_ENOUGH_POINTS
        p1, p2, p3 = self._calib_pts[:3]
        result = circle_calibration(
            (p1.x(), p1.y()),
            (p2.x(), p2.y()),
            (p3.x(), p3.y()),
            real_diam_um,
        )
        if result.um_per_px is None:
            return result.failure
        self._apply_calib(result.um_per_px, CALIB_MODE_CIRCLE)
        return ""

    def _apply_calib(self, um_per_px: float, mode_name: str):
        self.um_per_px = um_per_px
        self.calib_mode = CALIB_MODE_NONE
        self._calib_pts = []
        self.calib_pts_changed.emit([])
        self.calib_changed.emit(um_per_px, mode_name)
