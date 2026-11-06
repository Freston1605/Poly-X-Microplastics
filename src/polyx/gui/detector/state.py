"""Detector state shared by all the pages.

Inherits from QObject to emit signals when it changes (the pages subscribe).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from polyx.core.detect import DetectionSource
from polyx.core.detection_run import ImageResult as CoreImageResult
from polyx.core.detection_run import ModelInfo, RunParams
from polyx.core.i18n import tr

# Compatibility name used by the pages and by old headless folder readers.
InferenceParams = RunParams


# ────────────────────────────────────────────────────────────────────
@dataclass
class ModelSlot(ModelInfo):
    """A model slot in the `Modelos` tab (up to 3): metadata plus its backend."""

    loaded: DetectionSource | None = None


@dataclass
class ImageResult(CoreImageResult):
    """Per-image result plus the GUI review verdict."""

    # user verdict after visual review: None / `"buena"` / `"mala"`
    verdict: str | None = None


# ────────────────────────────────────────────────────────────────────
class DetectorState(QObject):
    """Global state of the Detector. The pages read/write here."""

    # Signals
    models_changed = Signal()
    images_changed = Signal()
    params_changed = Signal()
    run_progress = Signal(int, int, str)  # done, total, last_image
    run_started = Signal()
    run_finished = Signal()
    run_image_done = Signal(int, object)  # model_idx, ImageResult
    run_aborted = Signal()

    def __init__(self):
        super().__init__()
        self.model_slots: list[ModelSlot] = [
            ModelSlot(alias=tr("Modelo {n}").format(n=i + 1)) for i in range(3)
        ]
        self.images: list[Path] = []
        self.gt_folder: Path | None = None
        self.params = InferenceParams()
        # results[model_idx] -> List[ImageResult]
        self.results: dict[int, list[ImageResult]] = {}
        # Scale of each image and where it came from, by filename. The
        # runner fills it; the report reports it. Without this the sizes in um
        # would appear without saying what standard they were measured against.
        self.calibrations: dict[str, object] = {}
        # run timestamp/folder
        self.run_dir: Path | None = None
        self._running = False
        self._abort = False

    # ── helpers ──
    def active_models(self) -> list[ModelSlot]:
        return [s for s in self.model_slots if s.path is not None]

    def has_gt(self) -> bool:
        """Does any image have GT?"""
        for r_list in self.results.values():  # noqa: SIM110
            if any(r.has_gt for r in r_list):
                return True
        return False

    def is_running(self) -> bool:
        return self._running

    def set_running(self, v: bool):
        self._running = v

    def request_abort(self):
        self._abort = True

    def consume_abort(self) -> bool:
        a = self._abort
        self._abort = False
        return a

    def reset_results(self):
        self.results = {}
        self.calibrations = {}
        self.run_dir = None
