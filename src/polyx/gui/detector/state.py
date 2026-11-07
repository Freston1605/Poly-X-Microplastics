"""Detector state shared by all the pages.

Inherits from QObject to emit signals when it changes (the pages subscribe).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from polyx.core.detect import DetectionSource
from polyx.core.detection_run import (
    CalibrationPolicy,
    ImageResult,
    ModelInfo,
    RunParams,
    RunResult,
)
from polyx.core.i18n import _

# Compatibility name used by the pages and by old headless folder readers.
InferenceParams = RunParams


# ────────────────────────────────────────────────────────────────────
@dataclass
class ModelSlot(ModelInfo):
    """A model slot in the `Modelos` tab (up to 3): metadata plus its backend."""

    loaded: DetectionSource | None = None
    #: The alias is still the generated "Model N", so it follows the interface
    #: language; a name typed by the user is kept as written.
    alias_default: bool = True


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
            ModelSlot(alias=_("Model %(n)s") % {"n": i + 1}) for i in range(3)
        ]
        self.images: list[Path] = []
        self.gt_folder: Path | None = None
        # Image currently open in the Manual GT page. Kept in the state so a
        # language change can rebuild the page without moving the user away
        # from the annotation they were working on.
        self.gt_image: Path | None = None
        self.params = InferenceParams()
        # Calibration has one owner and it is not RunParams: the pages edit
        # this policy, the runner hands it to the engine and the run records
        # the effective per-photo values it produced.
        self.calibration = CalibrationPolicy()
        # results[real_model_slot_idx] -> List[ImageResult]. The engine numbers
        # its active models densely; the runner maps those back to these slots.
        self.results: dict[int, list[ImageResult]] = {}
        # Scale of each image and where it came from, by filename. The
        # runner fills it; the report reports it. Without this the sizes in um
        # would appear without saying what standard they were measured against.
        self.calibrations: dict[str, object] = {}
        # The canonical run record backing this GUI state, once a run exists.
        self.run_result: RunResult | None = None
        # Review verdicts are GUI state, keyed by real model slot and image.
        self.verdicts: dict[tuple[int, Path], str] = {}
        # run timestamp/folder
        self.run_dir: Path | None = None
        self._running = False
        self._abort = False

    # ── helpers ──
    def active_models(self) -> list[ModelSlot]:
        return [s for s in self.model_slots if s.path is not None]

    def set_verdict(self, model_idx: int, image_path: Path, verdict: str | None) -> None:
        """Store (or clear) the review verdict of one image for one model slot."""
        key = (model_idx, Path(image_path))
        if verdict is None:
            self.verdicts.pop(key, None)
        else:
            self.verdicts[key] = verdict

    def verdict_for(self, model_idx: int, image_path: Path) -> str | None:
        """The stored review verdict, or ``None`` when still unreviewed."""
        return self.verdicts.get((model_idx, Path(image_path)))

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
        self.run_result = None
        self.verdicts = {}
        self.run_dir = None
