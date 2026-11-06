"""Headless detection: the canonical run engine, its folder and legacy reads.

The canonical engine lives in :mod:`polyx.core.detection_run.engine`; it is
Qt-free and owns decode, ground-truth reading, measurement, matching, drawing
and the run folder. ``load_run`` reopens a canonical folder and
``load_legacy_run`` rebuilds a ``RunResult`` from a pre-canonical headless
folder, both without Qt.
"""

from polyx.core.paths import IMAGE_EXTS

from .config import CALIBRATION_MODES, CalibrationPolicy, ModelSpec, RunConfig
from .engine import RunDirNotEmpty, RunFailed, run_detection
from .legacy_reader import load_legacy_run
from .result import ImageResult, ModelExecution, ModelInfo, RunParams, RunResult
from .schema import DETECTION_COLUMNS, SCHEMA_VERSION, load_run

__all__ = [
    "CALIBRATION_MODES",
    "DETECTION_COLUMNS",
    "IMAGE_EXTS",
    "SCHEMA_VERSION",
    "CalibrationPolicy",
    "ImageResult",
    "ModelExecution",
    "ModelInfo",
    "ModelSpec",
    "RunConfig",
    "RunDirNotEmpty",
    "RunFailed",
    "RunParams",
    "RunResult",
    "load_legacy_run",
    "load_run",
    "run_detection",
]
