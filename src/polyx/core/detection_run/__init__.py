"""Headless detection: the canonical run engine, its folder and legacy support.

The canonical engine lives in :mod:`polyx.core.detection_run.engine`; it is
Qt-free and owns decode, ground-truth reading, measurement, matching, drawing
and the run folder. The old ``run_batch`` stays in ``legacy`` until its callers
migrate, and ``load_run`` reopens a canonical folder without Qt.
"""

from .config import CALIBRATION_MODES, CalibrationPolicy, ModelSpec, RunConfig
from .engine import RunDirNotEmpty, RunFailed, run_detection
from .legacy import IMAGE_EXTS, run_batch
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
    "load_run",
    "run_batch",
    "run_detection",
]
