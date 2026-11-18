"""Headless detection: the canonical run engine, its folder and run reader.

The canonical engine lives in :mod:`polyx.core.detection_run.engine`; it is
Qt-free and owns decode, ground-truth reading, measurement, matching, drawing
and the run folder. ``load_run`` reopens a canonical folder without Qt.
``reload_ground_truth`` re-reads the labels of a finished run through the same
rules, so correcting an annotation does not need a second inference pass nor
a second implementation of the sequence. Pre-canonical folders are not read:
old runs must be opened with an older version of Poly-X.
"""

from polyx.core.paths import IMAGE_EXTS

from .config import CALIBRATION_MODES, CalibrationPolicy, ModelSpec, RunConfig
from .engine import (
    RunDirNotEmpty,
    RunFailed,
    RunOutputError,
    RunOutputNotADirectory,
    run_detection,
)
from .identity import (
    ModelIdentity,
    display_name,
    generated_alias,
    resolve_identity,
    sanitize_alias,
)
from .reload import ReloadSummary, reload_ground_truth
from .result import ImageResult, ModelExecution, ModelInfo, RunParams, RunResult, RunStatus
from .schema import (
    DETECTION_COLUMNS,
    PAGE_COLUMNS,
    SCHEMA_VERSION,
    VIEWER_COLUMNS,
    detection_row,
    load_run,
)

__all__ = [
    "CALIBRATION_MODES",
    "DETECTION_COLUMNS",
    "IMAGE_EXTS",
    "PAGE_COLUMNS",
    "SCHEMA_VERSION",
    "VIEWER_COLUMNS",
    "CalibrationPolicy",
    "ImageResult",
    "ModelExecution",
    "ModelIdentity",
    "ModelInfo",
    "ModelSpec",
    "ReloadSummary",
    "RunConfig",
    "RunDirNotEmpty",
    "RunFailed",
    "RunOutputError",
    "RunOutputNotADirectory",
    "RunParams",
    "RunResult",
    "RunStatus",
    "detection_row",
    "display_name",
    "generated_alias",
    "load_run",
    "reload_ground_truth",
    "resolve_identity",
    "run_detection",
    "sanitize_alias",
]
