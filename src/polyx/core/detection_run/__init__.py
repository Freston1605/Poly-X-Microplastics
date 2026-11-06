"""Headless detection: the legacy batch runner and the shared run data shapes.

The canonical detection engine will live in this package as it is built; the
old ``run_batch`` stays here until its callers migrate. ``result`` is the
Qt-free data model the engine, the CLI and the GUI share.
"""

from .legacy import IMAGE_EXTS, run_batch
from .result import ImageResult, ModelInfo, RunParams

__all__ = ["IMAGE_EXTS", "ImageResult", "ModelInfo", "RunParams", "run_batch"]
