"""Configuration of one canonical detection run.

The engine receives already-loaded detection sources, the exact run folder and
the parameters, so this module only describes what to run -- it never loads
weights and never touches Qt. Calibration is its own policy because which photo
gets which scale is a run decision, not an inference parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from polyx.core.detect.protocol import DetectionSource

from .result import RunParams

CALIBRATION_MODES = ("auto", "manual", "index", "none")


@dataclass
class ModelSpec:
    """One loaded model taking part in the run.

    ``detector`` is already loaded by the caller: the engine never calls
    ``load()``. ``alias`` names the model's folder; it defaults to the weights
    stem, then to ``model_<n>``. ``names`` overrides the detector's own class
    map when a caller has a reason to, and is recorded in ``run.json`` either
    way.
    """

    detector: DetectionSource
    alias: str = ""
    path: Path | None = None
    model_type: str = "yolo"
    names: dict[int, str] = field(default_factory=dict)
    sha256: str = ""

    def folder_name(self, index: int) -> str:
        """The folder alias for this model, derived when the caller left it empty."""
        alias = self.alias.strip()
        if not alias and self.path is not None:
            alias = self.path.stem.strip()
        if not alias:
            alias = f"model_{index + 1}"
        return alias


@dataclass
class CalibrationPolicy:
    """How each photo gets its micrometres-per-pixel scale.

    Ticket #58 wires the ``manual`` and ``none`` paths. ``auto`` and ``index``
    are part of the shape because the shared contract already names them, but
    their fallback order (explicit index, searched index, plate measurement,
    manual value) belongs to ticket #60.
    """

    mode: str = "manual"
    um_per_px: float = 0.0
    index_path: Path | None = None
    measure_plate: bool = False
    diameter_plate_mm: float = 100.0
    height_plate_mm: float = 0.0
    distance_camera_mm: float = 0.0


@dataclass
class RunConfig:
    """Everything one call to :func:`run_detection` needs.

    ``output_dir`` is the exact folder the caller chose; the engine does not
    invent a timestamped name. A non-empty folder is refused unless ``force``
    is set.
    """

    images: list[Path]
    models: list[ModelSpec]
    output_dir: Path
    params: RunParams = field(default_factory=RunParams)
    gt_folder: Path | None = None
    calibration: CalibrationPolicy = field(default_factory=CalibrationPolicy)
    force: bool = False


__all__ = ["CALIBRATION_MODES", "CalibrationPolicy", "ModelSpec", "RunConfig"]
