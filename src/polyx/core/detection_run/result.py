"""Qt-free data shapes shared by the detection run engine and the GUI.

``RunParams`` is the configuration of a run, ``ImageResult`` is one model's
result on one image and ``ModelInfo`` identifies a model without its loaded
backend. Review verdicts are GUI state and deliberately do not live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from polyx.core.detect.yolo import Detection
from polyx.core.measure.calibration import Calibration, summary_batch
from polyx.core.platform import device_available


class RunStatus(StrEnum):
    """Lifecycle state of one detection run.

    A ``str`` enum so ``json.dumps`` keeps writing ``"completed"`` instead of
    an object, and old folders keep loading without a schema change.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"

    @classmethod
    def parse(cls, value: object, default: RunStatus | None = None) -> RunStatus:
        """Read a persisted value, tolerating unknown strings for forward compatibility."""
        try:
            return cls(str(value))
        except ValueError:
            return default or cls.COMPLETED


@dataclass
class RunParams:
    """Parameters of one detection run (formerly ``InferenceParams``).

    Calibration deliberately does not live here: how each photo gets its
    micrometres-per-pixel scale is a run-level policy (``CalibrationPolicy``),
    and the effective values are recorded per image in ``RunResult``.
    """

    conf: float = 0.25
    iou_nms: float = 0.45
    iou_tp: float = 0.50  # for error analysis
    # 1280 and not 640: microplastics occupy ~12 px in 4096 photos, and at 640
    # they collapse below the network stride, so by default nothing was detected.
    # Matches the `Fast` profile on the Parameters page.
    imgsz: int = 1280
    # It does not simply set "0" (CUDA): on a Mac that card does not exist and the
    # analysis failed at startup. It is resolved when creating the parameters
    # by choosing the best available -- CUDA, then MPS (Mac GPU), then CPU.
    device: str = field(default_factory=device_available)
    size_min_um: float = 0.0  # 0 = lower filter disabled
    size_max_um: float = 0.0  # 0 = upper filter disabled
    # -- Abstention: how far the polymer holds --
    # Minimum confidence to assign a polymer to the particle. Below it, the
    # particle is still counted but reported as "not assignable", instead of
    # inventing a class for it. 0 = disabled, which is the usual behavior.
    conf_assignment: float = 0.0
    # -- Particle shape --
    # It segments inside each box to measure real area, length and width instead
    # of the box's. Without this the area comes out 1.87x too large and a fibre
    # at a diagonal is reported as a fragment. It costs ~0.2 ms per particle.
    measure_shape: bool = True
    # -- Automatic tiling --
    # "auto" tiles only photos whose longest side exceeds tiling_threshold_px,
    # infers each tile at native resolution and merges with global NMS. The
    # threshold is 2000 so the study crops (1630 px per side) fit in one piece
    # and the full plates (~3260) get tiled.
    tiling: str = "auto"  # `"auto"` | `"siempre"` | `"nunca"`
    tiling_threshold_px: int = 2000
    tiling_tile: int = 0  # 0 = derive from min(threshold, imgsz)
    tiling_overlap: float = 0.25  # overlap between tiles; NMS removes the duplicate


@lru_cache(maxsize=12)
def _read_png(path: str, _mtime: float) -> bytes | None:
    """Read a run PNG, with a bounded cache.

    ``_mtime`` is not used inside: it takes part in the signature so the cache
    invalidates itself if the file changes on disk.
    """
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


@dataclass
class ImageResult:
    """Result of running a model on one image.

    The annotated images are stored on disk inside the run and only their
    paths live here. Holding the bytes cost ~6.4 GB of RAM on a batch of 552
    crops with three models, and it was pure waste: the PNGs were already
    written. The ``*_png`` properties still return bytes, so whoever consumed
    them does not change; what changes is that they are read on demand.
    """

    image_path: Path
    model_idx: int
    predictions: list[Detection] = field(default_factory=list)
    gt: list[Detection] = field(default_factory=list)
    has_gt: bool = False
    tp: int = 0
    fp: int = 0
    fn: int = 0
    miscls: int = 0
    # Paths of the annotated images saved in the run folder.
    annotated_path: Path | None = None  # prediction + GT (preview/errors)
    pred_path: Path | None = None  # model predictions only
    gt_path: Path | None = None  # GT only (None if absent)
    # tiling geometry if the photo was split; "" = inferred as one piece
    tiling: str = ""

    @staticmethod
    def _read_bytes(path: Path | None) -> bytes | None:
        if path is None:
            return None
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        return _read_png(str(path), mtime)

    @property
    def annotated_png(self) -> bytes | None:
        return self._read_bytes(self.annotated_path)

    @property
    def pred_png(self) -> bytes | None:
        return self._read_bytes(self.pred_path)

    @property
    def gt_png(self) -> bytes | None:
        return self._read_bytes(self.gt_path)


@dataclass
class ModelInfo:
    """Metadata that identifies one model in a run.

    It carries no loaded backend: callers load the detectors and pass them to
    the engine, so this shape stays cheap to serialise and importable without
    Qt.
    """

    alias: str = ""
    path: Path | None = None
    model_type: str = "yolo"
    names: dict[int, str] = field(default_factory=dict)
    sha256: str = ""


@dataclass
class ModelExecution:
    """What one model actually did during a run.

    ``imgsz_used`` and ``memory_fallback`` describe the inference that really
    happened, not what was requested: a GPU out of memory may have lowered the
    resolution, and the report must be able to declare it. ``shapeless`` counts
    the particles whose shape could not be segmented and kept the box fallback.
    """

    images: int = 0
    detections: int = 0
    tiled_images: int = 0
    imgsz_used: list[int] = field(default_factory=list)
    memory_fallback: list[str] = field(default_factory=list)
    shapeless: int = 0

    def record_imgsz(self, imgsz: int) -> None:
        """Keep the list sorted and unique without making callers dedupe."""
        value = int(imgsz)
        if value > 0 and value not in self.imgsz_used:
            self.imgsz_used = sorted([*self.imgsz_used, value])


@dataclass
class RunResult:
    """Everything a finished (or partial) detection run produced.

    ``results`` is keyed by the dense active-model index (``0..N-1``); the
    engine never leaks GUI slot indices. The rest is plain data so the CLI, the
    report and ``load_run`` can consume it without Qt.
    """

    run_dir: Path
    images: list[Path] = field(default_factory=list)
    models: list[ModelInfo] = field(default_factory=list)
    params: RunParams = field(default_factory=RunParams)
    calibrations: dict[Path, Calibration] = field(default_factory=dict)
    results: dict[int, list[ImageResult]] = field(default_factory=dict)
    status: RunStatus = RunStatus.COMPLETED
    error: str | None = None
    skipped_images: list[Path] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    execution: dict[int, ModelExecution] = field(default_factory=dict)
    metrics: dict[int, dict[str, int]] = field(default_factory=dict)
    # True for a result rebuilt from a pre-canonical headless folder: those
    # folders stored per-image counts but not the GT boxes.
    legacy: bool = False

    def __post_init__(self) -> None:
        # Callers written before the enum may still pass the plain JSON string.
        self.status = RunStatus.parse(self.status)

    @property
    def has_gt(self) -> bool:
        """Whether any image in the run carried ground truth."""
        return any(result.has_gt for results in self.results.values() for result in results)

    @property
    def calibrations_by_name(self) -> dict[str, Calibration]:
        """Filename-keyed view for callers that already worked that way.

        This is a convenience for the GUI state, which keys previews by file
        name. Callers that need an unambiguous lookup must use
        :meth:`calibration_for`: two folders in one batch can hold photos with
        the same basename.
        """
        return {path.name: calibration for path, calibration in self.calibrations.items()}

    def calibration_for(self, image: Path | str) -> Calibration | None:
        """Return the calibration of ``image`` without guessing across names.

        Exact paths win. A basename match is accepted only when it is unique;
        with two same-named photos the caller gets ``None`` instead of another
        image's scale.
        """
        path = Path(image)
        calibration = self.calibrations.get(path)
        if calibration is not None:
            return calibration
        matches = [value for key, value in self.calibrations.items() if key.name == path.name]
        return matches[0] if len(matches) == 1 else None

    @property
    def calibration_summary(self) -> dict:
        """Batch statistics over the valid calibrations.

        The report states the measurement standard from this: how many photos
        were calibrated, where each scale came from and which warnings were
        raised.
        """
        return summary_batch(list(self.calibrations.values()))


__all__ = [
    "ImageResult",
    "ModelExecution",
    "ModelInfo",
    "RunParams",
    "RunResult",
    "RunStatus",
]
