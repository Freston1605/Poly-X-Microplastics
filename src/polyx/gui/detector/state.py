"""Detector state shared by all the pages.

Inherits from QObject to emit signals when it changes (the pages subscribe).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from polyx.core.detect import DetectionSource
from polyx.core.detect.yolo import Detection
from polyx.core.i18n import tr
from polyx.core.platform import device_available


# ────────────────────────────────────────────────────────────────────
@dataclass
class ModelSlot:
    """A model slot in the `Modelos` tab (up to 3)."""

    alias: str = ""
    path: Optional[Path] = None  # path to the weights
    model_type: str = "yolo"  # "yolo" | "yolo-seg" | "sam2" | ...
    loaded: Optional[DetectionSource] = None


@dataclass
class InferenceParams:
    conf: float = 0.25
    iou_nms: float = 0.45
    iou_tp: float = 0.50  # for error analysis
    # 1280 and not 640: microplastics occupy ~12 px in 4096 photos, and at 640
    # they collapse below the network stride, so by default nothing was detected.
    # Matches the `Rápido` profile on the Parameters page.
    imgsz: int = 1280
    # It does not simply set "0" (CUDA): on a Mac that card does not exist and the
    # analysis failed at startup. It is resolved when creating the parameters
    # by choosing the best available -- CUDA, then MPS (Mac GPU), then CPU.
    device: str = field(default_factory=device_available)
    um_per_px: float = 0.0  # 0 = calibration disabled; manual fallback
    size_min_um: float = 0.0  # 0 = lower filter disabled
    size_max_um: float = 0.0  # 0 = upper filter disabled
    # ── Abstention: how far the polymer holds ──
    # Minimum confidence to assign a polymer to the particle. Below it, the
    # particle is still counted but reported as «not assignable», instead of
    # inventing a class for it. 0 = disabled, which is the
    # usual behavior. The reason -- Nile Red measures polarity, not chemical identity, and
    # PP and LDPE are the two polyolefins -- is in polyx/model/asignacion.py.
    conf_assignment: float = 0.0
    # ── Automatic calibration against the Petri dish ──
    # With this enabled each photo gets ITS scale by measuring the dish ring,
    # without marking anything by hand. It matters because the shooting distance varies between
    # shots: in this material the scale ranges from 31 to 49 µm/px, a factor of 1.5, and a
    # single value for the whole batch would give sizes with up to 50 % error.
    # Enabled by default. It used to be off, and the checkbox that controls it is
    # named `(recomendado)`: the application recommended something it did not do. The
    # effect was that the ENTIRE size section of the report came out with dashes
    # -- particles were measured in pixels, but without µm/px there is no way to turn them
    # to micrometers -- and nothing explained why.
    #
    # Failing here is cheap: if the dish is absent from the photo (crops, for
    # example) the calibration is declared invalid, inherited from index.csv if
    # present, and the analysis continues. It costs ~1-2 s per photo.
    measure_plate: bool = True
    diameter_plate_mm: float = 100.0
    # The dish rim is CLOSER to the camera than the background where the
    # particles rest, so it projects larger and the scale comes out small:
    # sizes are underestimated. With these two values it is corrected to the plane of
    # the base. At 0 no correction is applied and the report states so.
    height_plate_mm: float = 0.0
    distance_camera_mm: float = 0.0
    # CSV associating image name with already-calibrated µm/px. It is the only path
    # for crops, where the dish edge does not appear and therefore cannot
    # be re-measured: the scale is inherited from the photo they came from.
    index_calibration: str = ""
    # ── Particle shape ──
    # It segments inside each box to measure real area, length and width instead
    # of the box's. Without this the area comes out 1.87x too large and a fibre at
    # a diagonal is reported as a fragment. It costs ~0.2 ms per particle.
    measure_shape: bool = True
    # ── Automatic tiling ──
    # "auto" tiles only photos whose longest side exceeds troceo_umbral_px, infers
    # each tile at native resolution and merges with global NMS. The threshold is 2000
    # so the study crops (1630 px per side) fit in one piece and
    # the full plates (~3260) get tiled.
    tiling: str = "auto"  # `"auto"` | `"siempre"` | `"nunca"`
    tiling_threshold_px: int = 2000
    tiling_tile: int = 0  # 0 = derive from min(threshold, imgsz)
    tiling_overlap: float = 0.25  # overlap between tiles; NMS removes the duplicate


@lru_cache(maxsize=12)
def _read_png(path_2: str, _mtime: float) -> Optional[bytes]:
    """Reads a run PNG, with a bounded cache.

    ``_mtime`` is not used inside: it takes part in the signature so the cache
    invalidates itself if the file changes on disk.
    """
    try:
        return Path(path_2).read_bytes()
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
    predictions: List[Detection] = field(default_factory=list)
    gt: List[Detection] = field(default_factory=list)
    has_gt: bool = False
    tp: int = 0
    fp: int = 0
    fn: int = 0
    miscls: int = 0
    # Paths of the annotated images saved in the run folder.
    annotated_path: Optional[Path] = None  # prediction + GT (preview/errors)
    pred_path: Optional[Path] = None  # model predictions only
    gt_path: Optional[Path] = None  # GT only (None if absent)
    # user verdict after visual review: None / `"buena"` / `"mala"`
    verdict: Optional[str] = None
    # tiling geometry if the photo was split; "" = inferred as one piece
    tiling: str = ""

    @staticmethod
    def _bytes_de(path_2: Optional[Path]) -> Optional[bytes]:
        if path_2 is None:
            return None
        try:
            mtime = path_2.stat().st_mtime
        except OSError:
            return None
        return _read_png(str(path_2), mtime)

    @property
    def annotated_png(self) -> Optional[bytes]:
        return self._bytes_de(self.annotated_path)

    @property
    def pred_png(self) -> Optional[bytes]:
        return self._bytes_de(self.pred_path)

    @property
    def gt_png(self) -> Optional[bytes]:
        return self._bytes_de(self.gt_path)


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
        self.model_slots: List[ModelSlot] = [
            ModelSlot(alias=tr("Modelo {n}").format(n=i + 1)) for i in range(3)
        ]
        self.images: List[Path] = []
        self.gt_folder: Optional[Path] = None
        self.params = InferenceParams()
        # results[model_idx] -> List[ImageResult]
        self.results: Dict[int, List[ImageResult]] = {}
        # Scale of each image and where it came from, by filename. The
        # runner fills it; the report reports it. Without this the sizes in um
        # would appear without saying what standard they were measured against.
        self.calibrations: Dict[str, object] = {}
        # run timestamp/folder
        self.run_dir: Optional[Path] = None
        self._running = False
        self._abort = False

    # ── helpers ──
    def active_models(self) -> List[ModelSlot]:
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
