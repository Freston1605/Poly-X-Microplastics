"""Shared state of the Trainer across all the pages."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import re

from PySide6.QtCore import QObject, Signal

from polyx.model.plataforma import device_available
from polyx.model.dataset_audit import audit_dataset, LEVEL_ERROR, LEVEL_WARN

# Device accepted by Ultralytics: `"cpu"`, `"mps"`, `"cuda[:N]"` or `"N[,N...]"` (GPU by
# index). Anything else makes the run fail midway.
_DEVICE_RE = re.compile(r"^(cpu|mps|cuda(:\d+)?|\d+(,\d+)*)$", re.IGNORECASE)


# ────────────────────────────────────────────────────────────────────
# Training presets (`Suave`/`Balanceado`/`Estricto`)
# ────────────────────────────────────────────────────────────────────
PRESETS = {
    "Balanceado (recomendado)": {
        "epochs": 150, "imgsz": 1280, "batch": 8, "lr0": 0.01,
        "weight_decay": 0.0005, "momentum": 0.937, "warmup_epochs": 5,
        "close_mosaic": 15, "label_smoothing": 0.05, "patience": 50,
        "save_period": 20, "amp": True, "cos_lr": False, "cache": "disk",
        "aug_level": "Medio",
    },
    "Rápido (poco data)": {
        "epochs": 100, "imgsz": 960, "batch": 16, "lr0": 0.01,
        "weight_decay": 0.0005, "momentum": 0.937, "warmup_epochs": 3,
        "close_mosaic": 10, "label_smoothing": 0.0, "patience": 30,
        "save_period": 10, "amp": True, "cos_lr": False, "cache": "ram",
        "aug_level": "Fuerte",
    },
    "Largo (paper-quality)": {
        "epochs": 300, "imgsz": 1280, "batch": 8, "lr0": 0.005,
        "weight_decay": 0.0005, "momentum": 0.937, "warmup_epochs": 5,
        "close_mosaic": 20, "label_smoothing": 0.10, "patience": 100,
        "save_period": 25, "amp": True, "cos_lr": True, "cache": "disk",
        "aug_level": "Medio",
    },
    "Máximo imgsz (alta resolución)": {
        "epochs": 200, "imgsz": 1920, "batch": 4, "lr0": 0.01,
        "weight_decay": 0.0005, "momentum": 0.937, "warmup_epochs": 5,
        "close_mosaic": 15, "label_smoothing": 0.05, "patience": 60,
        "save_period": 20, "amp": True, "cos_lr": False, "cache": "disk",
        "aug_level": "Medio",
    },
    "Personalizado": {},  # marker
}

AUG_LEVELS = {
    "Ninguno":  dict(hsv_h=0.0,  hsv_s=0.0,  hsv_v=0.0,  fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0),
    "Suave":    dict(hsv_h=0.01, hsv_s=0.5,  hsv_v=0.3,  fliplr=0.5, mosaic=0.5, mixup=0.0, copy_paste=0.0),
    "Medio":    dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.1),
    "Fuerte":   dict(hsv_h=0.02, hsv_s=0.9,  hsv_v=0.5,  fliplr=0.5, mosaic=1.0, mixup=0.2, copy_paste=0.3),
}


# ────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    family: str = "v8"      # `"v8"` or `"v11"`
    size: str = "m"          # n / s / m / l / x
    custom_weights: Optional[Path] = None
    preset_name: str = "Balanceado (recomendado)"
    # Train v8 and v11 in sequence with the SAME configuration, to attribute
    # the metric difference to the architecture and not to the hyperparameters.
    compare_families: bool = False

    def families_to_train(self) -> list[str]:
        """Families that will be trained in this run, in order."""
        return ["v8", "v11"] if self.compare_families else [self.family]

    def weight_of(self, family_2: str) -> str:
        """Name of the base weight of a specific family, same size."""
        fam = family_2.lstrip("v") if family_2 != "v8" else family_2
        return f"yolo{fam}{self.size}.pt"

    @property
    def name(self) -> str:
        """Short name of the model, without extension: `yolov8m`, `yolo11n`."""
        return self.weight_of(self.family).removesuffix(".pt")

    @property
    def base_weights_name(self) -> str:
        """Name of the base weight exactly as Ultralytics publishes it (with extension).

        Watch out for the naming, which is inconsistent and unforgiving:
        family 8 carries "v" (yolov8m.pt) and 11 does not (yolo11m.pt).
        Building the name by gluing the family as-is gave "yolov11m.pt",
        which does not exist in the catalogue and made the whole v11 branch
        download fail.
        """
        return self.weight_of(self.family)


@dataclass
class DatasetConfig:
    yaml_path: Optional[Path] = None
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    class_names: List[str] = field(default_factory=list)


@dataclass
class TrainParams:
    imgsz: int = 1280              # default HIGH (practical max) per user requirement
    epochs: int = 150
    batch: int = 8
    lr0: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    warmup_epochs: int = 5
    close_mosaic: int = 15
    label_smoothing: float = 0.05
    patience: int = 50
    save_period: int = 20
    amp: bool = True
    cos_lr: bool = False
    cache: str = "disk"            # `"disk"`, `"ram"`, or `"False"`
    # After training, re-select the checkpoint by evaluating it only on the real
    # sediment of the validation split. best.pt comes from the global mAP, which in a mixed dataset
    # is dominated by the laboratory plates: the wrong domain for detection
    # on field photos. See polyx/gui/trainer/dominio.py.
    choose_by_real_domain: bool = True
    # See the equivalent comment in `polyx/gui/detector/state.py`: `"0"` is CUDA and does not
    # exist on a Mac. The best available device is chosen when building the state.
    device: str = field(default_factory=device_available)
    workers: int = 8
    project_name: str = "polyx_train"
    run_name: str = ""             # empty = auto-date


@dataclass
class AugParams:
    level: str = "Medio"
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1


# ────────────────────────────────────────────────────────────────────
@dataclass
class EpochMetrics:
    epoch: int
    box_loss: float = 0.0
    cls_loss: float = 0.0
    dfl_loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    lr: float = 0.0


# ────────────────────────────────────────────────────────────────────
class TrainerState(QObject):
    """Global state of the Trainer."""

    # Signals
    model_changed = Signal()
    dataset_changed = Signal()
    params_changed = Signal()
    aug_changed = Signal()

    train_started = Signal()
    train_epoch = Signal(object)        # EpochMetrics
    train_log = Signal(str)             # Ultralytics log line
    train_finished = Signal(str)        # path to best.pt
    train_aborted = Signal()
    train_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.model = ModelConfig()
        self.dataset = DatasetConfig()
        self.params = TrainParams()
        self.aug = AugParams()

        # epoch history during the current run
        self.history: List[EpochMetrics] = []
        self.best_map50: float = 0.0
        self.best_epoch: int = 0
        self.epochs_no_improve: int = 0

        self._running = False
        self._abort = False
        self.run_dir: Optional[Path] = None

    def is_running(self) -> bool: return self._running
    def set_running(self, v: bool): self._running = v
    def request_abort(self): self._abort = True
    def consume_abort(self) -> bool:
        a = self._abort; self._abort = False; return a

    def apply_preset(self, name: str):
        if name not in PRESETS or name == "Personalizado":
            return
        d = PRESETS[name]
        p = self.params
        a = self.aug
        for k, v in d.items():
            if k == "aug_level":
                a.level = v
                aug_vals = AUG_LEVELS.get(v, AUG_LEVELS["Medio"])
                for ak, av in aug_vals.items():
                    setattr(a, ak, av)
            elif hasattr(p, k):
                setattr(p, k, v)
        self.model.preset_name = name
        self.params_changed.emit()
        self.aug_changed.emit()
        self.model_changed.emit()


# ────────────────────────────────────────────────────────────────────
# Pre-flight validation: dataset + parameters before training
# ────────────────────────────────────────────────────────────────────
def _num(value) -> Optional[float]:
    """float(valor) or None if it is not numeric (None, empty str, text)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _in_range(value, lo: float, hi: float) -> bool:
    """True if valor is numeric and lo <= valor <= hi. Not numeric = out."""
    v = _num(value)
    return v is not None and lo <= v <= hi


def validate_training_config(state: TrainerState, *, con_dataset: bool = True) -> tuple[bool, list[str]]:
    """Strict pre-flight validation before launching a training.

    Returns (ok, messages). ok=False if something prevents or compromises
    the run: data.yaml missing/corrupt, missing splits or labels, or
    hyperparameters out of range. It reuses ``audit_dataset()`` for the
    dataset part (the same deep audit the Dataset tab shows) and checks the
    parameter limits separately, which Ultralytics validates late or with a
    one-line error mid-run.

    ``con_dataset=False`` is the cheap mode of the UI reactivity: it still
    requires a loaded data.yaml and validates all the parameters, but skips
    the disk audit (which reads one .txt per image). The pages use it when
    parameters, model or augmentation change, so the dataset is not scanned
    on every keypress; the full audit runs when ``dataset_changed`` arrives
    and always before starting the runner.
    """
    msgs: List[str] = []
    ok = True

    def _error(msg: str) -> None:
        nonlocal ok
        ok = False
        msgs.append(msg)

    # ── 1) Model ──
    if state.model.family not in ("v8", "v11"):
        _error(f"Familia de modelo desconocida: «{state.model.family}» (use «v8» o «v11»).")
    if state.model.size not in ("n", "s", "m", "l", "x"):
        _error(f"Tamaño de modelo desconocido: «{state.model.size}» (use n, s, m, l o x).")

    # ── 2) Hyperparameters ──
    p = state.params
    if not _in_range(p.epochs, 1, 10000):
        _error(f"Épocas fuera de rango: {p.epochs} (use un entero ≥ 1).")
    if not _in_range(p.batch, 1, 512):
        _error(f"Batch fuera de rango: {p.batch} (use un entero ≥ 1).")
    try:
        imgsz = int(p.imgsz)
    except (TypeError, ValueError):
        imgsz = -1
    if not (32 <= imgsz <= 8192) or imgsz % 32 != 0:
        _error(f"imgsz fuera de rango: {p.imgsz} (use un múltiplo de 32 entre 32 y 8192).")
    if not _DEVICE_RE.match((p.device or "").strip()):
        _error(f"Device inválido: «{p.device}» (use «cpu», «mps», «0» o «cuda:0»).")
    if not _in_range(p.lr0, 1e-9, 1.0):
        _error(f"lr0 fuera de rango: {p.lr0} (use un valor > 0).")
    if not _in_range(p.weight_decay, 0.0, 1.0):
        _error(f"weight_decay fuera de rango: {p.weight_decay} (use un valor en [0, 1]).")
    if not _in_range(p.momentum, 0.0, 1.0):
        _error(f"momentum fuera de rango: {p.momentum} (use un valor en [0, 1]).")
    if not _in_range(p.warmup_epochs, 0, 10**6):
        _error(f"warmup_epochs fuera de rango: {p.warmup_epochs} (use un entero ≥ 0).")
    if not _in_range(p.close_mosaic, 0, 10**6):
        _error(f"close_mosaic fuera de rango: {p.close_mosaic} (use un entero ≥ 0).")
    ls = _num(p.label_smoothing)
    if ls is None or not (0.0 <= ls < 1.0):
        _error(f"label_smoothing fuera de rango: {p.label_smoothing} (use un valor en [0, 1)).")
    if not _in_range(p.patience, 0, 10**6):
        _error(f"patience fuera de rango: {p.patience} (use un entero ≥ 0).")
    if p.save_period != -1 and not _in_range(p.save_period, 1, 10**6):
        _error(f"save_period fuera de rango: {p.save_period} (use -1 o un entero ≥ 1).")
    if not _in_range(p.workers, 0, 64):
        _error(f"workers fuera de rango: {p.workers} (use un entero entre 0 y 64).")
    cache = p.cache
    if not (isinstance(cache, bool)
            or (isinstance(cache, str) and cache.strip().lower() in ("ram", "disk", "true", "false"))):
        _error(f"cache inválido: {cache!r} (use «disk», «ram», «true» o «false»).")

    a = state.aug
    for name_2, value in (("hsv_h", a.hsv_h), ("hsv_s", a.hsv_s), ("hsv_v", a.hsv_v),
                         ("fliplr", a.fliplr), ("mosaic", a.mosaic),
                         ("mixup", a.mixup), ("copy_paste", a.copy_paste)):
        if not _in_range(value, 0.0, 1.0):
            _error(f"{name_2} fuera de rango: {value} (use un valor en [0, 1]).")

    # ── 3) Dataset: YAML integrity, presence of splits and labels ──
    yaml_path = state.dataset.yaml_path
    if yaml_path is None:
        _error("No hay data.yaml cargado (ve a la pestaña Dataset).")
        return ok, msgs
    if not con_dataset:
        # Cheap mode: the dataset is loaded and its structure is audited separately
        # (dataset_changed / `Validar ahora` / the startup gate). Here it only
        # reports what just changed: the parameters.
        return ok, msgs
    if not Path(yaml_path).exists():
        _error(f"data.yaml no encontrado: {yaml_path}")
        return ok, msgs

    try:
        audit = audit_dataset(yaml_path)
    except Exception as e:  # the audit must not bring down the pre-flight
        _error(f"No se pudo analizar el dataset: {e}")
        return ok, msgs

    for f in audit.findings:
        text_2 = f"{f.title}. {f.detail}".strip()
        if f.level == LEVEL_ERROR:
            _error(text_2)
        elif f.level == LEVEL_WARN:
            msgs.append(f"[aviso] {text_2}")
    # Train without any valid labels: the model would learn only background. The
    # audit flags it as a per-image warning; here it is blocking.
    labels_train = sum(audit.per_class.get("train", {}).values())
    if audit.ok and labels_train == 0:
        _error("No hay etiquetas válidas en train: el modelo aprendería solo fondo.")
    elif audit.ok:
        msgs.append("Dataset válido: splits, labels y clases correctos.")

    return ok, msgs
