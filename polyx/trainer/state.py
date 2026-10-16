"""Estado compartido del Entrenador entre todas las páginas."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import re

from PySide6.QtCore import QObject, Signal

from ..core.plataforma import dispositivo_disponible
from ..core.dataset_audit import audit_dataset, LEVEL_ERROR, LEVEL_WARN

# Device aceptado por Ultralytics: «cpu», «mps», «cuda[:N]» o «N[,N...]» (GPU por
# índice). Cualquier otra cosa hace fallar el run a mitad de camino.
_DEVICE_RE = re.compile(r"^(cpu|mps|cuda(:\d+)?|\d+(,\d+)*)$", re.IGNORECASE)


# ────────────────────────────────────────────────────────────────────
# Presets de entrenamiento (Suave/Balanceado/Estricto)
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
    "Personalizado": {},  # marcador
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
    family: str = "v8"      # "v8" o "v11"
    size: str = "m"          # n / s / m / l / x
    custom_weights: Optional[Path] = None
    preset_name: str = "Balanceado (recomendado)"
    # Entrena v8 y v11 seguidos con la MISMA configuracion, para poder atribuir
    # la diferencia de metricas a la arquitectura y no a los hiperparametros.
    comparar_familias: bool = False

    def familias_a_entrenar(self) -> list[str]:
        """Familias que se entrenaran en esta corrida, en orden."""
        return ["v8", "v11"] if self.comparar_familias else [self.family]

    def peso_de(self, familia: str) -> str:
        """Nombre del peso base de una familia concreta, mismo tamano."""
        fam = familia.lstrip("v") if familia != "v8" else familia
        return f"yolo{fam}{self.size}.pt"

    @property
    def name(self) -> str:
        """Nombre corto del modelo, sin extensión: «yolov8m», «yolo11n»."""
        return self.peso_de(self.family).removesuffix(".pt")

    @property
    def base_weights_name(self) -> str:
        """Nombre del peso base tal como lo publica Ultralytics (con extensión).

        Ojo con la nomenclatura, que es inconsistente y no perdona: la familia 8
        lleva "v" (yolov8m.pt) y la 11 no (yolo11m.pt). Construir el nombre
        pegando la familia tal cual daba "yolov11m.pt", que no existe en el
        catalogo y hacia fallar la descarga de toda la rama v11.
        """
        return self.peso_de(self.family)


@dataclass
class DatasetConfig:
    yaml_path: Optional[Path] = None
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    class_names: List[str] = field(default_factory=list)


@dataclass
class TrainParams:
    imgsz: int = 1280              # default ALTO (max practico) por requerimiento del usuario
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
    cache: str = "disk"            # "disk", "ram", o "False"
    # Tras entrenar, reelegir el checkpoint evaluandolo solo sobre el sedimento
    # real de la validacion. best.pt sale del mAP global, que en un dataset mixto
    # lo dominan las placas de laboratorio: el dominio equivocado para detectar
    # en fotos de terreno. Ver polyx/trainer/dominio.py.
    elegir_por_dominio_real: bool = True
    # Ver el comentario equivalente en detector/state.py: "0" es CUDA y no
    # existe en un Mac. Se elige el mejor disponible al construir el estado.
    device: str = field(default_factory=dispositivo_disponible)
    workers: int = 8
    project_name: str = "polyx_train"
    run_name: str = ""             # vacío = auto-fecha


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
    """Estado global del Entrenador."""

    # Señales
    model_changed = Signal()
    dataset_changed = Signal()
    params_changed = Signal()
    aug_changed = Signal()

    train_started = Signal()
    train_epoch = Signal(object)        # EpochMetrics
    train_log = Signal(str)             # línea de log de Ultralytics
    train_finished = Signal(str)        # ruta al best.pt
    train_aborted = Signal()
    train_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.model = ModelConfig()
        self.dataset = DatasetConfig()
        self.params = TrainParams()
        self.aug = AugParams()

        # historia de épocas durante el run actual
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
# Validación pre-vuelo: dataset + parámetros antes de entrenar
# ────────────────────────────────────────────────────────────────────
def _num(valor) -> Optional[float]:
    """float(valor) o None si no es numérico (None, str vacío, texto)."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _en_rango(valor, lo: float, hi: float) -> bool:
    """True si valor es numérico y lo <= valor <= hi. No numérico = fuera."""
    v = _num(valor)
    return v is not None and lo <= v <= hi


def validate_training_config(state: TrainerState, *, con_dataset: bool = True) -> tuple[bool, list[str]]:
    """Validación pre-vuelo estricta antes de lanzar un entrenamiento.

    Devuelve (ok, mensajes). ok=False si algo impide o compromete el run:
    data.yaml ausente/corrupto, splits o labels faltantes, o hiperparámetros
    fuera de rango. Reutiliza audit_dataset() para la parte de dataset (la
    misma auditoría profunda que muestra la pestaña Dataset) y chequea aparte
    los límites de parámetros, que Ultralytics valida tarde o con un error de
    una línea a mitad del run.

    ``con_dataset=False`` es el modo barato de la reactividad de la UI: sigue
    exigiendo un data.yaml cargado y valida todos los parámetros, pero omite
    la auditoría de disco (que lee un .txt por imagen). Las páginas lo usan
    cuando cambian parámetros, modelo o augmentación, para no escanear el
    dataset en cada tecla; la auditoría completa corre cuando llega
    ``dataset_changed`` y siempre antes de arrancar el runner.
    """
    msgs: List[str] = []
    ok = True

    def _error(msg: str) -> None:
        nonlocal ok
        ok = False
        msgs.append(msg)

    # ── 1) Modelo ──
    if state.model.family not in ("v8", "v11"):
        _error(f"Familia de modelo desconocida: «{state.model.family}» (use «v8» o «v11»).")
    if state.model.size not in ("n", "s", "m", "l", "x"):
        _error(f"Tamaño de modelo desconocido: «{state.model.size}» (use n, s, m, l o x).")

    # ── 2) Hiperparámetros ──
    p = state.params
    if not _en_rango(p.epochs, 1, 10000):
        _error(f"Épocas fuera de rango: {p.epochs} (use un entero ≥ 1).")
    if not _en_rango(p.batch, 1, 512):
        _error(f"Batch fuera de rango: {p.batch} (use un entero ≥ 1).")
    try:
        imgsz = int(p.imgsz)
    except (TypeError, ValueError):
        imgsz = -1
    if not (32 <= imgsz <= 8192) or imgsz % 32 != 0:
        _error(f"imgsz fuera de rango: {p.imgsz} (use un múltiplo de 32 entre 32 y 8192).")
    if not _DEVICE_RE.match((p.device or "").strip()):
        _error(f"Device inválido: «{p.device}» (use «cpu», «mps», «0» o «cuda:0»).")
    if not _en_rango(p.lr0, 1e-9, 1.0):
        _error(f"lr0 fuera de rango: {p.lr0} (use un valor > 0).")
    if not _en_rango(p.weight_decay, 0.0, 1.0):
        _error(f"weight_decay fuera de rango: {p.weight_decay} (use un valor en [0, 1]).")
    if not _en_rango(p.momentum, 0.0, 1.0):
        _error(f"momentum fuera de rango: {p.momentum} (use un valor en [0, 1]).")
    if not _en_rango(p.warmup_epochs, 0, 10**6):
        _error(f"warmup_epochs fuera de rango: {p.warmup_epochs} (use un entero ≥ 0).")
    if not _en_rango(p.close_mosaic, 0, 10**6):
        _error(f"close_mosaic fuera de rango: {p.close_mosaic} (use un entero ≥ 0).")
    ls = _num(p.label_smoothing)
    if ls is None or not (0.0 <= ls < 1.0):
        _error(f"label_smoothing fuera de rango: {p.label_smoothing} (use un valor en [0, 1)).")
    if not _en_rango(p.patience, 0, 10**6):
        _error(f"patience fuera de rango: {p.patience} (use un entero ≥ 0).")
    if p.save_period != -1 and not _en_rango(p.save_period, 1, 10**6):
        _error(f"save_period fuera de rango: {p.save_period} (use -1 o un entero ≥ 1).")
    if not _en_rango(p.workers, 0, 64):
        _error(f"workers fuera de rango: {p.workers} (use un entero entre 0 y 64).")
    cache = p.cache
    if not (isinstance(cache, bool)
            or (isinstance(cache, str) and cache.strip().lower() in ("ram", "disk", "true", "false"))):
        _error(f"cache inválido: {cache!r} (use «disk», «ram», «true» o «false»).")

    a = state.aug
    for nombre, valor in (("hsv_h", a.hsv_h), ("hsv_s", a.hsv_s), ("hsv_v", a.hsv_v),
                         ("fliplr", a.fliplr), ("mosaic", a.mosaic),
                         ("mixup", a.mixup), ("copy_paste", a.copy_paste)):
        if not _en_rango(valor, 0.0, 1.0):
            _error(f"{nombre} fuera de rango: {valor} (use un valor en [0, 1]).")

    # ── 3) Dataset: integridad del YAML, presencia de splits y labels ──
    yaml_path = state.dataset.yaml_path
    if yaml_path is None:
        _error("No hay data.yaml cargado (ve a la pestaña Dataset).")
        return ok, msgs
    if not con_dataset:
        # Modo barato: el dataset está cargado y su estructura se audita aparte
        # (dataset_changed / «Validar ahora» / el gate de arranque). Aquí solo
        # reporta lo que acaba de cambiar: los parámetros.
        return ok, msgs
    if not Path(yaml_path).exists():
        _error(f"data.yaml no encontrado: {yaml_path}")
        return ok, msgs

    try:
        audit = audit_dataset(yaml_path)
    except Exception as e:  # la auditoría no debe tumbar el pre-flight
        _error(f"No se pudo analizar el dataset: {e}")
        return ok, msgs

    for f in audit.findings:
        texto = f"{f.title}. {f.detail}".strip()
        if f.level == LEVEL_ERROR:
            _error(texto)
        elif f.level == LEVEL_WARN:
            msgs.append(f"[aviso] {texto}")
    # Train sin ninguna etiqueta válida: el modelo aprendería solo fondo. La
    # auditoría lo marca como aviso por imagen; aquí es bloqueante.
    etiquetas_train = sum(audit.per_class.get("train", {}).values())
    if audit.ok and etiquetas_train == 0:
        _error("No hay etiquetas válidas en train: el modelo aprendería solo fondo.")
    elif audit.ok:
        msgs.append("Dataset válido: splits, labels y clases correctos.")

    return ok, msgs
