"""polyx.core — model-agnostic science engine.

This package contains the pure-Python, Qt-free, ultralytics-free
computational engine. It is safe to import in headless environments
(CI, servers, CLI tools) without a display or GPU.
"""

from .detect import (
    Instance,
    DetectionSource,
    Detections,
    make_detector,
    list_detectors,
    YoloModel,
    NativeDetector,
)
from .measure import (
    measure,
    segment,
    Calibration,
    calibrate_from_plate,
    resolver,
)
from .quality import (
    match_image,
    aggregate_per_class,
    audit_dataset,
)
from .provenance import (
    logger_for,
    event,
    timer,
    provenance,
    sha256_file,
)
from .provenance import observability
from . import measure
from . import quality
from . import detect
from . import data
from .measure import morphology
from .measure import calibration
from .quality import metrics
from .quality import assignment
from .quality import dataset_audit
from .provenance import provenance as provenance_mod
from .provenance import observability
from .training import train as train_model, TrainingConfig
from .evaluation import evaluate as eval_model, EvalConfig, write_report as write_eval_report
from .detection_run import run_batch
from .report_simple import generate as generate_simple_report
from .data.labels import (
    read_labels,
    write_labels,
    classes_txt_path,
    write_classes_txt,
    read_classes_txt,
)

__all__ = [
    # detect
    "Instance",
    "DetectionSource", 
    "Detections",
    "make_detector",
    "list_detectors",
    "YoloModel",
    "NativeDetector",
    # measure
    "measure",
    "segment", 
    "Calibration",
    "calibrate_from_plate",
    "resolver",
    # quality
    "match_image",
    "aggregate_per_class",
    "audit_dataset",
    "train_model",
    "TrainingConfig",
    "eval_model",
    "EvalConfig",
    "write_eval_report",
    "run_batch",
    "generate_simple_report",
    # provenance
    "logger_for",
    "event",
    "timer",
    "provenance",
    "sha256_file",
"observability",
    "measure",
    "quality",
    "detect",
    "data",
    "morphology",
    "calibration",
    "metrics",
    "assignment",
    "dataset_audit",
    # data
    "read_labels",
    "write_labels",
    "classes_txt_path",
    "write_classes_txt",
    "read_classes_txt",
]

__version__ = "2.0.0"
