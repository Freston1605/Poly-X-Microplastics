"""polyx.core — model-agnostic science engine.

This package contains the pure-Python, Qt-free, ultralytics-free
computational engine. It is safe to import in headless environments
(CI, servers, CLI tools) without a display or GPU.
"""

from . import data, detect, measure, quality
from .data.labels import (
    classes_txt_path,
    read_classes_txt,
    read_labels,
    write_classes_txt,
    write_labels,
)
from .detect import (
    Detections,
    DetectionSource,
    Instance,
    NativeDetector,
    YoloModel,
    list_detectors,
    make_detector,
)
from .detection_run import run_batch
from .evaluation import EvalConfig
from .evaluation import evaluate as eval_model
from .evaluation import write_report as write_eval_report
from .measure import (
    Calibration,
    calibrate_from_plate,
    calibration,
    measure,
    morphology,
    resolver,
    segment,
)
from .provenance import (
    event,
    logger_for,
    observability,
    provenance,
    sha256_file,
    timer,
)
from .provenance import provenance as provenance_mod
from .quality import (
    aggregate_per_class,
    assignment,
    audit_dataset,
    dataset_audit,
    match_image,
    metrics,
)
from .report_simple import generate as generate_simple_report
from .training import TrainingConfig
from .training import train as train_model

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
