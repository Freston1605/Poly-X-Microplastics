"""Hydra composition and validation of one headless training execution.

The configuration tree is data, not Python: ``conf/`` holds the framework-neutral
protocol, the execution backend and the model family. This module turns the
composed tree into plain dataclasses so the rest of the pipeline never sees an
OmegaConf object and the resolved protocol can be written to the artifacts
byte-for-byte as the adapter consumed it.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from polyx.core.paths import SEGMENTATION_DATASETS_DIR

CONF_DIR = Path(__file__).resolve().parent / "conf"

#: The two tasks an adapter can execute. Detection consumes YOLO box labels;
#: segmentation consumes the canonical COCO mask dataset.
_TASKS = ("detect", "segment")

#: Augmentations every first-wave family can express identically. The
#: benchmark campaign may not declare anything outside this set: a transform
#: one family skips would make the model result depend on the augmentation.
#: Family-specific transforms belong in an exploratory run.
BENCHMARK_AUGMENTATIONS = frozenset({"hsv", "flip_lr"})


def default_scratch_dir() -> Path:
    """Scratch root for ephemeral per-family conversion intermediates.

    ``POLYX_HPC_SCRATCH`` is where the cluster exposes scratch; without it the
    system temporary directory is used for local debugging. Nothing written
    there is an artifact: a missing cache is simply rebuilt.
    """
    override = os.environ.get("POLYX_HPC_SCRATCH")
    base = Path(override).expanduser() if override else Path(tempfile.gettempdir())
    return (base / "polyx" / "pipeline").resolve()


@dataclass
class AmpConfig:
    """Mixed-precision request from the protocol."""

    enabled: bool = True
    dtype: str = "fp16"


@dataclass
class EarlyStoppingConfig:
    """Early-stopping request from the protocol."""

    monitor: str = "with_class_score"
    mode: str = "max"
    patience: int = 30
    min_delta: float = 0.001
    restore_best: bool = True


@dataclass
class Protocol:
    """Framework-neutral part of the configuration, shared by every adapter.

    ``augmentation_subset`` is the benchmark's common subset: transforms every
    first-wave family applies with the same magnitudes. ``exploratory_augmentation``
    is for runs where the benchmark is not the point; a benchmark execution
    refuses it and refuses any subset entry outside the common set.
    """

    split: str = "val"
    seed: int = 42
    epochs: int = 150
    batch: int = 8
    image_size: int = 1280
    #: Confidence threshold for validation predictions and for the operating
    #: point the shared evaluator scores. Recorded in every metrics artifact.
    confidence: float = 0.25
    optimizer: str = "SGD"
    learning_rate: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    schedule: str = "cosine"
    warmup_epochs: int = 5
    augmentation_subset: list[str] = field(default_factory=lambda: ["hsv", "flip_lr"])
    exploratory_augmentation: list[str] = field(default_factory=list)
    benchmark: bool = True
    # Magnitudes both families apply identically. The arithmetic is
    # Ultralytics' ``RandomHSV``/``RandomFlip``, so the torchvision sibling
    # reproduces it instead of choosing its own; the numbers are the
    # protocol's own values, not either family's defaults.
    hsv_h: float = 0.015
    hsv_s: float = 0.4
    hsv_v: float = 0.4
    flip_lr_probability: float = 0.5
    amp: AmpConfig = field(default_factory=AmpConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class Backend:
    """Where one execution runs."""

    name: str = "local"
    device: str = "cpu"


@dataclass
class Family:
    """Which adapter executes the protocol.

    ``run_name`` is the directory this execution occupies inside a campaign
    root. It defaults to the family name; two detection variants (say v8-n and
    v11-n) coexist in one campaign by declaring different run names.
    """

    name: str = "yolo_detection"
    framework: str = "ultralytics"
    weights: str = "yolov8n.pt"
    task: str = "detect"
    run_name: str = ""


@dataclass
class ExecutionConfig:
    """A fully resolved training execution or a COCO dataset import.

    Exactly one dataset source is set. ``data_yaml`` declares a YOLO box
    dataset for a detection family; ``coco_dataset`` points at a versioned COCO
    mask tool output for a segmentation family; ``coco_import`` imports and
    audits a CVAT export as the canonical segmentation training dataset.
    """

    data_yaml: Path | None
    output_dir: Path
    protocol: Protocol
    backend: Backend
    family: Family
    dataset_name: str = "polyx"
    resume: bool = False
    coco_import: Path | None = None
    coco_dataset: Path | None = None
    dataset_root: Path = SEGMENTATION_DATASETS_DIR
    scratch_dir: Path = field(default_factory=default_scratch_dir)
    weights_dir: Path | None = None
    weights_manifest: Path | None = None


def _build_protocol(raw: dict) -> Protocol:
    amp = raw.get("amp") or {}
    stopping = raw.get("early_stopping") or {}
    subset = [str(item) for item in raw.get("augmentation_subset", ["hsv", "flip_lr"])]
    exploratory = [str(item) for item in raw.get("exploratory_augmentation", [])]
    benchmark = bool(raw.get("benchmark", True))
    if benchmark:
        outside = sorted((set(subset) | set(exploratory)) - BENCHMARK_AUGMENTATIONS)
        if outside:
            raise ValueError(
                "benchmark augmentation must be the family-neutral common subset "
                f"{sorted(BENCHMARK_AUGMENTATIONS)}; move {outside} to "
                "protocol.exploratory_augmentation and set protocol.benchmark=false"
            )
    probability = float(raw.get("flip_lr_probability", 0.5))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("protocol.flip_lr_probability must be within [0, 1]")
    confidence = float(raw.get("confidence", 0.25))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("protocol.confidence must be within [0, 1]")
    # Lightning's callbacks own early stopping and checkpoint selection; a
    # value they cannot honour is refused here instead of being reinterpreted.
    monitor = str(stopping.get("monitor", "with_class_score"))
    if monitor != "with_class_score":
        raise ValueError(f"unsupported early-stopping monitor: {monitor}")
    mode = str(stopping.get("mode", "max"))
    if mode != "max":
        raise ValueError(f"unsupported early-stopping mode: {mode}")
    if not bool(stopping.get("restore_best", True)):
        raise ValueError(
            "early_stopping.restore_best=false is not supported: the kept epoch "
            "is always exported as best.pt"
        )
    return Protocol(
        split=str(raw["split"]),
        seed=int(raw["seed"]),
        epochs=int(raw["epochs"]),
        batch=int(raw["batch"]),
        image_size=int(raw["image_size"]),
        confidence=confidence,
        optimizer=str(raw["optimizer"]),
        learning_rate=float(raw["learning_rate"]),
        momentum=float(raw["momentum"]),
        weight_decay=float(raw["weight_decay"]),
        schedule=str(raw["schedule"]),
        warmup_epochs=int(raw["warmup_epochs"]),
        augmentation_subset=subset,
        exploratory_augmentation=exploratory,
        benchmark=benchmark,
        hsv_h=float(raw.get("hsv_h", 0.015)),
        hsv_s=float(raw.get("hsv_s", 0.4)),
        hsv_v=float(raw.get("hsv_v", 0.4)),
        flip_lr_probability=probability,
        amp=AmpConfig(enabled=bool(amp.get("enabled", True)), dtype=str(amp.get("dtype", "fp16"))),
        early_stopping=EarlyStoppingConfig(
            monitor=str(stopping.get("monitor", "with_class_score")),
            mode=str(stopping.get("mode", "max")),
            patience=int(stopping.get("patience", 30)),
            min_delta=float(stopping.get("min_delta", 0.001)),
            restore_best=bool(stopping.get("restore_best", True)),
        ),
    )


def _run_name(raw: dict) -> str:
    """The single directory-component name one campaign execution occupies."""
    candidate = str(raw.get("run_name") or raw.get("name") or "run").strip()
    if candidate in {".", ".."} or "/" in candidate or "\\" in candidate:
        raise ValueError(f"family.run_name must be one directory name, got {candidate!r}")
    return candidate


def execution_config_from_raw(raw: dict[str, Any]) -> ExecutionConfig:
    """Validate a resolved config tree into the object the engine consumes."""
    output_dir = str(raw.get("output_dir") or "")
    if not output_dir:
        raise ValueError("output_dir is a required configuration override")
    data_yaml = str(raw.get("data_yaml") or "")
    coco_import = str(raw.get("coco_import") or "")
    coco_dataset = str(raw.get("coco_dataset") or "")
    sources = [bool(data_yaml), bool(coco_import), bool(coco_dataset)]
    # An import is exclusive; a campaign job may carry both training sources so
    # one multirun can serve detection and segmentation families, and the
    # family's task decides which source it consumes.
    if coco_import and (data_yaml or coco_dataset):
        raise ValueError("set exactly one of data_yaml, coco_import or coco_dataset")
    if not any(sources):
        raise ValueError(
            "data_yaml (a box dataset), coco_dataset (a COCO mask dataset) or "
            "coco_import is a required configuration override"
        )
    backend_raw = raw.get("backend") or {}
    family_raw = raw.get("family") or {}
    task = str(family_raw.get("task", "detect"))
    if task not in _TASKS:
        raise ValueError(f"unknown family task: {task}")
    if not coco_import:
        if coco_dataset and not data_yaml and task != "segment":
            raise ValueError(
                "a COCO mask dataset requires a segmentation family; set "
                "family=yolo_segmentation or family=mask_rcnn instead of a detection family"
            )
        if data_yaml and not coco_dataset and task != "detect":
            raise ValueError(
                "a segmentation family needs a COCO mask dataset; set coco_dataset "
                "instead of data_yaml"
            )
        if task == "detect" and not data_yaml:
            raise ValueError("a detection family needs a YOLO box dataset; set data_yaml")
        if task == "segment" and not coco_dataset:
            raise ValueError("a segmentation family needs a COCO mask dataset; set coco_dataset")
    dataset_root = str(raw.get("dataset_root") or "")
    scratch_dir = str(raw.get("scratch_dir") or "")
    weights_dir = str(raw.get("weights_dir") or "")
    weights_manifest = str(raw.get("weights_manifest") or "")
    return ExecutionConfig(
        data_yaml=Path(data_yaml).expanduser().resolve() if data_yaml else None,
        output_dir=Path(output_dir).expanduser().resolve(),
        protocol=_build_protocol(raw.get("protocol") or {}),
        backend=Backend(
            name=str(backend_raw.get("name", "local")),
            device=str(backend_raw.get("device", "cpu")),
        ),
        family=Family(
            name=str(family_raw.get("name", "yolo_detection")),
            framework=str(family_raw.get("framework", "ultralytics")),
            weights=str(family_raw.get("weights", "yolov8n.pt")),
            task=task,
            run_name=_run_name(family_raw),
        ),
        dataset_name=str(raw.get("dataset_name", "polyx")),
        resume=bool(raw.get("resume", False)),
        coco_import=Path(coco_import).expanduser().resolve() if coco_import else None,
        coco_dataset=Path(coco_dataset).expanduser().resolve() if coco_dataset else None,
        dataset_root=(
            Path(dataset_root).expanduser().resolve() if dataset_root else SEGMENTATION_DATASETS_DIR
        ),
        scratch_dir=(
            Path(scratch_dir).expanduser().resolve() if scratch_dir else default_scratch_dir()
        ),
        weights_dir=Path(weights_dir).expanduser().resolve() if weights_dir else None,
        weights_manifest=(
            Path(weights_manifest).expanduser().resolve() if weights_manifest else None
        ),
    )


def load_config(overrides: Sequence[str] | None = None) -> ExecutionConfig:
    """Compose ``conf/`` with Hydra and validate the required overrides."""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        composed = compose(config_name="config", overrides=list(overrides or []))
    raw = cast("dict[str, Any]", OmegaConf.to_container(composed, resolve=True))
    return execution_config_from_raw(raw)


def config_to_dict(config: ExecutionConfig) -> dict[str, Any]:
    """Plain, YAML-serialisable view of the resolved execution."""
    data: dict[str, Any] = asdict(config)
    data["data_yaml"] = str(config.data_yaml) if config.data_yaml else ""
    data["output_dir"] = str(config.output_dir)
    data["coco_import"] = str(config.coco_import) if config.coco_import else ""
    data["coco_dataset"] = str(config.coco_dataset) if config.coco_dataset else ""
    data["dataset_root"] = str(config.dataset_root)
    data["scratch_dir"] = str(config.scratch_dir)
    data["weights_dir"] = str(config.weights_dir) if config.weights_dir else ""
    data["weights_manifest"] = str(config.weights_manifest) if config.weights_manifest else ""
    return data


def protocol_to_dict(config: ExecutionConfig) -> dict[str, Any]:
    """The protocol exactly as it was handed to the adapter."""
    return asdict(config.protocol)
