"""Hydra composition and validation of one headless training execution.

The configuration tree is data, not Python: ``conf/`` holds the framework-neutral
protocol, the execution backend and the model family. This module turns the
composed tree into plain dataclasses so the rest of the pipeline never sees an
OmegaConf object and the resolved protocol can be written to the artifacts
byte-for-byte as the adapter consumed it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from polyx.core.paths import SEGMENTATION_DATASETS_DIR

CONF_DIR = Path(__file__).resolve().parent / "conf"


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
    """Framework-neutral part of the configuration, shared by every adapter."""

    split: str = "val"
    seed: int = 42
    epochs: int = 150
    batch: int = 8
    image_size: int = 1280
    optimizer: str = "SGD"
    learning_rate: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    schedule: str = "cosine"
    warmup_epochs: int = 5
    augmentation_subset: list[str] = field(default_factory=list)
    amp: AmpConfig = field(default_factory=AmpConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class Backend:
    """Where one execution runs."""

    name: str = "local"
    device: str = "cpu"


@dataclass
class Family:
    """Which adapter executes the protocol."""

    name: str = "yolo_detection"
    framework: str = "ultralytics"
    weights: str = "yolov8n.pt"
    task: str = "detect"


@dataclass
class ExecutionConfig:
    """A fully resolved training execution or a COCO dataset import.

    When ``coco_import`` is set the command imports and audits a CVAT COCO
    export as the canonical segmentation training dataset; ``data_yaml`` is
    then unused. Otherwise the fields describe one headless training execution.
    """

    data_yaml: Path
    output_dir: Path
    protocol: Protocol
    backend: Backend
    family: Family
    dataset_name: str = "polyx"
    resume: bool = False
    coco_import: Path | None = None
    dataset_root: Path = SEGMENTATION_DATASETS_DIR


def _build_protocol(raw: dict) -> Protocol:
    amp = raw.get("amp") or {}
    stopping = raw.get("early_stopping") or {}
    return Protocol(
        split=str(raw["split"]),
        seed=int(raw["seed"]),
        epochs=int(raw["epochs"]),
        batch=int(raw["batch"]),
        image_size=int(raw["image_size"]),
        optimizer=str(raw["optimizer"]),
        learning_rate=float(raw["learning_rate"]),
        momentum=float(raw["momentum"]),
        weight_decay=float(raw["weight_decay"]),
        schedule=str(raw["schedule"]),
        warmup_epochs=int(raw["warmup_epochs"]),
        augmentation_subset=[str(item) for item in raw.get("augmentation_subset", [])],
        amp=AmpConfig(enabled=bool(amp.get("enabled", True)), dtype=str(amp.get("dtype", "fp16"))),
        early_stopping=EarlyStoppingConfig(
            monitor=str(stopping.get("monitor", "with_class_score")),
            mode=str(stopping.get("mode", "max")),
            patience=int(stopping.get("patience", 30)),
            min_delta=float(stopping.get("min_delta", 0.001)),
            restore_best=bool(stopping.get("restore_best", True)),
        ),
    )


def execution_config_from_raw(raw: dict[str, Any]) -> ExecutionConfig:
    """Validate a resolved config tree into the object the engine consumes."""
    output_dir = str(raw.get("output_dir") or "")
    if not output_dir:
        raise ValueError("output_dir is a required configuration override")
    data_yaml = str(raw.get("data_yaml") or "")
    coco_import = str(raw.get("coco_import") or "")
    if data_yaml and coco_import:
        raise ValueError("set exactly one of data_yaml or coco_import")
    if not data_yaml and not coco_import:
        raise ValueError("data_yaml or coco_import is a required configuration override")
    backend_raw = raw.get("backend") or {}
    family_raw = raw.get("family") or {}
    dataset_root = str(raw.get("dataset_root") or "")
    return ExecutionConfig(
        data_yaml=Path(data_yaml).expanduser().resolve() if data_yaml else Path(),
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
            task=str(family_raw.get("task", "detect")),
        ),
        dataset_name=str(raw.get("dataset_name", "polyx")),
        resume=bool(raw.get("resume", False)),
        coco_import=Path(coco_import).expanduser().resolve() if coco_import else None,
        dataset_root=(
            Path(dataset_root).expanduser().resolve() if dataset_root else SEGMENTATION_DATASETS_DIR
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
    data["data_yaml"] = str(config.data_yaml)
    data["output_dir"] = str(config.output_dir)
    data["coco_import"] = str(config.coco_import) if config.coco_import else ""
    data["dataset_root"] = str(config.dataset_root)
    return data


def protocol_to_dict(config: ExecutionConfig) -> dict[str, Any]:
    """The protocol exactly as it was handed to the adapter."""
    return asdict(config.protocol)
