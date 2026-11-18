"""Model-family adapters under one Lightning loop.

Each adapter translates the framework-neutral protocol into its framework's
calls and reports back in the same currency: COCO predictions per epoch and a
checkpoint path. The YOLO detection adapter reuses the Ultralytics neural
module and dataloader inside a Lightning module instead of reimplementing YOLO
training. Lightning, Ultralytics and torch are imported lazily so importing the
headless command never loads a GPU stack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyx.core.evaluation import _metrics_from_results
from polyx.pipeline.coco import coco_predictions_from_results
from polyx.pipeline.config import ExecutionConfig, Family
from polyx.pipeline.dataset import class_names, read_data_yaml, split_images

EpochCallback = Callable[["EpochResult"], bool]


@dataclass
class EpochResult:
    """What one validation epoch hands back to the shared training engine."""

    epoch: int
    predictions: list[dict]
    adapter_figures: dict
    checkpoint: Path


@dataclass
class FitResult:
    """Observable state of the finished Lightning fit."""

    precision: str
    epochs_completed: int
    amp_active: bool
    device: str
    framework: str
    weights: str


class FamilyAdapter:
    """Contract every family implements; later families are siblings."""

    name = ""

    def fit(
        self,
        config: ExecutionConfig,
        run_dir: Path,
        on_epoch: EpochCallback,
    ) -> FitResult:
        raise NotImplementedError


class YoloDetectionAdapter(FamilyAdapter):
    """Ultralytics YOLO detection wrapped as one Lightning module."""

    name = "yolo_detection"

    def fit(
        self,
        config: ExecutionConfig,
        run_dir: Path,
        on_epoch: EpochCallback,
    ) -> FitResult:
        from lightning import Callback, LightningModule, Trainer, seed_everything
        from ultralytics import YOLO

        protocol = config.protocol
        device = config.backend.device
        seed_everything(protocol.seed, workers=True)
        model = YOLO(config.family.weights)
        checkpoint_dir = Path(run_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        class _YoloLightningModule(LightningModule):  # type: ignore[misc,valid-type]
            """One Ultralytics module trained by the Lightning loop."""

            def __init__(self) -> None:
                super().__init__()
                self.model = model
                self.coco_predictions: list[dict] = []
                self.adapter_figures: dict = {}
                self.checkpoint_path = checkpoint_dir / "epoch_000.pt"

            def on_validation_epoch_end(self) -> None:
                raw = model.val(
                    data=str(config.data_yaml),
                    split=protocol.split,
                    imgsz=protocol.image_size,
                    batch=protocol.batch,
                    device=device,
                )
                results = list(raw) if isinstance(raw, (list, tuple)) else [raw]
                if not results or getattr(results[0], "boxes", None) is None:
                    images = split_images(config.data_yaml, protocol.split)
                    raw = model.predict(
                        source=[str(image) for image in images],
                        imgsz=protocol.image_size,
                        batch=protocol.batch,
                        device=device,
                    )
                    results = list(raw) if isinstance(raw, (list, tuple)) else [raw]
                self.coco_predictions = coco_predictions_from_results(results)
                names = class_names(read_data_yaml(config.data_yaml))
                self.adapter_figures = _metrics_from_results(results[0], names) if results else {}
                epoch = int(getattr(self, "current_epoch", 0))
                self.checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                model.save(str(self.checkpoint_path))

        class _ProtocolMonitor(Callback):  # type: ignore[misc,valid-type]
            """Bridges the Lightning loop to the engine's shared evaluation."""

            def on_validation_epoch_end(self, trainer: Any, module: Any) -> None:
                result = EpochResult(
                    epoch=int(getattr(trainer, "current_epoch", 0)),
                    predictions=list(module.coco_predictions),
                    adapter_figures=dict(module.adapter_figures),
                    checkpoint=Path(module.checkpoint_path),
                )
                if on_epoch(result):
                    trainer.should_stop = True

        precision = (
            "16-mixed" if protocol.amp.enabled and protocol.amp.dtype == "fp16" else "32-true"
        )
        accelerator = "cpu" if device in ("cpu", "mps") else "gpu"
        trainer = Trainer(
            max_epochs=protocol.epochs,
            precision=precision,
            accelerator=accelerator,
            devices=1,
            callbacks=[_ProtocolMonitor()],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            default_root_dir=str(run_dir),
        )
        trainer.fit(_YoloLightningModule())
        return FitResult(
            precision=precision,
            epochs_completed=int(trainer.current_epoch),
            amp_active=precision == "16-mixed",
            device=device,
            framework=config.family.framework,
            weights=config.family.weights,
        )


_ADAPTERS: dict[str, type[FamilyAdapter]] = {YoloDetectionAdapter.name: YoloDetectionAdapter}


def build_adapter(family: Family) -> FamilyAdapter:
    """Select the adapter named by configuration, never by a code edit."""
    try:
        adapter_class = _ADAPTERS[family.name]
    except KeyError as exc:
        raise ValueError(f"unknown model family: {family.name}") from exc
    return adapter_class()
