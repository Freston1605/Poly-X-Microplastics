"""Model-family adapters under one Lightning loop.

Each adapter translates the framework-neutral protocol into its framework's
calls and reports back in the same currency: COCO predictions per epoch and a
checkpoint path. The YOLO detection adapter reuses the Ultralytics neural
module and dataloader inside a Lightning module instead of reimplementing YOLO
training; the YOLO-seg adapter does the same on top of a job-time conversion of
the canonical COCO mask dataset into scratch.

Lightning, Ultralytics and torch are imported lazily so importing the headless
command never loads a GPU stack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyx.core.evaluation import _metrics_from_results
from polyx.core.quality.dataset_audit import DatasetAudit, audit_coco_masks, audit_dataset
from polyx.pipeline.coco import (
    coco_predictions_from_results,
    coco_segmentation_predictions_from_results,
)
from polyx.pipeline.config import ExecutionConfig, Family, Protocol
from polyx.pipeline.dataset import (
    class_names,
    coco_ground_truth,
    coco_store_ground_truth,
    coco_store_hashes,
    dataset_hashes,
    ordered_categories,
    read_coco_store,
    read_data_yaml,
    split_directory,
    split_images,
)
from polyx.pipeline.resume import ResumeState
from polyx.pipeline.seg_conversion import convert_coco_dataset

EpochCallback = Callable[["EpochResult"], bool]


@dataclass
class EpochResult:
    """What one validation epoch hands back to the shared training engine."""

    epoch: int
    predictions: list[dict]
    adapter_figures: dict
    checkpoint: Path


@dataclass
class PreparedDataset:
    """The dataset one adapter trains on, resolved before the fit starts.

    ``data_yaml`` is whatever declaration the framework consumes; for a
    segmentation family it points at the scratch conversion, never at a
    canonical dataset. ``ground_truth`` stays COCO in memory, which is what the
    shared evaluator scores.
    """

    data_yaml: Path
    ground_truth: dict
    audit: DatasetAudit
    hashes: dict[str, str]
    class_names: dict[int, str]
    source_kind: str
    source_path: Path
    conversion: dict[str, Any] | None = None


@dataclass
class FitResult:
    """Observable state of the finished Lightning fit."""

    precision: str
    epochs_completed: int
    amp_active: bool
    device: str
    framework: str
    weights: str


def _rebuild_network(module: Any, task: str, class_count: int) -> Any:
    """Size the pretrained Ultralytics network for the protocol's dataset.

    Ultralytics builds the head when the model is created for a dataset, not
    when a checkpoint is loaded. Training the checkpoint's 80-class head on a
    three-class dataset would silently train the wrong thing, so the network is
    rebuilt from its own yaml with the dataset's class count and the pretrained
    weights are transferred into the matching layers.
    """
    cfg = getattr(module, "yaml", None)
    if not isinstance(cfg, dict):
        return module
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel

    model_class = SegmentationModel if task == "segment" else DetectionModel
    network = model_class(dict(cfg), ch=int(cfg.get("ch", 3)), nc=class_count, verbose=False)
    network.load(module)
    return network


def _ultralytics_cfg(task: str, image_size: int, batch: int) -> Any:
    """The Ultralytics argument namespace the dataloader and loss both read."""
    from ultralytics.cfg import get_cfg

    return get_cfg(
        overrides={
            "task": task,
            "imgsz": image_size,
            "batch": batch,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "fraction": 1.0,
            "deterministic": True,
            "workers": 0,
        }
    )


def _ultralytics_dataset(data_yaml: Path, cfg: Any, *, mode: str) -> Any:
    """Build one of the Ultralytics datasets the family's loader consumes.

    The conversion writes the family's folder contract; this reuses the
    Ultralytics dataset and batch collation instead of reimplementing them.
    """
    from ultralytics.data import build_yolo_dataset

    data = read_data_yaml(data_yaml)
    names = class_names(data)
    split = "train" if mode == "train" else "val"
    directory = split_directory(data_yaml, data, split)
    if directory is None or not directory.is_dir():
        raise ValueError(f"the dataset declaration has no usable '{split}' split: {data_yaml}")
    val_directory = split_directory(data_yaml, data, "val")
    payload: dict[str, Any] = {
        "names": names,
        "nc": len(names),
        "channels": 3,
        "train": str(directory) if split == "train" else "",
        "val": str(val_directory) if val_directory is not None else "",
    }
    return build_yolo_dataset(
        cfg,
        str(directory),
        int(cfg.batch),
        payload,
        mode=mode,
        rect=False,
        stride=32,
    )


def _build_optimizer(net: Any, protocol: Protocol) -> Any:
    """Protocol optimizer over the Ultralytics network's parameter groups."""
    import torch

    groups: dict[str, list[Any]] = {"bias": [], "norm": [], "weight": []}
    norms = tuple(value for name, value in torch.nn.__dict__.items() if "Norm" in name)
    for module_name, module in net.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:
                groups["bias"].append(param)
            elif isinstance(module, norms):
                groups["norm"].append(param)
            else:
                groups["weight"].append(param)

    name = protocol.optimizer.lower()
    parameter_groups = [
        {"params": groups["weight"], "weight_decay": protocol.weight_decay},
        {"params": groups["norm"], "weight_decay": 0.0},
        {"params": groups["bias"], "weight_decay": 0.0},
    ]
    non_empty = [group for group in parameter_groups if group["params"]]
    if not non_empty:
        raise ValueError("the network has no trainable parameters")
    first = non_empty[0]
    if name == "sgd":
        optimizer: Any = torch.optim.SGD(
            first["params"],
            lr=protocol.learning_rate,
            momentum=protocol.momentum,
            nesterov=True,
            weight_decay=float(first["weight_decay"]),
        )
    elif name in {"adam", "adamw"}:
        optimizer_class = torch.optim.Adam if name == "adam" else torch.optim.AdamW
        optimizer = optimizer_class(
            first["params"],
            lr=protocol.learning_rate,
            weight_decay=float(first["weight_decay"]),
        )
    else:
        raise ValueError(f"unsupported optimizer in the protocol: {protocol.optimizer}")
    for group in non_empty[1:]:
        optimizer.add_param_group(group)
    return optimizer


def _lr_factor(protocol: Protocol) -> Callable[[int], float]:
    """Warmup plus the protocol's schedule, as a per-epoch multiplier."""
    import math

    warmup = max(0, protocol.warmup_epochs)
    total = max(1, protocol.epochs)

    def factor(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(1.0, max(0.0, (epoch - warmup) / max(1, total - warmup)))
        if protocol.schedule == "cosine":
            # Decay to one percent, the same floor Ultralytics' cosine uses.
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
        if protocol.schedule == "constant":
            return 1.0
        raise ValueError(f"unsupported learning-rate schedule: {protocol.schedule}")

    return factor


class FamilyAdapter:
    """Contract every family implements; later families are siblings."""

    name = ""
    task = ""

    def prepare(self, config: ExecutionConfig) -> PreparedDataset:
        """Resolve the dataset and audit it, before any artifact is trained."""
        raise NotImplementedError

    def fit(
        self,
        config: ExecutionConfig,
        run_dir: Path,
        dataset: PreparedDataset,
        on_epoch: EpochCallback,
        resume: ResumeState | None = None,
    ) -> FitResult:
        raise NotImplementedError


class _UltralyticsAdapter(FamilyAdapter):
    """Shared Ultralytics wrapper: one Lightning module, two label currencies."""

    task = ""

    def _prepare_boxes(self, config: ExecutionConfig) -> PreparedDataset:
        if config.data_yaml is None:
            raise ValueError("a detection family needs a YOLO box dataset (data_yaml)")
        audit = audit_dataset(config.data_yaml)
        data = read_data_yaml(config.data_yaml)
        return PreparedDataset(
            data_yaml=config.data_yaml,
            ground_truth=coco_ground_truth(config.data_yaml, config.protocol.split),
            audit=audit,
            hashes=dataset_hashes(config.data_yaml),
            class_names=class_names(data),
            source_kind="yolo_boxes",
            source_path=config.data_yaml,
        )

    def _prepare_masks(self, config: ExecutionConfig) -> PreparedDataset:
        if config.coco_dataset is None:
            raise ValueError("a segmentation family needs a COCO mask dataset")
        store = config.coco_dataset
        coco = read_coco_store(store)
        audit = audit_coco_masks(coco)
        if not audit.ok:
            return PreparedDataset(
                data_yaml=Path(),
                ground_truth={},
                audit=audit,
                hashes={},
                class_names={},
                source_kind="coco_instances",
                source_path=store,
            )
        conversion = convert_coco_dataset(store, config.scratch_dir)
        return PreparedDataset(
            data_yaml=conversion.data_yaml,
            ground_truth=coco_store_ground_truth(coco, config.protocol.split),
            audit=audit,
            hashes=coco_store_hashes(store, coco),
            class_names={index: name for index, (_id, name) in enumerate(ordered_categories(coco))},
            source_kind="coco_instances",
            source_path=store,
            conversion=conversion.describe(store),
        )

    def _predictions(self, results: list[Any]) -> list[dict]:
        raise NotImplementedError

    def fit(
        self,
        config: ExecutionConfig,
        run_dir: Path,
        dataset: PreparedDataset,
        on_epoch: EpochCallback,
        resume: ResumeState | None = None,
    ) -> FitResult:
        from lightning import Callback, LightningModule, Trainer, seed_everything
        from ultralytics import YOLO

        protocol = config.protocol
        device = config.backend.device
        seed_everything(protocol.seed, workers=True)
        resumed_checkpoint = resume.last_checkpoint if resume is not None else None
        if resumed_checkpoint is not None and Path(resumed_checkpoint).is_file():
            model = YOLO(str(resumed_checkpoint))
        else:
            model = YOLO(config.family.weights)
        # The pretrained head is sized for its own dataset; the protocol's
        # dataset decides the class count, so the Ultralytics network is rebuilt
        # from its yaml and the pretrained weights are transferred into it.
        model.model = _rebuild_network(model.model, config.family.task, len(dataset.class_names))
        checkpoint_dir = Path(run_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        predict = self._predictions

        class _YoloLightningModule(LightningModule):  # type: ignore[misc,valid-type]
            """One Ultralytics module trained by the Lightning loop."""

            def __init__(self) -> None:
                super().__init__()
                # Only the torch network is a child module. The Ultralytics
                # wrapper subclasses nn.Module and overrides ``train`` to launch
                # its own trainer, so registering it would turn Lightning's
                # model.eval() into a full Ultralytics training job.
                self.net = model.model
                self.coco_predictions: list[dict] = []
                self.adapter_figures: dict = {}
                self.checkpoint_path = checkpoint_dir / "epoch_000.pt"
                self._train_dataset: Any | None = None
                self._val_dataset: Any | None = None
                self._cfg: Any | None = None
                self.stop_requested = False

            def _arguments(self) -> Any:
                """Build the Ultralytics args once, when training really starts.

                Doing it lazily keeps a fake ``lightning``/``ultralytics`` in a
                test from importing the real stack, and gives the loss the same
                arguments the dataloader was built with.
                """
                if self._cfg is None:
                    self._cfg = _ultralytics_cfg(
                        config.family.task, protocol.image_size, protocol.batch
                    )
                    self.net.args = self._cfg
                return self._cfg

            def train_dataloader(self) -> Any:
                from torch.utils.data import DataLoader

                if self._train_dataset is None:
                    self._train_dataset = _ultralytics_dataset(
                        dataset.data_yaml, self._arguments(), mode="train"
                    )
                return DataLoader(
                    self._train_dataset,
                    batch_size=protocol.batch,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=self._train_dataset.collate_fn,
                )

            def val_dataloader(self) -> Any:
                # The protocol's validation is Ultralytics' own validator in
                # ``on_validation_epoch_end``; this loader only makes Lightning
                # run the validation loop that invokes it.
                from torch.utils.data import DataLoader

                if self._val_dataset is None:
                    self._val_dataset = _ultralytics_dataset(
                        dataset.data_yaml, self._arguments(), mode="val"
                    )
                return DataLoader(
                    self._val_dataset,
                    batch_size=protocol.batch,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=self._val_dataset.collate_fn,
                )

            def training_step(self, batch: Any, batch_idx: int) -> Any:
                # The Ultralytics network computes its own task loss from the
                # batch its dataloader produced; Lightning owns the step. The
                # Ultralytics loader yields uint8 images, which its native loop
                # scales to floats before the forward pass.
                import torch

                images = batch["img"]
                if images.dtype == torch.uint8:
                    batch["img"] = images.float() / 255.0
                else:
                    batch["img"] = images.float()
                self.net.train()
                loss, _items = self.net.loss(batch)
                return loss

            def validation_step(self, batch: Any, batch_idx: int) -> None:
                return None

            def configure_optimizers(self) -> Any:
                from torch.optim.lr_scheduler import LambdaLR

                optimizer = _build_optimizer(self.net, protocol)
                scheduler = LambdaLR(optimizer, lr_lambda=_lr_factor(protocol))
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
                }

            def on_validation_epoch_end(self) -> None:
                # Ultralytics writes its own plots under ``project``; keep that
                # ephemeral so the artifact layout stays the only output and no
                # global ``runs_dir`` is touched.
                validation_root = Path(config.scratch_dir) / "validation" / Path(run_dir).name
                raw = model.val(
                    data=str(dataset.data_yaml),
                    split=protocol.split,
                    imgsz=protocol.image_size,
                    batch=protocol.batch,
                    device=device,
                    project=str(validation_root),
                    name=str(getattr(self, "current_epoch", 0)),
                    plots=False,
                    save_json=False,
                    exist_ok=True,
                )
                results = list(raw) if isinstance(raw, (list, tuple)) else [raw]
                if not results or getattr(results[0], "boxes", None) is None:
                    images = split_images(dataset.data_yaml, protocol.split)
                    raw = model.predict(
                        source=[str(image) for image in images],
                        imgsz=protocol.image_size,
                        batch=protocol.batch,
                        device=device,
                    )
                    results = list(raw) if isinstance(raw, (list, tuple)) else [raw]
                self.coco_predictions = predict(results)
                self.adapter_figures = (
                    _metrics_from_results(results[0], dataset.class_names) if results else {}
                )
                epoch = int(getattr(self, "current_epoch", 0))
                self.checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                model.save(str(self.checkpoint_path))
                # The engine scores this epoch and decides whether to stop.
                # Lightning calls callbacks before the module hook, so the
                # decision is made here and the callback only forwards the flag
                # to loop implementations that do not expose ``self.trainer``.
                result = EpochResult(
                    epoch=epoch,
                    predictions=list(self.coco_predictions),
                    adapter_figures=dict(self.adapter_figures),
                    checkpoint=Path(self.checkpoint_path),
                )
                if on_epoch(result):
                    self.stop_requested = True
                    trainer = getattr(self, "trainer", None)
                    if trainer is not None:
                        trainer.should_stop = True

        class _ProtocolMonitor(Callback):  # type: ignore[misc,valid-type]
            """Forwards the module's stop decision to the training loop."""

            def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
                if getattr(pl_module, "stop_requested", False):
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
            num_sanity_val_steps=0,
            deterministic=True,
            default_root_dir=str(run_dir),
        )
        fit_kwargs: dict[str, Any] = {}
        if resumed_checkpoint is not None and Path(resumed_checkpoint).is_file():
            fit_kwargs["ckpt_path"] = str(resumed_checkpoint)
        trainer.fit(_YoloLightningModule(), **fit_kwargs)
        return FitResult(
            precision=precision,
            epochs_completed=int(trainer.current_epoch),
            amp_active=precision == "16-mixed",
            device=device,
            framework=config.family.framework,
            weights=config.family.weights,
        )


class YoloDetectionAdapter(_UltralyticsAdapter):
    """Ultralytics YOLO detection wrapped as one Lightning module."""

    name = "yolo_detection"
    task = "detect"

    def prepare(self, config: ExecutionConfig) -> PreparedDataset:
        return self._prepare_boxes(config)

    def _predictions(self, results: list[Any]) -> list[dict]:
        return coco_predictions_from_results(results)


class YoloSegmentationAdapter(_UltralyticsAdapter):
    """Ultralytics YOLO-seg trained from the canonical COCO mask dataset."""

    name = "yolo_segmentation"
    task = "segment"

    def prepare(self, config: ExecutionConfig) -> PreparedDataset:
        return self._prepare_masks(config)

    def _predictions(self, results: list[Any]) -> list[dict]:
        return coco_segmentation_predictions_from_results(results)


_ADAPTERS: dict[str, type[FamilyAdapter]] = {
    YoloDetectionAdapter.name: YoloDetectionAdapter,
    YoloSegmentationAdapter.name: YoloSegmentationAdapter,
}


def build_adapter(family: Family) -> FamilyAdapter:
    """Select the adapter named by configuration, never by a code edit."""
    try:
        adapter_class = _ADAPTERS[family.name]
    except KeyError as exc:
        raise ValueError(f"unknown model family: {family.name}") from exc
    return adapter_class()
