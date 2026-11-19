"""Model-family adapters under one Lightning loop.

Each adapter translates the framework-neutral protocol into its framework's
calls and reports back in the same currency: COCO predictions per epoch and a
checkpoint path. The YOLO detection adapter reuses the Ultralytics neural
module and dataloader inside a Lightning module instead of reimplementing YOLO
training; the YOLO-seg adapter does the same on top of a job-time conversion of
the canonical COCO mask dataset into scratch. The Mask R-CNN adapter is the
sibling non-Ultralytics family: torchvision reads the canonical COCO instances
directly and the same Lightning loop, evaluator and artifact layout apply.

Lightning, Ultralytics, torch and torchvision are imported lazily so importing
the headless command never loads a GPU stack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from polyx.core.evaluation import _metrics_from_results
from polyx.core.quality.dataset_audit import DatasetAudit, audit_coco_masks, audit_dataset
from polyx.pipeline.coco import (
    coco_predictions_from_results,
    coco_segmentation_predictions_from_results,
    coco_segmentation_predictions_from_torchvision,
)
from polyx.pipeline.config import ExecutionConfig, Family, Protocol
from polyx.pipeline.dataset import (
    category_indices,
    class_names,
    coco_ground_truth,
    coco_store_ground_truth,
    coco_store_hashes,
    dataset_hashes,
    ordered_categories,
    read_coco_store,
    read_data_yaml,
    split_coco_images,
    split_directory,
    split_images,
)
from polyx.pipeline.evaluation import rasterize_segmentation
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

    ``data_yaml`` is whatever declaration the framework consumes: a YOLO
    declaration for detection, the scratch YOLO-seg conversion for Ultralytics
    segmentation, and the canonical ``dataset.json`` for torchvision, which
    reads COCO instances directly. ``ground_truth`` stays COCO in memory, which
    is what the shared evaluator scores.
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
    augmentation: dict[str, Any] = field(default_factory=dict)


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


def _augmentation_report(protocol: Protocol, supported: frozenset[str]) -> dict[str, Any]:
    """Which declared augmentations this family can apply without substitution.

    The protocol names one common subset for the benchmark campaign. A family
    reports what it applied and what it skipped; it never replaces a skipped
    transform with a framework-specific one, which would make the model the
    wrong variable.
    """
    declared = list(protocol.augmentation_subset)
    return {
        "subset": declared,
        "applied": [name for name in declared if name in supported],
        "skipped": [name for name in declared if name not in supported],
        "family_specific": False,
    }


def _ultralytics_cfg(task: str, protocol: Protocol) -> Any:
    """The Ultralytics argument namespace the dataloader and loss both read.

    Augmentation is restricted to the protocol's common subset: the declared
    transforms are enabled and every family-specific knob is switched off, so a
    YOLO run and a Mask R-CNN run differ in augmentation only where the
    protocol says so.
    """
    from ultralytics.cfg import get_cfg

    augment = protocol.augmentation_subset
    hsv = "hsv" in augment
    return get_cfg(
        overrides={
            "task": task,
            "imgsz": protocol.image_size,
            "batch": protocol.batch,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "fraction": 1.0,
            "deterministic": True,
            "workers": 0,
            "hsv_h": 0.015 if hsv else 0.0,
            "hsv_s": 0.7 if hsv else 0.0,
            "hsv_v": 0.4 if hsv else 0.0,
            "fliplr": 0.5 if "flip_lr" in augment else 0.0,
            "mosaic": 1.0 if "mosaic" in augment else 0.0,
            "close_mosaic": 0,
            # Everything outside the benchmark's common subset stays off.
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "bgr": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "crop_fraction": 1.0,
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


class _CocoMaskAdapter(FamilyAdapter):
    """Preparation shared by the families trained from canonical COCO masks."""

    #: Protocol augmentations this family can express without substitution.
    supported_augmentations: frozenset[str] = frozenset()

    def _prepare_masks(self, config: ExecutionConfig, *, convert: bool) -> PreparedDataset:
        """Audit the store, keep its COCO view in memory and resolve the input.

        Ultralytics needs the job-time YOLO-seg folder contract in scratch;
        torchvision reads the canonical ``dataset.json`` directly, so no second
        intermediate is written for it.
        """
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
        conversion = convert_coco_dataset(store, config.scratch_dir) if convert else None
        return PreparedDataset(
            data_yaml=conversion.data_yaml if conversion is not None else store / "dataset.json",
            ground_truth=coco_store_ground_truth(coco, config.protocol.split),
            audit=audit,
            hashes=coco_store_hashes(store, coco),
            class_names={index: name for index, (_id, name) in enumerate(ordered_categories(coco))},
            source_kind="coco_instances",
            source_path=store,
            conversion=conversion.describe(store) if conversion is not None else None,
        )


def _build_trainer(config: ExecutionConfig, run_dir: Path) -> tuple[Any, str]:
    """The one Lightning trainer and precision every family uses.

    The monitor forwards a module's stop decision, made from the shared
    evaluator's score, to the training loop.
    """
    from lightning import Callback, Trainer

    class _ProtocolMonitor(Callback):  # type: ignore[misc,valid-type]
        """Forwards the module's stop decision to the training loop."""

        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            if getattr(pl_module, "stop_requested", False):
                trainer.should_stop = True

    protocol = config.protocol
    precision = "16-mixed" if protocol.amp.enabled and protocol.amp.dtype == "fp16" else "32-true"
    device = config.backend.device
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
    return trainer, precision


def _torchvision_detector(weights_name: str, class_count: int, image_size: int) -> Any:
    """Build the torchvision detector and size its heads for the dataset.

    Like the Ultralytics wrapper, pretrained heads are rebuilt for the
    protocol's dataset: torchvision reserves label 0 for background, so the
    output layer needs ``class_count + 1`` classes. The protocol's image size
    replaces the pretrained transform's own 800/1333 defaults so the model
    family is not silently changing the input scale.
    """
    from torchvision.models import detection
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    size = {"min_size": int(image_size), "max_size": int(image_size)}
    name = str(weights_name or "").strip().lower()
    if name in {"", "none", "random"}:
        model = detection.maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **size)
    else:
        factory = getattr(detection, name, None)
        if factory is None:
            raise ValueError(f"unsupported torchvision segmentation model: {weights_name}")
        model = factory(weights="DEFAULT", **size)
    classes = class_count + 1
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, classes)
    return model


class _CocoMaskTorchvisionDataset:
    """torchvision view of one split of the canonical COCO mask dataset.

    Images are decoded with ``cv2.imdecode`` over ``np.fromfile`` so accented
    paths work on every platform. Polygons are rasterized into instance masks;
    torchvision reserves label 0 for background, so the dense class index is
    shifted by one in the targets and shifted back in the predictions.
    """

    def __init__(
        self,
        store: Path,
        coco: dict[str, Any],
        split: str,
        class_indices: dict[int, int],
        augmentation: list[str],
    ) -> None:
        self.store = Path(store)
        self.images = split_coco_images(coco, split)
        self.class_indices = dict(class_indices)
        self.augmentation = list(augmentation)
        by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in coco.get("annotations", []):
            by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        self._annotations = by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import cv2
        import torch

        record = self.images[index]
        source = self.store / str(record["file_name"])
        buffer = np.fromfile(source, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read a COCO image: {record['file_name']}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]

        masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        labels: list[int] = []
        annotations = sorted(
            self._annotations.get(int(record["id"]), []), key=lambda item: int(item["id"])
        )
        for annotation in annotations:
            category = self.class_indices.get(int(annotation["category_id"]))
            if category is None:
                continue
            mask = rasterize_segmentation(annotation.get("segmentation"), width, height)
            bbox = [float(value) for value in annotation.get("bbox", [])]
            if mask is None or len(bbox) != 4:
                continue
            masks.append(mask)
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(category + 1)

        rgb, masks, boxes = _augment(rgb, masks, boxes, self.augmentation)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
        areas = [max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) for box in boxes]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": (
                torch.from_numpy(np.stack(masks).astype(np.uint8))
                if masks
                else torch.zeros((0, height, width), dtype=torch.uint8)
            ),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return tensor, target


def _augment(
    image: np.ndarray,
    masks: list[np.ndarray],
    boxes: list[list[float]],
    augmentation: list[str],
) -> tuple[np.ndarray, list[np.ndarray], list[list[float]]]:
    """Apply only protocol-declared, framework-neutral transforms.

    ``flip_lr`` flips the image, its instance masks and its boxes together.
    ``hsv`` jitters colour only. Nothing else is applied: a transform the
    family cannot express is reported as skipped, never replaced.
    """
    import cv2

    if "flip_lr" in augmentation and np.random.random() < 0.5:
        width = image.shape[1]
        image = np.ascontiguousarray(image[:, ::-1])
        masks = [np.ascontiguousarray(mask[:, ::-1]) for mask in masks]
        boxes = [[width - box[2], box[1], width - box[0], box[3]] for box in boxes]
    if "hsv" in augmentation:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-9.0, 9.0)) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.5, 1.5), 0.0, 255.0)
        hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.5, 1.5), 0.0, 255.0)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return image, masks, boxes


def _image_target_pairs(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[list[Any], list[dict]]:
    """Collate the variable-size images and targets torchvision expects."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


class _UltralyticsAdapter(_CocoMaskAdapter):
    """Shared Ultralytics wrapper: one Lightning module, two label currencies."""

    task = ""
    supported_augmentations = frozenset({"hsv", "flip_lr", "mosaic"})

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
        from lightning import LightningModule, seed_everything
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
                    self._cfg = _ultralytics_cfg(config.family.task, protocol)
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

        trainer, precision = _build_trainer(config, run_dir)
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
            augmentation=_augmentation_report(protocol, self.supported_augmentations),
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
        return self._prepare_masks(config, convert=True)

    def _predictions(self, results: list[Any]) -> list[dict]:
        return coco_segmentation_predictions_from_results(results)


class MaskRcnnAdapter(_CocoMaskAdapter):
    """torchvision Mask R-CNN, the first non-Ultralytics sibling family."""

    name = "mask_rcnn"
    task = "segment"
    #: The protocol's common subset minus mosaic, which has no torchvision
    #: equivalent. No framework-specific transform is substituted for it.
    supported_augmentations = frozenset({"hsv", "flip_lr"})

    def prepare(self, config: ExecutionConfig) -> PreparedDataset:
        return self._prepare_masks(config, convert=False)

    def fit(
        self,
        config: ExecutionConfig,
        run_dir: Path,
        dataset: PreparedDataset,
        on_epoch: EpochCallback,
        resume: ResumeState | None = None,
    ) -> FitResult:
        import torch
        from lightning import LightningModule, seed_everything

        protocol = config.protocol
        device = config.backend.device
        seed_everything(protocol.seed, workers=True)
        coco = read_coco_store(dataset.source_path)
        indices = category_indices(coco)
        augmentation = _augmentation_report(protocol, self.supported_augmentations)

        model = _torchvision_detector(
            config.family.weights, len(dataset.class_names), protocol.image_size
        )
        resumed_checkpoint = resume.last_checkpoint if resume is not None else None
        if resumed_checkpoint is not None and Path(resumed_checkpoint).is_file():
            saved = torch.load(resumed_checkpoint, map_location="cpu", weights_only=True)
            state = saved.get("state_dict", saved) if isinstance(saved, dict) else saved
            model.load_state_dict(state)

        checkpoint_dir = Path(run_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        store = Path(dataset.source_path)

        class _MaskRcnnLightningModule(LightningModule):  # type: ignore[misc,valid-type]
            """One torchvision detector trained by the shared Lightning loop."""

            def __init__(self) -> None:
                super().__init__()
                self.net = model
                self.coco_predictions: list[dict] = []
                self.checkpoint_path = checkpoint_dir / "epoch_000.pt"
                self.stop_requested = False
                self._train_dataset: Any | None = None
                self._val_dataset: Any | None = None

            def _split_dataset(self, split: str) -> Any:
                applied = augmentation["applied"] if split == "train" else []
                return _CocoMaskTorchvisionDataset(store, coco, split, indices, applied)

            def train_dataloader(self) -> Any:
                from torch.utils.data import DataLoader

                if self._train_dataset is None:
                    self._train_dataset = self._split_dataset("train")
                return DataLoader(
                    self._train_dataset,
                    batch_size=protocol.batch,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=_image_target_pairs,
                )

            def val_dataloader(self) -> Any:
                from torch.utils.data import DataLoader

                if self._val_dataset is None:
                    self._val_dataset = self._split_dataset(protocol.split)
                return DataLoader(
                    self._val_dataset,
                    batch_size=protocol.batch,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=_image_target_pairs,
                )

            def training_step(self, batch: Any, batch_idx: int) -> Any:
                images, targets = batch
                self.net.train()
                losses = self.net(images, targets)
                return sum(losses.values())

            def validation_step(self, batch: Any, batch_idx: int) -> None:
                images, targets = batch
                self.net.eval()
                with torch.no_grad():
                    outputs = self.net(images)
                image_ids = [int(target["image_id"].reshape(-1)[0]) for target in targets]
                self.coco_predictions.extend(
                    coco_segmentation_predictions_from_torchvision(outputs, image_ids)
                )

            def configure_optimizers(self) -> Any:
                from torch.optim.lr_scheduler import LambdaLR

                optimizer = _build_optimizer(self.net, protocol)
                scheduler = LambdaLR(optimizer, lr_lambda=_lr_factor(protocol))
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
                }

            def on_validation_epoch_end(self) -> None:
                epoch = int(getattr(self, "current_epoch", 0))
                self.checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                torch.save(
                    {"state_dict": self.net.state_dict(), "epoch": epoch},
                    self.checkpoint_path,
                )
                result = EpochResult(
                    epoch=epoch,
                    predictions=list(self.coco_predictions),
                    adapter_figures={},
                    checkpoint=Path(self.checkpoint_path),
                )
                self.coco_predictions = []
                # The engine scores this epoch and decides whether to stop.
                if on_epoch(result):
                    self.stop_requested = True
                    trainer = getattr(self, "trainer", None)
                    if trainer is not None:
                        trainer.should_stop = True

        trainer, precision = _build_trainer(config, run_dir)
        # The per-epoch checkpoint holds the model state dict, not a Lightning
        # checkpoint; its weights were restored above. Handing that file to
        # ``Trainer.fit(ckpt_path=...)`` would make Lightning fail looking for
        # its own bookkeeping, so the loop restarts with restored weights and
        # the engine's progress snapshot keeps selection and early stopping
        # continuous.
        trainer.fit(_MaskRcnnLightningModule())
        return FitResult(
            precision=precision,
            epochs_completed=int(trainer.current_epoch),
            amp_active=precision == "16-mixed",
            device=device,
            framework=config.family.framework,
            weights=config.family.weights,
            augmentation=augmentation,
        )


_ADAPTERS: dict[str, type[FamilyAdapter]] = {
    YoloDetectionAdapter.name: YoloDetectionAdapter,
    YoloSegmentationAdapter.name: YoloSegmentationAdapter,
    MaskRcnnAdapter.name: MaskRcnnAdapter,
}


def build_adapter(family: Family) -> FamilyAdapter:
    """Select the adapter named by configuration, never by a code edit."""
    try:
        adapter_class = _ADAPTERS[family.name]
    except KeyError as exc:
        raise ValueError(f"unknown model family: {family.name}") from exc
    return adapter_class()
