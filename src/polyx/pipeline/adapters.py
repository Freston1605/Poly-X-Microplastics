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

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from polyx.core.labels import format_polygon_line
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
from polyx.pipeline.evaluation import (
    PREDICTION_FLOOR,
    prediction_budget,
    rasterize_segmentation,
)
from polyx.pipeline.resume import ResumeState
from polyx.pipeline.seg_conversion import convert_coco_dataset, image_hashes
from polyx.pipeline.weights import resolve_weights

EpochCallback = Callable[["EpochResult"], bool]


@dataclass
class EpochResult:
    """What one validation epoch hands back to the shared training engine.

    ``checkpoint`` is the family-native export kept for compatibility and for
    best-weight restoration; ``resume_checkpoint`` is the real Lightning
    checkpoint holding the weights, optimizer, scheduler and epoch progress a
    later invocation needs to continue this execution.
    """

    epoch: int
    predictions: list[dict]
    adapter_figures: dict
    checkpoint: Path
    resume_checkpoint: Path | None = None
    #: What the fit resolved (precision actually run, device, verified weights,
    #: augmentation). The engine writes it into the progress snapshot, so a
    #: publication recovered after the last epoch keeps the same record.
    fit_metadata: dict[str, Any] | None = None


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
    """Observable state of the finished Lightning fit.

    ``precision`` is what Lightning actually ran; ``amp_requested`` is the
    protocol's request. They differ when Lightning substitutes a precision the
    device supports (for example bf16-mixed for 16-mixed on CPU), and only the
    former can honestly drive ``amp_active``.
    """

    precision: str
    amp_requested: str
    epochs_completed: int
    amp_active: bool
    device: str
    framework: str
    weights: str
    weights_verified: bool
    augmentation: dict[str, Any] = field(default_factory=dict)


#: Precision strings that mean some form of automatic mixed precision.
_MIXED_PRECISION = frozenset({"16-mixed", "bf16-mixed", "16-true", "bf16-true"})


def _amp_active(precision: str) -> bool:
    """Whether the precision Lightning reports is a mixed-precision mode."""
    return precision in _MIXED_PRECISION


def _fit_metadata(
    config: ExecutionConfig,
    trainer: Any,
    device: str,
    requested_precision: str,
    weights: Any,
    augmentation: dict[str, Any],
) -> dict[str, Any]:
    """The fit state every epoch carries into the progress snapshot."""
    reported = str(trainer.precision)
    return {
        "precision": reported,
        "amp_requested": requested_precision,
        "amp_active": _amp_active(reported),
        "device": device,
        "framework": config.family.framework,
        "weights": weights.describe(),
        "weights_verified": weights.verified,
        "augmentation": augmentation,
    }


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


def _declared_augmentations(protocol: Protocol) -> list[str]:
    """Every transform this execution declared, benchmark and exploratory."""
    return [*protocol.augmentation_subset, *protocol.exploratory_augmentation]


def _augmentation_report(protocol: Protocol, supported: frozenset[str]) -> dict[str, Any]:
    """Which declared augmentations this family can apply without substitution.

    The protocol names one common subset for the benchmark campaign and an
    optional exploratory list for runs where comparability is not the point. A
    family reports what it applied and what it skipped; it never replaces a
    skipped transform with a framework-specific one, which would make the model
    the wrong variable. The magnitudes travel with the report so two families
    can be judged on the transform they actually applied.
    """
    declared = _declared_augmentations(protocol)
    return {
        "benchmark": protocol.benchmark,
        "subset": list(protocol.augmentation_subset),
        "exploratory": list(protocol.exploratory_augmentation),
        "declared": declared,
        "applied": [name for name in declared if name in supported],
        "skipped": [name for name in declared if name not in supported],
        "family_specific": False,
        "parameters": {
            "hsv_h": protocol.hsv_h,
            "hsv_s": protocol.hsv_s,
            "hsv_v": protocol.hsv_v,
            "flip_lr_probability": protocol.flip_lr_probability,
        },
    }


def _lightning_resume_path(resume: ResumeState | None) -> Path | None:
    """The real Lightning checkpoint a resumed fit must restore from.

    A progress snapshot created before Lightning checkpoints existed has no
    such path; resuming it with restored weights only would silently restart
    the epoch loop, so the execution fails loudly instead.
    """
    if resume is None:
        return None
    checkpoint = resume.resume_checkpoint
    if checkpoint is None:
        raise RuntimeError(
            "the progress snapshot has no Lightning checkpoint; resume cannot "
            "continue the optimizer and epoch state"
        )
    path = Path(checkpoint)
    if not path.is_file():
        raise RuntimeError(f"the Lightning checkpoint named by the snapshot is missing: {path}")
    return path


def _ultralytics_cfg(task: str, protocol: Protocol) -> Any:
    """The Ultralytics argument namespace the dataloader and loss both read.

    Augmentation is restricted to the protocol's declared list: the declared
    transforms are enabled at the protocol's magnitudes and every
    family-specific knob is switched off, so a YOLO run and a Mask R-CNN run
    differ in augmentation only where the protocol says so.
    """
    from ultralytics.cfg import get_cfg

    augment = _declared_augmentations(protocol)
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
            "hsv_h": protocol.hsv_h if hsv else 0.0,
            "hsv_s": protocol.hsv_s if hsv else 0.0,
            "hsv_v": protocol.hsv_v if hsv else 0.0,
            "fliplr": protocol.flip_lr_probability if "flip_lr" in augment else 0.0,
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
        # Provenance states the bytes training used, not the hashes the import
        # manifest recorded: a store modified under the same version would
        # otherwise be described by a set of images it no longer holds.
        measured = (
            conversion.content_hashes if conversion is not None else image_hashes(store, coco)
        )
        hashes = coco_store_hashes(store, coco)
        if measured:
            hashes = {**hashes, **measured}
        return PreparedDataset(
            data_yaml=conversion.data_yaml if conversion is not None else store / "dataset.json",
            ground_truth=coco_store_ground_truth(coco, config.protocol.split),
            audit=audit,
            hashes=hashes,
            class_names={index: name for index, (_id, name) in enumerate(ordered_categories(coco))},
            source_kind="coco_instances",
            source_path=store,
            conversion=conversion.describe(store) if conversion is not None else None,
        )


def _build_trainer(config: ExecutionConfig, run_dir: Path) -> tuple[Any, str]:
    """The one Lightning trainer and the precision that was requested.

    Lightning may substitute a precision the device supports; the caller
    records ``trainer.precision``, not this request. The monitor forwards a
    module's stop decision, made from the shared evaluator's score, to the
    training loop.
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


def _torchvision_detector(
    weights_name: str,
    class_count: int,
    image_size: int,
    weights_path: Path | None = None,
) -> Any:
    """Build the torchvision detector and size its heads for the dataset.

    Like the Ultralytics wrapper, pretrained heads are rebuilt for the
    protocol's dataset: torchvision reserves label 0 for background, so the
    output layer needs ``class_count + 1`` classes. The protocol's image size
    replaces the pretrained transform's own 800/1333 defaults so the model
    family is not silently changing the input scale.

    A pinned, hash-verified checkpoint is loaded from its local path; the
    factory is asked for an untrained model in that case so nothing downloads.
    """
    import torch
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
        if weights_path is not None:
            model = factory(weights=None, weights_backbone=None, **size)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        else:
            model = factory(weights="DEFAULT", **size)
    classes = class_count + 1
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, classes)
    # Keep the ranked tail for AP, like the YOLO adapter; the shared evaluator
    # applies the protocol's confidence threshold only to the operating point.
    # The 100-detection cap is part of the same collection policy.
    model.roi_heads.score_thresh = PREDICTION_FLOOR
    # One ranked list per image where every class competes for the slots, so
    # the budget is 100 per class: the evaluator then applies COCO's cap per
    # image and category without a busy polymer hiding another's candidates.
    model.roi_heads.detections_per_img = prediction_budget(class_count)
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
        protocol: Protocol,
    ) -> None:
        self.store = Path(store)
        self.images = split_coco_images(coco, split)
        self.class_indices = dict(class_indices)
        self.augmentation = list(augmentation)
        self.protocol = protocol
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
        segments: list[np.ndarray | None] = []
        annotations = sorted(
            self._annotations.get(int(record["id"]), []), key=lambda item: int(item["id"])
        )
        for annotation in annotations:
            category = self.class_indices.get(int(annotation["category_id"]))
            if category is None:
                continue
            # Train on the polygon YOLO actually reads: the label line is
            # normalized to six decimals, so deriving the box and the mask from
            # the original coordinates leaves this family keeping instances the
            # pinned loader quantizes below its retention threshold.
            ring = _serialized_ring(annotation, width, height)
            if ring is not None:
                mask = rasterize_segmentation([ring.reshape(-1).tolist()], width, height)
                if mask is None:
                    continue
                box = [
                    float(ring[:, 0].min()),
                    float(ring[:, 1].min()),
                    float(ring[:, 0].max()),
                    float(ring[:, 1].max()),
                ]
            else:
                # A mask made of several rings cannot be written as one YOLO
                # target; a YOLO-seg execution refuses that export, and this
                # family alone still trains on it as one COCO instance.
                mask = rasterize_segmentation(annotation.get("segmentation"), width, height)
                bbox = annotation.get("bbox")
                if mask is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                box = [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[0]) + float(bbox[2]),
                    float(bbox[1]) + float(bbox[3]),
                ]
            masks.append(mask)
            boxes.append(box)
            labels.append(category + 1)
            segments.append(ring)

        # The loader filters geometry before colour/flip transforms. Run the
        # same letterbox/segment path here so both families keep one set.
        boxes, masks, labels = _filter_tiny_instances(
            boxes,
            masks,
            labels,
            image_width=width,
            image_height=height,
            image_size=self.protocol.image_size,
            segments=segments,
        )
        rgb, masks, boxes = _augment(rgb, masks, boxes, self.augmentation, self.protocol)
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
    protocol: Protocol,
) -> tuple[np.ndarray, list[np.ndarray], list[list[float]]]:
    """Apply only protocol-declared transforms at the protocol's magnitudes.

    ``flip_lr`` flips the image, its instance masks and its boxes together at
    the protocol's probability. ``hsv`` reproduces Ultralytics' ``RandomHSV``
    exactly: the same multiplicative gains drawn in the same order, applied
    through the same channel LUTs. Nothing else is applied, and a transform the
    family cannot express is reported as skipped, never replaced.
    """
    import cv2

    if "flip_lr" in augmentation and np.random.random() < protocol.flip_lr_probability:
        width = image.shape[1]
        image = np.ascontiguousarray(image[:, ::-1])
        masks = [np.ascontiguousarray(mask[:, ::-1]) for mask in masks]
        boxes = [[width - box[2], box[1], width - box[0], box[3]] for box in boxes]
    if "hsv" in augmentation and (protocol.hsv_h or protocol.hsv_s or protocol.hsv_v):
        gains = (
            np.random.uniform(-1, 1, 3)
            * [
                protocol.hsv_h,
                protocol.hsv_s,
                protocol.hsv_v,
            ]
            + 1
        )
        hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        levels = np.arange(0, 256, dtype=gains.dtype)
        hue_lut = ((levels * gains[0]) % 180).astype(np.uint8)
        saturation_lut = np.clip(levels * gains[1], 0, 255).astype(np.uint8)
        value_lut = np.clip(levels * gains[2], 0, 255).astype(np.uint8)
        augmented = cv2.merge(
            (
                cv2.LUT(hue, hue_lut),
                cv2.LUT(saturation, saturation_lut),
                cv2.LUT(value, value_lut),
            )
        )
        image = cv2.cvtColor(augmented, cv2.COLOR_HSV2RGB)
    return image, masks, boxes


def _image_target_pairs(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[list[Any], list[dict]]:
    """Collate the variable-size images and targets torchvision expects."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def _loader_keep_mask(
    boxes: list[list[float]],
    image_width: int,
    image_height: int,
    image_size: int,
    *,
    segments: list[np.ndarray | None] | None = None,
) -> np.ndarray:
    """Return the instances the pinned YOLO loader keeps after its resize.

    The torchvision sibling must drop exactly the instances YOLO drops. Rather
    than approximating the loader with a scale factor, this walks the loader's
    own path: ``load_image`` first scales the long side to ``imgsz`` with
    ``ceil`` rounding (rect mode, aspect preserved), the segments are resampled
    to the loader's 1000 points, ``LetterBox`` pads to a square, and
    ``RandomPerspective.apply_segments`` derives the boxes from the resized
    polygon. The retention test is ``box_candidates`` with the loader's segment
    area threshold of 0.01, not the box-only 0.10.
    """
    if not boxes:
        return np.zeros((0,), dtype=bool)
    from ultralytics.data.augment import LetterBox, RandomPerspective
    from ultralytics.utils.instance import Instances
    from ultralytics.utils.ops import resample_segments

    # The loader's rect resize, with its ceil rounding and cap at imgsz.
    ratio = image_size / max(image_height, image_width)
    resized_width = min(math.ceil(image_width * ratio), image_size)
    resized_height = min(math.ceil(image_height * ratio), image_size)
    scale_x = resized_width / image_width
    scale_y = resized_height / image_height

    array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    array[:, [0, 2]] *= scale_x
    array[:, [1, 3]] *= scale_y
    valid_segments: list[np.ndarray] = []
    if segments is not None:
        for segment in segments:
            if segment is None:
                valid_segments = []
                break
            points = np.asarray(segment, dtype=np.float32).reshape(-1, 2).copy()
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            valid_segments.append(points)
    if valid_segments and len(valid_segments) == len(array):
        # The loader resamples every ring to the same number of points before
        # stacking, which is also what lets rings with different vertex counts
        # travel together.
        segment_array = np.stack(resample_segments(valid_segments, n=1000))
    else:
        segment_array = np.zeros((len(array), 0, 2), dtype=np.float32)
    labels: dict[str, Any] = {
        "img": np.zeros((resized_height, resized_width, 3), dtype=np.uint8),
        "cls": np.zeros((len(array), 1), dtype=np.float32),
        "instances": Instances(
            bboxes=array,
            segments=segment_array,
            bbox_format="xyxy",
            normalized=False,
        ),
    }
    labels = LetterBox(new_shape=(image_size, image_size))(labels)
    letterboxed = np.asarray(labels["instances"].bboxes, dtype=np.float32).copy()
    letterboxed_segments = np.asarray(labels["instances"].segments, dtype=np.float32)
    perspective = RandomPerspective(
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
    )
    if segment_array.size:
        perspective.size = (labels["img"].shape[1], labels["img"].shape[0])
        derived, _ = perspective.apply_segments(letterboxed_segments, np.eye(3, dtype=np.float32))
        clipped = np.asarray(derived, dtype=np.float32).copy()
        area_thr = 0.01
    else:
        clipped = letterboxed.copy()
        area_thr = 0.10
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, image_size)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, image_size)
    return np.asarray(
        perspective.box_candidates(box1=letterboxed.T, box2=clipped.T, area_thr=area_thr),
        dtype=bool,
    )


def _serialized_ring(annotation: dict[str, Any], width: int, height: int) -> np.ndarray | None:
    """The polygon the pinned loader reads back from the YOLO label file.

    The converter writes normalized coordinates with six decimals, so a value
    such as 2.00001 px can come back as exactly 2 px. The torchvision sibling
    must train on the same numbers or the two families keep different
    instances near the loader's two-pixel retention threshold.
    """
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, (list, tuple)) or len(segmentation) != 1:
        return None
    ring = segmentation[0]
    if not isinstance(ring, (list, tuple)) or len(ring) < 6 or len(ring) % 2 != 0:
        return None
    points = [(float(ring[index]), float(ring[index + 1])) for index in range(0, len(ring), 2)]
    line = format_polygon_line(0, points, width, height)
    values = [float(value) for value in line.split()[1:]]
    if len(values) < 6 or len(values) % 2 != 0:
        return None
    return np.asarray(
        [[values[index] * width, values[index + 1] * height] for index in range(0, len(values), 2)],
        dtype=np.float32,
    )


def _filter_tiny_instances(
    boxes: list[list[float]],
    masks: list[np.ndarray],
    labels: list[int],
    *,
    image_width: int,
    image_height: int,
    image_size: int,
    segments: list[np.ndarray | None] | None = None,
) -> tuple[list[list[float]], list[np.ndarray], list[int]]:
    """Apply the pinned loader's retention rule to the torchvision sibling.

    The model must be the only variable: one family cannot train on instances
    the other discarded. The kept set comes from the loader's own resize,
    segment-derived boxes and clipping, not from a second approximation.
    """
    keep = _loader_keep_mask(
        boxes,
        image_width,
        image_height,
        image_size,
        segments=segments,
    )
    kept_boxes = [box for box, keep_one in zip(boxes, keep, strict=True) if keep_one]
    kept_masks = [mask for mask, keep_one in zip(masks, keep, strict=True) if keep_one]
    kept_labels = [label for label, keep_one in zip(labels, keep, strict=True) if keep_one]
    return kept_boxes, kept_masks, kept_labels


def _save_live_checkpoint(network: Any, path: Path) -> None:
    """Export the network that actually trained, never a staged checkpoint EMA.

    An Ultralytics checkpoint may carry an ``ema`` entry, and its loader prefers
    that entry over ``model``. Re-saving the staged checkpoint would therefore
    validate (and keep as ``best.pt``) weights from before this epoch. The live
    network is deep-copied and written without an ``ema`` key, so every consumer
    loads exactly the weights the Lightning loop trained.
    """
    import torch

    payload = {"model": deepcopy(network)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _validation_predictions(
    checkpoint: Path,
    config: ExecutionConfig,
    dataset: PreparedDataset,
    device: str,
    convert: Callable[[list[Any]], list[dict]],
) -> list[dict]:
    """Score a fresh inference model loaded from this epoch's snapshot.

    Ultralytics' predictor fuses and freezes the network it is handed. Giving
    it Lightning's live network makes the next backward pass fail, so each
    epoch validates a separate wrapper created from the immutable checkpoint
    written immediately before this call. The checkpoint has no staged ``ema``
    entry: it names the live weights, so the copy validated here is the model
    that actually trained.
    """
    from ultralytics import YOLO

    snapshot = YOLO(str(checkpoint))
    images = split_images(dataset.data_yaml, config.protocol.split)
    raw = snapshot.predict(
        source=[str(image) for image in images],
        imgsz=config.protocol.image_size,
        batch=config.protocol.batch,
        # Keep the ranked tail for AP; the shared evaluator applies the
        # operating point's confidence threshold only to F1.
        conf=PREDICTION_FLOOR,
        max_det=prediction_budget(len(dataset.class_names)),
        device=device,
        verbose=False,
    )
    return convert(list(raw) if isinstance(raw, (list, tuple)) else [raw])


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
        resumed_checkpoint = _lightning_resume_path(resume)
        # Always start from the family's declared pretrained weights and let
        # Lightning restore the execution's own checkpoint. An Ultralytics .pt
        # is not a Lightning checkpoint, so handing it to ``Trainer.fit`` would
        # fail looking for Lightning's bookkeeping. A pinned weight is resolved
        # to its verified local file first, so an offline node never downloads.
        resolved_weights = resolve_weights(
            config.family,
            weights_dir=config.weights_dir,
            manifest=config.weights_manifest,
        )
        model = YOLO(str(resolved_weights.path) if resolved_weights.path else config.family.weights)
        # The pretrained head is sized for its own dataset; the protocol's
        # dataset decides the class count, so the Ultralytics network is rebuilt
        # from its yaml and the pretrained weights are transferred into it.
        model.model = _rebuild_network(model.model, config.family.task, len(dataset.class_names))
        checkpoint_dir = Path(run_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resume_dir = checkpoint_dir / "lightning"
        resume_dir.mkdir(parents=True, exist_ok=True)
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
                self.checkpoint_path = checkpoint_dir / "epoch_000.pt"
                self._train_dataset: Any | None = None
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

            def configure_optimizers(self) -> Any:
                from torch.optim.lr_scheduler import LambdaLR

                optimizer = _build_optimizer(self.net, protocol)
                scheduler = LambdaLR(optimizer, lr_lambda=_lr_factor(protocol))
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
                }

            def on_train_epoch_end(self) -> None:
                epoch = int(getattr(self, "current_epoch", 0))
                self.checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                # Export the live Lightning network, not the staged checkpoint's
                # EMA. Both validation and `best.pt` read this artifact.
                _save_live_checkpoint(self.net, self.checkpoint_path)
                # Validation runs on a fresh model loaded from this immutable
                # snapshot. Ultralytics fuses and freezes whatever network it
                # predicts with, so using the live training module would break
                # the next backward pass. The shared evaluator remains the only
                # scorer; no second Ultralytics validation is run.
                self.coco_predictions = _validation_predictions(
                    self.checkpoint_path, config, dataset, device, predict
                )
                # A real Lightning checkpoint carries the weights, optimizer,
                # scheduler and epoch progress. One file per epoch is published
                # before the progress record points at it, so a preemption can
                # never pair a new checkpoint with an older progress record.
                resume_checkpoint: Path | None = None
                current_trainer = getattr(self, "trainer", None)
                if current_trainer is not None:
                    resume_checkpoint = resume_dir / f"checkpoint_{epoch:03d}.ckpt"
                    current_trainer.save_checkpoint(str(resume_checkpoint))
                # The engine scores this epoch and decides whether to stop.
                # Setting ``trainer.should_stop`` ends the loop after this
                # epoch; the registered callback is the same decision for the
                # validation-driven sibling adapter.
                result = EpochResult(
                    epoch=epoch,
                    predictions=list(self.coco_predictions),
                    adapter_figures={},
                    checkpoint=Path(self.checkpoint_path),
                    resume_checkpoint=resume_checkpoint,
                    fit_metadata=dict(fit_metadata),
                )
                if on_epoch(result):
                    self.stop_requested = True
                    if current_trainer is not None:
                        current_trainer.should_stop = True

        trainer, requested_precision = _build_trainer(config, run_dir)
        augmentation = _augmentation_report(protocol, self.supported_augmentations)
        fit_metadata = _fit_metadata(
            config, trainer, device, requested_precision, resolved_weights, augmentation
        )
        fit_kwargs: dict[str, Any] = {}
        if resumed_checkpoint is not None:
            fit_kwargs["ckpt_path"] = str(resumed_checkpoint)
        trainer.fit(_YoloLightningModule(), **fit_kwargs)
        reported_precision = str(trainer.precision)
        return FitResult(
            precision=reported_precision,
            amp_requested=requested_precision,
            epochs_completed=int(trainer.current_epoch),
            amp_active=_amp_active(reported_precision),
            device=device,
            framework=config.family.framework,
            weights=resolved_weights.describe(),
            weights_verified=resolved_weights.verified,
            augmentation=augmentation,
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

        resolved_weights = resolve_weights(
            config.family,
            weights_dir=config.weights_dir,
            manifest=config.weights_manifest,
        )
        model = _torchvision_detector(
            config.family.weights,
            len(dataset.class_names),
            protocol.image_size,
            resolved_weights.path,
        )
        # The checkpoint's weights, optimizer, scheduler and epoch progress are
        # restored by Lightning from its own checkpoint; a pre-fix snapshot has
        # none, and the helper refuses it instead of restarting the loop.
        resumed_checkpoint = _lightning_resume_path(resume)

        checkpoint_dir = Path(run_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resume_dir = checkpoint_dir / "lightning"
        resume_dir.mkdir(parents=True, exist_ok=True)
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
                return _CocoMaskTorchvisionDataset(store, coco, split, indices, applied, protocol)

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
                # The family-native export above keeps its historical shape;
                # resume uses a real Lightning checkpoint so the optimizer,
                # scheduler and epoch loop continue instead of restarting.
                resume_checkpoint: Path | None = None
                current_trainer = getattr(self, "trainer", None)
                if current_trainer is not None:
                    resume_checkpoint = resume_dir / f"checkpoint_{epoch:03d}.ckpt"
                    current_trainer.save_checkpoint(str(resume_checkpoint))
                result = EpochResult(
                    epoch=epoch,
                    predictions=list(self.coco_predictions),
                    adapter_figures={},
                    checkpoint=Path(self.checkpoint_path),
                    resume_checkpoint=resume_checkpoint,
                    fit_metadata=dict(fit_metadata),
                )
                self.coco_predictions = []
                # The engine scores this epoch and decides whether to stop.
                if on_epoch(result):
                    self.stop_requested = True
                    if current_trainer is not None:
                        current_trainer.should_stop = True

        trainer, requested_precision = _build_trainer(config, run_dir)
        fit_metadata = _fit_metadata(
            config, trainer, device, requested_precision, resolved_weights, augmentation
        )
        fit_kwargs: dict[str, Any] = {}
        if resumed_checkpoint is not None:
            fit_kwargs["ckpt_path"] = str(resumed_checkpoint)
        trainer.fit(_MaskRcnnLightningModule(), **fit_kwargs)
        reported_precision = str(trainer.precision)
        return FitResult(
            precision=reported_precision,
            amp_requested=requested_precision,
            epochs_completed=int(trainer.current_epoch),
            amp_active=_amp_active(reported_precision),
            device=device,
            framework=config.family.framework,
            weights=resolved_weights.describe(),
            weights_verified=resolved_weights.verified,
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
