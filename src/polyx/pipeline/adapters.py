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
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from polyx.core.labels import format_polygon_line
from polyx.core.quality.dataset_audit import DatasetAudit, audit_coco_masks
from polyx.pipeline.coco import (
    coco_predictions_from_results,
    coco_segmentation_predictions_from_results,
    coco_segmentation_predictions_from_torchvision,
)
from polyx.pipeline.config import ExecutionConfig, Family, Protocol
from polyx.pipeline.dataset import (
    ResolvedBoxDataset,
    category_indices,
    class_names,
    coco_store_ground_truth,
    coco_store_hashes,
    ordered_categories,
    read_coco_store,
    read_data_yaml,
    resolve_box_dataset,
    split_coco_images,
    split_directory,
    split_images,
)
from polyx.pipeline.evaluation import (
    PREDICTION_FLOOR,
    prediction_budget,
    rasterize_segmentation,
)
from polyx.pipeline.lightning_loop import ExecutionRecorder, FamilyRun, PolyxModule
from polyx.pipeline.masks import PackedMask
from polyx.pipeline.seg_conversion import convert_coco_dataset, image_hashes
from polyx.pipeline.weights import resolve_weights


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
    resolved_boxes: ResolvedBoxDataset | None = None


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


def _resolved_ultralytics_dataset(dataset: ResolvedBoxDataset, cfg: Any) -> Any:
    """Use native transforms and collation without a second scan or a sidecar cache."""
    from ultralytics.data.dataset import YOLODataset

    training = dataset.splits["train"]

    class _ResolvedDataset(YOLODataset):
        def get_img_files(self, img_path: Any) -> list[str]:
            return [str(image.path) for image in training]

        def get_labels(self) -> list[dict[str, Any]]:
            labels = []
            for image in training:
                rows = np.asarray(
                    [(box.class_id, box.cx, box.cy, box.w, box.h) for box in image.boxes],
                    dtype=np.float32,
                ).reshape(-1, 5)
                # Preserve the native loader's exact-row deduplication. The
                # shared audit and evaluator still see the original boxes.
                _, positions = np.unique(rows, axis=0, return_index=True)
                if len(positions) != len(rows):
                    rows = rows[positions]
                width, height = image.size
                labels.append(
                    {
                        "im_file": str(image.path),
                        "shape": (height, width),
                        "cls": rows[:, :1],
                        "bboxes": rows[:, 1:],
                        "segments": [],
                        "keypoints": None,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                )
            return labels

    native = _ResolvedDataset(
        img_path=str(dataset.data_yaml.parent),
        imgsz=cfg.imgsz,
        batch_size=cfg.batch,
        augment=True,
        hyp=cfg,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        task="detect",
        classes=cfg.classes,
        data={"names": dataset.class_names, "nc": len(dataset.class_names), "channels": 3},
    )
    # Even with cache=False the pinned loader probes disk sidecars. A path
    # under the image file cannot exist, so it always decodes the source.
    native.npy_files = [image.path / ".npy" for image in training]
    return native


class FamilyAdapter:
    """Contract every family implements; later families are siblings.

    An adapter resolves the dataset and builds the two Lightning pieces the
    shared loop drives: a module (model, steps, native export) and a data
    module. It does not own the loop, the optimizer, the schedule, the
    checkpoint, the stopping rule or the artifact record.
    """

    name = ""
    task = ""

    def prepare(self, config: ExecutionConfig) -> PreparedDataset:
        """Resolve the dataset and audit it, before any artifact is trained."""
        raise NotImplementedError

    def build(
        self,
        config: ExecutionConfig,
        dataset: PreparedDataset,
        recorder: ExecutionRecorder,
    ) -> FamilyRun:
        """Build the module and data module the shared Lightning loop drives."""
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
    _install_chunked_mask_postprocess(model)
    return model


#: How many full-resolution masks the torchvision postprocess may build at
#: once. torchvision pastes every detection's mask at the original image size in
#: one call; on a 4K plate with the per-class candidate budget that is hundreds
#: of 4K masks at tens of gigabytes, before the model's own memory is counted.
MASK_PASTE_CHUNK = 8


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """A tensor or array as a float numpy array on the CPU."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _install_chunked_mask_postprocess(model: Any) -> None:
    """Paste predicted masks a few at a time and pack each chunk immediately.

    ``GeneralizedRCNNTransform.postprocess`` resizes every box and pastes every
    mask at the original image resolution in one call. The adapter keeps the
    ranked tail the shared evaluator needs for AP, so that single call can ask
    for hundreds of full-resolution masks on a 4K plate and exhaust the V100
    before validation scores anything.

    Replacing the instance's method keeps torchvision's own paste -- no
    reimplementation of the model or its transforms -- while bounding the peak
    to ``MASK_PASTE_CHUNK`` masks: each chunk is turned into a packed mask at
    one bit per pixel before the next one is built. A stand-in without a
    transform is left alone, and its outputs already arrive as prediction dicts.
    """
    transform = getattr(model, "transform", None)
    original = getattr(transform, "postprocess", None)
    if original is None:
        return

    def postprocess(
        detections: list[dict],
        image_sizes: list[tuple[int, int]],
        original_sizes: list[tuple[int, int]],
    ) -> list[dict]:
        pasted_images: list[dict] = []
        for detection, image_size, original_size in zip(
            detections, image_sizes, original_sizes, strict=True
        ):
            total = len(detection["boxes"])
            masks: list[PackedMask] = []
            boxes: list[Any] = []
            labels: list[Any] = []
            scores: list[Any] = []
            for start in range(0, total, MASK_PASTE_CHUNK):
                chunk = {
                    key: value[start : start + MASK_PASTE_CHUNK] for key, value in detection.items()
                }
                pasted = original([chunk], [image_size], [original_size])[0]
                # The paste also resizes the boxes back to the original image.
                # Taking them from this call keeps boxes and masks in one frame;
                # reading them off ``detection`` would leave every box in the
                # model's letterboxed input geometry.
                boxes.append(pasted["boxes"])
                labels.append(pasted["labels"])
                scores.append(pasted["scores"])
                masks.extend(
                    PackedMask.pack(_tensor_to_numpy(mask) >= 0.5) for mask in pasted["masks"]
                )
                # Release this chunk's full-resolution tensors before the next
                # one is built; they are the memory this wrapper exists to bound.
                del chunk, pasted
            merged: dict[str, Any] = {
                "boxes": _concatenate(boxes),
                "labels": _concatenate(labels),
                "scores": _concatenate(scores),
                "masks": masks,
            }
            for key, value in detection.items():
                # Any framework key this wrapper does not touch passes through.
                merged.setdefault(key, value)
            pasted_images.append(merged)
        return pasted_images

    transform.postprocess = postprocess


def _concatenate(parts: list[Any]) -> Any:
    """One tensor from the chunks' pieces; the empty list when no chunk ran."""
    if not parts or not hasattr(parts[0], "shape"):
        return parts
    import torch

    return torch.cat(parts, dim=0)


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


class _YoloValidationPaths:
    """The validation split as paths and their stable COCO ids.

    The Lightning loop owns batching and the epoch; the Ultralytics predictor
    owns letterboxing, NMS and polygon decoding. Passing it the paths keeps the
    framework's own inference path in charge instead of reimplementing it, and
    the id travels beside the path so a batch never has to guess its place in
    the split.
    """

    def __init__(self, paths: Iterable[Path]) -> None:
        self.items = [{"image_id": index, "path": str(path)} for index, path in enumerate(paths)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.items[index])


class _YoloDataModule:
    """The family's data as a LightningDataModule.

    Training reuses the Ultralytics dataset and batch collation. Validation
    enumerates the split's paths for the Ultralytics predictor, so the same
    Lightning validation loop drives a YOLO family and a torchvision one.
    """

    def __init__(self, *, dataset: PreparedDataset, cfg: Any, protocol: Protocol) -> None:
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.protocol = protocol
        self._train_dataset: Any | None = None
        self._val_dataset: _YoloValidationPaths | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit") and self._train_dataset is None:
            boxes = self.dataset.resolved_boxes
            self._train_dataset = (
                _resolved_ultralytics_dataset(boxes, self.cfg)
                if boxes is not None
                else _ultralytics_dataset(self.dataset.data_yaml, self.cfg, mode="train")
            )

    def train_dataloader(self) -> Any:
        from torch.utils.data import DataLoader

        self.setup("fit")
        return DataLoader(
            self._train_dataset,
            batch_size=self.protocol.batch,
            shuffle=True,
            num_workers=0,
            collate_fn=self._train_dataset.collate_fn,
        )

    def val_dataloader(self) -> Any:
        from torch.utils.data import DataLoader

        if self._val_dataset is None:
            boxes = self.dataset.resolved_boxes
            split = self.protocol.split
            paths = (
                (image.path for image in boxes.splits[split])
                if boxes is not None
                else split_images(self.dataset.data_yaml, split)
            )
            self._val_dataset = _YoloValidationPaths(paths)
        return DataLoader(
            self._val_dataset,
            batch_size=self.protocol.batch,
            shuffle=False,
            num_workers=0,
        )


class _YoloModule(PolyxModule):
    """One Ultralytics module trained by the shared Lightning loop.

    Only the torch network is a child module. The Ultralytics wrapper itself
    subclasses ``nn.Module`` and overrides ``train`` to launch its own trainer,
    so registering it would turn Lightning's ``model.eval()`` into a full
    Ultralytics training job.
    """

    def __init__(
        self,
        *,
        net: Any,
        convert: Callable[[list[Any], list[int]], list[dict]],
        image_size: int,
        class_count: int,
        device: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.convert = convert
        self.image_size = image_size
        self.class_count = class_count
        self.device_name = device
        #: The Ultralytics predictor wrapper is itself an ``nn.Module``. Held
        #: directly on this module it would become a child, and Lightning's
        #: ``model.eval()`` would recurse into it and launch the framework's own
        #: trainer. A plain dict keeps it outside the module tree.
        self._predictor: dict[str, Any] = {}

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        # The Ultralytics network computes its own task loss from the batch its
        # dataloader produced; Lightning owns the step. The Ultralytics loader
        # yields uint8 images, which its native loop scales to floats first.
        import torch

        images = batch["img"]
        if images.dtype == torch.uint8:
            batch["img"] = images.float() / 255.0
        else:
            batch["img"] = images.float()
        self.net.train()
        loss, _items = self.net.loss(batch)
        return loss

    def on_validation_start(self) -> None:
        """Export this epoch's weights and point the predictor at the export.

        Ultralytics' predictor fuses and freezes the network it is handed, so
        it never receives the network Lightning is training. The snapshot is
        written first and predicted from second, which also makes the scored
        weights and the exported artifact the same bytes.
        """
        super().on_validation_start()
        from ultralytics import YOLO

        if self.fitting():
            path = self.recorder.layout.native_checkpoint(int(self.current_epoch))
            self.native_checkpoint_path = path
        else:
            path = self.recorder.layout.publication_snapshot
        self.export_native(path)
        self._predictor["model"] = YOLO(str(path))

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        paths = [str(path) for path in batch["path"]]
        image_ids = [int(image_id) for image_id in batch["image_id"]]
        raw = self._predictor["model"].predict(
            source=paths,
            imgsz=self.image_size,
            batch=len(paths),
            # Keep the ranked tail for AP; the shared evaluator applies the
            # operating point's confidence threshold only to F1.
            conf=PREDICTION_FLOOR,
            max_det=prediction_budget(self.class_count),
            device=self.device_name,
            verbose=False,
        )
        results = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        self.coco_predictions.extend(self.convert(results, image_ids))

    def export_native(self, path: Path, epoch: int | None = None) -> None:
        """Write the Ultralytics network that actually trained.

        An Ultralytics checkpoint may carry an ``ema`` entry, and its loader
        prefers that entry over ``model``. Re-saving a staged checkpoint would
        therefore validate weights from before this epoch; the live network is
        deep-copied and written without an ``ema`` key. The payload has no
        epoch field; the file name is what names the epoch.
        """
        _save_live_checkpoint(self.net, path)


class _UltralyticsAdapter(_CocoMaskAdapter):
    """Shared Ultralytics wrapper: one Lightning module, two label currencies."""

    task = ""
    supported_augmentations = frozenset({"hsv", "flip_lr", "mosaic"})

    def _predictions(self, results: list[Any], image_ids: list[int]) -> list[dict]:
        raise NotImplementedError

    def build(
        self,
        config: ExecutionConfig,
        dataset: PreparedDataset,
        recorder: ExecutionRecorder,
    ) -> FamilyRun:
        from lightning import LightningDataModule, LightningModule, seed_everything
        from ultralytics import YOLO

        protocol = config.protocol
        device = config.backend.device
        seed_everything(protocol.seed, workers=True)
        # A pinned weight is resolved to its verified local file first, so an
        # offline node never downloads.
        resolved_weights = resolve_weights(
            config.family,
            weights_dir=config.weights_dir,
            manifest=config.weights_manifest,
        )
        model = YOLO(str(resolved_weights.path) if resolved_weights.path else config.family.weights)
        # The pretrained head is sized for its own dataset; the protocol's
        # dataset decides the class count, so the network is rebuilt from its
        # yaml and the pretrained weights are transferred into it.
        net = _rebuild_network(model.model, config.family.task, len(dataset.class_names))
        # The loss and the dataloader read one argument namespace: the
        # protocol's, with every family-specific augmentation knob off.
        cfg = _ultralytics_cfg(config.family.task, protocol)
        net.args = cfg
        convert = self._predictions

        class _Data(_YoloDataModule, LightningDataModule):  # type: ignore[misc,valid-type]
            """The family's data module; the loop is shared."""

        datamodule = _Data(dataset=dataset, cfg=cfg, protocol=protocol)

        class _Module(_YoloModule, LightningModule):  # type: ignore[misc,valid-type]
            """The family's Lightning module; the loop is shared."""

            def __init__(self) -> None:
                super().__init__(
                    net=net,
                    convert=convert,
                    image_size=protocol.image_size,
                    class_count=len(dataset.class_names),
                    device=device,
                    protocol=protocol,
                    evaluator=recorder.evaluator,
                    recorder=recorder,
                )

        return FamilyRun(
            module=_Module(),
            datamodule=datamodule,
            weights=resolved_weights,
            augmentation=_augmentation_report(protocol, self.supported_augmentations),
            device=device,
        )


class YoloDetectionAdapter(_UltralyticsAdapter):
    """Ultralytics YOLO detection wrapped as one Lightning module."""

    name = "yolo_detection"
    task = "detect"

    def __init__(self, resolved_boxes: ResolvedBoxDataset | None = None) -> None:
        self._resolved_boxes = resolved_boxes

    def prepare(self, config: ExecutionConfig) -> PreparedDataset:
        if config.data_yaml is None:
            raise ValueError("a detection family needs a YOLO box dataset (data_yaml)")
        split = config.protocol.split
        boxes = self._resolved_boxes or resolve_box_dataset(config.data_yaml, split)
        return PreparedDataset(
            data_yaml=boxes.data_yaml,
            ground_truth=boxes.ground_truth(split),
            audit=boxes.audit(),
            hashes=boxes.hashes,
            class_names=boxes.class_names,
            source_kind="yolo_boxes",
            source_path=boxes.data_yaml,
            resolved_boxes=boxes,
        )

    def _predictions(self, results: list[Any], image_ids: list[int]) -> list[dict]:
        return coco_predictions_from_results(results, image_ids=image_ids)


class YoloSegmentationAdapter(_UltralyticsAdapter):
    """Ultralytics YOLO-seg trained from the canonical COCO mask dataset."""

    name = "yolo_segmentation"
    task = "segment"

    def prepare(self, config: ExecutionConfig) -> PreparedDataset:
        return self._prepare_masks(config, convert=True)

    def _predictions(self, results: list[Any], image_ids: list[int]) -> list[dict]:
        return coco_segmentation_predictions_from_results(results, image_ids=image_ids)


class _MaskDataModule:
    """The COCO mask store as a LightningDataModule.

    torchvision reads the canonical instances directly. Validation runs one
    image per step because the model pastes every detection's mask at the
    original resolution: a batch of eight 4K plates would multiply the peak the
    chunked postprocess exists to bound.
    """

    def __init__(
        self,
        *,
        store: Path,
        coco: dict[str, Any],
        indices: dict[int, int],
        protocol: Protocol,
        augmentation: dict[str, Any],
    ) -> None:
        super().__init__()
        self.store = store
        self.coco = coco
        self.indices = indices
        self.protocol = protocol
        self.augmentation = augmentation
        self._train_dataset: Any | None = None
        self._val_dataset: Any | None = None

    def _dataset(self, split: str) -> Any:
        applied = self.augmentation["applied"] if split == "train" else []
        return _CocoMaskTorchvisionDataset(
            self.store, self.coco, split, self.indices, applied, self.protocol
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit") and self._train_dataset is None:
            self._train_dataset = self._dataset("train")
        if self._val_dataset is None:
            self._val_dataset = self._dataset(self.protocol.split)

    def train_dataloader(self) -> Any:
        from torch.utils.data import DataLoader

        self.setup("fit")
        return DataLoader(
            self._train_dataset,
            batch_size=self.protocol.batch,
            shuffle=True,
            num_workers=0,
            collate_fn=_image_target_pairs,
        )

    def val_dataloader(self) -> Any:
        from torch.utils.data import DataLoader

        if self._val_dataset is None:
            self._val_dataset = self._dataset(self.protocol.split)
        return DataLoader(
            self._val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=_image_target_pairs,
        )


class _MaskModule(PolyxModule):
    """One torchvision detector trained by the shared Lightning loop."""

    def __init__(self, *, net: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.net = net

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        images, targets = batch
        self.net.train()
        losses = self.net(images, targets)
        return sum(losses.values())

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        import torch

        images, targets = batch
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(images)
        image_ids = [int(target["image_id"].reshape(-1)[0]) for target in targets]
        self.coco_predictions.extend(
            coco_segmentation_predictions_from_torchvision(outputs, image_ids)
        )

    def on_validation_start(self) -> None:
        super().on_validation_start()
        if not self.fitting():
            return
        path = self.recorder.layout.native_checkpoint(int(self.current_epoch))
        self.native_checkpoint_path = path
        self.export_native(path, epoch=int(self.current_epoch))

    def export_native(self, path: Path, epoch: int | None = None) -> None:
        """Write the detector's weights in the family's own format."""
        import torch

        payload = {
            "state_dict": self.net.state_dict(),
            "epoch": int(self.current_epoch if epoch is None else epoch),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)


class MaskRcnnAdapter(_CocoMaskAdapter):
    """torchvision Mask R-CNN, the first non-Ultralytics sibling family."""

    name = "mask_rcnn"
    task = "segment"
    #: The protocol's common subset minus mosaic, which has no torchvision
    #: equivalent. No framework-specific transform is substituted for it.
    supported_augmentations = frozenset({"hsv", "flip_lr"})

    def prepare(self, config: ExecutionConfig) -> PreparedDataset:
        return self._prepare_masks(config, convert=False)

    def build(
        self,
        config: ExecutionConfig,
        dataset: PreparedDataset,
        recorder: ExecutionRecorder,
    ) -> FamilyRun:
        from lightning import LightningDataModule, LightningModule, seed_everything

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

        class _Data(_MaskDataModule, LightningDataModule):  # type: ignore[misc,valid-type]
            """The family's data module; the loop is shared."""

        datamodule = _Data(
            store=Path(dataset.source_path),
            coco=coco,
            indices=indices,
            protocol=protocol,
            augmentation=augmentation,
        )

        class _Module(_MaskModule, LightningModule):  # type: ignore[misc,valid-type]
            """The family's Lightning module; the loop is shared."""

            def __init__(self) -> None:
                super().__init__(
                    net=model,
                    protocol=protocol,
                    evaluator=recorder.evaluator,
                    recorder=recorder,
                )

        return FamilyRun(
            module=_Module(),
            datamodule=datamodule,
            weights=resolved_weights,
            augmentation=augmentation,
            device=device,
        )


_ADAPTERS: dict[str, type[FamilyAdapter]] = {
    YoloDetectionAdapter.name: YoloDetectionAdapter,
    YoloSegmentationAdapter.name: YoloSegmentationAdapter,
    MaskRcnnAdapter.name: MaskRcnnAdapter,
}


def build_adapter(
    family: Family, *, resolved_boxes: ResolvedBoxDataset | None = None
) -> FamilyAdapter:
    """Select the adapter named by configuration, never by a code edit."""
    try:
        adapter_class = _ADAPTERS[family.name]
    except KeyError as exc:
        raise ValueError(f"unknown model family: {family.name}") from exc
    if adapter_class is YoloDetectionAdapter:
        return YoloDetectionAdapter(resolved_boxes)
    return adapter_class()
