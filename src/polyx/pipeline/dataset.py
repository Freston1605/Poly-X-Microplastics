"""Dataset resolution shared by the audit preflight, the evaluator and the adapters.

A headless box execution resolves its inputs once, here: the declared splits,
their images, the boxes each label file holds and one digest over exactly those
bytes. The audit, the evaluator's COCO ground truth, the native training loader
and the resume fingerprint then read that one file list instead of walking the
folder again. Segmentation datasets keep their own path: the canonical COCO
store and its job-time conversion.

The label codec stays the sole owner of YOLO label I/O; this module only turns
its boxes into the COCO instance view the shared evaluator consumes. Dataset
hashes are produced with the project's existing SHA-256 helper.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from polyx.core.i18n import N_
from polyx.core.labels import BBox, find_label_file, read_training_boxes
from polyx.core.paths import IMAGE_EXTS, find_images
from polyx.core.provenance import sha256_file
from polyx.core.quality.dataset_audit import DatasetAudit, audit_resolved_boxes, coco_image_split
from polyx.pipeline.errors import PipelineError


@dataclass(frozen=True)
class BoxImage:
    """One resolved image and the boxes read from its matching label file."""

    path: Path
    size: tuple[int, int]
    label: Path | None
    boxes: tuple[BBox, ...]
    sha256: str


@dataclass(frozen=True)
class ResolvedBoxDataset:
    """One file list for a headless box execution, in stable split order."""

    data_yaml: Path
    real_domain: bool
    class_names: dict[int, str]
    splits: dict[str, tuple[BoxImage, ...]]
    split_paths: dict[str, Path | None]
    hashes: dict[str, str]

    def audit(self) -> DatasetAudit:
        """Apply the shared audit to the resolved labels, retaining missing-label warnings."""
        records = {
            split: tuple((image.path, image.boxes if image.label else None) for image in images)
            for split, images in self.splits.items()
        }
        return audit_resolved_boxes(list(self.class_names.values()), self.split_paths, records)

    def ground_truth(self, split: str) -> dict[str, Any]:
        """The evaluator's COCO instance view, numbered in the loader's split order."""
        images = []
        annotations = []
        for image_id, image in enumerate(self.splits[split]):
            width, height = image.size
            images.append(
                {"id": image_id, "file_name": image.path.name, "width": width, "height": height}
            )
            for box in image.boxes:
                box_width, box_height = box.w * width, box.h * height
                annotations.append(
                    {
                        "id": len(annotations),
                        "image_id": image_id,
                        "category_id": box.class_id,
                        "bbox": [
                            (box.cx - box.w / 2) * width,
                            (box.cy - box.h / 2) * height,
                            box_width,
                            box_height,
                        ],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                    }
                )
        return {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": cid, "name": name} for cid, name in self.class_names.items()],
        }


def read_data_yaml(data_yaml: Path) -> dict[str, Any]:
    """Parse the dataset declaration using the project's YAML dependency."""
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid data.yaml: {data_yaml}")
    return data


def class_names(data: dict[str, Any]) -> dict[int, str]:
    """Class id to name, in the order declared by ``names``."""
    raw = data.get("names") or {}
    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}
    return {int(index): str(name) for index, name in raw.items()}


def split_directory(data_yaml: Path, data: dict[str, Any], split: str) -> Path | None:
    """Resolve one split's image folder relative to the data.yaml location."""
    value = data.get(split)
    if not value:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    root_value = data.get("path")
    if root_value:
        root = Path(str(root_value))
        if not root.is_absolute():
            root = (Path(data_yaml).parent / root).resolve()
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path(data_yaml).parent / candidate).resolve()


def split_images(data_yaml: Path, split: str) -> list[Path]:
    """Images of one split in stable order; ids are their positions here."""
    data = read_data_yaml(data_yaml)
    directory = split_directory(data_yaml, data, split)
    if directory is None or not directory.exists():
        return []
    return find_images(directory)


def _is_real_domain(declaration: Any) -> bool:
    """Whether an explicit dataset declaration names real-domain imagery."""
    if not isinstance(declaration, dict):
        return False
    for key in ("domain", "validation_domain", "image_domain"):
        if str(declaration.get(key, "")).strip().lower() in {"real", "real_domain"}:
            return True
    info = declaration.get("info")
    return _is_real_domain(info) if isinstance(info, dict) else False


def _resolve_box_image(image: Path, native_label: Path, names: dict[int, str]) -> BoxImage:
    import cv2
    import numpy as np
    from PIL import Image

    label = find_label_file(image)
    if label != (native_label if native_label.is_file() else None):
        raise PipelineError(
            "dataset.box_label_location",
            N_(
                "image %(image)s has a label at a location the two readers do not agree on; "
                "place its box labels at %(expected)s using the conventional layout"
            ),
            image=str(image),
            expected=str(native_label),
        )
    try:
        boxes = read_training_boxes(label, names) if label is not None else ()
    except (OSError, ValueError, OverflowError) as exc:
        raise PipelineError(
            "dataset.invalid_box_label",
            N_(
                "%(path)s must contain readable five-column boxes with a known class and normalized coordinates"
            ),
            path=str(label),
        ) from exc
    try:
        with Image.open(image) as source:
            source.verify()
            jpeg = source.format == "JPEG"
        contents = np.fromfile(image, dtype=np.uint8)
        # The native verifier would rewrite a JPEG without an end marker.
        if jpeg and contents[-2:].tobytes() != b"\xff\xd9":
            raise ValueError("incomplete JPEG")
        pixels = cv2.imdecode(contents, cv2.IMREAD_COLOR)
        if pixels is None or min(pixels.shape[:2]) < 10:
            raise ValueError("undecodable image or side smaller than 10 pixels")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, cv2.error) as exc:
        raise PipelineError(
            "dataset.unreadable_box_image",
            N_(
                "cannot use %(path)s as a training image: it must be readable and at least 10 pixels on each side"
            ),
            path=str(image),
        ) from exc
    height, width = pixels.shape[:2]
    return BoxImage(image, (width, height), label, boxes, hashlib.sha256(contents).hexdigest())


def resolve_box_dataset(data_yaml: Path, split: str) -> ResolvedBoxDataset:
    """Resolve, validate and fingerprint the actual inputs of a headless box execution."""
    from ultralytics.data.utils import IMG_FORMATS, img2label_paths

    data_yaml = data_yaml.resolve()
    declaration = read_data_yaml(data_yaml)
    names = dict(sorted(class_names(declaration).items()))
    if not names or list(names) != list(range(len(names))):
        raise PipelineError(
            "dataset.invalid_box_classes",
            N_("%(path)s must declare consecutive box class IDs starting at zero"),
            path=str(data_yaml),
        )
    root = data_yaml.parent
    hashes = {"data.yaml": sha256_file(data_yaml)}
    split_paths: dict[str, Path | None] = {}
    splits: dict[str, tuple[BoxImage, ...]] = {}
    resolved: dict[Path, BoxImage] = {}

    def hash_key(path: Path, split: str, directory: Path) -> str:
        """A stable, machine-independent key: the dataset-relative path, or the split's own."""
        if path.is_relative_to(root):
            return path.relative_to(root).as_posix()
        if path.is_relative_to(directory):
            return f"{split}/{path.relative_to(directory).as_posix()}"
        return f"{split}/{path.name}"

    for name in dict.fromkeys(("train", "val", "test", split)):
        directory = split_directory(data_yaml, declaration, name)
        split_paths[name] = directory
        images = []
        if directory is not None:
            if directory.is_file() or (
                not directory.is_dir() and directory.suffix.lower() == ".txt"
            ):
                raise PipelineError(
                    "dataset.box_split_not_directory",
                    N_("box split %(split)s must name an image directory, not a file: %(path)s"),
                    split=name,
                    path=str(directory),
                )
            if directory.is_dir():
                for path in sorted(directory.rglob("*")):
                    suffix = path.suffix.lower()
                    if not path.is_file() or suffix[1:] not in IMG_FORMATS:
                        continue
                    if suffix not in IMAGE_EXTS:
                        raise PipelineError(
                            "dataset.unsupported_box_image_format",
                            N_("unsupported headless box image format: %(path)s"),
                            path=str(path),
                        )
                    if path not in resolved:
                        native_label = Path(img2label_paths([str(path)])[0])
                        sample = _resolve_box_image(path, native_label, names)
                        resolved[path] = sample
                        hashes[hash_key(path, name, directory)] = sample.sha256
                        if sample.label is not None:
                            key = hash_key(sample.label, name, directory)
                            hashes[key] = sha256_file(sample.label)
                    images.append(resolved[path])
        if name == split and not images:
            raise PipelineError(
                "dataset.box_split_empty",
                N_("evaluation split %(split)s in %(path)s needs a directory with images"),
                split=name,
                path=str(data_yaml),
            )
        splits[name] = tuple(images)
    return ResolvedBoxDataset(
        data_yaml, _is_real_domain(declaration), names, splits, split_paths, hashes
    )


def validate_campaign_cohort(
    data_yaml: Path,
    coco_dataset: Path,
    split: str,
    *,
    resolved_boxes: ResolvedBoxDataset | None = None,
) -> dict[str, Any]:
    """Prove the two campaign sources describe the same real-domain cohort.

    A benchmark may carry a YOLO box dataset and a COCO mask dataset so the
    detection and segmentation families each consume the source they need.
    Sharing a protocol is not enough: if those two sources hold different
    images or different folds, the model is no longer the only variable. Both
    must declare real-domain imagery as well; a synthetic validation fold
    cannot support a paper-quality comparison.
    """
    box_dataset = (
        resolved_boxes if resolved_boxes is not None else resolve_box_dataset(data_yaml, split)
    )
    coco = read_coco_store(coco_dataset)
    if not box_dataset.real_domain:
        raise PipelineError(
            "campaign.domain_not_real",
            N_(
                "%(path)s must declare domain: real (or validation_domain: real) "
                "for a benchmark campaign"
            ),
            path=str(data_yaml),
        )
    if not _is_real_domain(coco):
        raise PipelineError(
            "campaign.domain_not_real",
            N_("%(path)s must declare info.domain: real for a benchmark campaign"),
            path=str(coco_dataset / "dataset.json"),
        )
    boxes: dict[str, tuple[str, str]] = {}
    masks: dict[str, tuple[str, str]] = {}
    for name in box_dataset.splits:
        for sample in box_dataset.splits[name]:
            path = sample.path
            if path.name in boxes:
                raise PipelineError(
                    "campaign.duplicate_box_image",
                    N_(
                        "the box dataset reuses the image name %(name)s across "
                        "folds; basenames must be unique for a cohort comparison"
                    ),
                    name=path.name,
                )
            boxes[path.name] = (name, sample.sha256)
        for image in split_coco_images(coco, name):
            path = coco_dataset / str(image["file_name"])
            if path.name in masks:
                raise PipelineError(
                    "campaign.duplicate_mask_image",
                    N_(
                        "the mask dataset reuses the image name %(name)s across "
                        "folds; basenames must be unique for a cohort comparison"
                    ),
                    name=path.name,
                )
            masks[path.name] = (name, sha256_file(path))
    if set(boxes) != set(masks):
        missing_boxes = sorted(set(masks) - set(boxes))
        missing_masks = sorted(set(boxes) - set(masks))
        raise PipelineError(
            "campaign.cohort_mismatch",
            N_(
                "the cohorts differ between %(boxes)s and %(masks)s: "
                "only-box=%(only_box)s only-mask=%(only_mask)s"
            ),
            boxes=str(data_yaml),
            masks=str(coco_dataset),
            only_box=missing_masks,
            only_mask=missing_boxes,
        )
    for name in boxes:
        box_fold, box_hash = boxes[name]
        mask_fold, mask_hash = masks[name]
        if box_fold != mask_fold:
            raise PipelineError(
                "campaign.fold_mismatch",
                N_(
                    "image %(name)s is in the %(box_fold)s fold of the box dataset "
                    "and the %(mask_fold)s fold of the mask dataset"
                ),
                name=name,
                box_fold=box_fold,
                mask_fold=mask_fold,
            )
        if box_hash != mask_hash:
            raise PipelineError(
                "campaign.content_mismatch",
                N_(
                    "image %(name)s has different content in the box and mask "
                    "datasets (%(box_hash)s != %(mask_hash)s)"
                ),
                name=name,
                box_hash=box_hash[:12],
                mask_hash=mask_hash[:12],
            )
    if not any(fold == split for fold, _hash in boxes.values()):
        raise PipelineError(
            "campaign.split_missing",
            N_("the benchmark split %(split)s has no images in either dataset"),
            split=split,
        )
    signature = {
        name: {"split": fold, "sha256": digest} for name, (fold, digest) in sorted(boxes.items())
    }
    return {"split": split, "images": signature, "domain": "real"}


def dataset_fingerprint(hashes: dict[str, str]) -> str:
    """One digest over every per-file hash, for resume identity.

    A resumed execution must be scoring the byte-identical dataset it was
    interrupted on: weights and the best epoch's figures belong to those files.
    The protocol comparison alone cannot see a relabelled image or an edited
    COCO export, so the training state carries this digest and resuming against
    a different one is refused. Paths are sorted, so two walks of the same tree
    always produce the same fingerprint.
    """
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(hashes[name]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# ── The canonical COCO mask dataset (ADR-0009) ────────────────────────────


def read_coco_store(store: Path) -> dict[str, Any]:
    """Load the canonical ``dataset.json`` of one versioned mask tool output."""
    dataset_json = Path(store).expanduser() / "dataset.json"
    if not dataset_json.is_file():
        raise ValueError(f"not a COCO segmentation dataset: {dataset_json} is missing")
    try:
        coco = json.loads(dataset_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read the COCO segmentation dataset: {exc}") from exc
    if not isinstance(coco, dict) or any(
        not isinstance(coco.get(key), list) for key in ("images", "annotations", "categories")
    ):
        raise ValueError("the COCO segmentation dataset needs images, annotations and categories")
    return coco


def ordered_categories(coco: dict[str, Any]) -> list[tuple[int, str]]:
    """Canonical categories as ``(id, name)`` in the dense order families use.

    COCO category ids are arbitrary — CVAT starts at 1 — while every family
    expects the dense ``0..N-1`` class indices its label files hold. Sorting by
    canonical id is the one mapping used by the import audit, the job-time
    conversion and the evaluator's ground-truth view.
    """
    names = {int(item["id"]): str(item["name"]) for item in coco.get("categories", [])}
    return sorted(names.items())


def category_indices(coco: dict[str, Any]) -> dict[int, int]:
    """Canonical category id to the family's dense class index."""
    return {category_id: index for index, (category_id, _) in enumerate(ordered_categories(coco))}


def split_coco_images(coco: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Images of one split in canonical file order.

    The conversion writes the family's images in exactly this order and the
    evaluator numbers the ground truth from it, so a prediction's position in
    the family's result list is its COCO ``image_id`` in both places.
    """
    return [dict(image) for image in coco.get("images", []) if coco_image_split(image) == split]


def coco_store_ground_truth(coco: dict[str, Any], split: str) -> dict[str, Any]:
    """COCO instance view of one split of the canonical mask dataset.

    The view keeps the canonical polygons and remaps ids to the dense class
    indices the family reports, and to the zero-based image positions that
    order the split. Masks never leave COCO on disk; this is what the shared
    evaluator scores against.
    """
    images = split_coco_images(coco, split)
    positions = {int(image["id"]): index for index, image in enumerate(images)}
    indices = category_indices(coco)
    annotations: list[dict[str, Any]] = []
    for annotation in coco.get("annotations", []):
        position = positions.get(int(annotation["image_id"]))
        if position is None:
            continue
        category = indices.get(int(annotation["category_id"]))
        if category is None:
            continue
        item: dict[str, Any] = {
            "id": int(annotation["id"]),
            "image_id": position,
            "category_id": category,
            "bbox": list(annotation.get("bbox", [])),
            "area": float(annotation.get("area", 0.0)),
            "iscrowd": int(annotation.get("iscrowd", 0)),
        }
        if "segmentation" in annotation:
            item["segmentation"] = annotation["segmentation"]
        annotations.append(item)
    return {
        "images": [
            {
                "id": index,
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
            for index, image in enumerate(images)
        ],
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name}
            for index, (_category_id, name) in enumerate(ordered_categories(coco))
        ],
    }


def coco_store_hashes(store: Path, coco: dict[str, Any] | None = None) -> dict[str, str]:
    """SHA-256 of the canonical dataset: the JSON and images as they are on disk.

    ``dataset.json`` is always measured from the file. The import manifest's
    recorded digests describe the bytes that were imported, so trusting them for
    the annotations would let an edited export — relabelled, added or removed
    instances — keep a fingerprint claiming nothing changed. The manifest is
    still the fallback for an entry whose file is no longer where it said.
    """
    store = Path(store)
    manifest_path = store / "manifest.json"
    recorded: dict[str, str] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if isinstance(manifest, dict):
            hashes = manifest.get("dataset_hashes")
            if isinstance(hashes, dict):
                recorded = {str(key): str(value) for key, value in hashes.items()}
    dataset = read_coco_store(store) if coco is None else coco
    result = dict(recorded)
    dataset_json = store / "dataset.json"
    if dataset_json.is_file():
        result["dataset.json"] = sha256_file(dataset_json)
    for image in dataset.get("images", []):
        relative = str(image.get("file_name", ""))
        path = store / relative
        if relative and path.is_file():
            result[relative] = sha256_file(path)
    return result
