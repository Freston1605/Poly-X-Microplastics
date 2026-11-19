"""Dataset reads shared by the audit pre-flight, the evaluator and the adapter.

The label codec stays the sole owner of YOLO label I/O; this module only turns
its boxes into the COCO instance view the shared evaluator consumes. Dataset
hashes are produced with the project's existing SHA-256 helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from polyx.core.detect.protocol import size_image
from polyx.core.i18n import N_
from polyx.core.labels import find_label_file, read_boxes
from polyx.core.paths import find_images
from polyx.core.provenance import sha256_file
from polyx.core.quality.dataset_audit import coco_image_split
from polyx.pipeline.errors import PipelineError


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


def validate_campaign_cohort(data_yaml: Path, coco_dataset: Path, split: str) -> dict[str, Any]:
    """Prove the two campaign sources describe the same real-domain cohort.

    A benchmark may carry a YOLO box dataset and a COCO mask dataset so the
    detection and segmentation families each consume the source they need.
    Sharing a protocol is not enough: if those two sources hold different
    images or different folds, the model is no longer the only variable. Both
    must declare real-domain imagery as well; a synthetic validation fold
    cannot support a paper-quality comparison.
    """
    declaration = read_data_yaml(data_yaml)
    coco = read_coco_store(coco_dataset)
    if not _is_real_domain(declaration):
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
    for name in ("train", "val", "test"):
        for path in split_images(data_yaml, name):
            if path.name in boxes:
                raise PipelineError(
                    "campaign.duplicate_box_image",
                    N_(
                        "the box dataset reuses the image name %(name)s across "
                        "folds; basenames must be unique for a cohort comparison"
                    ),
                    name=path.name,
                )
            boxes[path.name] = (name, sha256_file(path))
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


def coco_ground_truth(data_yaml: Path, split: str) -> dict[str, Any]:
    """COCO instance view of the split's YOLO box labels."""
    data = read_data_yaml(data_yaml)
    names = class_names(data)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 0
    for image_id, image in enumerate(split_images(data_yaml, split)):
        size = size_image(image)
        if size is None:
            continue
        width, height = size
        images.append({"id": image_id, "file_name": image.name, "width": width, "height": height})
        label_path = find_label_file(image)
        if label_path is None:
            continue
        boxes, _ = read_boxes(label_path)
        for box in boxes:
            x = (box.cx - box.w / 2) * width
            y = (box.cy - box.h / 2) * height
            box_width = box.w * width
            box_height = box.h * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(box.class_id),
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": name} for cid, name in sorted(names.items())],
    }


def dataset_hashes(data_yaml: Path) -> dict[str, str]:
    """SHA-256 of the declaration and every image and label it uses."""
    data_yaml = Path(data_yaml)
    root = data_yaml.parent
    result = {"data.yaml": sha256_file(data_yaml)}
    candidates = sorted({*find_images(root), *root.rglob("*.txt")})
    for path in candidates:
        if path == data_yaml or not path.is_file():
            continue
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


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
    """SHA-256 of the canonical dataset, preferring the import manifest's own."""
    store = Path(store)
    manifest_path = store / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        recorded = manifest.get("dataset_hashes") if isinstance(manifest, dict) else None
        if isinstance(recorded, dict) and recorded:
            return {str(key): str(value) for key, value in recorded.items()}
    dataset = read_coco_store(store) if coco is None else coco
    result = {"dataset.json": sha256_file(store / "dataset.json")}
    for image in dataset.get("images", []):
        relative = str(image.get("file_name", ""))
        path = store / relative
        if relative and path.is_file():
            result[relative] = sha256_file(path)
    return result
