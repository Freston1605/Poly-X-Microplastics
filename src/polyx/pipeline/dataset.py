"""Dataset reads shared by the audit pre-flight, the evaluator and the adapter.

The label codec stays the sole owner of YOLO label I/O; this module only turns
its boxes into the COCO instance view the shared evaluator consumes. Dataset
hashes are produced with the project's existing SHA-256 helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from polyx.core.detect.protocol import size_image
from polyx.core.labels import find_label_file, read_boxes
from polyx.core.paths import find_images
from polyx.core.provenance import sha256_file


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
