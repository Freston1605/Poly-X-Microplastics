"""Job-time conversion of the canonical COCO mask dataset to YOLO-seg labels.

The canonical mask format is COCO instance JSON (ADR-0009). A YOLO-seg family
expects the Ultralytics folder contract instead: ``images/<split>`` beside
``labels/<split>`` holding normalized polygon lines and a ``data.yaml``
declaring the classes. This module writes exactly that, at job time, into the
scratch filesystem.

Nothing here is an artifact or a second canonical dataset: the output is a
cache keyed by the canonical ``dataset.json``, it lives under the configured
scratch root, and deleting it only costs a rebuild. Images are copied, never
moved or linked back into the store, and polygons are written through the
project's only YOLO label codec.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from polyx.core.i18n import N_
from polyx.core.labels import format_polygon_line, label_path_for, write_polygon_lines
from polyx.core.provenance import sha256_file
from polyx.pipeline.dataset import (
    ordered_categories,
    read_coco_store,
    split_coco_images,
)
from polyx.pipeline.errors import PipelineError

#: Name written into the conversion metadata; a change means a new cache key.
CONVERSION_FORMAT = "yolo-seg-polygons"
#: Bump when the produced layout changes, so old caches are not reused.
CONVERSION_VERSION = 3
#: Splits the canonical dataset may carry; only these are written out.
_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ConvertedDataset:
    """Where the family-facing fallback of the canonical masks lives."""

    root: Path
    data_yaml: Path
    cache_key: str
    cached: bool
    images: int
    labels: int
    polygons: int
    #: SHA-256 of each image whose pixels this conversion actually used.
    content_hashes: dict[str, str] = field(default_factory=dict)

    def describe(self, store: Path) -> dict[str, Any]:
        """Provenance view: where the intermediate is, never a canonical path."""
        return {
            "format": CONVERSION_FORMAT,
            "version": CONVERSION_VERSION,
            "root": str(self.root),
            "data_yaml": str(self.data_yaml),
            "cache_key": self.cache_key,
            "cached": self.cached,
            "source": str(store),
            "counts": {
                "images": self.images,
                "labels": self.labels,
                "polygons": self.polygons,
            },
            "ephemeral": True,
        }


def image_hashes(store: Path, coco: dict[str, Any]) -> dict[str, str]:
    """SHA-256 of every image as it exists on disk right now."""
    return {
        str(image["file_name"]): sha256_file(store / str(image["file_name"]))
        for image in coco.get("images", [])
    }


def _cache_key(store: Path, coco: dict[str, Any], images: dict[str, str]) -> str:
    """Key the cache by the JSON and every image's actual bytes.

    Two stores can share a dataset.json and still hold different pixels. A
    JSON-only key would let the second execution silently train on the first
    store's cached images.
    """
    payload = json.dumps(
        {"dataset": sha256_file(store / "dataset.json"), "images": images},
        sort_keys=True,
    ).encode()
    return f"{CONVERSION_FORMAT}-v{CONVERSION_VERSION}-{hashlib.sha256(payload).hexdigest()[:16]}"


def _rings(annotation: dict[str, Any]) -> list[list[tuple[float, float]]]:
    """Valid polygon rings of one COCO annotation, in pixel coordinates."""
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, (list, tuple)):
        return []
    rings: list[list[tuple[float, float]]] = []
    for ring in segmentation:
        if not isinstance(ring, (list, tuple)) or len(ring) < 6 or len(ring) % 2 != 0:
            continue
        numbers = [float(value) for value in ring]
        rings.append([(numbers[index], numbers[index + 1]) for index in range(0, len(numbers), 2)])
    return rings


def _collision_key(
    class_id: int,
    ring: list[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[int, float, float, float, float]:
    """The key the pinned loader deduplicates label rows by.

    ``verify_image_label`` parses each line as ``float32``, converts the
    normalized polygon to a box with ``segments2boxes`` and removes rows that
    are equal **after that conversion**. Two rings with different vertices but
    the same extrema therefore collapse into one training target even though
    their label lines differ. The key is built from the line this converter is
    about to write, so the check cannot drift from the file.
    """
    line = format_polygon_line(class_id, ring, width, height)
    values = np.asarray([float(value) for value in line.split()[1:]], dtype=np.float64)
    normalized = values.reshape(-1, 2)
    x_min = float(normalized[:, 0].min())
    y_min = float(normalized[:, 1].min())
    x_max = float(normalized[:, 0].max())
    y_max = float(normalized[:, 1].max())
    box = np.asarray([x_min, y_min, x_max - x_min, y_max - y_min])
    derived = np.asarray([class_id, *box], dtype=np.float32)
    return (
        int(derived[0]),
        float(derived[1]),
        float(derived[2]),
        float(derived[3]),
        float(derived[4]),
    )


def _write_split(
    store: Path,
    coco: dict[str, Any],
    split: str,
    root: Path,
    indices: dict[int, int],
) -> tuple[int, int, int]:
    """Convert one split; return ``(images, labels, polygons)``."""
    images = split_coco_images(coco, split)
    annotations: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)

    images_dir = root / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)

    polygon_count = 0
    label_count = 0
    for position, image in enumerate(images):
        source = store / str(image["file_name"])
        if not source.is_file():
            raise ValueError(f"COCO image is missing from the dataset: {image['file_name']}")
        suffix = Path(str(image["file_name"])).suffix
        stem = f"{position:06d}"
        (images_dir / f"{stem}{suffix}").write_bytes(source.read_bytes())

        # Positions, not canonical annotation ids, are the label file's order.
        ordered = sorted(annotations.get(int(image["id"]), []), key=lambda item: item["id"])
        polygons: list[tuple[int, list[tuple[float, float]]]] = []
        seen: dict[tuple[int, float, float, float, float], int] = {}
        for annotation in ordered:
            class_id = indices[int(annotation["category_id"])]
            rings = _rings(annotation)
            if len(rings) != 1:
                raise PipelineError(
                    "seg_conversion.multipart_annotation",
                    N_(
                        "a COCO mask annotation must have exactly one polygon ring to "
                        "stay one training instance; annotation %(id)s has %(count)d. "
                        "Split the particle into separate annotations in the "
                        "annotation tool, or export a single exterior polygon."
                    ),
                    id=annotation.get("id", "?"),
                    count=len(rings),
                )
            key = _collision_key(class_id, rings[0], int(image["width"]), int(image["height"]))
            if key in seen:
                raise PipelineError(
                    "seg_conversion.colliding_target",
                    N_(
                        "image %(image)s has annotations %(first)s and %(second)s "
                        "that derive the same YOLO box; the pinned loader would "
                        "collapse them into one target. Merge or remove the "
                        "duplicate in the annotation tool."
                    ),
                    image=image["file_name"],
                    first=seen[key],
                    second=annotation.get("id", "?"),
                )
            seen[key] = int(annotation.get("id", -1))
            polygons.append((class_id, rings[0]))
        label_path = label_path_for(images_dir / f"{stem}{suffix}")
        write_polygon_lines(
            label_path,
            polygons,
            int(image["width"]),
            int(image["height"]),
        )
        label_count += 1
        polygon_count += len(polygons)
    return len(images), label_count, polygon_count


def convert_coco_dataset(store: Path, scratch_root: Path) -> ConvertedDataset:
    """Convert one canonical COCO store into the YOLO-seg folder contract.

    The result is cached under ``scratch_root/seg_datasets/<cache key>``. A
    second execution over the same canonical dataset reuses it; a different
    dataset hashes to a different key and cannot pick up stale labels.
    """
    store = Path(store).expanduser().resolve()
    scratch_root = Path(scratch_root).expanduser().resolve()
    coco = read_coco_store(store)
    hashes = image_hashes(store, coco)
    cache_key = _cache_key(store, coco, hashes)
    target = scratch_root / "seg_datasets" / cache_key
    data_yaml = target / "data.yaml"
    if data_yaml.is_file():
        counts = _read_counts(target / "conversion.json")
        return ConvertedDataset(
            root=target,
            data_yaml=data_yaml,
            cache_key=cache_key,
            cached=True,
            images=counts.get("images", 0),
            labels=counts.get("labels", 0),
            polygons=counts.get("polygons", 0),
            content_hashes=hashes,
        )

    splits = [split for split in _SPLITS if split_coco_images(coco, split)]
    if "train" not in splits or "val" not in splits:
        raise ValueError(
            "the COCO mask dataset needs train and val images to train and select a "
            f"checkpoint; found splits: {', '.join(splits) or 'none'}. Re-import the "
            "export with image-level split markers."
        )

    partial = target.parent / f".{cache_key}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    indices = {
        category_id: index for index, (category_id, _) in enumerate(ordered_categories(coco))
    }
    images = labels = polygons = 0
    for split in splits:
        split_images, split_labels, split_polygons = _write_split(
            store, coco, split, partial, indices
        )
        images += split_images
        labels += split_labels
        polygons += split_polygons

    declaration: dict[str, Any] = {
        "path": str(target),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, (_id, name) in enumerate(ordered_categories(coco))},
    }
    if "test" in splits:
        declaration["test"] = "images/test"
    (partial / "data.yaml").write_text(
        yaml.safe_dump(declaration, sort_keys=False), encoding="utf-8"
    )
    (partial / "conversion.json").write_text(
        json.dumps(
            {
                "format": CONVERSION_FORMAT,
                "version": CONVERSION_VERSION,
                "source": str(store),
                "cache_key": cache_key,
                "counts": {"images": images, "labels": labels, "polygons": polygons},
                "image_hashes": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if target.exists():
        # A cache of a different version that shares the key cannot exist; a
        # half-written folder without data.yaml is rebuilt.
        shutil.rmtree(target)
    partial.rename(target)
    return ConvertedDataset(
        root=target,
        data_yaml=target / "data.yaml",
        cache_key=cache_key,
        cached=False,
        images=images,
        labels=labels,
        polygons=polygons,
        content_hashes=hashes,
    )


def _read_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    counts = loaded.get("counts") if isinstance(loaded, dict) else None
    if not isinstance(counts, dict):
        return {}
    return {str(key): int(value) for key, value in counts.items()}
