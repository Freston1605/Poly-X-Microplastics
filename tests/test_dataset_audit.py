"""The dataset audit reads its labels through the label codec.

A conventional dataset (``images/<split>`` beside ``labels/<split>``) must
report the class and box statistics the audit has always reported, and a
YOLO-seg polygon line must not be counted as a detection box: the polygon
dialect is valid YOLO but not a box, so the codec's box reader counts it
malformed instead of mangling its first four coordinates into a box.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polyx.core.quality.dataset_audit import LEVEL_WARN, audit_dataset

PET_BOX = "0 0.5 0.5 0.10 0.10"
PP_BOX = "1 0.5 0.5 0.30 0.30"
POLYGON = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"


def make_dataset(
    root: Path,
    *,
    train_labels: list[str],
    val_labels: list[str] | None = None,
) -> Path:
    """A conventional YOLO dataset: ``images/<split>`` + ``labels/<split>``."""
    for split, labels in (("train", train_labels), ("val", val_labels or train_labels)):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(labels):
            (images_dir / f"{split}_{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (labels_dir / f"{split}_{i}.txt").write_text(text, encoding="utf-8")

    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: PET\n  1: PP\n",
        encoding="utf-8",
    )
    return yaml_path


def test_audit_finds_conventional_split_labels(tmp_path: Path):
    """``labels/<split>`` is the codec's conventional location; the audit uses it."""
    yaml_path = make_dataset(
        tmp_path / "dataset",
        train_labels=[f"{PET_BOX}\n{PP_BOX}\n", f"{PET_BOX}\n{PP_BOX}\n"],
        val_labels=[f"{PET_BOX}\n{PP_BOX}\n", f"{PET_BOX}\n{PP_BOX}\n"],
    )

    audit = audit_dataset(yaml_path)

    assert audit.per_class["train"] == {"PET": 2, "PP": 2}
    assert audit.per_class["val"] == {"PET": 2, "PP": 2}
    assert audit.box_area_median["PET"] == pytest.approx(0.01)
    assert audit.box_area_median["PP"] == pytest.approx(0.09)
    # Labels were found: no "images with no label" warning.
    assert not any(
        finding.level == LEVEL_WARN and "no label" in finding.title.lower()
        for finding in audit.findings
    ), audit.findings


def test_audit_does_not_count_a_polygon_line_as_a_box(tmp_path: Path):
    """A YOLO-seg polygon is not a detection box and must not pollute the stats."""
    yaml_path = make_dataset(
        tmp_path / "dataset",
        train_labels=[f"{PET_BOX}\n", f"{POLYGON}\n"],
        val_labels=[f"{PET_BOX}\n", f"{POLYGON}\n"],
    )

    audit = audit_dataset(yaml_path)

    # Only the one real box exists in each split; the polygon adds nothing.
    assert audit.per_class["train"] == {"PET": 1}
    assert audit.per_class["val"] == {"PET": 1}
    # If the polygon were misread as a box its 0.9 x 0.1 area would shift this.
    assert audit.box_area_median["PET"] == pytest.approx(0.01)
