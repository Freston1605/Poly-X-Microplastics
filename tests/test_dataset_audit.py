"""The dataset audit reads its labels through the label codec.

A conventional dataset (``images/<split>`` beside ``labels/<split>``) must
report the class and box statistics the audit has always reported, and a
YOLO-seg polygon line must not be counted as a detection box: the polygon
dialect is valid YOLO but not a box, so the codec's box reader counts it
malformed instead of mangling its first four coordinates into a box. The
audit reports those lines as a warning rather than dropping them silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polyx.core.quality.dataset_audit import LEVEL_WARN, audit_dataset
from tests.conftest import split_dataset

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
    val = train_labels if val_labels is None else val_labels
    splits = {
        "train": {f"train_{i}": text for i, text in enumerate(train_labels)},
        "val": {f"val_{i}": text for i, text in enumerate(val)},
    }
    split_dataset(root, splits, suffix=".jpg", data_yaml=True)
    return root / "data.yaml"


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
    # The one polygon per split is reported instead of silently discarded.
    malformed = [
        finding
        for finding in audit.findings
        if finding.level == LEVEL_WARN and "YOLO-seg" in finding.detail
    ]
    assert malformed, audit.findings
    assert "2" in malformed[0].title, malformed[0].title
    # The explanation names the dialect; it survives the locale.
    assert "YOLO-seg" in malformed[0].detail
