"""A CVAT COCO export driven through the one headless command seam.

A tiny export stands in for the first stratified plate batch. The command must
import and validate it, hand the masks to the *existing* dataset audit (not a
second validator), store the result as a versioned tool output away from every
detection run folder, and refuse machine-generated masks as ground truth.

Every assertion reads an artifact the command wrote or a value it returned; no
private helper of the pipeline is imported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import write_image

NAMES = ("PET", "PP", "LDPE")
# Four distinct, non-collinear corners: a valid polygon on a 64x48 image.
_POLYGON = [10.0, 8.0, 30.0, 8.0, 30.0, 28.0, 10.0, 28.0]


def _polygon(dx: float = 0.0) -> list[float]:
    return [value + (dx if index % 2 == 0 else 0.0) for index, value in enumerate(_POLYGON)]


def _build_export(
    root: Path,
    *,
    degenerate: bool = False,
    invalid: bool = False,
    rle: bool = False,
    machine_source: str | None = None,
    split: bool = True,
    image_name: str = "train/plate_t1.png",
) -> Path:
    """Write a CVAT-shaped COCO export with train and val images.

    ``split=False`` writes an export without any split marker, which is what a
    tool that does not carry the project's convention emits.
    """
    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images = [
        {"id": 1, "file_name": "train/plate_t1.png", "width": 64, "height": 48},
        {"id": 2, "file_name": "train/plate_t2.png", "width": 64, "height": 48},
        {"id": 3, "file_name": "val/plate_v1.png", "width": 64, "height": 48},
        {"id": 4, "file_name": "val/plate_v2.png", "width": 64, "height": 48},
    ]
    if split:
        for image in images:
            image["split"] = "train" if image["file_name"].startswith("train") else "val"
    images[0]["file_name"] = image_name
    for image in images:
        write_image(root / "images" / image["file_name"], width=64, height=48)

    # (image_id, category_id, x-offset): every class appears in both splits.
    planned = [
        (1, 1, 0.0),
        (1, 2, 32.0),
        (2, 3, 0.0),
        (2, 1, 32.0),
        (3, 1, 0.0),
        (3, 2, 32.0),
        (4, 3, 0.0),
        (4, 2, 32.0),
    ]
    annotations: list[dict] = []
    for annotation_id, (image_id, category_id, dx) in enumerate(planned, start=1):
        segmentation: object = [_polygon(dx)]
        if invalid and annotation_id == 1:
            segmentation = [[10.0, 8.0, 30.0]]  # odd coordinate count
        if degenerate and annotation_id == 1:
            segmentation = [[10.0, 8.0, 20.0, 8.0, 30.0, 8.0]]  # collinear, zero area
        if rle and annotation_id == 1:
            segmentation = {"size": [48, 64], "counts": [10, 20]}
        annotation: dict = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "bbox": [10.0 + dx, 8.0, 20.0, 20.0],
            "area": 400.0,
            "iscrowd": 0,
        }
        if machine_source is not None and annotation_id == 2:
            annotation["attributes"] = {"source": machine_source}
        annotations.append(annotation)

    export = {
        "info": {"description": "CVAT export of the stratified first plate batch"},
        "images": images,
        "categories": [{"id": index, "name": name} for index, name in enumerate(NAMES, start=1)],
        "annotations": annotations,
    }
    (annotations_dir / "instances_default.json").write_text(
        json.dumps(export, indent=2) + "\n", encoding="utf-8"
    )
    return root


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


def _overrides(
    export: Path, dataset_root: Path, output: Path, name: str = "first_batch"
) -> list[str]:
    return [
        f"coco_import={export}",
        f"dataset_root={dataset_root}",
        f"output_dir={output}",
        f"dataset_name={name}",
    ]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_coco_import_emits_a_versioned_tool_output_through_the_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Import, audit and artifact are all observable from the one command."""
    export = _build_export(tmp_path / "export")
    dataset_root = tmp_path / "user_data" / "segmentation_datasets"
    output = tmp_path / "executions" / "import"

    code = _command(*_overrides(export, dataset_root, output))

    assert code == 0, capsys.readouterr().err
    store = dataset_root / "first_batch" / "v1"
    for relative in ("dataset.json", "audit.json", "manifest.json"):
        assert (store / relative).is_file(), f"missing tool output: {relative}"
    assert (store / "images" / "train" / "plate_t1.png").is_file()
    assert (store / "images" / "val" / "plate_v2.png").is_file()

    # The canonical COCO JSON is rewritten so its file_name values resolve
    # inside the versioned output; the masks never depend on the export folder.
    canonical = _read(store / "dataset.json")
    file_names = {image["file_name"] for image in canonical["images"]}
    assert file_names == {
        "images/train/plate_t1.png",
        "images/train/plate_t2.png",
        "images/val/plate_v1.png",
        "images/val/plate_v2.png",
    }

    manifest = _read(store / "manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["version"] == "v1"
    assert manifest["format"] == "coco_instances"
    assert manifest["artifact_class"] == "tool_output"
    assert manifest["source_tool"] == "cvat"
    assert manifest["ground_truth"]["policy"] == "human-only"
    assert manifest["ground_truth"]["machine_generated_rejected"] == 0
    assert manifest["class_distribution"] == {
        "train": {"PET": 2, "PP": 1, "LDPE": 1},
        "val": {"PET": 1, "PP": 2, "LDPE": 1},
    }
    assert "ten plates" in manifest["first_batch_target"]
    assert manifest["counts"] == {"images": 4, "annotations": 8, "categories": 3}
    assert manifest["dataset_hashes"]["dataset.json"]
    assert manifest["dataset_hashes"]["images/train/plate_t1.png"]
    assert len(manifest["source_export_sha256"]) == 64

    audit = _read(store / "audit.json")
    assert audit["ok"] is True
    assert audit["per_class"]["train"] == {"PET": 2, "PP": 1, "LDPE": 1}
    assert audit["per_class"]["val"] == {"PET": 1, "PP": 2, "LDPE": 1}

    # The command's own artifacts repeat the distribution and name the store.
    assert {path.name for path in output.iterdir()} == {
        "config.yaml",
        "import_manifest.json",
        "dataset_audit.json",
        "report.md",
    }
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Class distribution" in report
    assert "PET" in report and "LDPE" in report
    assert str(store) in report
    assert "ten plates" in report

    printed = capsys.readouterr().out
    assert str(store) in printed
    for marker in ("DATASET:", "VERSION: v1", "CLASSES:", "AUDIT: ok=true"):
        assert marker in printed


def test_default_dataset_root_is_a_tool_output_under_the_user_data_dir(tmp_path: Path) -> None:
    """The storage location is outside every detection run folder by default."""
    from polyx.core.paths import RUNS_DIR, RUNS_TRAIN_DIR, USER_DATA_DIR
    from polyx.pipeline.config import load_config

    export = _build_export(tmp_path / "export")
    config = load_config([f"coco_import={export}", f"output_dir={tmp_path / 'out'}"])

    assert config.dataset_root == USER_DATA_DIR / "segmentation_datasets"
    assert config.dataset_root.is_relative_to(USER_DATA_DIR)
    assert not config.dataset_root.is_relative_to(RUNS_DIR)
    assert not config.dataset_root.is_relative_to(RUNS_TRAIN_DIR)


def test_degenerate_mask_polygon_is_rejected_by_the_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], selected_language
) -> None:
    """A zero-area polygon fails the audit and no canonical version is stored."""
    selected_language("en")  # the finding text below is asserted in English
    export = _build_export(tmp_path / "export", degenerate=True)
    dataset_root = tmp_path / "user_data" / "segmentation_datasets"
    output = tmp_path / "executions" / "rejected"

    code = _command(*_overrides(export, dataset_root, output))

    assert code == 1
    assert "degenerate mask polygons" in capsys.readouterr().err.lower()
    assert not (dataset_root / "first_batch" / "v1").exists()


def test_invalid_mask_polygon_is_rejected_by_the_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], selected_language
) -> None:
    """An odd coordinate count is an invalid polygon and stops the import."""
    selected_language("en")  # the finding text below is asserted in English
    export = _build_export(tmp_path / "export", invalid=True)
    dataset_root = tmp_path / "user_data" / "segmentation_datasets"
    output = tmp_path / "executions" / "rejected"

    code = _command(*_overrides(export, dataset_root, output))

    assert code == 1
    assert "invalid mask polygons" in capsys.readouterr().err.lower()
    assert not (dataset_root / "first_batch" / "v1").exists()


def test_rle_masks_are_rejected_because_there_is_no_rle_codec(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], selected_language
) -> None:
    """COCO RLE is not a polygon; the canonical dataset is polygon-only."""
    selected_language("en")  # the finding text below is asserted in English
    export = _build_export(tmp_path / "export", rle=True)
    dataset_root = tmp_path / "user_data" / "segmentation_datasets"
    output = tmp_path / "executions" / "rle"

    code = _command(*_overrides(export, dataset_root, output))

    assert code == 1
    assert "rle segmentation" in capsys.readouterr().err.lower()
    assert not (dataset_root / "first_batch" / "v1").exists()


@pytest.mark.parametrize("source", ["SAM2", "sam2", "Otsu", "yolo-seg", "YOLO_seg"])
def test_machine_generated_mask_cannot_be_recorded_as_ground_truth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str
) -> None:
    """Any machine generator marker rejects the whole export as GT."""
    export = _build_export(tmp_path / "export", machine_source=source)
    dataset_root = tmp_path / "user_data" / "segmentation_datasets"
    output = tmp_path / "executions" / "machine"

    code = _command(*_overrides(export, dataset_root, output))

    assert code == 1
    error = capsys.readouterr().err.lower()
    assert "machine-generated" in error
    assert source.lower().replace("_", "-") in error.replace("_", "-")
    assert not (dataset_root / "first_batch" / "v1").exists()


def test_a_second_import_increments_the_tool_output_version(tmp_path: Path) -> None:
    """Versioning is observable: an unchanged re-import does not overwrite v1."""
    export = _build_export(tmp_path / "export")
    dataset_root = tmp_path / "user_data" / "segmentation_datasets"

    assert _command(*_overrides(export, dataset_root, tmp_path / "run1")) == 0
    assert _command(*_overrides(export, dataset_root, tmp_path / "run2")) == 0

    assert (dataset_root / "first_batch" / "v1" / "manifest.json").is_file()
    assert (dataset_root / "first_batch" / "v2" / "manifest.json").is_file()
    assert _read(dataset_root / "first_batch" / "v2" / "manifest.json")["version"] == "v2"


@pytest.mark.parametrize(
    "image_name, expected",
    [("train/missing.png", "image is missing"), ("../x.png", "unsafe coco image file_name")],
)
def test_an_unreadable_or_unsafe_image_path_leaves_no_version_on_disk(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], image_name: str, expected: str
) -> None:
    """A failed import writes no canonical version, not even a partial one."""
    export = _build_export(tmp_path / "export", image_name=image_name)
    if expected == "image is missing":
        # The fixture wrote the referenced image; remove it to model an export
        # copied without its image folder.
        (export / "images" / image_name).unlink()
    dataset_root = tmp_path / "user_data" / "segmentation_datasets"
    output = tmp_path / "executions" / "broken"

    code = _command(*_overrides(export, dataset_root, output))

    assert code == 1
    assert expected in capsys.readouterr().err.lower()
    assert not (dataset_root / "first_batch").exists()


def test_an_export_without_split_markers_imports_with_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], selected_language
) -> None:
    """An unsplit export is accepted, but the audit says validation is missing."""
    selected_language("en")  # the finding text below is asserted in English
    export = _build_export(tmp_path / "export", split=False)
    dataset_root = tmp_path / "user_data" / "segmentation_datasets"
    output = tmp_path / "executions" / "unsplit"

    code = _command(*_overrides(export, dataset_root, output))

    assert code == 0, capsys.readouterr().err
    audit = _read(dataset_root / "first_batch" / "v1" / "audit.json")
    assert audit["ok"] is True
    assert audit["per_class"] == {
        "train": {"PET": 3, "PP": 3, "LDPE": 2},
        "val": {},
        "test": {},
    }
    titles = [finding["title"].lower() for finding in audit["findings"]]
    assert any("validation" in title for title in titles)


def test_ambiguous_input_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An execution and an import cannot share one invocation."""
    export = _build_export(tmp_path / "export")
    code = _command(
        f"coco_import={export}",
        f"data_yaml={tmp_path / 'data.yaml'}",
        f"output_dir={tmp_path / 'out'}",
    )

    assert code == 1
    assert "exactly one" in capsys.readouterr().err.lower()
