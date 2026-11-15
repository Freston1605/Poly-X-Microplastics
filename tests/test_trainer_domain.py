"""``create_real_only_yaml`` links validation labels through the codec.

The real-subset yaml builds its own ``images/val`` + ``labels/val`` tree by
hard-linking (or copying) the source images and their labels. The source
labels must be located by the label codec's split convention, and an image
without a label must simply contribute no label file instead of failing.
"""

from __future__ import annotations

from pathlib import Path

from polyx.gui.trainer.dominio import create_real_only_yaml

REAL = "real__"
LAB = "lab__"


def make_dataset(root: Path, images: dict[str, str | None]) -> Path:
    """A split dataset; the value is the label text, or ``None`` for none."""
    images_dir = root / "images" / "val"
    labels_dir = root / "labels" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for name, label_text in images.items():
        (images_dir / f"{name}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        if label_text is not None:
            (labels_dir / f"{name}.txt").write_text(label_text, encoding="utf-8")

    yaml_path = root / "data.yaml"
    yaml_path.write_text("path: .\nval: images/val\nnames:\n  0: PET\n", encoding="utf-8")
    return yaml_path


def test_split_layout_label_is_linked_into_the_real_subset(tmp_path: Path):
    """``images/val/...`` resolves to ``labels/val/...`` through the codec."""
    yaml_path = make_dataset(
        tmp_path / "dataset",
        {f"{REAL}a": "0 0.5 0.5 0.2 0.2\n", f"{LAB}b": "0 0.5 0.5 0.2 0.2\n"},
    )

    output = create_real_only_yaml(yaml_path, tmp_path / "out")

    assert output is not None
    subset_labels = tmp_path / "out" / "labels" / "val"
    assert (subset_labels / f"{REAL}a.txt").read_text(encoding="utf-8") == ("0 0.5 0.5 0.2 0.2\n")
    assert not (subset_labels / f"{LAB}b.txt").exists(), "only real images are linked"


def test_image_without_label_is_linked_without_a_label_file(tmp_path: Path):
    """A missing label is not an error: the image is linked, the label is not."""
    yaml_path = make_dataset(
        tmp_path / "dataset",
        {f"{REAL}a": None, f"{REAL}b": "0 0.5 0.5 0.2 0.2\n"},
    )

    output = create_real_only_yaml(yaml_path, tmp_path / "out")

    assert output is not None
    subset_images = tmp_path / "out" / "images" / "val"
    subset_labels = tmp_path / "out" / "labels" / "val"
    assert (subset_images / f"{REAL}a.jpg").exists()
    assert (subset_images / f"{REAL}b.jpg").exists()
    assert not (subset_labels / f"{REAL}a.txt").exists()
    assert (subset_labels / f"{REAL}b.txt").exists()


def test_legacy_same_directory_label_is_found_by_the_codec(tmp_path: Path):
    """A label next to the image (no conventional ``labels/val``) is still linked."""
    root = tmp_path / "dataset"
    images_dir = root / "images" / "val"
    images_dir.mkdir(parents=True)
    (images_dir / f"{REAL}a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (images_dir / f"{REAL}a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text("path: .\nval: images/val\nnames:\n  0: PET\n", encoding="utf-8")

    output = create_real_only_yaml(yaml_path, tmp_path / "out")

    assert output is not None
    assert (tmp_path / "out" / "labels" / "val" / f"{REAL}a.txt").exists()
