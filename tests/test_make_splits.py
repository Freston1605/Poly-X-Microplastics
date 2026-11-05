"""Tests for tools/data/make_splits.py.

The fixture builds a minimal but structurally valid copy of the real YOLO
dataset (every image paired with a label, two exact byte duplicates) and
verifies that the split is applied atomically, that the duplicate files are
removed, and that the script fails closed when anything is missing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "data" / "make_splits.py"
MANIFEST = ROOT / "tools" / "data" / "split_manifest.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_dataset(tmp_path: Path) -> Path:
    # Load the manifest by path: it is untracked-data tooling, deliberately
    # outside src/polyx/.
    manifest = _load_module("split_manifest", MANIFEST)

    root = tmp_path / "data"
    source_train = "train"
    source_val = "val"
    for split in (source_train, source_val):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    # The initial copy of the dataset (before the split is applied) lives in
    # images/train and images/val only: every file is created there.
    initial_val = {
        "30-03-23 sed marino (6) + PET (F1) 100 mm",
        "30-03-23 sed marino (6.1) + PET (F1) 100 mm",
        "30-03-23 sed marino (7) + PET (F2) 100 mm",
        "30-03-23 sed marino (7.1) + PET (F2) 100 mm",
        "Mezc 4",
        "Mezc 5",
    }
    all_stems = [stem for images in manifest.SPLIT_IMAGES.values() for stem in images]
    for stem in all_stems:
        split = source_val if stem in initial_val else source_train
        image = root / "images" / split / f"{stem}.jpg"
        image.write_bytes(f"fixture-image-{stem}".encode())
        label = root / "labels" / split / f"{stem}.txt"
        class_id = 2 if stem.startswith("LDPE") else 1 if stem.startswith("PP") else 0
        label.write_text(f"{class_id} 0.5 0.5 0.05 0.05\n", encoding="utf-8")

    # Two exact duplicates that the script must refuse to keep.
    def find_image(stem: str) -> Path:
        for split in ("train", "val"):
            candidate = root / "images" / split / f"{stem}.jpg"
            if candidate.exists():
                return candidate
        raise AssertionError(f"fixture source missing: {stem}")

    for duplicate in manifest.REMOVED_DUPLICATES:
        retained = {"LDPE 10": "LDPE 8", "LDPE 12": "LDPE 9"}[duplicate]
        retained_path = find_image(retained)
        (root / "images" / "train" / f"{duplicate}.jpg").write_bytes(retained_path.read_bytes())
        label = root / "labels" / "train" / f"{duplicate}.txt"
        if duplicate == "LDPE 10":
            label.write_text("2 0.5 0.5 0.05 0.05\n", encoding="utf-8")
        # LDPE 12 is intentionally label-less: it is an exact copy of LDPE 9.

    (root / "labels/train.cache").write_bytes(b"fake-cache")
    (root / "labels/val.cache").write_bytes(b"fake-cache")
    (root / "dataset.yaml").write_text(
        "# Rutas RELATIVAS a la ubicación de este archivo (data_microplastico/).\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: PET\n"
        "  1: PP\n"
        "  2: LDPE\n",
        encoding="utf-8",
    )
    return root


def _load_make_splits():
    return _load_module("make_splits", SCRIPT)


def test_dry_run_is_inert(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    exit_code = _load_make_splits()._main(["--root", str(root)])

    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert exit_code == 0
    assert before == after


def test_apply_repartitions_and_deletes_duplicates(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    manifest = _load_module("split_manifest_apply", MANIFEST)

    exit_code = _load_make_splits()._main(["--root", str(root), "--apply"])

    assert exit_code == 0
    assert (root / "images" / "test").is_dir()
    assert (root / "labels" / "test").is_dir()
    for split, stems in manifest.SPLIT_IMAGES.items():
        for stem in stems:
            assert (root / "images" / split / f"{stem}.jpg").is_file()
            assert (root / "labels" / split / f"{stem}.txt").is_file()
    for duplicate in manifest.REMOVED_DUPLICATES:
        assert not list((root / "images").rglob(f"{duplicate}.jpg"))
        assert not list((root / "labels").rglob(f"{duplicate}.txt"))
    assert not list(root.rglob("*.cache"))
    yaml_text = (root / "dataset.yaml").read_text(encoding="utf-8")
    assert "images/test" in yaml_text
    assert "data_microplastico" not in yaml_text


def test_apply_is_idempotent(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    script = _load_make_splits()

    assert script._main(["--root", str(root), "--apply"]) == 0
    files_after_first = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }

    assert script._main(["--root", str(root)]) == 0
    files_after_second = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert files_after_first == files_after_second


def test_missing_label_blocks_changes(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    (root / "labels" / "train" / "LDPE 8.txt").unlink()
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    assert _load_make_splits()._main(["--root", str(root), "--apply"]) == 1
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_changed_duplicate_is_not_removed(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    (root / "images" / "train" / "LDPE 10.jpg").write_bytes(b"different-content")
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    assert _load_make_splits()._main(["--root", str(root), "--apply"]) == 1
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_unexpected_file_blocks_changes(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    (root / "images" / "train" / "classes.txt").write_text("PET\nPP\nLDPE\n", encoding="utf-8")
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    assert _load_make_splits()._main(["--root", str(root), "--apply"]) == 1
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_swapped_class_names_abort_before_changes(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    yaml_path = root / "dataset.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    swapped = text.replace(
        "0: PET\n  1: PP\n  2: LDPE\n",
        "0: PP\n  1: PET\n  2: LDPE\n",
    )
    assert swapped != text
    yaml_path.write_text(swapped, encoding="utf-8")
    before = _snapshot(root)

    assert _load_make_splits()._main(["--root", str(root), "--apply"]) == 1
    assert _snapshot(root) == before


def test_nested_file_aborts_before_changes(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    nested = root / "images" / "val" / "nested"
    nested.mkdir()
    (root / "images" / "val" / "Mezc 5.jpg").rename(nested / "Mezc 5.jpg")
    before = _snapshot(root)

    assert _load_make_splits()._main(["--root", str(root), "--apply"]) == 1
    assert _snapshot(root) == before


def test_unknown_class_id_aborts_before_changes(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    label = root / "labels" / "train" / "Mezc 1.txt"
    label.write_text(
        label.read_text(encoding="utf-8") + "9 0.5 0.5 0.05 0.05\n",
        encoding="utf-8",
    )
    before = _snapshot(root)

    assert _load_make_splits()._main(["--root", str(root), "--apply"]) == 1
    assert _snapshot(root) == before


def test_failed_apply_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_dataset(tmp_path)
    before = _snapshot(root)
    script = _load_make_splits()
    real_rename = Path.rename

    def fail_first_rename_into_test_split(self: Path, target: Path) -> Path:
        if Path(target).parent == root / "images" / "test":
            raise OSError("simulated mid-apply failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_first_rename_into_test_split)
    assert script._main(["--root", str(root), "--apply"]) != 0
    assert _snapshot(root) == before
