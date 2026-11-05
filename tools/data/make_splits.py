"""Re-partition the Poly-X labelled dataset by physical plate.

The labelled dataset is not committed to the repository (see .gitignore:
``/data/``). This script re-partitions a local copy of it into the standard
Ultralytics layout ``images/{train,val,test}`` and ``labels/{train,val,test}``,
using the tracked membership in ``split_manifest.py``.

Two byte-identical images are removed (``LDPE 10`` and ``LDPE 12``); all other
files are moved, never copied, and their bytes are verified before and after
(images and labels alike).

Usage:

    .venv/bin/python tools/data/make_splits.py --root data
    .venv/bin/python tools/data/make_splits.py --root data --apply

``--dry-run`` is the default and writes nothing. The script exits non-zero
and changes nothing when any check fails: it never guesses. If the apply
phase itself fails midway, everything is moved back, so the dataset is left
exactly as it was. Should a restore itself fail, the recovery files are
kept and their location printed instead of being silently discarded.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a declared dependency
    yaml = None

try:
    from . import split_manifest as manifest
except ImportError:  # Script is run directly: add tools/data to the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import split_manifest as manifest  # type: ignore[no-redef]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXT = ".txt"
EXPECTED_NAMES = {"0": "PET", "1": "PP", "2": "LDPE"}

SPLIT_DIRS = ("train", "val", "test")


class DatasetError(RuntimeError):
    """A precondition failed; no changes may be applied."""


@dataclasses.dataclass
class _Report:
    """The collected state of the dataset tree."""

    images: dict[str, Path] = dataclasses.field(default_factory=dict)
    labels: dict[str, Path] = dataclasses.field(default_factory=dict)
    unexpected: list[str] = dataclasses.field(default_factory=list)
    duplicates: list[str] = dataclasses.field(default_factory=list)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_tree(root: Path) -> _Report:
    """Collect images and labels, or fail with a complete list of problems."""
    report = _Report()
    for split in SPLIT_DIRS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.exists() and not label_dir.exists():
            continue
        if image_dir.exists() != label_dir.exists():
            report.unexpected.append(f"half-created split '{split}'")
            continue
        if not image_dir.is_dir() or not label_dir.is_dir():
            report.unexpected.append(f"'{split}' is not a directory")
            continue
        _collect_split(image_dir, "images", report)
        _collect_split(label_dir, "labels", report)

    if report.unexpected:
        return report

    _verify_duplicate_copies(report)
    _verify_expected_sets(report)
    _verify_class_ids(report)
    return report


def _collect_split(directory: Path, kind: str, report: _Report) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        if path.parent != directory:
            # The Ultralytics layout is flat; a nested file would break the
            # destination computation and confuse every consumer of the set.
            report.unexpected.append(f"nested file: {path} (the YOLO layout is flat)")
            continue
        if kind == "images" and path.suffix.lower() in IMAGE_EXTS:
            _remember(report.images, path, report)
        elif kind == "labels" and path.suffix.lower() == LABEL_EXT:
            _remember(report.labels, path, report)
        else:
            report.unexpected.append(str(path))


def _remember(inventory: dict[str, Path], path: Path, report: _Report) -> None:
    stem = path.stem
    other = inventory.get(stem)
    if other is not None:
        report.duplicates.append(f"duplicate {stem} in {other.parent} and {path.parent}")
        return
    inventory[stem] = path


def _verify_duplicate_copies(report: _Report) -> None:
    # The dataset ships two byte-identical plates; both must match their retained twin.
    for duplicate, retained in manifest.DUPLICATE_PAIRS:
        dup = report.images.get(duplicate)
        keep = report.images.get(retained)
        if dup is None:
            continue
        if keep is None:
            raise DatasetError(
                f"{duplicate} exists but the retained duplicate {retained} is missing."
            )
        if _sha256_file(dup) != _sha256_file(keep):
            raise DatasetError(
                f"{duplicate} is not identical to {retained}; refusing to remove it."
            )


def _verify_expected_sets(report: _Report) -> None:
    images = {
        stem: path
        for stem, path in report.images.items()
        if stem not in manifest.REMOVED_DUPLICATES
    }
    labels = {
        stem: path
        for stem, path in report.labels.items()
        if stem not in manifest.REMOVED_DUPLICATES
    }

    expected = {stem for images in manifest.SPLIT_IMAGES.values() for stem in images}
    problems = []
    for label, actual in (("missing images", set(images)), ("missing labels", set(labels))):
        missing = sorted(expected - actual)
        if missing:
            problems.append(f"{label}: {len(missing)}: {', '.join(missing)}")
    unexpected_images = sorted(set(images) - expected)
    unexpected_labels = sorted(set(labels) - expected)
    if unexpected_images:
        problems.append(
            f"unexpected images ({len(unexpected_images)}): {', '.join(unexpected_images)}"
        )
    if unexpected_labels:
        problems.append(
            f"unexpected labels ({len(unexpected_labels)}): {', '.join(unexpected_labels)}"
        )
    if problems:
        raise DatasetError("\n".join(problems))


def _verify_class_ids(report: _Report) -> None:
    problems = [
        f"{path}: unknown class id {class_id}"
        for path in sorted(report.labels.values())
        for class_id in _label_stats(path)
        if str(class_id) not in EXPECTED_NAMES
    ]
    if problems:
        raise DatasetError("\n".join(problems))


def _label_stats(path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            class_id = int(float(fields[0]))
        except ValueError:
            continue
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def _class_counts(report: _Report) -> dict[str, dict[str, int]]:
    names = list(EXPECTED_NAMES.values())
    per_split: dict[str, dict[str, int]] = {}
    for split, stems in manifest.SPLIT_IMAGES.items():
        rows = dict.fromkeys(names, 0)
        for stem in stems:
            label = report.labels[stem]
            for class_id, count in _label_stats(label).items():
                rows[names[class_id]] += count
        per_split[split] = rows
    return per_split


def _expected_names(root: Path) -> dict[str, str]:
    yaml_path = root / "dataset.yaml"
    if yaml is None or not yaml_path.exists():
        return dict(EXPECTED_NAMES)
    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    raw_names = parsed.get("names")
    if not raw_names:
        return dict(EXPECTED_NAMES)
    names = {str(k): str(v) for k, v in dict(raw_names).items()}
    if names != EXPECTED_NAMES:
        raise DatasetError(
            f"unexpected class names in {yaml_path}: {names} (expected {EXPECTED_NAMES})"
        )
    return names


def _moves_from_report(report: _Report) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for inventory in (report.images, report.labels):
        for split, stems in manifest.SPLIT_IMAGES.items():
            for stem in stems:
                path = inventory[stem]
                destination_dir = path.parents[1] / split  # flat layout, checked at collect
                if path.parent != destination_dir:
                    moves.append((path, destination_dir / path.name))
    return moves


def _patch_dataset_yaml(root: Path, names: dict[str, str]) -> None:
    names_lines = []
    for index, name in sorted(names.items(), key=lambda item: int(item[0])):
        names_lines.append(f"  {index}: {name}\n")

    text = (
        "# Dataset Poly-X — microplastics Nile Red fluorescence (PET/PP/LDPE)\n"
        "#\n"
        "# Paths are relative to this file (data/). Without 'path', Ultralytics uses\n"
        "# the YAML directory as the root, so the dataset works on any PC.\n"
        "# Maintained by tools/data/make_splits.py; do not edit split values by hand.\n"
        "\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n" + "".join(names_lines)
    )
    (root / "dataset.yaml").write_text(text, encoding="utf-8")


def _ensure_split_dirs(root: Path, created: list[Path]) -> None:
    """Create the split directories, appending each one to ``created`` as made.

    The caller owns the list so that a failure part-way through can still be
    rolled back: whatever this call managed to create is removed again.
    """
    for split in SPLIT_DIRS:
        for kind in ("images", "labels"):
            directory = root / kind / split
            if not directory.exists():
                directory.mkdir(parents=True)
                created.append(directory)


def _stage_file(path: Path, backup: Path, root: Path, bucket: str) -> tuple[Path, Path]:
    target = backup / bucket / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    return path, target


def _stage_caches(root: Path, backup: Path) -> list[tuple[Path, Path]]:
    labels_dir = root / "labels"
    if not labels_dir.is_dir():
        return []
    return [
        _stage_file(cache, backup, root, "caches")
        for cache in sorted(labels_dir.rglob("*.cache"))
        if cache.is_file()
    ]


def _stage_duplicates(root: Path, backup: Path) -> list[tuple[Path, Path]]:
    staged = []
    for stem in manifest.REMOVED_DUPLICATES:
        for split in SPLIT_DIRS:
            label = root / "labels" / split / f"{stem}{LABEL_EXT}"
            if label.exists():
                staged.append(_stage_file(label, backup, root, "duplicates"))
            for ext in IMAGE_EXTS:
                image = root / "images" / split / f"{stem}{ext}"
                if image.exists():
                    staged.append(_stage_file(image, backup, root, "duplicates"))
    return staged


def _backup_dataset_yaml(root: Path, backup: Path) -> bool:
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        return False
    shutil.copy2(yaml_path, backup / "dataset.yaml")
    return True


def _rollback(
    completed_moves: list[tuple[Path, Path]],
    staged: list[tuple[Path, Path]],
    yaml_backed_up: bool,
    backup: Path,
    root: Path,
    created_dirs: list[Path],
) -> int:
    """Undo every mutation this apply performed, best effort.

    Returns the number of restore operations that failed; the caller must
    keep the backup directory when that is non-zero, because it still holds
    the only copy of whatever could not be put back.
    """
    failures = 0
    for source, destination in reversed(completed_moves):
        try:
            destination.rename(source)
        except OSError as exc:
            failures += 1
            print(f"warning: could not restore {destination} -> {source}: {exc}",
                  file=sys.stderr)
    if yaml_backed_up:
        try:
            shutil.copy2(backup / "dataset.yaml", root / "dataset.yaml")
        except OSError as exc:
            failures += 1
            print(f"warning: could not restore dataset.yaml: {exc}", file=sys.stderr)
    for original, backup_path in reversed(staged):
        try:
            original.parent.mkdir(parents=True, exist_ok=True)
            backup_path.rename(original)
        except OSError as exc:
            failures += 1
            print(f"warning: could not restore {original}: {exc}", file=sys.stderr)
    for directory in reversed(created_dirs):
        try:
            directory.rmdir()  # only removes it if the rollback left it empty
        except OSError as exc:
            failures += 1
            print(f"warning: could not remove {directory}: {exc}", file=sys.stderr)
    return failures


def _summary_lines(report: _Report) -> list[str]:
    lines = [
        f"found {len(report.images)} image files",
        (
            f"  target: {len(manifest.SPLIT_IMAGES['train'])} train / "
            f"{len(manifest.SPLIT_IMAGES['val'])} val / "
            f"{len(manifest.SPLIT_IMAGES['test'])} test"
        ),
        f"  removing duplicates: {', '.join(manifest.REMOVED_DUPLICATES)}",
    ]
    return lines


def _verify_after_apply(root: Path, before: dict[str, str], before_labels: dict[str, str]) -> None:
    found_images: dict[str, Path] = {}
    found_labels: dict[str, Path] = {}
    for split in SPLIT_DIRS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if image_dir.is_dir():
            for path in sorted(image_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    found_images[path.stem] = path
        if label_dir.is_dir():
            for path in sorted(label_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() == LABEL_EXT:
                    found_labels[path.stem] = path

    problems = []
    for stem, expected_hash in before.items():
        path = found_images.get(stem)
        if path is None:
            problems.append(f"{stem}: missing after apply")
        elif _sha256_file(path) != expected_hash:
            problems.append(f"{stem}: image bytes changed during apply")
    for stem, expected_hash in before_labels.items():
        path = found_labels.get(stem)
        if path is None:
            problems.append(f"{stem}: label missing after apply")
        elif _sha256_file(path) != expected_hash:
            problems.append(f"{stem}: label bytes changed during apply")
    if problems:
        raise DatasetError("post-apply verification failed:\n" + "\n".join(problems))


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_splits.py", description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="Dataset root directory (the data/ folder).")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the split. Default: verify and report only.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: dataset root does not exist: {root}", file=sys.stderr)
        return 2

    try:
        names = _expected_names(root)
        report = _collect_tree(root)
    except DatasetError as exc:
        print(f"dataset preflight failed; NO changes made:\n{exc}", file=sys.stderr)
        return 1

    if report.unexpected:
        print("dataset preflight failed; NO changes made:", file=sys.stderr)
        for item in report.unexpected:
            print(f"  - {item}", file=sys.stderr)
        return 1
    if report.duplicates:
        print("duplicate basenames found; NO changes made:", file=sys.stderr)
        for item in report.duplicates:
            print(f"  - {item}", file=sys.stderr)
        return 1

    moves = _moves_from_report(report)
    for line in _summary_lines(report):
        print(line)

    if moves:
        print(f"\n{len(moves)} file(s) would move:")
        for source, destination in moves:
            print(f"  {source.relative_to(root)} -> {destination.relative_to(root)}")
    else:
        print("\nthe tree is already in the requested layout; no moves needed")

    total_bytes = sum(path.stat().st_size for path in report.images.values())
    print(
        f"\n{len(report.images)} images / {len(report.labels)} labels, "
        f"{total_bytes / 1e6:.1f} MB payload"
    )

    print("\nclass boxes per split:")
    for split, per_class in _class_counts(report).items():
        summary = "  ".join(f"{name}={count}" for name, count in per_class.items())
        print(f"  {split:<5} {summary}")

    if not args.apply:
        print("\ndry run; re-run with --apply to write the split")
        return 0

    retained = {stem for images in manifest.SPLIT_IMAGES.values() for stem in images}
    before = {stem: _sha256_file(report.images[stem]) for stem in retained}
    before_labels = {stem: _sha256_file(report.labels[stem]) for stem in retained}

    backup = Path(tempfile.mkdtemp(prefix=".make_splits_backup_", dir=root.parent))
    staged: list[tuple[Path, Path]] = []
    completed_moves: list[tuple[Path, Path]] = []
    created_dirs: list[Path] = []
    yaml_backed_up = False
    try:
        _ensure_split_dirs(root, created_dirs)
        caches = _stage_caches(root, backup)
        duplicates = _stage_duplicates(root, backup)
        staged += caches
        staged += duplicates
        yaml_backed_up = _backup_dataset_yaml(root, backup)

        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise DatasetError(f"refusing to overwrite {destination}")
            source.rename(destination)
            completed_moves.append((source, destination))

        _patch_dataset_yaml(root, names)
        _verify_after_apply(root, before, before_labels)
    except (DatasetError, OSError) as exc:
        failures = _rollback(
            completed_moves, staged, yaml_backed_up, backup, root, created_dirs
        )
        if failures:
            print(
                f"apply failed and {failures} restore operation(s) did not complete; "
                f"the recovery files are kept at {backup}:\n{exc}",
                file=sys.stderr,
            )
        else:
            shutil.rmtree(backup, ignore_errors=True)
            print(
                f"apply failed; the dataset was restored to its previous state:\n{exc}",
                file=sys.stderr,
            )
        return 1
    shutil.rmtree(backup, ignore_errors=True)

    print("\napplied:")
    print(f"  {len(completed_moves)} file(s) moved")
    print(f"  {len(duplicates)} duplicate file(s) removed (LDPE 10/12)")
    print(f"  {len(caches)} ultralytics cache(s) removed")
    print("  dataset.yaml rewritten: test added, header comment fixed")
    print("  all retained image and label bytes verified after apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
