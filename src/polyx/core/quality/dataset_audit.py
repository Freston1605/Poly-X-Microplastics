"""Deep audit of a YOLO dataset before training.

It goes beyond "does the file exist?": it analyzes the class distribution
between train and val, detects classes that are never validated (the classic
error that inflates the metrics), dominance by a few images,
degenerate/out-of-range boxes, and classes indistinguishable by size. Designed
so the lab chemist does not train blindly on a bad split.

Public function: ``audit_dataset(yaml_path) -> DatasetAudit``.
Pure Python (no Qt) → testable headless."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polyx.core.i18n import _
from polyx.core.labels import find_label_file, read_boxes
from polyx.core.paths import IMAGE_EXTS

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"


@dataclass
class Finding:
    level: str  # ok | warn | error
    title: str
    detail: str = ""


@dataclass
class DatasetAudit:
    findings: list[Finding] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)  # split -> {class: n}
    box_area_median: dict[str, float] = field(
        default_factory=dict
    )  # class -> median normalised area
    ok: bool = True  # False if there is any Finding at error level

    def add(self, level: str, title: str, detail: str = ""):
        self.findings.append(Finding(level, title, detail))
        if level == LEVEL_ERROR:
            self.ok = False


# ──────────────────────────────────────────────────────────────────
def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_split(yaml_dir: Path, data: dict, split_value) -> Path | None:
    if not split_value:
        return None
    p = Path(split_value)
    if p.is_absolute():
        return p
    root_value = data.get("path")
    if root_value:
        root = Path(root_value)
        if not root.is_absolute():
            root = (yaml_dir / root).resolve()
        cand = (root / p).resolve()
        if cand.exists():
            return cand
    return (yaml_dir / p).resolve()


def _iter_images(split_path: Path | None, yaml_dir: Path, root: Path) -> list[Path]:
    if split_path is None or not split_path.exists():
        return []
    if split_path.is_dir():
        out: list[Path] = []
        for ext in IMAGE_EXTS:
            out.extend(split_path.rglob(f"*{ext}"))
        return out
    # .txt listing file
    out = []
    for line in split_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()  # noqa: PLW2901
        if not line or line.startswith("#"):
            continue
        pp = Path(line)
        if not pp.is_absolute():
            cand = (root / pp).resolve()
            pp = cand if cand.exists() else (yaml_dir / line).resolve()
        if pp.exists():
            out.append(pp)
    return out


def _scan_split(images: list[Path]) -> dict:
    """Reads labels of a split. Returns per-class counts, box stats and issues.

    Labels are located and parsed by the label codec, so the split layout and
    the polygon dialect are understood in one place: a YOLO-seg polygon line
    is valid YOLO but not a box, and the codec's box reader counts it
    malformed instead of misreading its first four coordinates as a box. The
    count is returned so the audit can report it instead of dropping it.
    """
    per_class: dict[int, int] = defaultdict(int)
    areas: dict[int, list[float]] = defaultdict(list)
    per_class_per_img: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )  # class -> {img: n}
    degenerate = out_range = duplicates = empty = no_label = malformed = 0
    n_imgs = len(images)
    for img in images:
        lp = find_label_file(img)
        if lp is None:
            no_label += 1
            continue
        boxes, malformed_lines = read_boxes(lp)
        malformed += malformed_lines
        seen = set()
        for box in boxes:
            c = box.class_id
            per_class[c] += 1
            per_class_per_img[c][img.name] += 1
            areas[c].append(box.w * box.h)
            if box.w <= 0 or box.h <= 0 or box.w < 1e-4 or box.h < 1e-4:
                degenerate += 1
            if not (0 <= box.cx <= 1 and 0 <= box.cy <= 1 and 0 <= box.w <= 1 and 0 <= box.h <= 1):
                out_range += 1
            key = (c, round(box.cx, 5), round(box.cy, 5), round(box.w, 5), round(box.h, 5))
            if key in seen:
                duplicates += 1
            seen.add(key)
        if not boxes:
            empty += 1
    return dict(  # noqa: C408
        per_class=dict(per_class),
        areas=dict(areas),
        per_class_per_img={k: dict(v) for k, v in per_class_per_img.items()},
        degenerate=degenerate,
        out_range=out_range,
        duplicates=duplicates,
        empty=empty,
        no_label=no_label,
        malformed=malformed,
        n_imgs=n_imgs,
    )


# ── Auditor: method object holding the shared context ────────────
class _Auditor:
    """Holds the mutable state shared by all the audit checks.

    Replaces the free functions that used to receive ``audit``/``tr``/``vl``/
    ``cls_name`` as arguments. Each check is now a method reading ``self.*``.
    """

    def __init__(self, yaml_path: Path | None = None):
        self.yaml_path = yaml_path or Path()
        self.audit = DatasetAudit()
        self.data: dict = {}
        self.yaml_dir: Path = self.yaml_path.parent if yaml_path is not None else Path()
        self.root: Path = self.yaml_dir
        self.scans: dict[str, dict[str, Any]] = {}
        self.tr: dict[str, Any] = {}
        self.vl: dict[str, Any] = {}
        self.source = "yolo"

    @classmethod
    def from_scans(cls, scans: dict[str, dict[str, Any]], class_names: list[str]) -> _Auditor:
        """An auditor over label scans built from another label source.

        The COCO-mask entry point builds the same per-split scan dictionaries a
        YOLO dataset produces, then hands them to the same check methods. It is
        an alternate *input* to the auditor, not a second validator.
        """
        auditor = cls()
        auditor.source = "coco"
        auditor.audit = DatasetAudit(class_names=list(class_names))
        auditor.scans = scans
        auditor.tr = scans["train"]
        auditor.vl = scans["val"]
        return auditor

    # ── orchestration ──
    def run(self) -> DatasetAudit:
        audit = self.audit
        if not self.yaml_path.exists():
            audit.add(LEVEL_ERROR, _("data.yaml not found"), str(self.yaml_path))
            return audit

        try:
            self.data = _load_yaml(self.yaml_path)
        except Exception as e:  # noqa: BLE001
            audit.add(LEVEL_ERROR, _("Could not read data.yaml"), str(e))
            return audit

        self.yaml_dir = self.yaml_path.parent
        root_value = self.data.get("path")
        if root_value:
            root = Path(root_value)
            if not root.is_absolute():
                root = (self.yaml_dir / root).resolve()
            self.root = root
        else:
            self.root = self.yaml_dir

        self._resolve_names()
        self._scan_all_splits()
        return self.run_checks()

    def run_checks(self) -> DatasetAudit:
        """The checks shared by YOLO scans and COCO-mask scans."""
        self._check_structure()
        self._check_class_coverage()
        self._check_imbalance()
        self._check_dominance()
        self._check_size_similarity()
        self._check_mechanical()
        self._check_polygons()
        self._check_val_size()
        self._finalize()
        return self.audit

    # ── helpers ──
    def _cls_name(self, cid: int) -> str:
        audit = self.audit
        return (
            audit.class_names[cid]
            if 0 <= cid < len(audit.class_names)
            else _("class %(id)s") % {"id": cid}
        )

    def _resolve_names(self) -> None:
        names = self.data.get("names") or []
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]  # noqa: PLW0108
        self.audit.class_names = [str(n) for n in names]

    def _scan_all_splits(self) -> None:
        audit = self.audit
        data, yaml_dir, root = self.data, self.yaml_dir, self.root
        scans = {}
        for split in ("train", "val", "test"):
            split_value = data.get(split)
            resolved = _resolve_split(yaml_dir, data, split_value)
            if not split_value and split in ("train", "val"):
                audit.add(
                    LEVEL_ERROR,
                    _("No %(split)s split declared") % {"split": split},
                    _(
                        "The data.yaml has no '%(split)s' key. Ultralytics needs it "
                        "to train and measure metrics."
                    )
                    % {"split": split},
                )
            elif split_value and (resolved is None or not resolved.exists()):
                lvl = LEVEL_ERROR if split in ("train", "val") else LEVEL_WARN
                audit.add(
                    lvl,
                    _("%(split)s path does not exist") % {"split": split},
                    _(
                        "The data.yaml points to “%(path)s”, which does not exist on this "
                        "computer. This usually happens when the project is moved to another "
                        "folder or PC: fix the paths in the data.yaml (or use relative paths)."
                    )
                    % {"path": resolved},
                )
            imgs = _iter_images(resolved, yaml_dir, root)
            scans[split] = _scan_split(imgs)
            scans[split]["path_exists"] = bool(resolved and resolved.exists())
            audit.per_class[split] = {
                self._cls_name(c): n for c, n in scans[split]["per_class"].items()
            }

        self.scans = scans
        self.tr = scans["train"]
        self.vl = scans["val"]

        for c, ar in self.tr["areas"].items():
            if ar:
                audit.box_area_median[self._cls_name(c)] = statistics.median(ar)

    def _check_structure(self) -> None:
        audit, tr, vl = self.audit, self.tr, self.vl
        if not audit.class_names:
            audit.add(LEVEL_ERROR, _("No classes defined"), _("The data.yaml has no 'names' key."))
        if tr["n_imgs"] == 0 and tr.get("path_exists"):
            audit.add(
                LEVEL_ERROR,
                _("Empty train"),
                _("The train folder exists but contains no images."),
            )
        if vl["n_imgs"] == 0 and vl.get("path_exists"):
            audit.add(
                LEVEL_ERROR,
                _("Empty val"),
                _("The val folder exists but contains no images → metrics cannot be measured."),
            )
        if self.source == "coco" and tr["n_imgs"] > 0 and vl["n_imgs"] == 0:
            audit.add(
                LEVEL_WARN,
                _("No validation split in the COCO export"),
                _(
                    "Every mask was imported into train because no image carries a split "
                    "marker. Mark the validation images before training: class coverage and "
                    "metrics cannot be audited without them."
                ),
            )

    def _check_class_coverage(self) -> None:
        audit, tr, vl = self.audit, self.tr, self.vl
        train_classes = set(tr["per_class"].keys())
        val_classes = set(vl["per_class"].keys())

        missing_in_val = train_classes - val_classes
        if missing_in_val and vl["n_imgs"] > 0:
            names_2 = ", ".join(sorted(self._cls_name(c) for c in missing_in_val))
            audit.add(
                LEVEL_ERROR,
                _("Classes without validation: %(classes)s") % {"classes": names_2},
                _(
                    "These classes have examples in train but ZERO in val. mAP will not "
                    "measure their performance (it will give you a false sense of success). "
                    "Add images of these classes to the validation split."
                ),
            )
        if len(val_classes) == 1 and len(train_classes) > 1:
            audit.add(
                LEVEL_ERROR,
                _("Single-class validation (%(class)s)")
                % {"class": self._cls_name(next(iter(val_classes)))},
                _(
                    "The validation set contains a single class even though the model "
                    "learns several. Multiclass metrics are not valid."
                ),
            )

    def _check_imbalance(self) -> None:
        audit, tr = self.audit, self.tr
        counts = tr["per_class"]
        if len(counts) >= 2:
            mx, mn = max(counts.values()), min(counts.values())
            if mn > 0 and mx / mn >= 10:
                top = self._cls_name(max(counts, key=counts.get))
                bot = self._cls_name(min(counts, key=counts.get))
                audit.add(
                    LEVEL_WARN,
                    _("Strong class imbalance in train"),
                    _(
                        "“%(top)s” has %(top_count)d boxes and “%(bottom)s” only "
                        "%(bottom_count)d (%(ratio).0f× more). Consider balancing or using "
                        "class weights."
                    )
                    % {
                        "top": top,
                        "top_count": mx,
                        "bottom": bot,
                        "bottom_count": mn,
                        "ratio": mx / mn,
                    },
                )

    def _check_dominance(self) -> None:
        audit, tr = self.audit, self.tr
        for c, per_img in tr["per_class_per_img"].items():
            total = tr["per_class"].get(c, 0)
            if total >= 50 and per_img:
                top_img, top_n = max(per_img.items(), key=lambda kv: kv[1])
                if top_n / total >= 0.40:
                    audit.add(
                        LEVEL_WARN,
                        _("“%(class)s”: little diversity") % {"class": self._cls_name(c)},
                        _(
                            "%(percent).0f%% of its boxes come from a single image "
                            "(%(image)s). The model may overfit to that scene."
                        )
                        % {
                            "percent": top_n / total * 100,
                            "image": top_img,
                        },
                    )

    def _check_size_similarity(self) -> None:
        audit = self.audit
        meds = audit.box_area_median
        cls_list = list(meds.items())
        for i in range(len(cls_list)):
            for j in range(i + 1, len(cls_list)):
                (na, aa), (nb, ab) = cls_list[i], cls_list[j]
                if aa > 0 and ab > 0:
                    ratio = max(aa, ab) / min(aa, ab)
                    if ratio < 1.25:
                        audit.add(
                            LEVEL_WARN,
                            _("“%(first)s” and “%(second)s” are almost identical in size")
                            % {"first": na, "second": nb},
                            _(
                                "Their boxes are the same size, so the model can only tell "
                                "them apart by colour/texture. If they also look alike, "
                                "expect confusion between the two."
                            ),
                        )

    def _check_mechanical(self) -> None:
        audit, tr, vl = self.audit, self.tr, self.vl
        for split, sc in (("train", tr), ("val", vl)):
            if sc["degenerate"]:
                audit.add(
                    LEVEL_WARN,
                    _("Degenerate boxes in %(split)s: %(count)d")
                    % {"split": split, "count": sc["degenerate"]},
                    _("Boxes with width/height ~0. Check the annotation."),
                )
            if sc["out_range"]:
                audit.add(
                    LEVEL_ERROR,
                    _("Coordinates outside [0,1] in %(split)s: %(count)d")
                    % {"split": split, "count": sc["out_range"]},
                    _("Badly normalised labels; YOLO will interpret them wrongly."),
                )
            if sc["duplicates"]:
                audit.add(
                    LEVEL_WARN,
                    _("Duplicate boxes in %(split)s: %(count)d")
                    % {"split": split, "count": sc["duplicates"]},
                    _("The same box repeated within one file."),
                )
            if sc["no_label"]:
                audit.add(
                    LEVEL_WARN,
                    _("Images with no label in %(split)s: %(count)d")
                    % {"split": split, "count": sc["no_label"]},
                    _(
                        "They will be trained as background (no objects). Confirm this is "
                        "intentional."
                    ),
                )
        malformed = sum(scan["malformed"] for scan in self.scans.values())
        if malformed:
            audit.add(
                LEVEL_WARN,
                _("Label lines that are not detection boxes: %(count)d") % {"count": malformed},
                _(
                    "Those lines were skipped. A YOLO-seg polygon line is valid YOLO "
                    "but not a detection box, so the audit counts boxes only."
                ),
            )

    def _check_polygons(self) -> None:
        """Degenerate or structurally invalid mask polygons reject a dataset.

        Only COCO-mask scans carry these counters; YOLO box scans keep them at
        zero, so the shared check runs harmlessly for both label sources.
        """
        audit = self.audit
        for split, scan in self.scans.items():
            invalid = int(scan.get("invalid_polygons", 0))
            degenerate = int(scan.get("degenerate_polygons", 0))
            unknown = int(scan.get("unknown_categories", 0))
            if invalid:
                audit.add(
                    LEVEL_ERROR,
                    _("Invalid mask polygons in %(split)s: %(count)d")
                    % {"split": split, "count": invalid},
                    _(
                        "A mask polygon needs at least three finite, in-bounds points and "
                        "an even coordinate count. RLE segmentation is not a polygon; "
                        "export polygons from the annotation tool."
                    ),
                )
            if degenerate:
                audit.add(
                    LEVEL_ERROR,
                    _("Degenerate mask polygons in %(split)s: %(count)d")
                    % {"split": split, "count": degenerate},
                    _(
                        "These polygons enclose zero area (collinear or repeated points) "
                        "and cannot be used as segmentation ground truth."
                    ),
                )
            if unknown:
                audit.add(
                    LEVEL_ERROR,
                    _("Annotations with an unknown category in %(split)s: %(count)d")
                    % {"split": split, "count": unknown},
                    _(
                        "Every mask must reference a category declared in the export's "
                        "categories list."
                    ),
                )
        orphans = sum(int(scan.get("orphan_annotations", 0)) for scan in self.scans.values())
        if orphans:
            audit.add(
                LEVEL_ERROR,
                _("Annotations with an unknown image: %(count)d") % {"count": orphans},
                _(
                    "These masks reference an image that is not declared in the export's "
                    "images list."
                ),
            )

    def _check_val_size(self) -> None:
        audit, vl = self.audit, self.vl
        if 0 < vl["n_imgs"] < 5:
            audit.add(
                LEVEL_WARN,
                _("Very small validation set (%(count)d images)") % {"count": vl["n_imgs"]},
                _("With so few images the metrics are unstable. Aim for ≥ 10-20."),
            )

    def _finalize(self) -> None:
        audit = self.audit
        if audit.ok and not any(f.level == LEVEL_WARN for f in audit.findings):
            audit.add(
                LEVEL_OK,
                _("No dataset problems detected"),
                _("Class distribution, validation and format all look correct."),
            )


# ──────────────────────────────────────────────────────────────────
def audit_dataset(yaml_path: str | Path) -> DatasetAudit:
    return _Auditor(Path(yaml_path)).run()


# ── COCO mask polygons ─────────────────────────────────────────────────────
# The canonical segmentation ground truth is an external CVAT COCO export.
# The geometry rules below feed the same auditor as the YOLO box path: one
# validator, one DatasetAudit result, two label sources.

_MIN_POLYGON_AREA = 1e-6  # px²; a zero-area ring is not a mask

#: Split names a COCO export may carry as an image-level attribute. Without a
#: marker an image belongs to train, which the audit reports as a warning.
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Absolute shoelace area of one polygon, in pixel squared."""
    total = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def validate_polygon(
    segmentation: Any, width: int, height: int
) -> tuple[list[list[tuple[float, float]]], int, int]:
    """Validate a COCO ``segmentation`` field.

    Returns ``(polygons, invalid_count, degenerate_count)``. A dictionary
    (RLE) is invalid because the project has no RLE codec and the annotation
    tool is asked to export polygons; a flat coordinate list must hold at
    least three finite, in-bounds points and enclose a non-zero area.
    """
    if not isinstance(segmentation, (list, tuple)) or not segmentation:
        return [], 1, 0
    polygons: list[list[tuple[float, float]]] = []
    invalid = degenerate = 0
    for ring in segmentation:
        if not isinstance(ring, (list, tuple)):
            invalid += 1
            continue
        numbers = list(ring)
        if (
            len(numbers) < 6
            or len(numbers) % 2 != 0
            or any(not isinstance(value, (int, float)) for value in numbers)
            or any(not math.isfinite(float(value)) for value in numbers)
        ):
            invalid += 1
            continue
        points = [
            (float(numbers[index]), float(numbers[index + 1]))
            for index in range(0, len(numbers), 2)
        ]
        if any(x < 0 or x > width or y < 0 or y > height for x, y in points):
            invalid += 1
            continue
        if len(set(points)) < 3 or polygon_area(points) <= _MIN_POLYGON_AREA:
            degenerate += 1
            continue
        polygons.append(points)
    return polygons, invalid, degenerate


def _image_split(image: dict[str, Any]) -> str:
    """The normalised split marker of one COCO image, ``train`` by default."""
    value: Any = image.get("split")
    attributes = image.get("attributes")
    if value is None and isinstance(attributes, dict):
        value = attributes.get("split")
    elif value is None and isinstance(attributes, list):
        for item in attributes:
            if isinstance(item, dict) and str(item.get("name", "")).lower() == "split":
                value = item.get("value")
                break
    if value is None or not str(value).strip():
        return "train"
    return _SPLIT_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())


def _empty_coco_scan() -> dict[str, Any]:
    """A scan with the keys the box checks read, plus polygon counters."""
    return {
        "per_class": defaultdict(int),
        "areas": defaultdict(list),
        "per_class_per_img": defaultdict(lambda: defaultdict(int)),
        "degenerate": 0,
        "out_range": 0,
        "duplicates": 0,
        "empty": 0,
        "no_label": 0,
        "malformed": 0,
        "n_imgs": 0,
        "path_exists": False,
        "invalid_polygons": 0,
        "degenerate_polygons": 0,
        "unknown_categories": 0,
        "orphan_annotations": 0,
    }


def audit_coco_masks(coco: dict[str, Any]) -> DatasetAudit:
    """Audit the polygons of a COCO instance dataset with the shared checks.

    The split is read from an image-level ``split`` attribute (CVAT metadata
    or ``attributes``); an image without one belongs to train. COCO category
    ids are arbitrary (CVAT starts at 1), so they are mapped to the sorted
    positions the auditor indexes class names by.
    """
    images: dict[int, dict[str, Any]] = {}
    for item in coco.get("images", []):
        images[int(item["id"])] = item
    category_names = {int(item["id"]): str(item["name"]) for item in coco.get("categories", [])}
    ordered_ids = sorted(category_names)
    id_to_index = {category_id: index for index, category_id in enumerate(ordered_ids)}
    class_names = [category_names[category_id] for category_id in ordered_ids]

    scans: dict[str, dict[str, Any]] = {
        split: _empty_coco_scan() for split in ("train", "val", "test")
    }
    image_split: dict[int, str] = {}
    for image_id, item in images.items():
        split = _image_split(item)
        image_split[image_id] = split
        scan = scans.setdefault(split, _empty_coco_scan())
        scan["n_imgs"] += 1
        scan["path_exists"] = True

    for annotation in coco.get("annotations", []):
        try:
            image_id = int(annotation["image_id"])
        except (KeyError, TypeError, ValueError):
            scans["train"]["orphan_annotations"] += 1
            continue
        image = images.get(image_id)
        if image is None:
            # An annotation that belongs to no image cannot be audited per
            # split; it is an error of the export, not a mask to repair.
            scans["train"]["orphan_annotations"] += 1
            continue
        scan = scans[image_split[image_id]]
        try:
            class_id = id_to_index.get(int(annotation["category_id"]))
        except (KeyError, TypeError, ValueError):
            class_id = None
        if class_id is None:
            scan["unknown_categories"] += 1
            continue
        width, height = int(image["width"]), int(image["height"])
        polygons, invalid, degenerate = validate_polygon(
            annotation.get("segmentation"), width, height
        )
        scan["invalid_polygons"] += invalid
        scan["degenerate_polygons"] += degenerate
        image_area = max(width * height, 1)
        for points in polygons:
            area = polygon_area(points) / image_area
            scan["per_class"][class_id] += 1
            scan["areas"][class_id].append(area)
            scan["per_class_per_img"][class_id][str(image_id)] += 1
        if not polygons and not invalid and not degenerate:
            scan["empty"] += 1

    for scan in scans.values():
        scan["per_class"] = dict(scan["per_class"])
        scan["areas"] = dict(scan["areas"])
        scan["per_class_per_img"] = {
            key: dict(value) for key, value in scan["per_class_per_img"].items()
        }

    auditor = _Auditor.from_scans(scans, class_names)
    audit = auditor.audit
    for split, scan in scans.items():
        audit.per_class[split] = {
            auditor._cls_name(class_id): count for class_id, count in scan["per_class"].items()
        }
        for class_id, areas in scan["areas"].items():
            if areas:
                audit.box_area_median[auditor._cls_name(class_id)] = statistics.median(areas)
    return auditor.run_checks()
