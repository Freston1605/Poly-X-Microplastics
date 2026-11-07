"""Deep audit of a YOLO dataset before training.

It goes beyond "does the file exist?": it analyzes the class distribution
between train and val, detects classes that are never validated (the classic
error that inflates the metrics), dominance by a few images,
degenerate/out-of-range boxes, and classes indistinguishable by size. Designed
so the lab chemist does not train blindly on a bad split.

Public function: ``audit_dataset(yaml_path) -> DatasetAudit``.
Pure Python (no Qt) → testable headless."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from polyx.core.i18n import _

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

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


def _label_for(img: Path) -> Path | None:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            cand = Path(*(parts[:i] + ["labels"] + parts[i + 1 :])).with_suffix(".txt")  # noqa: RUF005
            return cand if cand.exists() else None
    sib = img.parent.parent / "labels" / (img.stem + ".txt")
    if sib.exists():
        return sib
    same = img.with_suffix(".txt")
    return same if same.exists() else None


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
    """Reads labels of a split. Returns per-class counts, box stats and issues."""
    per_class: dict[int, int] = defaultdict(int)
    areas: dict[int, list[float]] = defaultdict(list)
    per_class_per_img: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )  # class -> {img: n}
    degenerate = out_range = duplicates = empty = no_label = 0
    n_imgs = len(images)
    for img in images:
        lp = _label_for(img)
        if lp is None:
            no_label += 1
            continue
        seen = set()
        n_boxes = 0
        for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                c = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:5])
            except ValueError:
                continue
            n_boxes += 1
            per_class[c] += 1
            per_class_per_img[c][img.name] += 1
            areas[c].append(w * h)
            if w <= 0 or h <= 0 or w < 1e-4 or h < 1e-4:
                degenerate += 1
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                out_range += 1
            key = (c, round(cx, 5), round(cy, 5), round(w, 5), round(h, 5))
            if key in seen:
                duplicates += 1
            seen.add(key)
        if n_boxes == 0:
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
        n_imgs=n_imgs,
    )


# ── Auditor: method object holding the shared context ────────────
class _Auditor:
    """Holds the mutable state shared by all the audit checks.

    Replaces the free functions that used to receive ``audit``/``tr``/``vl``/
    ``cls_name`` as arguments. Each check is now a method reading ``self.*``.
    """

    def __init__(self, yaml_path: Path):
        self.yaml_path = yaml_path
        self.audit = DatasetAudit()
        self.data: dict = {}
        self.yaml_dir: Path = yaml_path.parent
        self.root: Path = yaml_path.parent
        self.scans: dict = {}
        self.tr: dict = {}
        self.vl: dict = {}

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
        self._check_structure()
        self._check_class_coverage()
        self._check_imbalance()
        self._check_dominance()
        self._check_size_similarity()
        self._check_mechanical()
        self._check_val_size()
        self._finalize()
        return audit

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
