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


def _iter_images(split_path: Path, yaml_dir: Path, root: Path) -> list[Path]:
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
    per_class = defaultdict(int)
    areas = defaultdict(list)
    per_class_per_img = defaultdict(lambda: defaultdict(int))  # class -> {img: n}
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


# ── Helpers for audit_dataset ────────────────────────────────────
def _cls_name(audit: DatasetAudit, cid: int) -> str:
    return audit.class_names[cid] if 0 <= cid < len(audit.class_names) else f"clase {cid}"


def _setup_context(yaml_path: Path, audit: DatasetAudit):
    """Load yaml and resolve root. Returns (data, yaml_dir, root) or None on failure."""
    try:
        data = _load_yaml(yaml_path)
    except Exception as e:  # noqa: BLE001
        audit.add(LEVEL_ERROR, "No se pudo leer data.yaml", str(e))
        return None
    yaml_dir = yaml_path.parent
    root_value = data.get("path")
    if root_value:
        root = Path(root_value)
        if not root.is_absolute():
            root = (yaml_dir / root).resolve()
    else:
        root = yaml_dir
    return data, yaml_dir, root


def _resolve_names(data: dict, audit: DatasetAudit) -> None:
    names = data.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]  # noqa: PLW0108
    audit.class_names = [str(n) for n in names]


def _scan_all_splits(data, yaml_dir, root, audit):
    scans = {}
    for split in ("train", "val", "test"):
        split_value = data.get(split)
        resolved = _resolve_split(yaml_dir, data, split_value)
        if not split_value and split in ("train", "val"):
            audit.add(
                LEVEL_ERROR,
                f"Sin split {split} declarado",
                f"El data.yaml no tiene clave '{split}'. Ultralytics la "
                "necesita para entrenar y medir métricas.",
            )
        elif split_value and (resolved is None or not resolved.exists()):
            lvl = LEVEL_ERROR if split in ("train", "val") else LEVEL_WARN
            audit.add(
                lvl,
                f"Ruta de {split} no existe",
                f"El data.yaml apunta a «{resolved}», que no existe en este equipo. "
                "Suele pasar al mover el proyecto de carpeta o de PC: corrige las "
                "rutas del data.yaml (o usa rutas relativas).",
            )
        imgs = _iter_images(resolved, yaml_dir, root)
        scans[split] = _scan_split(imgs)
        scans[split]["path_exists"] = bool(resolved and resolved.exists())
        audit.per_class[split] = {_cls_name(audit, c): n for c, n in scans[split]["per_class"].items()}

    tr, vl = scans["train"], scans["val"]

    for c, ar in tr["areas"].items():
        if ar:
            audit.box_area_median[_cls_name(audit, c)] = statistics.median(ar)

    return scans, tr, vl


def _check_structure(audit, tr, vl):
    if not audit.class_names:
        audit.add(LEVEL_ERROR, "Sin clases definidas", "El data.yaml no tiene clave 'names'.")
    if tr["n_imgs"] == 0 and tr.get("path_exists"):
        audit.add(
            LEVEL_ERROR, "Train vacío", "La carpeta de train existe pero no contiene imágenes."
        )
    if vl["n_imgs"] == 0 and vl.get("path_exists"):
        audit.add(
            LEVEL_ERROR,
            "Val vacío",
            "La carpeta de val existe pero no contiene imágenes → no se pueden medir métricas.",
        )


def _check_class_coverage(audit, tr, vl):
    train_classes = set(tr["per_class"].keys())
    val_classes = set(vl["per_class"].keys())

    missing_in_val = train_classes - val_classes
    if missing_in_val and vl["n_imgs"] > 0:
        names_2 = ", ".join(sorted(_cls_name(audit, c) for c in missing_in_val))
        audit.add(
            LEVEL_ERROR,
            f"Clases sin validación: {names_2}",
            "Estas clases tienen ejemplos en train pero CERO en val. El mAP "
            "no medirá su rendimiento (te dará una falsa sensación de éxito). "
            "Agrega imágenes de estas clases al split de validación.",
        )
    if len(val_classes) == 1 and len(train_classes) > 1:
        audit.add(
            LEVEL_ERROR,
            f"Validación de una sola clase ({_cls_name(audit, next(iter(val_classes)))})",
            "El set de validación contiene una única clase pese a que el "
            "modelo aprende varias. Las métricas multiclase no son válidas.",
        )


def _check_imbalance(audit, tr, cls_name_fn):
    counts = tr["per_class"]
    if len(counts) >= 2:
        mx, mn = max(counts.values()), min(counts.values())
        if mn > 0 and mx / mn >= 10:
            top = cls_name_fn(max(counts, key=counts.get))
            bot = cls_name_fn(min(counts, key=counts.get))
            audit.add(
                LEVEL_WARN,
                "Desbalance fuerte de clases en train",
                f"«{top}» tiene {mx} cajas y «{bot}» solo {mn} "
                f"({mx / mn:.0f}× más). Considera balancear o usar pesos por clase.",
            )


def _check_dominance(audit, tr, cls_name_fn):
    for c, per_img in tr["per_class_per_img"].items():
        total = tr["per_class"].get(c, 0)
        if total >= 50 and per_img:
            top_img, top_n = max(per_img.items(), key=lambda kv: kv[1])
            if top_n / total >= 0.40:
                audit.add(
                    LEVEL_WARN,
                    f"«{cls_name_fn(c)}»: poca diversidad",
                    f"El {top_n / total * 100:.0f}% de sus cajas viene de una sola "
                    f"imagen ({top_img}). El modelo puede sobreajustar a esa escena.",
                )


def _check_size_similarity(audit):
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
                        f"«{na}» y «{nb}» tienen tamaño casi idéntico",
                        "Sus cajas son del mismo tamaño, así que el modelo solo "
                        "puede distinguirlas por color/textura. Si además se "
                        "parecen visualmente, esperá confusión entre ambas.",
                    )


def _check_mechanical(audit, tr, vl):
    for split, sc in (("train", tr), ("val", vl)):
        if sc["degenerate"]:
            audit.add(
                LEVEL_WARN,
                f"Cajas degeneradas en {split}: {sc['degenerate']}",
                "Cajas con ancho/alto ~0. Revisa la anotación.",
            )
        if sc["out_range"]:
            audit.add(
                LEVEL_ERROR,
                f"Coordenadas fuera de [0,1] en {split}: {sc['out_range']}",
                "Labels mal normalizadas; YOLO las interpretará mal.",
            )
        if sc["duplicates"]:
            audit.add(
                LEVEL_WARN,
                f"Cajas duplicadas en {split}: {sc['duplicates']}",
                "Misma caja repetida en un archivo.",
            )
        if sc["no_label"]:
            audit.add(
                LEVEL_WARN,
                f"Imágenes sin label en {split}: {sc['no_label']}",
                "Se entrenarán como fondo (sin objetos). Confirma que es intencional.",
            )


def _check_val_size(audit, vl):
    if 0 < vl["n_imgs"] < 5:
        audit.add(
            LEVEL_WARN,
            f"Validación muy pequeña ({vl['n_imgs']} imágenes)",
            "Con tan pocas imágenes las métricas son inestables. Apunta a ≥ 10-20.",
        )


def _finalize(audit):
    if audit.ok and not any(f.level == LEVEL_WARN for f in audit.findings):
        audit.add(
            LEVEL_OK,
            "Dataset sin problemas detectados",
            "Distribución de clases, validación y formato se ven correctos.",
        )


# ──────────────────────────────────────────────────────────────────
def audit_dataset(yaml_path: str | Path) -> DatasetAudit:
    audit = DatasetAudit()
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        audit.add(LEVEL_ERROR, "data.yaml no encontrado", str(yaml_path))
        return audit

    ctx = _setup_context(yaml_path, audit)
    if ctx is None:
        return audit
    data, yaml_dir, root = ctx

    _resolve_names(data, audit)

    scans, tr, vl = _scan_all_splits(data, yaml_dir, root, audit)

    def cls_name(cid: int) -> str:
        return _cls_name(audit, cid)

    # ── Checks ──
    _check_structure(audit, tr, vl)
    _check_class_coverage(audit, tr, vl)
    _check_imbalance(audit, tr, cls_name)
    _check_dominance(audit, tr, cls_name)
    _check_size_similarity(audit)
    _check_mechanical(audit, tr, vl)
    _check_val_size(audit, vl)
    _finalize(audit)

    return audit
