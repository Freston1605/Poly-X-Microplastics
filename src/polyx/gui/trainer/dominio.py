"""Chooses the weight according to the real subset of the validation.

The dataset combines two subsets, marked by the file-name prefix that
``armar_dataset.py`` puts: ``lab__`` and ``real__``. The ``lab__`` subset
contributes many more boxes per image, so it dominates the global mAP and
``best.pt`` is chosen with it.

Here that choice is redone looking only at ``real__``, and the winner is
saved separately. The training leaves three weights:

    best_sintetico.pt   the one Ultralytics picks by global mAP
    best_real.pt        the best on the real subset
    last.pt             last epoch, for resuming

If a dataset does not carry both prefixes, the step is skipped silently.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

REAL_PREFIX = "real__"
LAB_PREFIX = "lab__"

# The three weights left at the end. Ultralytics' best.pt is preserved.
BEST_REAL_NAME = "best_real.pt"
BEST_LAB_NAME = "best_sintetico.pt"


def _val_images(data_yaml: Path) -> List[Path]:
    """Lists the images of the validation split declared in the yaml."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    val = cfg.get("val")
    if not val:
        return []
    path = Path(val)
    if not path.is_absolute():
        path = (root / path).resolve()

    if path.is_file():  # list of .txt paths
        return [Path(l.strip()) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)
    return []


def domains_present(data_yaml: Path) -> Dict[str, int]:
    """Counts how many validation images there are of each domain."""
    count = {"real": 0, "lab": 0, "sin_marcar": 0}
    for p in _val_images(Path(data_yaml)):
        if p.name.startswith(REAL_PREFIX):
            count["real"] += 1
        elif p.name.startswith(LAB_PREFIX):
            count["lab"] += 1
        else:
            count["sin_marcar"] += 1
    return count


def can_reselect(data_yaml: Path) -> bool:
    """True if the validation mixes domains, so re-choosing makes sense."""
    c = domains_present(Path(data_yaml))
    return c["real"] > 0 and c["lab"] > 0


def _label_for(image: Path) -> Path:
    """Path of an image's .txt, with the images/ -> labels/ convention."""
    parts_2 = list(image.parts)
    for i in range(len(parts_2) - 1, -1, -1):
        if parts_2[i] == "images":
            parts_2[i] = "labels"
            break
    return Path(*parts_2).with_suffix(".txt")


def _link(origin: Path, destination: Path) -> None:
    """Hard link when possible; copy otherwise (other volume, no permissions)."""
    import os
    import shutil

    if destination.exists():
        return
    try:
        os.link(origin, destination)
    except (OSError, NotImplementedError):
        shutil.copy2(origin, destination)


def create_real_only_yaml(data_yaml: Path, destination_dir: Path) -> Optional[Path]:
    """Writes a dataset.yaml whose validation set is only the real images.

    The subset lives in its **own tree** images/ + labels/, not in a list
    pointing at the original split. Ultralytics derives the label cache path
    from the folder that contains them, so validating a subset of
    ``images/val`` rewrites ``labels/val.cache`` with only those images: the
    next training would silently validate on the subset and its best.pt would
    mean something else. With its own tree, the cache that gets touched is
    ``_seleccion/labels/val.cache`` and the dataset's cache stays intact.

    It is hard-linked instead of copied: same volume, no duplicated bytes.
    """
    data_yaml = Path(data_yaml)
    real = [p for p in _val_images(data_yaml) if p.name.startswith(REAL_PREFIX)]
    if not real:
        return None

    img_dir = destination_dir / "images" / "val"
    lbl_dir = destination_dir / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img in real:
        if not img.exists():
            continue
        _link(img, img_dir / img.name)
        label_path = _label_for(img)
        if label_path.exists():
            _link(label_path, lbl_dir / label_path.name)

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    new = {
        "path": str(destination_dir.resolve()),
        # train is not used when validating, but Ultralytics expects the key.
        # It points to the same subset so the large split is not dragged in.
        "train": "images/val",
        "val": "images/val",
        "names": cfg.get("names", {}),
    }
    output = destination_dir / "dataset_solo_real.yaml"
    output.write_text(yaml.safe_dump(new, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output


def candidates(run_dir: Path) -> List[Path]:
    """Checkpoints to evaluate: best, last and the periodic ones if any."""
    weights = Path(run_dir) / "weights"
    if not weights.is_dir():
        return []
    # Weights produced by this same function are excluded: when retraining over
    # an existing run they would be there and evaluated as if they were candidates.
    own = {BEST_REAL_NAME, BEST_LAB_NAME}
    seen, output = set(), []
    for p in [weights / "best.pt", weights / "last.pt"] + sorted(weights.glob("epoch*.pt")):
        if p.exists() and p.name not in seen and p.name not in own:
            seen.add(p.name)
            output.append(p)
    return output


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def choose_better_in_real(
    run_dir: Path,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[Path], List[Dict]]:
    """Evaluates the checkpoints on real sediment and copies the best one.

    Returns (path of best_real.pt, results table). If nothing can be
    evaluated, it returns (None, []) and the training goes on with its usual
    best.pt.
    """

    def _log(m: str) -> None:
        if log:
            log(m)

    run_dir = Path(run_dir)
    weights = candidates(run_dir)
    if not weights:
        _log("[INFO] No hay checkpoints que evaluar; se conserva best.pt.")
        return None, []

    yaml_real = create_real_only_yaml(Path(data_yaml), run_dir / "_seleccion")
    if yaml_real is None:
        _log("[INFO] La validacion no tiene subconjunto real; se conserva best.pt.")
        return None, []

    from ultralytics import YOLO

    _log("")
    _log("=" * 60)
    _log("  SELECCION DE PESOS")
    _log("=" * 60)
    _log(f"  Evaluando {len(weights)} checkpoint(s) sobre el subconjunto real.")

    table: List[Dict] = []
    for p in weights:
        try:
            res = YOLO(str(p)).val(
                data=str(yaml_real),
                imgsz=imgsz,
                batch=batch,
                device=device,
                plots=False,
                verbose=False,
            )
            pr, rc = float(res.box.mp), float(res.box.mr)
            row = {
                "checkpoint": p.name,
                "ruta": p,
                "precision": pr,
                "recall": rc,
                "f1": _f1(pr, rc),
                "map50": float(res.box.map50),
            }
        except Exception as exc:
            _log(f"  [WARN] no se pudo evaluar {p.name}: {exc}")
            continue
        table.append(row)
        _log(
            f"    {row['checkpoint']:<14} F1={row['f1']:.4f}  "
            f"P={pr:.4f}  R={rc:.4f}  mAP50={row['map50']:.4f}"
        )

    if not table:
        _log("  No se pudo evaluar ningun checkpoint; se conserva best.pt.")
        return None, []

    # F1 as the criterion, mAP50 as the tie-breaker: getting the number of
    # particles right matters, not refining the box.
    table.sort(key=lambda d: (d["f1"], d["map50"]), reverse=True)
    best = table[0]

    destination = run_dir / "weights" / BEST_REAL_NAME
    try:
        shutil.copy2(best["ruta"], destination)
    except OSError as exc:
        _log(f"  [WARN] no se pudo copiar el ganador: {exc}")
        return None, table

    _log("")
    _log(f"  Mejor en el subconjunto real: {best['checkpoint']} (F1={best['f1']:.4f})")
    origin_best = next((d for d in table if d["checkpoint"] == "best.pt"), None)
    if origin_best and origin_best["checkpoint"] != best["checkpoint"]:
        _log(f"  best.pt daba F1={origin_best['f1']:.4f}.")
    elif origin_best:
        _log("  Coincide con best.pt.")

    # Copy with its own name of the weight Ultralytics chooses, so all three
    # are named equally clearly in the run folder.
    origin = run_dir / "weights" / "best.pt"
    if origin.exists():
        try:
            shutil.copy2(origin, run_dir / "weights" / BEST_LAB_NAME)
        except OSError as exc:
            _log(f"  [WARN] no se pudo crear {BEST_LAB_NAME}: {exc}")

    _log("")
    _log("  Pesos disponibles en weights/:")
    _log(f"    {BEST_LAB_NAME:<20} mejor por mAP global")
    _log(f"    {BEST_REAL_NAME:<20} mejor sobre el subconjunto real")
    _log(f"    {'last.pt':<20} ultima epoca, para reanudar")
    _log("=" * 60)
    return destination, table
