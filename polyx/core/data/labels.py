"""Label/annotation format — single source of truth for YOLO-txt I/O."""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from polyx.core.detect.protocol import Instance, Detections


# YOLO-txt format: class_id cx_n cy_n w_n h_n [conf]
# All coordinates normalized to [0, 1] relative to image dimensions.


def read_labels(
    txt_path: str | Path,
    img_w: int,
    img_h: int,
    class_names: Dict[int, str],
    include_conf: bool = False,
) -> List["Instance"]:
    """Read YOLO-txt label file and return Instance list.

    Args:
        txt_path: Path to .txt label file.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        class_names: Mapping from class_id to class name.
        include_conf: If True, expect 6th column (confidence) in file.

    Returns:
        List of Instance objects.
    """
    from polyx.core.detect.protocol import Instance
    instances: List[Instance] = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                expected = 6 if include_conf else 5
                if len(parts) != expected:
                    continue
                cid = int(parts[0])
                cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
                conf = float(parts[5]) if include_conf else 1.0

                cx = cx_n * img_w
                cy = cy_n * img_h
                w = w_n * img_w
                h = h_n * img_h
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                instances.append(Instance(
                    class_id=cid,
                    class_name=class_names.get(cid, str(cid)),
                    conf=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
    except Exception:
        pass
    return instances


def write_labels(
    instances: List["Instance"],
    txt_path: str | Path,
    img_w: int,
    img_h: int,
    include_conf: bool = False,
) -> None:
    """Write Instance list to YOLO-txt label file.

    Args:
        instances: List of Instance objects.
        txt_path: Output .txt file path.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        include_conf: If True, write confidence as 6th column.
    """
    from polyx.core.detect.protocol import Instance
    lines = []
    for inst in instances:
        cx_n = inst.cx / img_w
        cy_n = inst.cy / img_h
        w_n = inst.w / img_w
        h_n = inst.h / img_h
        if include_conf:
            lines.append(f"{inst.class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f} {inst.conf:.6f}")
        else:
            lines.append(f"{inst.class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

    Path(txt_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def classes_txt_path(run_dir: str | Path) -> Path:
    """Return the path to classes.txt for a run directory."""
    return Path(run_dir) / "classes.txt"


def write_classes_txt(class_names: Dict[int, str], txt_path: str | Path) -> None:
    """Write class names to classes.txt (one per line, ordered by class_id)."""
    max_id = max(class_names.keys()) if class_names else 0
    lines = [class_names.get(i, f"class_{i}") for i in range(max_id + 1)]
    Path(txt_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_classes_txt(txt_path: str | Path) -> Dict[int, str]:
    """Read class names from classes.txt."""
    class_names = {}
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name:
                    class_names[i] = name
    except Exception:
        pass
    return class_names
