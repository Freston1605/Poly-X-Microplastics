"""Machine provenance for the report: equipment table and checkpoint training info.

Both readers are pure I/O over the machine (OS, RAM, GPU) and over the model
checkpoint, so both keep their heavy imports (``psutil``, ``torch``) lazy
inside the body: the import-laziness gate
(``tests/test_importes_diferidos_informe.py``) requires that importing the
report package does not pay for them.

They are re-exported from ``polyx.gui.report._common`` on purpose: the
section builders read ``_team`` and ``_training_of`` through that module at
call time, which is what lets the byte-identity fixtures pin the live
equipment table to canonical rows (``tools/audit/capturar_fixtures_informe.py``
and ``tests/test_render_reporte_benigno.py`` patch exactly that attribute).
"""

from __future__ import annotations

import platform
from pathlib import Path

from polyx.core.i18n import tr


def _team() -> list[tuple]:
    """Machine components, so the report can state where it was run."""

    rows_2 = [
        (tr("Sistema operativo"), f"{platform.system()} {platform.release()}"),
        (tr("Procesador"), platform.processor() or "—"),
    ]
    try:
        import psutil

        rows_2.append((tr("Memoria RAM"), f"{psutil.virtual_memory().total / 1024**3:.1f} GB"))
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            rows_2 += [
                ("GPU", props.name),
                ("VRAM", f"{props.total_memory / 1024**3:.1f} GB"),
                ("Compute capability", f"{props.major}.{props.minor}"),
                (
                    "PyTorch",
                    f"{torch.__version__} ({'ROCm' if getattr(torch.version, 'hip', None) else 'CUDA'} {getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None)})",
                ),
            ]
        else:
            rows_2 += [("GPU", tr("no disponible")), ("PyTorch", torch.__version__)]
    except Exception:  # noqa: BLE001, S110
        pass
    return rows_2


def _training_of(weight: Path) -> dict | None:
    """Reads from the checkpoint with which configuration and result the model
    was trained.

    Ultralytics keeps ``train_args`` and ``train_metrics`` inside the .pt
    itself, so the report can declare imgsz, batch, epochs and metrics
    without requiring the run folder to still exist.
    """
    try:
        import torch

        ck = torch.load(str(weight), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    args = ck.get("train_args") or {}
    met = ck.get("train_metrics") or {}
    if not args and not met:
        return None
    return {
        "base": args.get("model", "—"),
        "imgsz": args.get("imgsz", "—"),
        "batch": args.get("batch", "—"),
        "epocas": args.get("epochs", "—"),
        "optimizador": args.get("optimizer", "—"),
        "lr0": args.get("lr0", "—"),
        "amp": args.get("amp", "—"),
        "precision": met.get("metrics/precision(B)"),
        "recall": met.get("metrics/recall(B)"),
        "map50": met.get("metrics/mAP50(B)"),
        "map": met.get("metrics/mAP50-95(B)"),
    }
