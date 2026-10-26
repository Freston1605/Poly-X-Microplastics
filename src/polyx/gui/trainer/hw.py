"""Hardware detection and imgsz / batch recommendations.

Heuristic to answer quickly "what is the highest imgsz I can train on
without OOM?". By default the user wants to train as large as possible.
Here we compute a sensible value from the available VRAM.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    available: bool
    name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    # ── Diagnostics ──
    index: int = 0
    n_gpus: int = 0
    capacity: str = ""          # compute capability, e.g. "8.9"
    torch_version: str = ""
    torch_cuda: str = ""         # CUDA version torch was built with
    driver: str = ""
    # The card exists but torch cannot use it (torch installed without CUDA).
    # This is the most common and most confusing failure: the machine has a GPU and the
    # trainer used to report `no GPU` without explaining why.
    gpu_sin_torch: bool = False
    # Integrated GPU of an Apple Silicon Mac, through Metal (MPS). It is not CUDA:
    # it lacks dedicated VRAM (it shares system RAM) and lacks compute capability,
    # so VRAM-based recommendations are interpreted differently.
    is_mps: bool = False
    detail: str = ""

    @property
    def device(self) -> str:
        """The value that must be passed to YOLO as ``device``."""
        if self.is_mps:
            return "mps"
        return str(self.index) if self.available else "cpu"


def _nvidia_smi() -> Optional[dict]:
    """Consults the GPU through nvidia-smi, outside of torch.

    It distinguishes "no card" from "there is a card but torch was installed
    without CUDA", which are different problems with different solutions.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        # With several cards, the one with the most VRAM is kept
        best = None
        for line in out.stdout.strip().splitlines():
            parts_2 = [p.strip() for p in line.split(",")]
            if len(parts_2) < 4:
                continue
            total = float(parts_2[1]) / 1024.0
            if best is None or total > best["total_gb"]:
                best = {"name": parts_2[0], "total_gb": total,
                         "free_gb": float(parts_2[2]) / 1024.0,
                         "driver": parts_2[3],
                         "n": len(out.stdout.strip().splitlines())}
        return best
    except Exception:
        return None


def detect_gpu() -> GPUInfo:
    """Identifies the card the training will run on.

    With several GPUs the one with the most VRAM is chosen, not 0: on a
    machine with integrated and dedicated graphics, 0 can be the bad one.
    """
    torch_v = torch_cuda = ""
    try:
        import torch
        torch_v = torch.__version__
        torch_cuda = getattr(torch.version, "cuda", None) or ""
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            # Choose the card with the most total VRAM
            idx, best_total = 0, -1.0
            for i in range(n):
                try:
                    total_i = torch.cuda.get_device_properties(i).total_memory
                except Exception:
                    continue
                if total_i > best_total:
                    idx, best_total = i, float(total_i)
            props = torch.cuda.get_device_properties(idx)
            free, total = torch.cuda.mem_get_info(idx)
            smi = _nvidia_smi()
            return GPUInfo(
                available=True,
                name=props.name,
                vram_total_gb=total / (1024**3),
                vram_free_gb=free / (1024**3),
                index=idx, n_gpus=n,
                capacity=f"{props.major}.{props.minor}",
                torch_version=torch_v, torch_cuda=torch_cuda,
                driver=(smi or {}).get("driver", ""),
            )
        # ── Apple Silicon Mac: the GPU is used through Metal (MPS) ──
        # It comes after CUDA because a machine does not have both, and before
        # nvidia-smi because on a Mac that binary does not exist and would only waste
        # a subprocess timeout.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            # Unified memory: the GPU shares system RAM, it lacks its own
            # VRAM to query. Total RAM is reported as the upper bound,
            # which is what actually limits training.
            total_gb = free_gb = 0.0
            try:
                import psutil
                vm = psutil.virtual_memory()
                total_gb = vm.total / (1024 ** 3)
                free_gb = vm.available / (1024 ** 3)
            except Exception:
                pass
            import platform as _pl
            return GPUInfo(
                available=True, is_mps=True,
                name=f"GPU integrada de Apple ({_pl.machine()})",
                vram_total_gb=total_gb, vram_free_gb=free_gb,
                n_gpus=1, torch_version=torch_v,
                detail=("Memoria unificada: la GPU comparte la RAM del sistema, "
                         "asi que la cifra de arriba es la RAM total y no una VRAM "
                         "dedicada. MPS no soporta todas las operaciones de CUDA; "
                         "si el entrenamiento falla, prueba con CPU."),
            )
    except Exception as e:
        torch_v = torch_v or "no importable"
        return GPUInfo(available=False, torch_version=torch_v,
                       detail=f"{type(e).__name__}: {e}")

    # torch does not see CUDA: is the card missing, or was torch built without CUDA?
    smi = _nvidia_smi()
    if smi:
        return GPUInfo(
            available=False, gpu_sin_torch=True,
            name=smi["name"], vram_total_gb=smi["total_gb"],
            vram_free_gb=smi["free_gb"], driver=smi["driver"],
            n_gpus=smi.get("n", 1),
            torch_version=torch_v, torch_cuda=torch_cuda,
            detail=(f"nvidia-smi ve la tarjeta (driver {smi['driver']}) pero "
                     f"torch {torch_v} está compilado "
                     f"{'sin CUDA' if not torch_cuda else f'para CUDA {torch_cuda}'}."),
        )
    # On a Mac talking about `NVIDIA GPU` means nothing: there the path is MPS and
    # it only exists with Apple Silicon. Saying so keeps someone from going
    # looking for NVIDIA drivers on a machine that will never have them.
    import sys as _sys
    if _sys.platform == "darwin":
        import platform as _pl
        intel = _pl.machine().lower() not in ("arm64", "aarch64")
        return GPUInfo(
            available=False, torch_version=torch_v,
            detail=("Este Mac tiene procesador Intel: no dispone de MPS, que "
                     "requiere Apple Silicon (M1 o posterior). Se entrena y "
                     "detecta por CPU." if intel else
                     "Apple Silicon detectado pero torch no expone MPS. "
                     "Reinstala PyTorch desde el instalador de Poly-X."))
    return GPUInfo(available=False, torch_version=torch_v, torch_cuda=torch_cuda,
                   detail="Ni torch ni nvidia-smi encuentran una GPU NVIDIA.")


# Approximate VRAM footprint per architecture.
#   static_gb      = weights + optimizer + gradients (does NOT scale with batch)
#   act_gb_640_b1  = activations per image at imgsz=640 with AMP
#
# Calibrated against real measurements (yolov8m 640/16 AMP ≈ 6 GB,
# 1280/8 AMP ≈ 10 GB) on RTX 30xx/40xx. Activations scale with
# batch * (imgsz/640)^2.
_FOOTPRINT = {
    # size: (static_gb, act_gb_640_b1)
    "n": (0.30, 0.10),
    "s": (0.50, 0.18),
    "m": (1.00, 0.30),
    "l": (1.70, 0.50),
    "x": (2.80, 0.80),
}


def estimate_vram_gb(size: str, imgsz: int, batch: int, amp: bool = True) -> float:
    """Approximation of the VRAM (GB) needed to train.

    Formula: static_gb + act_gb_640_b1 * batch * (imgsz/640)^2 + overhead.
    The static term (weights+optim+grad) is NOT multiplied by batch — that
    was the bug that gave absurd estimates (hundreds of GB).
    """
    static_gb, act_gb = _FOOTPRINT.get(size, _FOOTPRINT["m"])
    if not amp:
        # FP32 ~ doubles activations and increases optimizer
        act_gb *= 1.8
        static_gb *= 1.5
    activations = act_gb * max(1, batch) * (imgsz / 640.0) ** 2
    overhead = 0.5   # CUDA/cuDNN/allocator
    return static_gb + activations + overhead


def recommend_max_imgsz(size: str, batch: int, vram_free_gb: float,
                        amp: bool = True,
                        safety: float = 0.85) -> int:
    """Returns the highest imgsz in steps of 32 that fits in VRAM.

    Args:
        size: architecture ("n","s","m","l","x")
        batch: batch size
        vram_free_gb: available VRAM (GB)
        safety: safety factor (0.85 = use 85% of the VRAM)
    """
    cap = vram_free_gb * safety
    candidates = [3840, 2560, 1920, 1600, 1280, 1024, 960, 800, 640, 512, 416]
    for sz in candidates:
        if estimate_vram_gb(size, sz, batch, amp) <= cap:
            return sz
    return 320


def recommend_batch(size: str, imgsz: int, vram_free_gb: float,
                    amp: bool = True, safety: float = 0.85) -> int:
    """Suggests the maximum batch for a given imgsz."""
    cap = vram_free_gb * safety
    for b in (64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1):
        if estimate_vram_gb(size, imgsz, b, amp) <= cap:
            return b
    return 1


IMGSZ_LADDER = [4096, 3840, 3200, 2560, 2080, 1920, 1664, 1600, 1280,
                  1024, 960, 800, 640, 512, 416, 320]


@dataclass
class Recommendation:
    """Suggested configuration and what limited each part."""
    imgsz: int
    batch: int
    amp: bool
    workers: int
    cache: str                  # "ram" | "disk" | ""
    vram_est_gb: float
    vram_cap_gb: float
    limited_by: str           # "vram" | "resolucion_nativa" | "tope_batch"
    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def recommend_config(size: str, vram_free_gb: float,
                      imgsz_native: int = 0,
                      batch_min: int = 2, batch_max: int = 32,
                      n_cpus: int = 0, dataset_gb: float = 0.0,
                      ram_free_gb: float = 0.0,
                      safety: float = 0.85) -> Recommendation:
    """Training config with priority **imgsz → batch → speed**.

    The order is not negotiable and follows the physics of the problem: the
    particles of this study measure ~12-19 px, and what makes them disappear
    is resolution, not batch. Therefore:

    1. **imgsz at the maximum** that fits in VRAM with the minimum usable batch.
    2. **batch** is raised afterwards, with imgsz already fixed, to use up
       what is left over.
    3. **speed** (AMP, workers, cache) is tuned last, and only with what does
       not cost resolution or batch.

    ``imgsz_nativo`` is the larger side of the dataset images: going above it
    adds no information, it only interpolates and burns VRAM. With 0 the cap
    is not applied.

    AMP stays always on: it is not just speed, it is what makes the high
    imgsz possible, so turning it off would violate priority 1.
    """
    amp = True
    cap = max(0.5, vram_free_gb * safety)
    notes: list[str] = []

    max_size = 0
    if imgsz_native and imgsz_native > 0:
        # multiple of 32 (network stride) above the native side
        max_size = int(((int(imgsz_native) + 31) // 32) * 32)

    # ── 1. imgsz rules ──
    ladder = [s for s in IMGSZ_LADDER if not max_size or s <= max_size]
    if not ladder:
        ladder = [min(IMGSZ_LADDER)]
    imgsz = 0
    for sz in ladder:
        if estimate_vram_gb(size, sz, batch_min, amp) <= cap:
            imgsz = sz
            break

    limited = "vram"
    if imgsz == 0:
        # Even the smallest imgsz does not fit with the minimum batch: the batch is lowered,
        # which is priority 2, before lowering the resolution further.
        request = batch_min
        for b in (2, 1):
            if b >= request:
                continue
            for sz in ladder:
                if estimate_vram_gb(size, sz, b, amp) <= cap:
                    imgsz, batch_min = sz, b
                    notes.append(
                        f"No cabía batch {request} ni al mínimo de resolución: se bajó "
                        f"el batch a {b} para no sacrificar imgsz.")
                    break
            if imgsz:
                break
    if imgsz == 0:
        imgsz, batch_min = min(ladder), 1
        notes.append("La VRAM no alcanza ni para el caso mínimo; el entrenamiento "
                     "probablemente dé OOM. Considera un modelo más pequeño.")
    elif max_size and imgsz == max(ladder):
        # The dataset ceiling was reached, not the card's: the leftover VRAM
        # must go to batch, not to interpolating invented pixels.
        limited = "resolucion_nativa"
        notes.append(
            f"imgsz topado en {imgsz} px porque las imágenes miden {imgsz_native} px "
            f"de lado: más allá solo se interpola y se gasta VRAM sin añadir señal.")

    # ── 2. batch afterwards, with imgsz already fixed ──
    batch = batch_min
    for b in (batch_max, 24, 16, 12, 8, 6, 4, 2, 1):
        if b < batch_min:
            break
        if estimate_vram_gb(size, imgsz, b, amp) <= cap:
            batch = b
            break
    if batch >= batch_max:
        if limited == "vram":
            limited = "tope_batch"
        notes.append(
            f"batch topado en {batch_max}: por encima, las mejoras son marginales y "
            f"el batch grande promedia gradientes de objetos pequeños. "
            f"Ultralytics acumula a batch nominal 64, así que un batch chico "
            f"penaliza menos de lo que parece.")

    # ── 3. speed, at the end and without touching the previous steps ──
    import os
    cpus = int(n_cpus) if n_cpus else (os.cpu_count() or 8)
    # 8 workers is the useful ceiling: more loading threads do not speed things up if the GPU is already
    # saturated, and each one reserves page memory.
    workers = max(2, min(8, cpus - 2))
    cache = ""
    if dataset_gb > 0 and ram_free_gb > 0:
        if dataset_gb * 1.3 <= ram_free_gb * 0.6:
            cache = "ram"
        else:
            cache = "disk"
            notes.append(
                f"El dataset (~{dataset_gb:.1f} GB) no cabe holgado en RAM: cache en "
                f"disco en vez de RAM, para no arriesgar el swap a mitad del entreno.")

    return Recommendation(
        imgsz=imgsz, batch=batch, amp=amp, workers=workers, cache=cache,
        vram_est_gb=estimate_vram_gb(size, imgsz, batch, amp),
        vram_cap_gb=cap, limited_by=limited, notes=notes,
    )


def humanize_gb(x: float) -> str:
    if x >= 10:
        return f"{x:.0f} GB"
    return f"{x:.1f} GB"
