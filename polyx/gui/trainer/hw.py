"""Detección de hardware y recomendaciones de imgsz / batch.

Heurística para responder rápidamente "¿cuál es el imgsz más alto al que
puedo entrenar sin OOM?". El usuario quiere por defecto entrenar al máximo
posible. Aquí calculamos un valor sensato a partir de la VRAM disponible.
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
    # ── Diagnóstico ──
    index: int = 0
    n_gpus: int = 0
    capacidad: str = ""          # compute capability, ej. "8.9"
    torch_version: str = ""
    torch_cuda: str = ""         # versión CUDA con que se compiló torch
    driver: str = ""
    # La tarjeta existe pero torch no la puede usar (torch instalado sin CUDA).
    # Es el fallo más común y el más confuso: la máquina tiene GPU y el
    # entrenador decía "no hay GPU" sin explicar por qué.
    gpu_sin_torch: bool = False
    # GPU integrada de un Mac con Apple Silicon, vía Metal (MPS). No es CUDA:
    # no tiene VRAM propia (comparte la RAM del sistema) ni compute capability,
    # así que las recomendaciones basadas en VRAM se interpretan distinto.
    es_mps: bool = False
    detalle: str = ""

    @property
    def dispositivo(self) -> str:
        """El valor que hay que pasarle a YOLO como ``device``."""
        if self.es_mps:
            return "mps"
        return str(self.index) if self.available else "cpu"


def _nvidia_smi() -> Optional[dict]:
    """Consulta la tarjeta por nvidia-smi, al margen de torch.

    Sirve para distinguir "no hay tarjeta" de "hay tarjeta pero torch se instaló
    sin CUDA", que son problemas distintos con soluciones distintas.
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
        # Con varias tarjetas se queda la de más VRAM
        mejor = None
        for linea in out.stdout.strip().splitlines():
            partes = [p.strip() for p in linea.split(",")]
            if len(partes) < 4:
                continue
            total = float(partes[1]) / 1024.0
            if mejor is None or total > mejor["total_gb"]:
                mejor = {"name": partes[0], "total_gb": total,
                         "free_gb": float(partes[2]) / 1024.0,
                         "driver": partes[3],
                         "n": len(out.stdout.strip().splitlines())}
        return mejor
    except Exception:
        return None


def detect_gpu() -> GPUInfo:
    """Identifica la tarjeta con la que se va a entrenar.

    Con varias GPU se elige la de más VRAM, no la 0: en un equipo con gráfica
    integrada y dedicada, la 0 puede ser la mala.
    """
    torch_v = torch_cuda = ""
    try:
        import torch
        torch_v = torch.__version__
        torch_cuda = getattr(torch.version, "cuda", None) or ""
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            # Elegir la tarjeta con más VRAM total
            idx, mejor_total = 0, -1.0
            for i in range(n):
                try:
                    total_i = torch.cuda.get_device_properties(i).total_memory
                except Exception:
                    continue
                if total_i > mejor_total:
                    idx, mejor_total = i, float(total_i)
            props = torch.cuda.get_device_properties(idx)
            free, total = torch.cuda.mem_get_info(idx)
            smi = _nvidia_smi()
            return GPUInfo(
                available=True,
                name=props.name,
                vram_total_gb=total / (1024**3),
                vram_free_gb=free / (1024**3),
                index=idx, n_gpus=n,
                capacidad=f"{props.major}.{props.minor}",
                torch_version=torch_v, torch_cuda=torch_cuda,
                driver=(smi or {}).get("driver", ""),
            )
        # ── Mac con Apple Silicon: la GPU se usa por Metal (MPS) ──
        # Va despues de CUDA porque una maquina no tiene las dos, y antes de
        # nvidia-smi porque en un Mac ese binario no existe y solo gastaria
        # un timeout de subprocess.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            # Memoria unificada: la GPU comparte la RAM del sistema, no tiene
            # una VRAM propia que consultar. Se reporta la RAM total como cota
            # superior, que es lo que de verdad limita el entrenamiento.
            total_gb = libre_gb = 0.0
            try:
                import psutil
                vm = psutil.virtual_memory()
                total_gb = vm.total / (1024 ** 3)
                libre_gb = vm.available / (1024 ** 3)
            except Exception:
                pass
            import platform as _pl
            return GPUInfo(
                available=True, es_mps=True,
                name=f"GPU integrada de Apple ({_pl.machine()})",
                vram_total_gb=total_gb, vram_free_gb=libre_gb,
                n_gpus=1, torch_version=torch_v,
                detalle=("Memoria unificada: la GPU comparte la RAM del sistema, "
                         "asi que la cifra de arriba es la RAM total y no una VRAM "
                         "dedicada. MPS no soporta todas las operaciones de CUDA; "
                         "si el entrenamiento falla, prueba con CPU."),
            )
    except Exception as e:
        torch_v = torch_v or "no importable"
        return GPUInfo(available=False, torch_version=torch_v,
                       detalle=f"{type(e).__name__}: {e}")

    # torch no ve CUDA: ¿es que no hay tarjeta, o que torch viene sin CUDA?
    smi = _nvidia_smi()
    if smi:
        return GPUInfo(
            available=False, gpu_sin_torch=True,
            name=smi["name"], vram_total_gb=smi["total_gb"],
            vram_free_gb=smi["free_gb"], driver=smi["driver"],
            n_gpus=smi.get("n", 1),
            torch_version=torch_v, torch_cuda=torch_cuda,
            detalle=(f"nvidia-smi ve la tarjeta (driver {smi['driver']}) pero "
                     f"torch {torch_v} está compilado "
                     f"{'sin CUDA' if not torch_cuda else f'para CUDA {torch_cuda}'}."),
        )
    # En un Mac hablar de "GPU NVIDIA" no significa nada: ahi la via es MPS y
    # solo existe con Apple Silicon. Decirlo asi evita que alguien se ponga a
    # buscar drivers de NVIDIA en un equipo que nunca los va a tener.
    import sys as _sys
    if _sys.platform == "darwin":
        import platform as _pl
        intel = _pl.machine().lower() not in ("arm64", "aarch64")
        return GPUInfo(
            available=False, torch_version=torch_v,
            detalle=("Este Mac tiene procesador Intel: no dispone de MPS, que "
                     "requiere Apple Silicon (M1 o posterior). Se entrena y "
                     "detecta por CPU." if intel else
                     "Apple Silicon detectado pero torch no expone MPS. "
                     "Reinstala PyTorch desde el instalador de Poly-X."))
    return GPUInfo(available=False, torch_version=torch_v, torch_cuda=torch_cuda,
                   detalle="Ni torch ni nvidia-smi encuentran una GPU NVIDIA.")


# Footprint aproximado de VRAM por arquitectura.
#   static_gb      = pesos + optimizador + gradientes (NO escala con batch)
#   act_gb_640_b1  = activaciones por imagen a imgsz=640 con AMP
#
# Calibrado contra mediciones reales (yolov8m 640/16 AMP ≈ 6 GB,
# 1280/8 AMP ≈ 10 GB) en RTX 30xx/40xx. Activaciones escalan con
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
    """Aproximación a la VRAM (GB) necesaria para entrenar.

    Fórmula: static_gb + act_gb_640_b1 * batch * (imgsz/640)^2 + overhead.
    El término estático (pesos+optim+grad) NO se multiplica por batch — ese
    era el bug que daba estimaciones absurdas (cientos de GB).
    """
    static_gb, act_gb = _FOOTPRINT.get(size, _FOOTPRINT["m"])
    if not amp:
        # FP32 ~ duplica activaciones y aumenta optimizador
        act_gb *= 1.8
        static_gb *= 1.5
    activations = act_gb * max(1, batch) * (imgsz / 640.0) ** 2
    overhead = 0.5   # CUDA/cuDNN/allocator
    return static_gb + activations + overhead


def recommend_max_imgsz(size: str, batch: int, vram_free_gb: float,
                        amp: bool = True,
                        safety: float = 0.85) -> int:
    """Devuelve el imgsz más alto en pasos de 32 que cabe en VRAM.

    Args:
        size: arquitectura ("n","s","m","l","x")
        batch: tamaño de batch
        vram_free_gb: VRAM disponible (GB)
        safety: factor de seguridad (0.85 = usar 85% del VRAM)
    """
    cap = vram_free_gb * safety
    candidates = [3840, 2560, 1920, 1600, 1280, 1024, 960, 800, 640, 512, 416]
    for sz in candidates:
        if estimate_vram_gb(size, sz, batch, amp) <= cap:
            return sz
    return 320


def recommend_batch(size: str, imgsz: int, vram_free_gb: float,
                    amp: bool = True, safety: float = 0.85) -> int:
    """Sugiere batch máximo para un imgsz dado."""
    cap = vram_free_gb * safety
    for b in (64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1):
        if estimate_vram_gb(size, imgsz, b, amp) <= cap:
            return b
    return 1


ESCALERA_IMGSZ = [4096, 3840, 3200, 2560, 2080, 1920, 1664, 1600, 1280,
                  1024, 960, 800, 640, 512, 416, 320]


@dataclass
class Recomendacion:
    """Configuración sugerida y qué fue lo que limitó cada cosa."""
    imgsz: int
    batch: int
    amp: bool
    workers: int
    cache: str                  # "ram" | "disk" | ""
    vram_est_gb: float
    vram_cap_gb: float
    limitado_por: str           # "vram" | "resolucion_nativa" | "tope_batch"
    notas: list = None

    def __post_init__(self):
        if self.notas is None:
            self.notas = []


def recomendar_config(size: str, vram_free_gb: float,
                      imgsz_nativo: int = 0,
                      batch_min: int = 2, batch_max: int = 32,
                      n_cpus: int = 0, dataset_gb: float = 0.0,
                      ram_libre_gb: float = 0.0,
                      safety: float = 0.85) -> Recomendacion:
    """Config de entrenamiento con prioridad **imgsz → batch → velocidad**.

    El orden no es negociable y responde a la física del problema: las
    partículas de este estudio ocupan ~12-19 px, y lo que las hace desaparecer
    es la resolución, no el batch. Por eso:

    1. **imgsz al máximo** que quepa en VRAM con el batch mínimo utilizable.
    2. **batch** se sube después, con imgsz ya fijo, hasta llenar lo que sobra.
    3. **velocidad** (AMP, workers, cache) se ajusta al final, y solo con lo
       que no cueste resolución ni batch.

    ``imgsz_nativo`` es el lado mayor de las imágenes del dataset: subir por
    encima de eso no añade información, solo interpola y gasta VRAM. Con 0 no se
    aplica el tope.

    AMP queda siempre encendido: no es solo velocidad, es lo que permite el
    imgsz alto, así que apagarlo violaría la prioridad 1.
    """
    amp = True
    cap = max(0.5, vram_free_gb * safety)
    notas: list[str] = []

    tope = 0
    if imgsz_nativo and imgsz_nativo > 0:
        # múltiplo de 32 (stride de la red) por encima del lado nativo
        tope = int(((int(imgsz_nativo) + 31) // 32) * 32)

    # ── 1. imgsz manda ──
    escalera = [s for s in ESCALERA_IMGSZ if not tope or s <= tope]
    if not escalera:
        escalera = [min(ESCALERA_IMGSZ)]
    imgsz = 0
    for sz in escalera:
        if estimate_vram_gb(size, sz, batch_min, amp) <= cap:
            imgsz = sz
            break

    limitado = "vram"
    if imgsz == 0:
        # Ni el imgsz más chico cabe con el batch mínimo: se cede batch, que es
        # la prioridad 2, antes que bajar más la resolución.
        pedido = batch_min
        for b in (2, 1):
            if b >= pedido:
                continue
            for sz in escalera:
                if estimate_vram_gb(size, sz, b, amp) <= cap:
                    imgsz, batch_min = sz, b
                    notas.append(
                        f"No cabía batch {pedido} ni al mínimo de resolución: se bajó "
                        f"el batch a {b} para no sacrificar imgsz.")
                    break
            if imgsz:
                break
    if imgsz == 0:
        imgsz, batch_min = min(escalera), 1
        notas.append("La VRAM no alcanza ni para el caso mínimo; el entrenamiento "
                     "probablemente dé OOM. Considera un modelo más pequeño.")
    elif tope and imgsz == max(escalera):
        # Se llegó al techo que impone el dataset, no al de la tarjeta: la VRAM
        # que sobra debe irse a batch, no a interpolar píxeles inventados.
        limitado = "resolucion_nativa"
        notas.append(
            f"imgsz topado en {imgsz} px porque las imágenes miden {imgsz_nativo} px "
            f"de lado: más allá solo se interpola y se gasta VRAM sin añadir señal.")

    # ── 2. batch después, con imgsz ya fijo ──
    batch = batch_min
    for b in (batch_max, 24, 16, 12, 8, 6, 4, 2, 1):
        if b < batch_min:
            break
        if estimate_vram_gb(size, imgsz, b, amp) <= cap:
            batch = b
            break
    if batch >= batch_max:
        if limitado == "vram":
            limitado = "tope_batch"
        notas.append(
            f"batch topado en {batch_max}: por encima, las mejoras son marginales y "
            f"el batch grande promedia gradientes de objetos pequeños. "
            f"Ultralytics acumula a batch nominal 64, así que un batch chico "
            f"penaliza menos de lo que parece.")

    # ── 3. velocidad, al final y sin tocar lo anterior ──
    import os
    cpus = int(n_cpus) if n_cpus else (os.cpu_count() or 8)
    # 8 workers es el techo útil: más hilos de carga no aceleran si la GPU ya va
    # saturada, y cada uno reserva memoria de página.
    workers = max(2, min(8, cpus - 2))
    cache = ""
    if dataset_gb > 0 and ram_libre_gb > 0:
        if dataset_gb * 1.3 <= ram_libre_gb * 0.6:
            cache = "ram"
        else:
            cache = "disk"
            notas.append(
                f"El dataset (~{dataset_gb:.1f} GB) no cabe holgado en RAM: cache en "
                f"disco en vez de RAM, para no arriesgar el swap a mitad del entreno.")

    return Recomendacion(
        imgsz=imgsz, batch=batch, amp=amp, workers=workers, cache=cache,
        vram_est_gb=estimate_vram_gb(size, imgsz, batch, amp),
        vram_cap_gb=cap, limitado_por=limitado, notas=notas,
    )


def humanize_gb(x: float) -> str:
    if x >= 10:
        return f"{x:.0f} GB"
    return f"{x:.1f} GB"
