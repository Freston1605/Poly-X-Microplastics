"""Wrapper delgado sobre Ultralytics YOLO para inferencia + utilidades YOLO-txt."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import os

import logging

import numpy as np
import cv2

from .observabilidad import logger_para, evento

@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    # caja en coordenadas de píxeles absolutas (x1,y1,x2,y2)
    x1: float
    y1: float
    x2: float
    y2: float
    # tamaño equivalente (μm) si se calibró
    diam_um: Optional[float] = None
    area_um2: Optional[float] = None
    # ── Forma real, medida sobre la máscara de la partícula ──
    # La caja no describe la forma: en este material sobreestima el área 1.87x y
    # el largo un 14%, y una fibra en diagonal tiene caja cuadrada. Estos campos
    # los llena morfologia.medir_deteccion(); quedan en None si no se midió.
    largo_um: Optional[float] = None      # dimensión mayor, siguiendo la curva
    ancho_um: Optional[float] = None
    aspecto: Optional[float] = None       # largo/ancho reales
    curvatura: Optional[float] = None     # 1.0 = recta; >1.15 = curva
    morfotipo: Optional[str] = None       # "fibra" | "fragmento"
    # Las dos cotas del largo por separado, y el rectángulo equivalente. Se
    # guardan porque comparadas informan de lo irregular que es la partícula.
    feret_um: Optional[float] = None      # cuerda: mayor distancia del borde
    geodesico_um: Optional[float] = None  # camino más largo por dentro
    largo_rect_eq_um: Optional[float] = None
    metodo_largo: Optional[str] = None
    # La medida salio, pero pide un vistazo: tipicamente varias particulas en
    # contacto medidas como una, lo que INFLA la talla. Viaja en la Detection
    # -- y no solo en la Morfologia que la produjo -- porque si no, el informe
    # no tiene forma de avisarlo sin volver a medir todo el lote.
    revisar: bool = False
    aviso_forma: Optional[str] = None
    # Numero de la partícula dentro de SU imagen, empezando en 1. Es lo que
    # permite ir de una fila de la tabla a la partícula concreta en la foto: sin
    # él, «la mayor mide 5878 µm» no se puede ir a comprobar.
    numero: Optional[int] = None
    # ¿Se sostiene el polímero de esta partícula, o solo su detección?
    # Lo decide core.asignacion tras la inferencia, comparando conf contra el
    # umbral de asignación. En False la partícula sigue contando, pero se
    # reporta como «no asignable» en vez de atribuirle un polímero.
    # De fábrica True: sin umbral configurado, todo se asigna como siempre.
    asignable: bool = True

    @property
    def w(self) -> float: return self.x2 - self.x1
    @property
    def h(self) -> float: return self.y2 - self.y1
    @property
    def cx(self) -> float: return 0.5 * (self.x1 + self.x2)
    @property
    def cy(self) -> float: return 0.5 * (self.y1 + self.y2)


def _is_oom(e: Exception) -> bool:
    """¿La excepción es por falta de memoria GPU?"""
    try:
        import torch
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(e).lower()


def _cuda_empty_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _log():
    """Logger del modulo, resuelto tarde para no exigir escritura al importar."""
    return logger_para("detector")


class YoloModel:
    """Carga perezosa de Ultralytics YOLO. Permite múltiples modelos en paralelo."""

    # Escalera descendente para el auto-fallback cuando la GPU se queda sin
    # memoria: en vez de fallar, se reintenta con el siguiente tamaño menor.
    FALLBACK_SIZES = (8192, 7168, 6144, 5120, 4096, 3008, 2048, 1536, 1280, 1024, 640)

    # Tope de detecciones por pase. Ultralytics trae 300, que aqui trunca en
    # silencio: una placa de desembocadura pasa de 500 particulas, y a confianza
    # baja el modelo propone bastante mas. Al recortar el sobrante se perderian
    # justo las cajas de menor confianza sin aviso, de modo que un barrido de
    # umbrales dejaria de ser comparable con una corrida hecha a ese umbral.
    MAX_DET = 5000

    def __init__(self, weights_path: str | Path, alias: str = ""):
        self.weights_path = str(weights_path)
        self.alias = alias or Path(self.weights_path).stem
        self._model = None
        self.names: Dict[int, str] = {}
        # imgsz realmente usado en la última inferencia (puede ser menor al
        # pedido si actuó el auto-fallback por falta de memoria GPU)
        self.last_imgsz: int = 0
        self.last_fallback: bool = False

    def load(self):
        if self._model is not None:
            return self
        from ultralytics import YOLO
        self._model = YOLO(self.weights_path)
        nm = getattr(self._model, "names", None) or {}
        self.names = dict(nm)
        return self

    def probe_max_imgsz(self, image_path: str | Path, device: str = "0",
                        candidates: Tuple[int, ...] = (2048, 3008, 4096, 5120, 6144, 7168, 8192),
                        progress=None) -> Tuple[int, Dict[int, str]]:
        """Encuentra el imgsz más grande que la GPU aguanta para ESTE modelo.

        Prueba inferencia real a tamaños ascendentes hasta que falle por memoria.
        Devuelve (max_ok, detalle) donde detalle es {imgsz: "ok"/"oom"/"err"}.
        En CPU no hay tope de VRAM → devuelve el mayor candidato sin probar.
        """
        self.load()
        detail: Dict[int, str] = {}
        dev = str(device).strip().lower()
        if dev in ("cpu", "-1"):
            return candidates[-1], {c: "cpu" for c in candidates}
        try:
            import torch
        except Exception:
            return 1920, {}
        max_ok = 0
        for sz in candidates:
            if progress is not None:
                progress(sz)
            try:
                self._model.predict(source=str(image_path), imgsz=sz, conf=0.25,
                                    device=device, verbose=False, save=False)
                detail[sz] = "ok"
                max_ok = sz
            except torch.cuda.OutOfMemoryError:
                detail[sz] = "oom"
                break
            except Exception as e:
                detail[sz] = f"err: {type(e).__name__}"
                break
            finally:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return (max_ok or 1920), detail

    def predict(self, image_path: str | Path, conf: float = 0.25, iou: float = 0.45,
                imgsz: int = 640, device: str = "0",
                auto_fallback: bool = True) -> List[Detection]:
        self.load()
        # Auto-fallback: si el imgsz pedido agota la VRAM, reintenta con cada
        # tamaño menor de la escalera en vez de abortar el análisis completo.
        sizes = [int(imgsz)]
        if auto_fallback:
            sizes += [s for s in self.FALLBACK_SIZES if s < int(imgsz)]
        res = None
        for i, sz in enumerate(sizes):
            try:
                res = self._model.predict(
                    source=str(image_path), conf=conf, iou=iou, imgsz=sz,
                    device=device, verbose=False, save=False,
                    max_det=self.MAX_DET,
                )
                self.last_imgsz = sz
                self.last_fallback = (i > 0)
                break
            except Exception as e:
                if not auto_fallback or i == len(sizes) - 1 or not _is_oom(e):
                    raise
                evento(_log(), "imgsz.fallback", level=logging.WARNING,
                       alias=self.alias, archivo=str(image_path),
                       intento_previo=sz,
                       imgsz_siguiente=(sizes[i + 1]
                                        if i + 1 < len(sizes) else None),
                       razon="gpu_oom")
                _cuda_empty_cache()
        out: List[Detection] = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
            out.append(Detection(
                class_id=int(cls),
                class_name=self.names.get(int(cls), str(int(cls))),
                conf=float(c),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            ))
        return out

    def predict_sliced(self, image_path: str | Path, conf: float = 0.10,
                       iou: float = 0.45, tile: int = 1024, overlap: float = 0.25,
                       device: str = "0", agnostic_nms: bool = True,
                       imgsz: Optional[int] = None, batch: int = 8) -> List[Detection]:
        """Inferencia por tiles (técnica SAHI/Slicing Aided Hyper Inference).

        Divide la imagen en recortes solapados de ``tile``×``tile`` px, infiere
        cada uno a resolución nativa del tile y reproyecta las cajas a la imagen
        completa. Pensado para partículas diminutas en fotos de alta resolución,
        donde un único pase a imgsz bajo las haría desaparecer.

        Las cajas de todos los tiles se fusionan con NMS global (``agnostic_nms``
        suprime duplicados aunque difieran en clase, útil cuando interesa la
        detección por encima de la clase).
        """
        self.load()
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return []
        H, W = img.shape[:2]
        overlap = min(max(overlap, 0.0), 0.9)
        step = max(1, int(round(tile * (1.0 - overlap))))
        xs = _tile_starts(W, tile, step)
        ys = _tile_starts(H, tile, step)

        crops: List[np.ndarray] = []
        origins: List[Tuple[int, int]] = []
        for y0 in ys:
            for x0 in xs:
                crops.append(img[y0:y0 + tile, x0:x0 + tile])
                origins.append((x0, y0))
        if not crops:
            return []

        # imgsz controla la resolución de entrada del tile. Si es mayor que el
        # tile, el recorte se upscalea → partículas diminutas se agrandan hacia
        # la escala con que se entrenó el modelo (clave para objetos pequeños).
        infer_sz = int(imgsz) if imgsz else tile

        dets: List[Detection] = []
        bs = max(1, int(batch))
        # Procesar en sub-lotes para no agotar la VRAM (muchos tiles a imgsz alto).
        for start in range(0, len(crops), bs):
            chunk = crops[start:start + bs]
            chunk_org = origins[start:start + bs]
            results = self._model.predict(
                source=chunk, conf=conf, iou=iou, imgsz=infer_sz,
                device=device, verbose=False, save=False,
                max_det=self.MAX_DET,
            )
            for (x0, y0), r in zip(chunk_org, results):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                for (a, b, c, d), cc, k in zip(xyxy, cf, cl):
                    dets.append(Detection(
                        class_id=int(k),
                        class_name=self.names.get(int(k), str(int(k))),
                        conf=float(cc),
                        x1=float(a) + x0, y1=float(b) + y0,
                        x2=float(c) + x0, y2=float(d) + y0,
                    ))
        return _nms(dets, iou_thr=iou, agnostic=agnostic_nms)

    def predict_auto(self, image_path: str | Path, conf: float = 0.25,
                     iou: float = 0.45, imgsz: int = 1280, device: str = "0",
                     troceo: str = "auto", umbral_px: int = 2000, tile: int = 0,
                     overlap: float = 0.25, batch: int = 8,
                     agnostic_nms: bool = True,
                     registro: Optional[Dict] = None) -> List[Detection]:
        """Infiere troceando o no segun el tamano de la foto, sin intervencion.

        ``troceo``: ``"auto"`` decide por el lado mayor contra ``umbral_px``;
        ``"siempre"`` fuerza el troceado; ``"nunca"`` fuerza el pase unico.

        Si se pasa ``registro``, se rellena con la decision tomada para poder
        declararla en el informe: contar 500 particulas con troceado o sin el no
        es el mismo metodo, y el numero no significa lo mismo.
        """
        wh = tamano_imagen(image_path)
        plan = None
        if troceo == "nunca" or wh is None:
            plan = None
        elif troceo == "siempre":
            plan = politica_troceado(wh[0], wh[1], imgsz,
                                     umbral_px=0, tile=tile, overlap=overlap)
        else:
            plan = politica_troceado(wh[0], wh[1], imgsz,
                                     umbral_px=umbral_px, tile=tile, overlap=overlap)

        if registro is not None:
            registro.clear()
            registro.update({"ancho": wh[0] if wh else 0, "alto": wh[1] if wh else 0,
                             "troceado": plan is not None, "plan": plan})

        if plan is None:
            return self.predict(image_path, conf=conf, iou=iou, imgsz=imgsz,
                                device=device)
        return self.predict_sliced(image_path, conf=conf, iou=iou,
                                   tile=plan["tile"], overlap=plan["overlap"],
                                   device=device, agnostic_nms=agnostic_nms,
                                   imgsz=imgsz, batch=batch)


def tamano_imagen(image_path: str | Path) -> Optional[Tuple[int, int]]:
    """(ancho, alto) leyendo solo la cabecera, sin decodificar la imagen entera.

    Se consulta antes de cada inferencia, asi que decodificar una foto de 3260 px
    solo para medirla costaria mas que la propia decision.

    Devuelve el tamano YA ROTADO segun EXIF, porque ese es el marco en el que
    trabaja el resto del programa: ``cv2.imdecode`` aplica la orientacion y
    Pillow no. Sin corregirlo, una foto marcada con orientation 6 se medi­a
    4096x3072 y se decodificaba 3072x4096, de modo que las cajas normalizadas
    contra un tamano se desnormalizaban contra el otro y acababan desplazadas.
    """
    try:
        from PIL import Image
        with Image.open(str(image_path)) as im:
            ancho, alto = int(im.size[0]), int(im.size[1])
            # 5,6,7,8 son las orientaciones que intercambian los ejes.
            if im.getexif().get(274) in (5, 6, 7, 8):
                ancho, alto = alto, ancho
            return ancho, alto
    except Exception:
        # Rutas con acentos o formatos que Pillow no abre: cae al camino de cv2,
        # que ya se usa en todo el resto del proyecto por esa misma razon.
        try:
            img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                return None
            return int(img.shape[1]), int(img.shape[0])
        except Exception:
            return None


def politica_troceado(ancho: int, alto: int, imgsz: int, umbral_px: int = 2000,
                      tile: int = 0, overlap: float = 0.25) -> Optional[Dict]:
    """Decide si la foto se trocea y con que geometria, mirando solo su tamano.

    ``None`` significa un unico pase sobre la imagen completa.

    El umbral no se compara contra el area sino contra el **lado mayor**, porque
    lo que hace desaparecer una particula es el reescalado a ``imgsz``: una foto
    de 3260 px entrando a 2080 encoge cada particula a 0.64x, y por debajo del
    stride de la red deja de existir. Trocear evita ese reescalado.

    Con ``tile=0`` el lado del tile sale de ``min(umbral_px, imgsz)``: asi el
    recorte entra a la red sin reducirse, que es justamente el objetivo.

    CUANDO trocear y DE QUE TAMAÑO son dos preguntas distintas, y mezclarlas
    costo un fallo: para forzar el troceo, ``predict_auto`` pasa
    ``umbral_px=0`` -- su manera de decir "no decidas, trocea" --, y de ahi
    salia ``min(0, imgsz) = 0``, que el clamp subia a 256. Sobre una foto de
    4096x3072 eso son 336 teselas de 256 px: lentisimo y con CERO detecciones,
    porque cada tesela se reescala 8x hasta imgsz y el modelo no ve nada
    parecido a lo que entreno. Con umbral_px en 0 el tamaño lo manda imgsz, que
    es lo unico que de verdad lo determina.
    """
    lado = max(int(ancho), int(alto))
    if lado <= int(umbral_px):
        return None
    if int(tile) > 0:
        t = int(tile)
    else:
        candidatos = [v for v in (int(umbral_px), int(imgsz)) if v > 0]
        t = min(candidatos) if candidatos else 1280
    t = max(256, min(t, lado))
    overlap = min(max(float(overlap), 0.0), 0.9)
    step = max(1, int(round(t * (1.0 - overlap))))
    n = len(_tile_starts(int(ancho), t, step)) * len(_tile_starts(int(alto), t, step))
    return {"tile": t, "overlap": overlap, "n_tiles": n,
            "lado": lado, "umbral_px": int(umbral_px)}


def _tile_starts(total: int, tile: int, step: int) -> List[int]:
    """Posiciones de inicio de los tiles a lo largo de un eje (último pegado al borde)."""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, step))
    if not starts or starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def _box_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1); y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2); y2 = min(a.y2, b.y2)
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: List[Detection], iou_thr: float = 0.45,
         agnostic: bool = True) -> List[Detection]:
    """NMS greedy sobre detecciones de todos los tiles. Conserva la de mayor conf."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i].conf)
    keep: List[int] = []
    suppressed = [False] * len(dets)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        for jdx in order:
            if jdx == idx or suppressed[jdx]:
                continue
            if not agnostic and dets[jdx].class_id != dets[idx].class_id:
                continue
            if _box_iou(dets[idx], dets[jdx]) >= iou_thr:
                suppressed[jdx] = True
    return [dets[i] for i in keep]


def read_yolo_txt(txt_path: str | Path, img_w: int, img_h: int,
                  class_names: Dict[int, str]) -> List[Detection]:
    """Lee un .txt YOLO (class cx cy w h normalizados) y devuelve Detection con conf=1."""
    out: List[Detection] = []
    p = Path(txt_path)
    if not p.exists():
        return out
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = [float(x) for x in parts[1:5]]
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            out.append(Detection(
                class_id=cls,
                class_name=class_names.get(cls, str(cls)),
                conf=1.0,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
    except Exception:
        pass
    return out


def find_gt_for_image(image_path: Path, gt_folder: Optional[Path]) -> Optional[Path]:
    """Busca el .txt GT junto a la imagen, en /labels hermana, o en gt_folder."""
    stem = image_path.stem
    candidates = []
    if gt_folder is not None:
        candidates.append(gt_folder / f"{stem}.txt")
    # mismo directorio
    candidates.append(image_path.parent / f"{stem}.txt")
    # hermana labels/
    if image_path.parent.name.lower() == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{stem}.txt")
    else:
        candidates.append(image_path.parent / "labels" / f"{stem}.txt")
    for c in candidates:
        if c.exists():
            return c
    return None


def compute_box_size_um(det: Detection, um_per_px: Optional[float]) -> None:
    """Talla aproximada a partir de la CAJA, no de la partícula.

    Es un respaldo: sobreestima el área 1.87x en la mediana de este material
    porque la caja de una partícula alargada está casi vacía, y depende de la
    orientación. Solo se usa cuando la segmentación no pudo aislar la partícula.
    Lo bueno lo hace ``morfologia.medir_deteccion``.
    """
    if um_per_px is None or um_per_px <= 0:
        return
    area_px2 = det.w * det.h
    area_um2 = area_px2 * (um_per_px ** 2)
    # diámetro equivalente (círculo con misma área)
    diam_um = 2.0 * (area_um2 / np.pi) ** 0.5
    det.diam_um = diam_um
    det.area_um2 = area_um2
