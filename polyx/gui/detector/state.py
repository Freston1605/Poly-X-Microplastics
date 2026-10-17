"""Estado compartido del Detector entre todas las páginas.

Hereda de QObject para emitir señales cuando cambia (las páginas se suscriben).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional

from PySide6.QtCore import QObject, Signal

from polyx.model.yolo_wrap import Detection, YoloModel
from polyx.model.i18n import tr
from polyx.model.plataforma import dispositivo_disponible


# ────────────────────────────────────────────────────────────────────
@dataclass
class ModelSlot:
    """Un slot de modelo en la pestaña 'Modelos' (hasta 3)."""
    alias: str = ""
    path: Optional[Path] = None      # ruta al .pt
    loaded: Optional[YoloModel] = None


@dataclass
class InferenceParams:
    conf: float = 0.25
    iou_nms: float = 0.45
    iou_tp: float = 0.50            # para análisis de errores
    # 1280 y no 640: los microplásticos ocupan ~12 px en fotos de 4096, y a 640
    # colapsan bajo el stride de la red, así que de fábrica no se detectaba nada.
    # Coincide con el perfil «Rápido» de la página de Parámetros.
    imgsz: int = 1280
    # No se fija "0" (CUDA) a secas: en un Mac no existe esa tarjeta y el
    # analisis fallaba al arrancar. Se resuelve al crear los parametros
    # eligiendo lo mejor que haya -- CUDA, luego MPS (GPU del Mac), luego CPU.
    device: str = field(default_factory=dispositivo_disponible)
    um_per_px: float = 0.0          # 0 = no hay calibración; respaldo manual
    size_min_um: float = 0.0        # 0 = sin filtro inferior
    size_max_um: float = 0.0        # 0 = sin filtro superior
    # ── Abstención: hasta dónde se sostiene el polímero ──
    # Confianza mínima para atribuir un polímero a la partícula. Por debajo, la
    # partícula se cuenta igual pero se reporta como «no asignable», en vez de
    # inventarle una clase. 0 = desactivado, que es el comportamiento de
    # siempre. El porqué —el Nile Red mide polaridad, no identidad química, y
    # PP y LDPE son las dos poliolefinas— está en core/asignacion.py.
    conf_asignacion: float = 0.0
    # ── Calibración automática contra la placa Petri ──
    # Con esto activo cada foto obtiene SU escala midiendo el anillo de la placa,
    # sin marcar nada a mano. Importa porque la distancia de disparo varía entre
    # tomas: en este material la escala va de 31 a 49 µm/px, un factor 1.5, y un
    # valor único para todo el lote daría tamaños con hasta 50 % de error.
    # Encendido de fabrica. Estuvo apagado, y la casilla que lo controla se
    # llama "(recomendado)": la aplicacion recomendaba algo que no hacia. El
    # efecto era que TODA la seccion de tallas del informe salia con guiones
    # -- las particulas se median en pixeles pero sin µm/px no hay como pasarlas
    # a micrometros -- y nada explicaba por que.
    #
    # Fallar aqui es barato: si la placa no se ve en la foto (recortes, por
    # ejemplo) la calibracion se declara invalida, se hereda del indice.csv si
    # lo hay, y el analisis sigue. Cuesta ~1-2 s por foto.
    medir_placa: bool = True
    diametro_placa_mm: float = 100.0
    # El aro de la placa esta MAS CERCA de la cámara que el fondo donde reposan
    # las partículas, así que se proyecta más grande y la escala sale pequeña:
    # las tallas quedan subestimadas. Con estos dos datos se corrige al plano de
    # la base. En 0 no se corrige y el informe lo declara.
    altura_placa_mm: float = 0.0
    distancia_camara_mm: float = 0.0
    # CSV que asocia nombre de imagen con µm/px ya calibrado. Es la única vía
    # para los recortes, donde el borde de la placa no aparece y por tanto no se
    # puede volver a medir: la escala se hereda de la foto de la que salieron.
    indice_calibracion: str = ""
    # ── Forma de la partícula ──
    # Segmenta dentro de cada caja para medir área, largo y ancho reales en vez
    # de los de la caja. Sin esto el área sale 1.87x de más y una fibra en
    # diagonal se reporta como fragmento. Cuesta ~0.2 ms por partícula.
    medir_forma: bool = True
    # ── Troceado automático ──
    # "auto" trocea sola la foto cuyo lado mayor pase de troceo_umbral_px, infiere
    # cada tile a resolución nativa y fusiona con NMS global. El umbral va a 2000
    # para que los recortes del estudio (1630 px de lado) entren de una pieza y
    # las placas completas (~3260) se troceen.
    troceo: str = "auto"            # "auto" | "siempre" | "nunca"
    troceo_umbral_px: int = 2000
    troceo_tile: int = 0            # 0 = derivar de min(umbral, imgsz)
    troceo_overlap: float = 0.25    # solape entre tiles; el NMS quita el duplicado


@lru_cache(maxsize=12)
def _leer_png(ruta: str, _mtime: float) -> Optional[bytes]:
    """Lee un PNG del run, con caché acotada.

    ``_mtime`` no se usa dentro: entra en la firma para que la caché se
    invalide sola si el archivo cambia en disco.
    """
    try:
        return Path(ruta).read_bytes()
    except OSError:
        return None


@dataclass
class ImageResult:
    """Resultado de inferir un modelo sobre una imagen.

    Las imágenes anotadas se guardan en disco dentro del run y aquí solo viven
    sus rutas. Retener los bytes costaba ~6.4 GB de RAM en un lote de 552
    recortes con tres modelos, y era gasto puro: los PNG ya estaban escritos.
    Las propiedades ``*_png`` siguen devolviendo bytes, así que quien las
    consumía no cambia; lo que cambia es que se leen bajo demanda.
    """
    image_path: Path
    model_idx: int
    predictions: List[Detection] = field(default_factory=list)
    gt: List[Detection] = field(default_factory=list)
    has_gt: bool = False
    tp: int = 0
    fp: int = 0
    fn: int = 0
    miscls: int = 0
    # Rutas de las imágenes anotadas guardadas en la carpeta del run.
    annotated_path: Optional[Path] = None   # predicción + GT (preview/errores)
    pred_path: Optional[Path] = None        # solo predicciones del modelo
    gt_path: Optional[Path] = None          # solo GT (None si no hay)
    # veredicto del usuario tras revisión visual: None / "buena" / "mala"
    verdict: Optional[str] = None
    # geometría del troceado si la foto se partió; "" = se infirió de una pieza
    troceo: str = ""

    @staticmethod
    def _bytes_de(ruta: Optional[Path]) -> Optional[bytes]:
        if ruta is None:
            return None
        try:
            mtime = ruta.stat().st_mtime
        except OSError:
            return None
        return _leer_png(str(ruta), mtime)

    @property
    def annotated_png(self) -> Optional[bytes]:
        return self._bytes_de(self.annotated_path)

    @property
    def pred_png(self) -> Optional[bytes]:
        return self._bytes_de(self.pred_path)

    @property
    def gt_png(self) -> Optional[bytes]:
        return self._bytes_de(self.gt_path)


# ────────────────────────────────────────────────────────────────────
class DetectorState(QObject):
    """Estado global del Detector. Las páginas leen/escriben aquí."""

    # Señales
    models_changed = Signal()
    images_changed = Signal()
    params_changed = Signal()
    run_progress = Signal(int, int, str)    # done, total, last_image
    run_started = Signal()
    run_finished = Signal()
    run_image_done = Signal(int, object)    # model_idx, ImageResult
    run_aborted = Signal()

    def __init__(self):
        super().__init__()
        self.model_slots: List[ModelSlot] = [
            ModelSlot(alias=tr("Modelo {n}").format(n=i + 1)) for i in range(3)]
        self.images: List[Path] = []
        self.gt_folder: Optional[Path] = None
        self.params = InferenceParams()
        # results[model_idx] -> List[ImageResult]
        self.results: Dict[int, List[ImageResult]] = {}
        # Escala de cada imagen y de donde salio, por nombre de archivo. La
        # llena el runner; el informe la reporta. Sin esto los tamanos en um
        # aparecerian sin decir contra que patron se midieron.
        self.calibraciones: Dict[str, object] = {}
        # run timestamp/folder
        self.run_dir: Optional[Path] = None
        self._running = False
        self._abort = False

    # ── helpers ──
    def active_models(self) -> List[ModelSlot]:
        return [s for s in self.model_slots if s.path is not None]

    def has_gt(self) -> bool:
        """¿Alguna imagen tiene GT?"""
        for r_list in self.results.values():
            if any(r.has_gt for r in r_list):
                return True
        return False

    def is_running(self) -> bool:
        return self._running

    def set_running(self, v: bool):
        self._running = v

    def request_abort(self):
        self._abort = True

    def consume_abort(self) -> bool:
        a = self._abort
        self._abort = False
        return a

    def reset_results(self):
        self.results = {}
        self.calibraciones = {}
        self.run_dir = None
