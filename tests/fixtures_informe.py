"""Estados DetectorState reutilizables para los fixtures del informe.

La captura del baseline pre-fix (scripts/capturar_fixtures_informe.py, T03) y
los tests de identidad de bytes e inyeccion muerta (T04-T07) necesitan todos
el MISMO par de estados de entrada: uno benigno y otro con inyeccion HTML en
los nombres de archivo y alias de modelo. Una receta repetida a mano en cada
consumidor se desincroniza y el baseline deja de describir el render real;
aqui esta la fuente comun.

La receta de construccion es la del lote minimo de tests/test_informe_biling
ue (_estado_minimo): una foto sintetica de 320x260 con un disco de r=22, una
Detection (x1=138, y1=108, x2=182, y2=152) medida con morphology y el
ImageResult publicado en el slot 0 de results. La variante hostil reutiliza
EXACTAMENTE los mismos datos medidos: solo cambia donde vive la inyeccion,
el alias del modelo y el nombre de archivo de la foto. Con la foto identica,
cualquier divergencia del render solo puede venir de esas dos cadenas hosti-
les, que es justo lo que el slice quiere aislar.

OJO: la foto se escribe via cv2.imencode + Path.write_bytes, nunca
cv2.imwrite. En Windows imwrite/imread fallan en silencio con rutas fuera
de ASCII, y un nombre de archivo hostil (<img src=x onerror=alert(2)>
.png) es justo el caso extremo que este fixture ejercita.

Matiz conocido del nombre hostil: NO lleva ``/`` embebido a proposito. El
ejemplo de planificacion (``<script>a</script>.png``) partia el nombre en
dos componentes y el ``<`` de apertura se perdia de ``image_path.name``
(quedaba ``script>.png``, sin etiqueta que escapar). Con el nombre actual
la inyeccion completa llega viva al sumidero, que es lo que T06/T07 cer-
rifican.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from polyx.core.detect.protocol import Instance as Detection

# ---------------------------------------------------------------------------
# Constantes de los datos sinteticos
# ---------------------------------------------------------------------------

# Nombre de archivo y alias en la variante benigna. Nombre corto, asci y
# sin caracteres especiales, para que elijo un color distinto por convencion
# del proyecto: ruta con pathlib.Path y datos reales fuera de los samples del
# repo (nada del lote oficial entra a los fixtures de hardening).
NOMBRE_FOTO_BENIGNO: str = "placa_muestra_a.png"
ALIAS_BENIGNO: str = "m1"

# Inyeccion de referencia. Cada sumidero del informe (alias de modelo, nombre
# de archivo) recibe una de estas cadenas para el lado hostil del render.
#
# OJO con el nombre de archivo: NO lleva ``/``. El ejemplo de planificacion
# era ``<script>a</script>.png``, pero la barra embebida parte el nombre en
# dos componentes (``<script>a`` como directorio y ``script>.png`` como
# archivo) y el ``<`` de apertura se pierde: el sumidero recibe ``script>
# .png``, sin etiqueta que escapar. Con ``<img src=x onerror=alert(2)>.png``
# la inyeccion completa llega viva a ``image_path.name``. El alert(2) distin-
# gue este sumidero del alias (alert(1)) al asertar en T06/T07.
NOMBRE_FOTO_HOSTIL: str = "<img src=x onerror=alert(2)>.png"
ALIAS_HOSTIL: str = "<img src=x onerror=alert(1)>"


def foto_sintetica(ancho: int = 320, alto: int = 260) -> "np.ndarray":
    """Build the synthetic BGR frame shared by all benign/hostile fixtures."""
    img = np.full((alto, ancho, 3), 18, np.uint8)
    cv2.circle(img, (ancho // 2, alto // 2), 22, (60, 210, 240), -1)
    return img


def _escribir_foto(ruta: Path) -> None:
    """Encode the synthetic frame and write it to ruta (never imwrite).

    Los nombres hostiles no llevan ``/`` (ver NOMBRE_FOTO_HOSTIL), pero el
    mkdir(parents=True) se deja igual: no cuesta y blinda contra futuros
    nombres con subdirectorio.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", foto_sintetica())
    if not ok:      # pragma: no cover - cv2 nunca falla con tipos correctos
        raise RuntimeError(f"cv2.imencode fallo para {ruta}")
    ruta.write_bytes(buf.tobytes())


def deteccion_medida(img: "np.ndarray") -> Detection:
    """Build the measured Detection (disc fit to the synthetic frame)."""
    d = Detection(class_id=0, class_name="PET", conf=0.9,
                  x1=138, y1=108, x2=182, y2=152)
    from polyx.core import morphology
    morphology.apply_to_detection(d, img, 30.0)
    return d


def _normalizar(valor: Union[str, Sequence[str]]) -> list[str]:
    """Accept one filename/alias or a whole batch; always return a list."""
    if isinstance(valor, str):
        return [valor]
    return list(valor)


def _construir_estado(directorio: Path,
                      nombres: Sequence[str],
                      aliases: Sequence[str],
                      escribir_foto: bool = True):
    """Build the DetectorState shared by every fixture variant.

    Identical recipe to test_informe_bilingue._estado_minimo: one synthetic
    frame written to directorio/nombre, one predicted Detection measured by
    morfologia at 30.0 um/px, and one ImageResult published on each model
    slot. Multiple versions only differ in the injected strings.
    """
    from polyx.gui.detector.state import DetectorState, ImageResult, ModelSlot

    directorio.mkdir(parents=True, exist_ok=True)
    if escribir_foto:
        for nombre in nombres:
            _escribir_foto(directorio / nombre)

    st = DetectorState()
    # model_slots es una LISTA (ver state.py, active_models itera sobre ella);
    # no se reemplaza el contenedor, solo se llenan los indices pedidos.
    assert len(st.model_slots) >= len(aliases), (
        "DetectorState no tiene slots para tantos modelos")
    st.results = {}
    numero = 1
    for idx, alias in enumerate(aliases):
        st.model_slots[idx] = ModelSlot(alias=alias, path=Path("fake.pt"))
        filas = []
        for nombre in nombres:
            ruta = directorio / nombre
            d = deteccion_medida(foto_sintetica())
            d.number = numero
            numero += 1
            filas.append(ImageResult(image_path=ruta, model_idx=idx,
                                     predictions=[d], gt=[], has_gt=False))
        st.results[idx] = filas
    return st


def estado_benigno(directorio: Optional[Path] = None):
    """Return a clean, benign DetectorState for the report fixtures.

    The photo written is also benign (``placa_muestra_a.png``), so downstream
    renders must reproduce the baseline byte-for-byte.
    """
    directorio = directorio or Path(tempfile.mkdtemp(prefix="polyx-fixture-benigna-"))
    return _construir_estado(Path(directorio),
                             [NOMBRE_FOTO_BENIGNO], [ALIAS_BENIGNO])


def estado_hostil(nombres: Union[str, Sequence[str], None] = None,
                  alias: Union[str, Sequence[str], None] = None,
                  directorio: Optional[Path] = None):
    """Return a DetectorState carrying HTML injection in its strings.

    Injection lives in two sinks of the report: the model alias ("m1" beco-
    mes ``<img src=x onerror=alert(1)>``) and the image file name (becomes
    ``<img src=x onerror=alert(2)>.png``). The filename deliberately avo-
    ids ``/`` so the full payload survives ``image_path.name``. If no injec-
    tion strings are supplied, the defaults from NOMBRE_FOTO_HOSTIL /
    ALIAS_HOSTIL are used so callers can simply ask for estado_hostil()
    and still get the worst case.
    """
    nombres = _normalizar(nombres) if nombres is not None else [NOMBRE_FOTO_HOSTIL]
    if alias is None:
        aliases = [ALIAS_HOSTIL]
    else:
        aliases = _normalizar(alias)
    directorio = directorio or Path(tempfile.mkdtemp(prefix="polyx-fixture-hostil-"))
    return _construir_estado(Path(directorio), nombres, aliases)
