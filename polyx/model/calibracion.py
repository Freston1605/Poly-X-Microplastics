"""Fija los um/px de una foto usando la placa Petri como patron de longitud.

El diametro externo de la placa es conocido -- 100 mm por defecto -- asi que su
radio en pixeles determina la escala de ESA foto. Hace falta porque la distancia
de disparo cambia entre tomas: en este material los um/px van de 31.3 a 33.9, y
sin corregirlo una misma particula se reportaria con tamanos distintos segun en
que foto haya caido.

Hough por si solo no sirve como patron: en algunas placas erraba el radio hasta
un 12%, y ese error entra entero en la escala. Aqui Hough aporta solo el centro
aproximado; el radio sale de buscar el borde del anillo en 720 direcciones y
ajustar un circulo por minimos cuadrados descartando atipicos. Se ajusta al
borde EXTERNO del anillo, que es el que corresponde al diametro nominal.

El algoritmo estaba en ``paper/detectar_placa.py`` y se movio aqui para que la
aplicacion y el script del paper midan con el mismo codigo. Dos
implementaciones de la misma medida divergen, y entonces la cifra del informe
deja de ser la cifra del paper.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

# Direcciones en que se busca el borde del anillo. 720 = una cada medio grado:
# suficientes para que el ajuste sobreviva a reflejos y oclusiones parciales.
N_ANGULOS = 720

# Diametro de la placa Petri del estudio, en mm. Medido sobre las placas reales.
# Se anota porque las Petri estandar vienen en 90 y en 100 mm, y confundirlas
# metaria un 11% en todas las tallas sin dar ningun sintoma.
#
# ESTE VALOR VA CON EL BORDE **EXTERNO** DEL ANILLO, que es al que ajusta
# puntos_del_borde() -- camina hacia AFUERA desde el pico de brillo. Cambiar uno
# sin el otro descoloca toda la escala del estudio.
DIAMETRO_PLACA_MM = 100.0

# Espesor de la pared del anillo, medido sobre las fotos del estudio: el borde
# interno cae en 0.960 del radio ajustado y el externo en 1.000, unos 2.0 mm de
# pared (1.95 / 2.02 / 2.02 mm en tres placas distintas).
#
# DE AQUI SALE LA INCERTIDUMBRE DE LA ESCALA, y no es simetrica. El nominal de
# una Petri es ambiguo a nivel de la pared: puede referirse al diametro externo
# o al util interior. Aqui se toma el EXTERNO. Si en realidad se refiriera al
# interno -- 96 mm en vez de 100 --, la escala correcta seria un 4.2% mayor
# (100/96) y TODAS las tallas del estudio estarian subestimadas en esa cifra.
#
# El error solo puede ir en ese sentido, nunca al reves, porque el externo es el
# mayor de los dos bordes posibles.
#
# No se resolvio a favor del interno porque: (a) las fotos con huincha dan
# 98-100 mm en el borde externo, (b) el catalogo de placas Petri suele nombrar
# el diametro externo, y (c) el borde interno no sirve para calibrar -- es una
# transicion gradual que no se ve en 3 de cada 14 placas, tapada por el
# sedimento. Queda declarado en el informe en vez de disimulado.
ESPESOR_PARED_MM = 2.0
SESGO_MAXIMO_PARED_PCT = 4.2

# De donde salio la escala, de mas fiable a menos. Va al informe: una medida sin
# su procedencia no es verificable.
ORIGEN_PLACA = "placa"      # ajustada sobre el anillo de esta misma foto
ORIGEN_INDICE = "indice"    # heredada del recorte via indice.csv
ORIGEN_MANUAL = "manual"    # tecleada a mano en Parametros
ORIGEN_NINGUNA = "ninguna"  # sin calibrar; los tamanos no se pueden reportar


@dataclass
class Calibracion:
    """Escala de una imagen y como se obtuvo."""
    um_por_px: float = 0.0
    origen: str = ORIGEN_NINGUNA
    cx: float = 0.0
    cy: float = 0.0
    radio_px: float = 0.0
    n_puntos: int = 0
    desvio_hough_pct: float = 0.0
    diametro_mm: float = DIAMETRO_PLACA_MM
    # Correccion de paralaje aro->base. 1.0 = no se aplico, y entonces las
    # tallas quedan subestimadas en una cantidad que depende de la distancia de
    # disparo. Se guarda para poder declararlo en el informe.
    factor_paralaje: float = 1.0
    aviso: str = ""

    @property
    def valida(self) -> bool:
        return self.um_por_px > 0

    def descripcion(self) -> str:
        """Frase corta para la interfaz y el informe."""
        if not self.valida:
            return "sin calibrar"
        if self.origen == ORIGEN_PLACA:
            return (f"{self.um_por_px:.4f} µm/px — placa de "
                    f"{self.diametro_mm:g} mm, r={self.radio_px:.0f} px "
                    f"({self.n_puntos} puntos de borde)")
        if self.origen == ORIGEN_INDICE:
            return f"{self.um_por_px:.4f} µm/px — heredada del recorte"
        return f"{self.um_por_px:.4f} µm/px — introducida a mano"


def leer_imagen(ruta: str | Path) -> Optional[np.ndarray]:
    """Lee con ``fromfile`` porque las carpetas traen tildes y parentesis.

    ``cv2.imread`` falla en silencio con rutas no ASCII en Windows y devuelve
    None, que aqui se confundiria con "no hay placa".
    """
    try:
        return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8),
                            cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def hough_aproximado(bgr: np.ndarray) -> Optional[tuple]:
    """Centro y radio aproximados. Solo semilla para el ajuste fino."""
    alto, ancho = bgr.shape[:2]
    escala = 900 / max(alto, ancho)
    chico = cv2.resize(bgr, (int(ancho * escala), int(alto * escala)))
    gris = cv2.medianBlur(cv2.cvtColor(chico, cv2.COLOR_BGR2GRAY), 5)
    lado = min(gris.shape)
    c = cv2.HoughCircles(gris, cv2.HOUGH_GRADIENT, dp=1.5, minDist=lado,
                         param1=120, param2=60,
                         minRadius=int(lado * 0.22), maxRadius=int(lado * 0.52))
    if c is None:
        return None
    mejor = max(np.round(c[0]).astype(float), key=lambda v: v[2])
    return mejor[0] / escala, mejor[1] / escala, mejor[2] / escala


def puntos_del_borde(gris: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """Para cada angulo, el radio del borde externo del anillo brillante."""
    ang = np.linspace(0, 2 * np.pi, N_ANGULOS, endpoint=False)
    radios = np.arange(max(1, int(r * 0.80)), int(r * 1.25))
    xs = np.clip(cx + np.outer(radios, np.cos(ang)), 0, gris.shape[1] - 1).astype(int)
    ys = np.clip(cy + np.outer(radios, np.sin(ang)), 0, gris.shape[0] - 1).astype(int)
    perfiles = gris[ys, xs].astype(float)          # (radio, angulo)

    base = np.median(gris[int(max(0, cy - r * 0.5)):int(cy + r * 0.5),
                          int(max(0, cx - r * 0.5)):int(cx + r * 0.5)])
    puntos = []
    for k in range(N_ANGULOS):
        col = perfiles[:, k]
        pico = int(np.argmax(col))
        if col[pico] <= base + 12:                 # sin anillo visible aqui
            continue
        umbral = base + (col[pico] - base) * 0.5
        j = pico
        while j < len(col) - 1 and col[j] > umbral:
            j += 1
        rr = radios[j]
        puntos.append((cx + rr * np.cos(ang[k]), cy + rr * np.sin(ang[k])))
    return np.array(puntos)


def ajustar_circulo(pts: np.ndarray) -> tuple:
    """Ajuste algebraico de circulo por minimos cuadrados (Kasa)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    return cx, cy, r


def ajuste_robusto(pts: np.ndarray, iteraciones: int = 3) -> Optional[tuple]:
    """Ajusta y descarta puntos lejanos; el anillo tiene reflejos y oclusiones."""
    for _ in range(iteraciones):
        if len(pts) < 30:
            return None
        cx, cy, r = ajustar_circulo(pts)
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        pts = pts[d < max(3.0, 2.0 * np.median(d))]
    return ajustar_circulo(pts) + (len(pts),)


def corregir_paralaje(um_por_px: float, altura_placa_mm: float,
                      distancia_camara_mm: float) -> tuple:
    """Lleva la escala del ARO de la placa al PLANO DE LA BASE.

    El aro esta mas cerca de la camara que el fondo donde reposan las
    particulas, asi que se proyecta mas grande: 100 mm medidos sobre el aro
    ocupan mas pixeles que 100 mm apoyados en la base. Como
    ``um/px = 100000 / diametro_px``, un diametro inflado da un um/px pequeno, y
    como ``talla = px * um/px``, **todas las tallas salen subestimadas**.

    Con D la distancia de la camara a la base y h la altura de la placa, el aro
    esta a D-h y el factor entre ambas escalas es D/(D-h).

    MEDIDO SOBRE ESTE MATERIAL, EL EFECTO ES DESPRECIABLE Y NO HACE FALTA
    CORREGIR. El razonamiento vale en general -- disparando cerca el error si
    seria grande, un 16% con la placa a 15 mm y la camara a 100 mm -- pero aqui
    se comprobo asi: el borde del suelo de la placa se ve, y su radio es 0.863
    veces el del aro. Ese encogimiento podria ser perspectiva o podria ser la
    forma de la placa (el suelo va rehundido). Se distinguen porque la
    perspectiva depende de la distancia de disparo y la forma no, y hay placas
    fotografiadas a dos distancias que difieren 1.43x:

        cerca (n=14):  base/aro = 0.8634 +- 0.0058
        lejos (n=14):  base/aro = 0.8643 +- 0.0077
        cociente entre grupos: 1.0010 +- 0.0030  ->  indistinguible de 1.0

    Al no cambiar con la distancia, el encogimiento es la forma de la placa y la
    perspectiva queda acotada por debajo del 1.3% incluso en el borde de la
    incertidumbre. Por eso los dos parametros vienen en 0 y no se corrige nada.

    Se deja implementado porque otro montaje -- una camara mas cerca, o placas
    mas altas -- si lo necesitaria, y porque conviene que quede escrito como se
    descarto en vez de repetir la duda.

    Devuelve (um_por_px_corregido, factor). Sin datos suficientes devuelve el
    valor sin tocar y factor 1.0: es preferible una escala sin corregir y
    declarada que una corregida con numeros inventados.
    """
    if altura_placa_mm <= 0 or distancia_camara_mm <= 0:
        return um_por_px, 1.0
    if distancia_camara_mm <= altura_placa_mm:
        # La camara estaria dentro de la placa; el dato es erroneo.
        return um_por_px, 1.0
    factor = distancia_camara_mm / (distancia_camara_mm - altura_placa_mm)
    return um_por_px * factor, factor


def calibrar_desde_placa(bgr: np.ndarray,
                         diametro_mm: float = DIAMETRO_PLACA_MM,
                         altura_placa_mm: float = 0.0,
                         distancia_camara_mm: float = 0.0) -> Calibracion:
    """Mide el anillo de la placa y devuelve la escala de esta foto.

    Devuelve una Calibracion sin validez -- con ``aviso`` explicando por que --
    cuando no encuentra la placa, en vez de lanzar: en un lote de 69 fotos hay
    que poder seguir con las demas.
    """
    if bgr is None or bgr.size == 0:
        return Calibracion(aviso="imagen vacia o ilegible")
    if diametro_mm <= 0:
        return Calibracion(aviso="diametro nominal invalido")

    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    semilla = hough_aproximado(bgr)
    if semilla is None:
        return Calibracion(diametro_mm=diametro_mm,
                           aviso="no se encontro la placa en la foto")
    hx, hy, hr = semilla

    pts = puntos_del_borde(gris, hx, hy, hr)
    fino = ajuste_robusto(pts) if len(pts) >= 30 else None
    if fino is None:
        # Hough como ultimo recurso: sirve para no perder la foto, pero su radio
        # es justo lo que no es fiable, asi que queda avisado.
        cx, cy, r, n = hx, hy, hr, 0
        aviso = "el ajuste fino del anillo fallo; escala tomada de Hough (poco fiable)"
    else:
        cx, cy, r, n = fino
        aviso = ""

    desvio = 100.0 * (r - hr) / hr if hr > 0 else 0.0
    if not aviso and abs(desvio) > 8:
        aviso = f"Hough erraba {desvio:+.1f}% respecto del ajuste fino"

    if r <= 0:
        return Calibracion(diametro_mm=diametro_mm, aviso="radio ajustado nulo")

    um_aro = (diametro_mm * 1000.0) / (2 * r)
    um_base, factor = corregir_paralaje(um_aro, altura_placa_mm,
                                        distancia_camara_mm)
    # Sin aviso cuando no se corrige: se midio sobre este material y el efecto
    # es indetectable (ver corregir_paralaje). Advertirlo en cada foto seria
    # alarmar por un sesgo que los datos acotan por debajo del 1.3%.
    return Calibracion(um_por_px=um_base,
                       origen=ORIGEN_PLACA, cx=cx, cy=cy, radio_px=r,
                       n_puntos=n, desvio_hough_pct=desvio,
                       diametro_mm=diametro_mm, factor_paralaje=factor,
                       aviso=aviso)


def cargar_indice(ruta_csv: str | Path) -> Dict[str, float]:
    """Lee un CSV que asocie nombre de imagen con su um/px ya calibrado.

    Sirve para los recortes: la placa entera no aparece en el recorte, de modo
    que su escala no se puede volver a medir sobre el y tiene que heredarse de
    la foto de la que salio. ``cortar_placas.py`` deja esa correspondencia en
    ``indice.csv``, y ``detectar_placa.py`` la de las placas completas en
    ``calibracion_placas.csv``.

    Se aceptan las dos cabeceras porque son los dos archivos que produce el
    pipeline; devuelve ``{}`` si el archivo no existe o no trae la columna.
    """
    ruta = Path(ruta_csv)
    if not ruta.is_file():
        return {}
    mapa: Dict[str, float] = {}
    try:
        with ruta.open(newline="", encoding="utf-8") as fh:
            for fila in csv.DictReader(fh):
                nombre = fila.get("recorte") or fila.get("placa") or ""
                try:
                    um = float(fila.get("um_por_px", "") or 0)
                except ValueError:
                    continue
                if nombre and um > 0:
                    mapa[nombre] = um
    except (OSError, csv.Error):
        return {}
    return mapa


NOMBRES_INDICE = ("indice.csv", "calibracion_placas.csv")


def buscar_indice(ruta_imagen: str | Path, niveles: int = 3) -> Dict[str, float]:
    """Busca un CSV de calibración junto a las imágenes, subiendo unos niveles.

    Se busca en vez de exigir la ruta porque el CSV lo escribe el pipeline del
    paper en la carpeta de los recortes o en la de arriba, y obligar a
    señalarlo a mano es justo el paso que se olvida; olvidarlo no da error, da
    un informe con la escala equivocada.
    """
    carpeta = Path(ruta_imagen)
    carpeta = carpeta.parent if carpeta.is_file() else carpeta
    for _ in range(max(1, niveles)):
        for nombre in NOMBRES_INDICE:
            mapa = cargar_indice(carpeta / nombre)
            if mapa:
                return mapa
        if carpeta.parent == carpeta:
            break
        carpeta = carpeta.parent
    return {}


def resolver(ruta_imagen: str | Path,
             indice: Optional[Dict[str, float]] = None,
             um_por_px_manual: float = 0.0,
             diametro_mm: float = DIAMETRO_PLACA_MM,
             medir_placa: bool = False,
             bgr: Optional[np.ndarray] = None,
             altura_placa_mm: float = 0.0,
             distancia_camara_mm: float = 0.0) -> Calibracion:
    """Escala de una imagen, por orden de fiabilidad.

    1. ``indice`` -- ya calibrada contra la placa, por foto. Es lo que hay para
       los recortes y no cuesta nada.
    2. medir la placa en la propia foto, si se pidio y esta visible.
    3. el valor tecleado en Parametros, que aplica uno solo a todo el lote.

    El orden importa: el valor manual es el ultimo porque es unico para el lote
    entero, y en este material la escala real varia hasta 1.5x entre fotos.

    ``bgr`` evita releer del disco cuando quien llama ya tiene la imagen
    decodificada, que es el caso del runner: son 12 MP por foto.
    """
    nombre = Path(ruta_imagen).name
    if indice and nombre in indice:
        return Calibracion(um_por_px=indice[nombre], origen=ORIGEN_INDICE,
                           diametro_mm=diametro_mm)

    if medir_placa:
        if bgr is None:
            bgr = leer_imagen(ruta_imagen)
        cal = calibrar_desde_placa(bgr, diametro_mm, altura_placa_mm,
                                   distancia_camara_mm)
        if cal.valida:
            return cal
        if um_por_px_manual > 0:
            return Calibracion(um_por_px=um_por_px_manual, origen=ORIGEN_MANUAL,
                               diametro_mm=diametro_mm,
                               aviso=f"no se pudo medir la placa ({cal.aviso})")
        return cal

    if um_por_px_manual > 0:
        return Calibracion(um_por_px=um_por_px_manual, origen=ORIGEN_MANUAL,
                           diametro_mm=diametro_mm)
    return Calibracion(diametro_mm=diametro_mm)


# Hasta que fraccion del radio ajustado se considera area util de la placa.
#
# El circulo se ajusta al borde EXTERNO del anillo, y la pared mide unos 2 mm
# sobre un radio de 50: el borde interno cae en 0.960. El corte va en 0.95, por
# dentro de esa pared.
#
# QUE NO SE LLEVA POR DELANTE NINGUNA PARTICULA ESTA MEDIDO, no supuesto: sobre
# las 100 particulas anotadas a mano de este estudio, la mas periferica esta en
# 0.828 del radio y ninguna pasa de ahi. Entre 0.83 y 0.95 hay margen de sobra,
# asi que el corte aparta el aro sin costar muestra. Si en otro material el
# sedimento llegara a apoyarse en la pared, este numero habria que revisarlo
# con la misma medicion.
FRACCION_AREA_UTIL = 0.95


def fraccion_del_radio(cal, x: float, y: float) -> Optional[float]:
    """A que fraccion del radio de la placa cae el punto (x, y).

    None si esa foto no tiene un circulo ajustado, que es lo que pasa cuando la
    escala vino a mano o heredada del indice.
    """
    if cal is None or not getattr(cal, "radio_px", 0):
        return None
    return float(np.hypot(x - cal.cx, y - cal.cy)) / float(cal.radio_px)


def sobre_el_anillo(cal, x1: float, y1: float, x2: float, y2: float) -> bool:
    """Si la caja (x1,y1,x2,y2) cae sobre la pared de la placa o fuera de ella.

    Se mira el CENTRO de la caja y no una esquina: una particula legitima
    pegada al borde util tiene esquinas que asoman, y descartarla por eso seria
    perder muestra de verdad. El anillo, en cambio, cae entero fuera.

    Devuelve False cuando no hay circulo ajustado: sin saber donde esta la
    placa no se puede afirmar que algo este fuera, y marcar por si acaso seria
    inventar un descarte.
    """
    f = fraccion_del_radio(cal, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return f is not None and f > FRACCION_AREA_UTIL


def resumen_lote(calibraciones) -> dict:
    """Estadistica de escala del lote, para la seccion de metodos del informe.

    La dispersion es el dato que se reporta: si las fotos no comparten escala,
    un histograma de tamanos hecho sobre todas juntas mezcla unidades.
    """
    ums = [c.um_por_px for c in calibraciones if c.valida]
    if not ums:
        return {"n": 0, "origenes": {}}
    a = np.array(ums, dtype=float)
    origenes: Dict[str, int] = {}
    for c in calibraciones:
        if c.valida:
            origenes[c.origen] = origenes.get(c.origen, 0) + 1
    return {
        "n": len(ums),
        "min": float(a.min()),
        "max": float(a.max()),
        "media": float(a.mean()),
        # ddof=1 (desviacion muestral): con n fotos se esta estimando la
        # dispersion de la poblacion de fotos, no describiendo solo estas n.
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "mediana": float(np.median(a)),
        "variacion": float(a.max() / a.min()) if a.min() > 0 else 0.0,
        "origenes": origenes,
        "avisos": [c.aviso for c in calibraciones if c.aviso],
    }
