"""Dos fallos que se vieron a ojo en un informe, sobre la placa 10.3x.

  * Una ficha de ejemplo era el BORDE DE LA PLACA, entrado como "PET fibra" de
    5978 um. El aro es una banda brillante y continua, el detector lo confunde
    con material y, por salir enorme, se colaba entre las mayores -- que es
    justo el criterio con que se eligen las fichas.

  * La recta de Feret cruzaba por fuera del contorno en las particulas
    irregulares. Eso NO es un error de medida: Feret es la separacion de dos
    mordazas de calibre, no un camino por dentro. Pero dibujada continua se lee
    como si la medida atravesara aire, y asi se leyo.

El circulo de la placa ya se conocia -- de el sale la escala --; lo que faltaba
era preguntarle.
"""
from __future__ import annotations

import cv2
import numpy as np

from polyx.ml import calibracion as C
from polyx.ml import morfologia as M
from polyx.ml.yolo_wrap import Detection


# ── El anillo de la placa ──────────────────────────────────────────────

def _placa(um_por_px=33.9, cx=2054.0, cy=1534.0, radio=1471.0):
    """La calibracion real de 10.3x.jpg, que es donde se vio el fallo."""
    return C.Calibration(um_por_px=um_por_px, origin=C.ORIGIN_PLATE,
                         cx=cx, cy=cy, radius_px=radio)


def test_una_caja_sobre_el_aro_se_reconoce():
    cal = _placa()
    # A ras del borde externo, que es donde el detector engancha el aro.
    x = cal.cx + cal.radius_px * 0.99
    assert C.about_the_ring(cal, x - 25, cal.cy - 25, x + 25, cal.cy + 25)


def test_una_particula_del_centro_no_se_aparta():
    cal = _placa()
    assert not C.about_the_ring(cal, cal.cx - 20, cal.cy - 20,
                                 cal.cx + 20, cal.cy + 20)


def test_una_particula_pegada_al_borde_util_se_conserva():
    """Se mira el CENTRO de la caja, no una esquina.

    Una particula legitima apoyada en el borde util tiene esquinas que asoman
    sobre el aro; descartarla por eso seria perder muestra de verdad, que es un
    fallo peor que el que se estaba arreglando.
    """
    cal = _placa()
    x = cal.cx + cal.radius_px * 0.93          # dentro del area util
    assert not C.about_the_ring(cal, x - 40, cal.cy - 40, x + 40, cal.cy + 40)


def test_sin_circulo_ajustado_no_se_aparta_nada():
    """Sin saber donde esta la placa no se puede afirmar que algo este fuera.

    Pasa con la escala tecleada a mano o heredada del indice: hay um/px pero no
    hay circulo. Marcar por si acaso seria inventarse un descarte.
    """
    a_mano = C.Calibration(um_por_px=30.0, origin=C.ORIGIN_MANUAL)
    assert not C.about_the_ring(a_mano, 0, 0, 50, 50)
    assert C.radius_fraction(a_mano, 10, 10) is None


def test_el_area_util_deja_fuera_la_pared_del_anillo():
    """El circulo se ajusta al borde EXTERNO y la pared cae en 0.960 del radio.

    El corte tiene que quedar por dentro de eso, o el aro entraria igual.
    """
    assert C.FRACTION_AREA_USEFUL < 0.960


# ── La recta de Feret en una particula concava ─────────────────────────

def _grumo_con_muesca():
    """Particula gruesa con una muesca profunda: la cuerda cruza el hueco.

    Tiene que ser GRUESA para que se mida por Feret y no por el geodesico: el
    geodesico solo se usa en particulas delgadas y dobladas. Es la forma de las
    fichas que se cuestionaron.
    """
    img = np.full((220, 240, 3), 18, np.uint8)
    pts = np.array([[40, 60], [200, 60], [200, 170], [150, 170],
                    [150, 110], [110, 110], [110, 170], [40, 170]])
    cv2.fillPoly(img, [pts], (60, 210, 240))
    return img, Detection(class_id=0, class_name="PET", conf=0.9,
                          x1=40, y1=60, x2=200, y2=170)


def test_la_recta_de_feret_puede_salirse_y_se_declara_cuanto():
    """No se corrige la medida: se mide cuanto sale, para poder decirlo."""
    img, d = _grumo_con_muesca()
    _, m = M.draw_measurement(img, d)
    assert m is not None and m.ok
    assert m.method.startswith("Feret"), (
        f"la figura tiene que medirse por Feret para probar esto, no por {m.method}")
    assert m.feret_outside > 0.05, (
        "en esta figura la cuerda cruza la muesca; deberia declararlo")


def test_en_una_particula_convexa_la_recta_no_se_sale():
    """El contraste: si saliera aqui, el numero no significaria nada."""
    img = np.full((200, 200, 3), 18, np.uint8)
    cv2.circle(img, (100, 100), 45, (60, 210, 240), -1)
    d = Detection(class_id=0, class_name="PET", conf=0.9,
                  x1=55, y1=55, x2=145, y2=145)
    _, m = M.draw_measurement(img, d)
    assert m is not None and m.ok
    assert m.feret_outside < 0.02


def test_el_tramo_de_fuera_se_dibuja_a_trazos():
    """Continuo por dentro, discontinuo por fuera.

    Se comprueba contando pixeles amarillos sobre el hueco de la muesca: si el
    tramo se dibujara continuo habria uno por columna, y a trazos hay menos.
    """
    img, d = _grumo_con_muesca()
    par, m = M.draw_measurement(img, d, zoom=1)
    assert par is not None
    # La mitad derecha es la que lleva las marcas.
    vis = par[:, par.shape[1] // 2:]
    # Amarillo del dibujo: (0, 210, 255) en BGR.
    amarillo = ((vis[:, :, 0] < 80) & (vis[:, :, 1] > 150) & (vis[:, :, 2] > 200))
    assert amarillo.any(), "no se dibujo la recta"
    assert m.feret_outside > 0.05


def test_el_corte_del_area_util_no_se_lleva_particulas_reales():
    """El riesgo del otro lado: pasarse de estricto y perder muestra.

    Medido sobre las 100 particulas anotadas a mano del estudio, la mas
    periferica cae en 0.828 del radio de la placa. El corte va en 0.95, o sea
    con 0.12 de radio de margen. Si alguien lo bajara para "asegurar" que no
    entra el aro, empezaria a descartar particulas de verdad -- que es un fallo
    peor que el que se estaba arreglando, porque no deja rastro.
    """
    cal = _placa()
    mas_periferica = 0.828
    x = cal.cx + cal.radius_px * mas_periferica
    assert not C.about_the_ring(cal, x - 30, cal.cy - 30, x + 30, cal.cy + 30)
    assert C.FRACTION_AREA_USEFUL > mas_periferica + 0.10, (
        "el corte se acerco demasiado a la particula real mas periferica")
