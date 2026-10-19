"""La medida de forma, contra figuras cuya talla se conoce de antemano.

Estas pruebas existen porque cada afirmación que hace `morfologia` sobre una
partícula acaba en una tabla del paper. Sin ellas, un cambio bienintencionado en
la segmentación puede desplazar todas las tallas sin que nada avise.

Cada prueba fija además el PORQUÉ de una decisión de diseño, de modo que si
alguien vuelve a intentar la variante descartada, la suite se lo dice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyx.model import morfologia as M  # noqa: E402
from tests.formas import (  # noqa: E402
    CASOS, arco, circulo, grumo_con_muesca, recta, recta_dentada, recta_girada,
)


# ── El largo ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("nombre,fabrica,tol", CASOS,
                         ids=[c[0] for c in CASOS])
def test_largo_contra_talla_conocida(nombre, fabrica, tol):
    mascara, real = fabrica()
    m = M.measure(mascara)
    assert m.ok, f"{nombre}: no se pudo medir"
    error = 100.0 * abs(m.length_px - real) / real
    assert error <= tol, (
        f"{nombre}: largo {m.length_px:.1f} px frente a {real:.1f} reales "
        f"({error:.1f} % de error, tolerancia {tol} %)")


def test_el_largo_no_depende_de_la_orientacion():
    """El defecto que motivó todo esto: la caja del detector da un número
    distinto según cómo haya caído la partícula, y eso no es una medida."""
    horizontal = M.measure(recta()[0]).length_px
    for grados in (15, 30, 45, 60, 90):
        girada = M.measure(recta_girada(grados)[0]).length_px
        desvio = 100.0 * abs(girada - horizontal) / horizontal
        assert desvio < 3.0, (
            f"a {grados}° la misma barra mide {girada:.1f} px en vez de "
            f"{horizontal:.1f} ({desvio:.1f} % de desvío)")


def test_en_fibra_curva_gana_el_geodesico():
    """En un arco, Feret da la cuerda y se queda corto; el camino geodésico
    rodea la curva. Si esta prueba falla, el método volvió a medir cuerdas."""
    m = M.measure(arco(180)[0])
    assert m.method.startswith("geodesico")
    assert m.geodesic_px > m.feret_px * 1.3


def test_en_particula_compacta_gana_feret():
    """En un círculo el camino más largo es el recto, y el geodésico no debe
    inflarlo rodeando nada."""
    m = M.measure(circulo()[0])
    assert m.method.startswith("Feret")


def test_el_geodesico_no_se_usa_en_particula_gruesa():
    """La condición de delgadez.

    Sin ella, un grumo con una concavidad daba un largo mayor que su propia
    extensión porque el camino la bordeaba. Se observó en una partícula real de
    44 px de extensión a la que se le atribuían 73.
    """
    mascara, real = grumo_con_muesca()
    m = M.measure(mascara)
    assert m.method.startswith("Feret"), (
        "un grumo grueso no debe medirse por el camino geodésico")
    assert m.length_px <= real * 1.10


# ── Lo que se descartó, y no debe volver ───────────────────────────────
def test_el_rectangulo_equivalente_no_sirve_como_talla():
    """Un borde dentado infla el perímetro sin alargar la partícula, y el
    modelo de rectángulo lo traduce en talla. Por eso no se reporta como tal."""
    mascara, real = recta_dentada()
    m = M.measure(mascara)
    assert m.length_rect_eq_px > real * 1.15, (
        "si esto falla, el caso dentado dejó de ser representativo")
    assert abs(m.length_px - real) / real < 0.06, (
        "el largo reportado no debe seguir al rectángulo equivalente")


def test_el_rectangulo_equivalente_no_existe_en_una_compacta():
    """Para un círculo P² = 4πA < 16A, así que el discriminante es negativo."""
    m = M.measure(circulo()[0])
    assert m.length_rect_eq_px == 0.0


# ── Forma y clasificación ──────────────────────────────────────────────
def test_el_circulo_tiene_aspecto_uno():
    """El ancho es el grosor máximo inscrito y no área/largo: con área/largo un
    círculo perfecto daba aspecto 1.27 y la clasificación arrancaba sesgada."""
    m = M.measure(circulo()[0])
    assert m.aspect == pytest.approx(1.0, abs=0.25)
    assert m.morphotype == M.FRAGMENT


def test_una_barra_larga_se_clasifica_como_fibra():
    m = M.measure(recta(largo=200, grosor=10)[0])
    assert m.aspect > M.ASPECT_FIBER
    assert m.morphotype == M.FIBER


def test_la_curvatura_distingue_recta_de_arco():
    assert M.measure(recta()[0]).curvature == pytest.approx(1.0, abs=0.05)
    assert M.measure(arco(180)[0]).curvature > 1.2


# ── Segmentación ───────────────────────────────────────────────────────
def _foto_con_particula(color=(60, 210, 240), fondo=18):
    """Una partícula brillante sobre fondo oscuro, como bajo Nile Red."""
    img = np.full((200, 200, 3), fondo, np.uint8)
    cv2.circle(img, (100, 100), 22, color, -1)
    return img


class _Caja:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


def test_la_mascara_no_se_traga_el_fondo():
    """El umbral sale del contraste anillo/caja, no de Otsu sobre el recorte.

    Con Otsu, la mediana de la máscara que era fondo oscuro llegaba al 5.4 % y
    en un 7.4 % de las partículas pasaba del 20 %, inflando el área.
    """
    img = _foto_con_particula()
    x1, y1, x2, y2 = 78, 78, 122, 122
    mascara = M.segment(img, x1, y1, x2, y2)
    assert mascara is not None
    # La máscara viene en el marco del RECORTE, no de la foto: hay que
    # desplazarla con el mismo margen que usó segmentar. Se pide al módulo en
    # vez de repetir el número, que es como una prueba deja de probar lo que
    # cree que prueba.
    margen = M.box_margin(x2 - x1, y2 - y1)
    ox, oy = max(0, x1 - margen), max(0, y1 - margen)
    gris = img.max(axis=2)[oy:oy + mascara.shape[0], ox:ox + mascara.shape[1]]
    dentro = gris[mascara > 0]
    assert dentro.min() > 100, "hay píxeles de fondo dentro de la máscara"


def test_la_mascara_no_se_escapa_de_la_caja():
    """Sin el recorte a la caja, una partícula pegada a otra forma con ella una
    sola componente conexa y se mide el conjunto."""
    img = np.full((200, 300, 3), 18, np.uint8)
    cv2.circle(img, (100, 100), 20, (60, 210, 240), -1)
    cv2.circle(img, (150, 100), 20, (60, 210, 240), -1)   # vecina pegada
    cv2.rectangle(img, (118, 95), (134, 105), (60, 210, 240), -1)  # puente
    mascara = M.segment(img, 80, 80, 120, 120)
    assert mascara is not None
    ys, xs = np.nonzero(mascara)
    ancho = xs.max() - xs.min() + 1
    assert ancho < 40 * 1.6, (
        f"la máscara mide {ancho} px de ancho para una caja de 40")


def _foto_dos_pegadas(d=60, solape=8, lienzo=340):
    """Dos partículas brillantes en contacto, y la caja de la primera."""
    img = np.full((lienzo, lienzo, 3), 18, np.uint8)
    c1 = (lienzo // 2 - d // 2 + solape // 2, lienzo // 2)
    c2 = (lienzo // 2 + d // 2 - solape // 2, lienzo // 2)
    cv2.circle(img, c1, d // 2, (60, 210, 240), -1)
    cv2.circle(img, c2, d // 2, (60, 210, 240), -1)
    caja = (c1[0] - d // 2, c1[1] - d // 2, c1[0] + d // 2, c1[1] + d // 2)
    return img, caja, float(d)


@pytest.mark.parametrize("solape", [2, 6, 10, 16])
def test_separa_dos_particulas_en_contacto(solape):
    """Dos partículas que se tocan forman una sola componente conexa, y medirlas
    juntas suma sus tallas. Se separan por watershed sobre la transformada de
    distancia: cada una tiene su núcleo, y el corte cae por el cuello."""
    img, caja, real = _foto_dos_pegadas(60, solape)
    m = M.measure(M.segment(img, *caja))
    assert m.ok
    error = 100.0 * abs(m.length_px - real) / real
    assert error < 8.0, (
        f"con solape {solape} px mide {m.length_px:.1f} en vez de {real:.0f} "
        f"({error:.1f} % de error): no se separaron")


def test_no_parte_una_particula_sana():
    """El riesgo contrario: cortar donde no hay nada que cortar. Ninguna de las
    formas de una sola pieza debe verse afectada por la separación."""
    for nombre, fabrica in (("circulo", circulo), ("recta", recta),
                            ("arco 180", lambda: arco(180)),
                            ("dentada", recta_dentada)):
        mascara, real = fabrica()
        ys, xs = np.nonzero(mascara)
        semilla = ((xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2)
        antes = int(np.count_nonzero(mascara))
        despues = int(np.count_nonzero(M.separate_glued(mascara, semilla)))
        assert despues == antes, (
            f"{nombre}: la separación recortó {antes - despues} px de una "
            f"partícula de una sola pieza")


def test_una_particula_aislada_no_cambia():
    """Control de la tubería completa: una partícula sola mide lo mismo con la
    separación puesta que sin ella."""
    img = np.full((340, 340, 3), 18, np.uint8)
    cv2.circle(img, (170, 170), 30, (60, 210, 240), -1)
    m = M.measure(M.segment(img, 140, 140, 200, 200))
    assert m.ok
    assert m.length_px == pytest.approx(60.0, abs=2.0)


def test_medir_deteccion_nunca_lanza():
    """Un fallo en una partícula no puede tumbar un lote de 552 recortes."""
    m = M.measure_detection(None, _Caja(0, 0, 10, 10))
    assert m.ok is False
    img = _foto_con_particula()
    m = M.measure_detection(img, _Caja(-50, -50, -10, -10))
    assert m.ok is False


def test_marca_para_revisar_la_talla_implausible():
    """Una partícula no puede ser mucho más larga que la diagonal de su caja
    salvo que esté muy enroscada; lo más frecuente es que sean dos pegadas."""
    img = np.full((300, 300, 3), 18, np.uint8)
    ang = np.deg2rad(np.linspace(0, 300, 300))
    pts = np.c_[150 + 60 * np.cos(ang), 150 + 60 * np.sin(ang)].astype(np.int32)
    cv2.polylines(img, [pts], False, (60, 210, 240), 7)
    m = M.measure_detection(img, _Caja(110, 110, 150, 150))
    if m.ok and m.length_px > 1.5 * np.hypot(40, 40):
        assert m.review, "una talla implausible debe quedar marcada"


# ── Calibración a micrómetros ──────────────────────────────────────────
def test_las_tallas_escalan_con_la_calibracion():
    mascara, _ = recta()
    a = M.measure(mascara, um_por_px=10.0)
    b = M.measure(mascara, um_por_px=20.0)
    assert b.length_um == pytest.approx(2 * a.length_um, rel=1e-6)
    assert b.area_um2 == pytest.approx(4 * a.area_um2, rel=1e-6)


def test_sin_calibracion_no_hay_micrometros():
    m = M.measure(recta()[0])
    assert m.length_um is None and m.area_um2 is None
    assert m.length_px > 0, "en píxeles sí debe medir"


# ── Cajas sin particula ───────────────────────────────────────────────

def test_una_caja_sin_contraste_no_devuelve_mascara():
    """Sin particula que aislar no se inventa una talla.

    Aqui se caia a Otsu, que parte el histograma del RUIDO y devuelve una
    mascara fragmentada con una talla de aspecto normal -- indistinguible de un
    dato bueno salvo mirando la imagen. Se veia en la hoja de contacto: fondo
    oscuro y un contorno en zigzag con los extremos de Feret en esquinas
    opuestas.

    Medido sobre 100 cajas anotadas a mano, 7 caian por ahi, y las 7 eran de
    las dos placas anotadas en el marco de pixeles equivocado: la caja señalaba
    un sitio sin particula.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    # Ruido uniforme: ni particula ni fondo, solo grano.
    img = (18 + rng.normal(0, 2.0, (200, 200, 3))).clip(0, 255).astype(np.uint8)
    assert M.segment(img, 80, 80, 120, 120) is None


def test_una_particula_tenue_pero_visible_si_se_mide():
    """El otro lado: el corte no puede llevarse particulas de verdad.

    En los datos reales hay un salto claro -- las cajas vacias dan contraste
    2-7 y la particula real mas tenue da 34 --, asi que un contraste de 40 tiene
    que medirse sin problema.
    """
    import numpy as np
    img = np.full((200, 200, 3), 18, np.uint8)
    cv2.circle(img, (100, 100), 15, (18, 58, 58), -1)   # +40 sobre el fondo
    mascara = M.segment(img, 80, 80, 120, 120)
    assert mascara is not None
    assert M.measure(mascara).ok
