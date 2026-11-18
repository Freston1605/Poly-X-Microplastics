"""La escala en micrómetros por píxel, y de dónde sale.

Toda talla del estudio pasa por aquí: si la escala se desplaza, se desplazan
todas las cifras del paper a la vez y de forma coherente, que es el peor modo de
fallo posible porque nada parece roto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyx.core import calibration as C


def _foto_de_placa(radio=300, lienzo=900, diametro_mm=100.0):
    """Una placa Petri sintética: anillo brillante sobre fondo oscuro.

    Devuelve (imagen, µm/px que debería medirse).
    """
    img = np.full((lienzo, lienzo, 3), 12, np.uint8)
    centro = (lienzo // 2, lienzo // 2)
    cv2.circle(img, centro, radio, (210, 205, 200), 9)  # el aro
    cv2.circle(img, centro, radio - 40, (60, 55, 90), -1)  # el interior
    return img, diametro_mm * 1000.0 / (2 * radio)


# ── Medida del anillo ──────────────────────────────────────────────────
def test_mide_la_placa_sin_que_nadie_la_marque():
    img, esperado = _foto_de_placa(radio=300)
    cal = C.calibrate_from_plate(img, 100.0)
    assert cal.valid
    assert cal.origin == C.ORIGIN_PLATE
    assert cal.um_por_px == pytest.approx(esperado, rel=0.03)
    assert cal.n_points > 100, "el ajuste debe apoyarse en muchos puntos"


def test_la_escala_escala_con_el_radio():
    """Una placa que se ve más pequeña en el encuadre da más µm por píxel.

    Los dos radios se eligen dentro del rango en que Hough busca la placa
    (0.22 a 0.52 del lado menor), que es donde caen las placas reales: en este
    material ocupan entre el 26 % y el 49 % del lado. Un radio justo en el borde
    del rango hace fallar la búsqueda y probaría otra cosa.
    """
    grande = C.calibrate_from_plate(_foto_de_placa(radio=350)[0], 100.0)
    pequena = C.calibrate_from_plate(_foto_de_placa(radio=250)[0], 100.0)
    assert grande.valid and pequena.valid
    assert pequena.um_por_px > grande.um_por_px
    assert pequena.um_por_px / grande.um_por_px == pytest.approx(350 / 250, rel=0.06)


def test_el_diametro_nominal_entra_lineal():
    """Confundir una Petri de 90 mm con una de 100 desplaza todo un 11 %."""
    img, _ = _foto_de_placa(radio=300)
    a = C.calibrate_from_plate(img, 100.0)
    b = C.calibrate_from_plate(img, 90.0)
    assert b.um_por_px / a.um_por_px == pytest.approx(0.9, rel=1e-6)


def test_sin_placa_no_inventa_una_escala():
    vacia = np.full((400, 400, 3), 12, np.uint8)
    cal = C.calibrate_from_plate(vacia, 100.0)
    assert not cal.valid
    assert cal.warning, "debe explicar por qué no pudo"


def test_no_calibra_con_datos_absurdos():
    img, _ = _foto_de_placa()
    assert not C.calibrate_from_plate(img, 0.0).valid
    assert not C.calibrate_from_plate(None, 100.0).valid


# ── Corrección de paralaje ─────────────────────────────────────────────
def test_el_paralaje_agranda_la_escala():
    """El aro está más cerca de la cámara que la base, de modo que la escala
    medida sobre él es demasiado pequeña y las tallas salen cortas."""
    base, factor = C.correct_parallax(32.0, height_plate_mm=15, distance_camera_mm=300)
    assert factor == pytest.approx(300 / 285, rel=1e-9)
    assert base > 32.0


def test_sin_los_dos_datos_no_se_corrige():
    """Es preferible una escala sin corregir y declarada a una corregida con
    números inventados."""
    assert C.correct_parallax(32.0, 0, 300) == (32.0, 1.0)
    assert C.correct_parallax(32.0, 15, 0) == (32.0, 1.0)


def test_no_se_corrige_con_la_camara_dentro_de_la_placa():
    """Distancia menor que la altura es un dato mal tomado, no un caso límite."""
    assert C.correct_parallax(32.0, 15, 10) == (32.0, 1.0)


def test_el_paralaje_viene_desactivado():
    """Medido sobre este material el efecto es indetectable (< 1.3 %), así que
    de fábrica no se aplica: corregir por defecto sería inventar."""
    img, esperado = _foto_de_placa()
    cal = C.calibrate_from_plate(img, 100.0)
    assert cal.factor_parallax == 1.0
    assert cal.um_por_px == pytest.approx(esperado, rel=0.03)


# ── Orden de preferencia ───────────────────────────────────────────────
def test_medir_la_placa_manda_sobre_el_valor_manual(tmp_path):
    """El valor manual es uno solo para todo el lote, y en este material la
    escala real varía 1.59× entre fotos."""
    img, esperado = _foto_de_placa()
    cal = C.resolver(tmp_path / "x.jpg", um_por_px_manual=99.0, measure_plate=True, bgr=img)
    assert cal.origin == C.ORIGIN_PLATE
    assert cal.um_por_px == pytest.approx(esperado, rel=0.03)


def test_el_manual_es_el_ultimo_recurso(tmp_path):
    cal = C.resolver(tmp_path / "x.jpg", um_por_px_manual=31.85, measure_plate=False)
    assert cal.origin == C.ORIGIN_MANUAL
    assert cal.um_por_px == pytest.approx(31.85)


def test_sin_nada_queda_sin_calibrar(tmp_path):
    cal = C.resolver(tmp_path / "x.jpg", um_por_px_manual=0.0, measure_plate=False)
    assert not cal.valid
    assert cal.origin == C.ORIGIN_NONE


# ── Resumen del lote ───────────────────────────────────────────────────
def test_el_resumen_declara_la_dispersion():
    """Si las fotos no comparten escala, un histograma de tallas hecho sobre
    todas juntas mezcla unidades. El informe tiene que poder decirlo."""
    cals = [C.Calibration(31.3, C.ORIGIN_PLATE), C.Calibration(49.7, C.ORIGIN_PLATE)]
    r = C.summary_batch(cals)
    assert r["n"] == 2
    assert r["variation"] == pytest.approx(49.7 / 31.3, rel=1e-9)
    assert r["origins"] == {C.ORIGIN_PLATE: 2}


def test_el_resumen_ignora_las_no_validas():
    r = C.summary_batch([C.Calibration(), C.Calibration(32.0, C.ORIGIN_PLATE)])
    assert r["n"] == 1


def test_el_resumen_de_un_lote_vacio_no_revienta():
    assert C.summary_batch([])["n"] == 0
