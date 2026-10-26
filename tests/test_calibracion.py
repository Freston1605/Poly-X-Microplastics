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
    cv2.circle(img, centro, radio, (210, 205, 200), 9)   # el aro
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


# ── Índice de calibración ──────────────────────────────────────────────
def test_lee_el_indice_de_recortes(tmp_path):
    """En un recorte la placa no aparece, así que su escala solo puede
    heredarse de la foto de la que salió."""
    csv = tmp_path / "indice.csv"
    csv.write_text("recorte,origen,escala,um_por_px\n"
                   "a__f0c0.jpg,a.jpg,1.05,31.2815\n"
                   "a__f0c1.jpg,a.jpg,1.05,31.2815\n", encoding="utf-8")
    mapa = C.carry_index(csv)
    assert mapa["a__f0c0.jpg"] == pytest.approx(31.2815)


def test_lee_tambien_el_indice_de_placas(tmp_path):
    csv = tmp_path / "calibracion_placas.csv"
    csv.write_text("placa,testigo,um_por_px\n1.1.jpg,T1,32.8647\n", encoding="utf-8")
    assert C.carry_index(csv)["1.1.jpg"] == pytest.approx(32.8647)


def test_un_indice_que_no_existe_no_revienta(tmp_path):
    assert C.carry_index(tmp_path / "no_existe.csv") == {}


def test_busca_el_indice_subiendo_carpetas(tmp_path):
    """Se busca en vez de exigir la ruta porque señalarlo a mano es el paso que
    se olvida, y olvidarlo no da error: da tallas con la escala equivocada."""
    hondo = tmp_path / "recortes" / "testigo"
    hondo.mkdir(parents=True)
    (tmp_path / "recortes" / "indice.csv").write_text(
        "recorte,um_por_px\nx.jpg,31.5\n", encoding="utf-8")
    assert C.search_index(hondo / "x.jpg")["x.jpg"] == pytest.approx(31.5)


# ── Orden de preferencia ───────────────────────────────────────────────
def test_el_indice_manda_sobre_todo(tmp_path):
    img, _ = _foto_de_placa()
    foto = tmp_path / "1.1.jpg"
    cv2.imwrite(str(foto), img)
    cal = C.resolver(foto, index={"1.1.jpg": 31.0}, um_por_px_manual=99.0,
                     measure_plate=True, bgr=img)
    assert cal.origin == C.ORIGIN_INDEX
    assert cal.um_por_px == pytest.approx(31.0)


def test_medir_la_placa_manda_sobre_el_valor_manual(tmp_path):
    """El valor manual es uno solo para todo el lote, y en este material la
    escala real varía 1.59× entre fotos."""
    img, esperado = _foto_de_placa()
    cal = C.resolver(tmp_path / "x.jpg", index={}, um_por_px_manual=99.0,
                     measure_plate=True, bgr=img)
    assert cal.origin == C.ORIGIN_PLATE
    assert cal.um_por_px == pytest.approx(esperado, rel=0.03)


def test_el_manual_es_el_ultimo_recurso(tmp_path):
    cal = C.resolver(tmp_path / "x.jpg", index={}, um_por_px_manual=31.85,
                     measure_plate=False)
    assert cal.origin == C.ORIGIN_MANUAL
    assert cal.um_por_px == pytest.approx(31.85)


def test_sin_nada_queda_sin_calibrar(tmp_path):
    cal = C.resolver(tmp_path / "x.jpg", index={}, um_por_px_manual=0.0,
                     measure_plate=False)
    assert not cal.valid
    assert cal.origin == C.ORIGIN_NONE


# ── Resumen del lote ───────────────────────────────────────────────────
def test_el_resumen_declara_la_dispersion():
    """Si las fotos no comparten escala, un histograma de tallas hecho sobre
    todas juntas mezcla unidades. El informe tiene que poder decirlo."""
    cals = [C.Calibration(31.3, C.ORIGIN_PLATE), C.Calibration(49.7, C.ORIGIN_PLATE)]
    r = C.summary_batch(cals)
    assert r["n"] == 2
    assert r["variacion"] == pytest.approx(49.7 / 31.3, rel=1e-9)
    assert r["origenes"] == {C.ORIGIN_PLATE: 2}


def test_el_resumen_ignora_las_no_validas():
    r = C.summary_batch([C.Calibration(), C.Calibration(32.0, C.ORIGIN_PLATE)])
    assert r["n"] == 1


def test_el_resumen_de_un_lote_vacio_no_revienta():
    assert C.summary_batch([])["n"] == 0
