"""Que el contorno verde caiga SOBRE la particula, no al lado.

Este fallo ya se corrigio una vez, en ``dibujar_medicion``, y volvio por la
puerta de al lado: ``_ficha_particula`` hace lo mismo y tenia su propio margen
copiado a mano --  ``0.35 * min(lado)`` frente al ``0.50 * max(lado)`` real de
``margen_de_caja()``. Dos origenes distintos para la misma mascara, asi que el
contorno salia desplazado sobre la foto. Se vio a ojo en el informe, en el
apartado "la particula mayor, medida".

Por eso la prueba es de COMPORTAMIENTO y no de estructura, y recorre las dos
funciones: comprueba que el verde dibujado se solape con la particula
brillante. Si aparece una tercera copia con su propio margen, cae aqui.
"""

from __future__ import annotations

import base64
import re

import cv2
import numpy as np
import pytest

from polyx.core import morphology as M
from polyx.core.detect.protocol import Instance as Detection
from polyx.gui.report_html import _record_particle


def _foto_con_particula(tmp_path, cx=300, cy=200, r=34):
    """Una particula brillante sola, lejos del centro del recorte.

    Descentrada a proposito: con la particula en el centro, un desplazamiento
    del contorno se disimula y la prueba pasaria estando roto.
    """
    img = np.full((420, 620, 3), 18, np.uint8)
    cv2.circle(img, (cx, cy), r, (60, 210, 240), -1)
    ruta = tmp_path / "placa.png"
    cv2.imwrite(str(ruta), img)
    d = Detection(
        class_id=0, class_name="PET", conf=0.9, x1=cx - r, y1=cy - r, x2=cx + r, y2=cy + r
    )
    d.number = 1
    M.apply_to_detection(d, img, 30.0)
    return img, ruta, d


def _centroides(vis):
    """Centroide del verde dibujado y de la particula brillante, en px."""
    b, g, rr = vis[:, :, 0].astype(int), vis[:, :, 1].astype(int), vis[:, :, 2].astype(int)
    # El contorno se dibuja en verde puro; la particula es amarilla (R y G altos).
    verde = (g > 150) & (b < 90) & (rr < 90)
    brillante = (g > 120) & (rr > 120)
    if not verde.any() or not brillante.any():
        return None, None
    ys, xs = np.nonzero(verde)
    yb, xb = np.nonzero(brillante)
    return (xs.mean(), ys.mean()), (xb.mean(), yb.mean())


def test_dibujar_medicion_pone_el_contorno_sobre_la_particula(tmp_path):
    img, _, d = _foto_con_particula(tmp_path)
    par, m = M.draw_measurement(img, d, zoom=1)
    assert par is not None and m.ok
    vis = par[:, par.shape[1] // 2 :]
    c_verde, c_part = _centroides(vis)
    assert c_verde is not None, "no se dibujo el contorno"
    desvio = float(np.hypot(c_verde[0] - c_part[0], c_verde[1] - c_part[1]))
    assert desvio < 4, f"el contorno esta {desvio:.1f} px fuera de la particula"


def test_la_ficha_del_informe_pone_el_contorno_sobre_la_particula(tmp_path):
    """La copia que se habia desincronizado.

    Con el margen viejo (0.35 del lado menor frente al 0.50 del lado mayor) el
    contorno salia desplazado unos 10 px sobre una particula de 68 px.
    """
    _, ruta, d = _foto_con_particula(tmp_path)
    html = _record_particle(d, ruta, 30.0, "1 prueba")
    assert html, "la ficha no se genero"
    m64 = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
    assert m64, "la ficha no lleva imagen"
    datos = np.frombuffer(base64.b64decode(m64.group(1)), dtype=np.uint8)
    par = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    assert par is not None

    vis = par[:, par.shape[1] // 2 :]
    c_verde, c_part = _centroides(vis)
    assert c_verde is not None, "no se dibujo el contorno"
    # La ficha se reescala para el informe, asi que el desvio se mide en
    # fraccion del ancho y no en pixeles absolutos.
    desvio = float(np.hypot(c_verde[0] - c_part[0], c_verde[1] - c_part[1]))
    assert desvio < 0.06 * vis.shape[1], (
        f"el contorno esta {desvio:.0f} px fuera de la particula "
        f"(ancho {vis.shape[1]} px): el recorte y la mascara no comparten origen"
    )


@pytest.mark.parametrize("lado_x,lado_y", [(90, 30), (30, 90), (60, 60)])
def test_el_margen_del_recorte_es_el_mismo_en_los_dos_sitios(lado_x, lado_y):
    """La causa raiz, dicha en una linea.

    Cualquier funcion que dibuje una mascara de segmentar() sobre su foto tiene
    que recortar con margen_de_caja(). Se comprueba con cajas de proporciones
    distintas, que es donde min y max se separan.
    """
    import inspect

    from polyx.gui import report_html

    fuente = inspect.getsource(report_html._record_particle)
    assert "margen_de_caja(" in fuente, (
        "la ficha volvio a calcularse su propio margen; tiene que usar "
        "margen_de_caja() o el contorno saldra corrido"
    )
    # Y que ese margen no sea el que se usaba antes.
    assert M.box_margin(lado_x, lado_y) == int(max(3, round(0.50 * max(lado_x, lado_y))))
