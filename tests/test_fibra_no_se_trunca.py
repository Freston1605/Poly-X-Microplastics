"""Que una fibra no salga cortada por la caja del detector ni por el watershed.

Este fallo se vio en el informe, a ojo, sobre una fibra real de la placa 10.3:
el contorno medido terminaba en un borde RECTO, y una fibra no termina en
recto. Reconstruido paso a paso, se perdia asi:

    componente conexa .... 369 px   <- la mascara correcta
    tras watershed ....... 254 px   (-31%)
    tras recortar a caja .. 174 px   (-31%)

Las dos causas son distintas y hay una prueba para cada una:

  * El watershed partia la fibra por su propio adelgazamiento. Su umbral de
    nucleo se calibro sobre CIRCULOS pegados; una fibra con dos zonas gruesas
    genera dos nucleos y se corta sola.
  * El recorte a la caja usaba una holgura fija del 10%. El detector encajona
    corto las particulas alargadas -- 36 px de fibra en una caja de 29x24 --,
    asi que recortar a esa caja copiaba el error del detector.

Importa porque el largo subestimado va directo a la talla que se reporta, y
hacia abajo: una fibra cortada no se nota en las cifras, solo en la imagen.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from polyx.model import morfologia as M


# Cuanto se queda corta la caja del detector por cada extremo de una fibra.
# 0.10 = (36 - 29) / 2 / 36, medido sobre una fibra real de la placa 10.3.
RECORTE_REAL_DEL_DETECTOR = 0.10


def _foto_fibra_con_bultos(largo=120, grosor=9, bulto=7):
    """Fibra recta con dos ensanchamientos, y la caja CORTA que le pondria el
    detector.

    Los dos bultos son lo que dispara el watershed: cada uno es un maximo de la
    transformada de distancia y por tanto un nucleo. La caja se pone a
    proposito mas corta que la fibra, que es lo que hace el detector.
    """
    img = np.full((260, 320, 3), 18, np.uint8)
    y = 130
    x0 = (320 - largo) // 2
    cv2.rectangle(img, (x0, y - grosor // 2), (x0 + largo, y + grosor // 2),
                  (60, 210, 240), -1)
    # Zonas gruesas, separadas para que den dos nucleos distintos.
    cv2.circle(img, (x0 + largo // 4, y), bulto, (60, 210, 240), -1)
    cv2.circle(img, (x0 + 3 * largo // 4, y), bulto, (60, 210, 240), -1)

    # Caja del detector: corta por cada extremo en la proporcion que se
    # MIDIO sobre el material real -- una fibra de 36 px encajonada en 29 --,
    # no en la que me pareceria razonable. La diferencia importa: con un 20%
    # inventado, la unica forma de pasar la prueba era no recortar nada, y de
    # ahi salio la fuga que llego al informe.
    recorte = int(round(largo * RECORTE_REAL_DEL_DETECTOR))
    caja = (x0 + recorte, y - grosor, x0 + largo - recorte, y + grosor)
    return img, caja, float(largo)


def test_el_watershed_no_parte_una_fibra_con_dos_zonas_gruesas():
    """La causa del -31% en la fibra real. Cada bulto es un nucleo, y con eso
    el watershed la cortaba por el medio como si fueran dos particulas."""
    img, caja, _ = _foto_fibra_con_bultos()
    # Mascara antes de separar: la componente conexa entera.
    canal = img.max(axis=2)
    entera = (canal > 100).astype(np.uint8) * 255
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(entera, 8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    componente = (etiquetas == idx).astype(np.uint8) * 255
    semilla = (int(np.mean(np.nonzero(componente)[1])),
               int(np.mean(np.nonzero(componente)[0])))

    despues = M.separar_pegadas(componente.copy(), semilla)
    conserva = np.count_nonzero(despues) / max(1, np.count_nonzero(componente))
    assert conserva > 0.95, (
        f"el watershed se llevo el {100*(1-conserva):.0f}% de la fibra")


def test_la_fibra_no_se_recorta_a_la_caja_corta_del_detector():
    """La otra mitad del fallo. Con la holgura fija del 10%, la mascara salia
    cortada por un borde recto donde terminaba la caja."""
    img, caja, largo_real = _foto_fibra_con_bultos()
    mascara = M.segmentar(img, *caja)
    assert mascara is not None
    m = M.medir(mascara)
    assert m.ok
    # Se admite el margen de discretizacion habitual del metodo (~5%), pero no
    # una amputacion: antes de arreglarlo aqui salia en torno al 60% del largo.
    assert m.largo_px > largo_real * 0.90, (
        f"la fibra mide {m.largo_px:.0f} px y deberia rondar {largo_real:.0f}")


def test_una_particula_compacta_sigue_recortandose_a_su_caja():
    """La proteccion que NO se puede perder: una vecina pegada por un puente
    no debe entrar en la medida. Es el caso que motivo el recorte, y de el
    salian tallas de 8.9 mm."""
    img = np.full((200, 300, 3), 18, np.uint8)
    cv2.circle(img, (100, 100), 20, (60, 210, 240), -1)
    cv2.circle(img, (150, 100), 20, (60, 210, 240), -1)
    cv2.rectangle(img, (118, 95), (134, 105), (60, 210, 240), -1)
    mascara = M.segmentar(img, 80, 80, 120, 120)
    assert mascara is not None
    xs = np.nonzero(mascara)[1]
    assert (xs.max() - xs.min() + 1) < 40 * 1.6


def test_una_compacta_no_se_considera_alargada():
    """Si lo fuera, dejaria de recortarse y volverian las vecinas pegadas."""
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 30, 255, -1)
    assert M._elongacion(m) < M.ELONGACION_ALARGADA


def test_dos_pegadas_no_se_consideran_alargadas():
    """El caso exacto que protege el recorte: al unirse quedan rechonchas."""
    m = np.zeros((200, 300), np.uint8)
    cv2.circle(m, (100, 100), 20, 255, -1)
    cv2.circle(m, (150, 100), 20, 255, -1)
    cv2.rectangle(m, (118, 95), (134, 105), 255, -1)
    assert M._elongacion(m) < M.ELONGACION_ALARGADA


def test_una_fibra_si_se_considera_alargada():
    m = np.zeros((200, 200), np.uint8)
    cv2.rectangle(m, (40, 98), (160, 102), 255, -1)
    assert M._elongacion(m) > M.ELONGACION_ALARGADA


def test_la_elongacion_no_depende_de_la_curvatura():
    """El motivo de no usar minAreaRect: el rectangulo que envuelve un arco es
    casi cuadrado, y con el la fibra curva de la placa 10.3 se seguia
    recortando por parecer compacta."""
    recta = np.zeros((240, 240), np.uint8)
    cv2.line(recta, (40, 120), (200, 120), 255, 7)
    arco = np.zeros((240, 240), np.uint8)
    cv2.ellipse(arco, (120, 120), (80, 80), 0, 200, 340, 255, 7)
    assert M._elongacion(recta) > M.ELONGACION_ALARGADA
    assert M._elongacion(arco) > M.ELONGACION_ALARGADA, (
        f"un arco debe seguir siendo alargado, dio {M._elongacion(arco):.1f}")


def test_la_elongacion_no_depende_de_la_orientacion():
    recta = np.zeros((200, 200), np.uint8)
    cv2.rectangle(recta, (40, 98), (160, 102), 255, -1)
    diagonal = np.zeros((200, 200), np.uint8)
    cv2.line(diagonal, (50, 50), (150, 150), 255, 5)
    assert M._elongacion(recta) > M.ELONGACION_ALARGADA
    assert M._elongacion(diagonal) > M.ELONGACION_ALARGADA


def test_el_aviso_llega_a_la_deteccion():
    """El informe lee el aviso de la Detection, no de la Morfologia.

    Es una distincion facil de pasar por alto y ya costo un fallo: la
    Morfologia que devuelve dibujar_medicion() viene de medir(), y ahi el
    campo esta siempre en False -- quien lo pone es medir_deteccion(), al
    comparar el largo con la diagonal de la caja. Mirando el objeto
    equivocado, el aviso no aparecia nunca en el informe.

    El caso tiene que ser ALARGADO. Un racimo compacto no sirve: el recorte a
    la caja lo protege, no hay inflacion, y entonces no hay nada que avisar
    -- que es justamente el comportamiento correcto.
    """
    from polyx.model.yolo_wrap import Detection

    img = np.full((200, 260, 3), 18, np.uint8)
    # La barra sobresale por AMBOS lados de la caja: es el caso real, el
    # detector encajona un trozo del medio de algo mas largo.
    cv2.rectangle(img, (20, 96), (240, 104), (60, 210, 240), -1)

    d = Detection(class_id=0, class_name="PET", conf=0.9,
                  x1=100, y1=92, x2=130, y2=108)
    assert M.aplicar_a_deteccion(d, img, 30.0)
    assert d.revisar is True, "el aviso no llego a la Detection"
    assert d.aviso_forma, "y sin texto no hay nada que enseñar"


def test_una_particula_sana_no_se_marca():
    """Si se marcara todo, el aviso dejaria de significar algo."""
    from polyx.model.yolo_wrap import Detection

    img = np.full((200, 200, 3), 18, np.uint8)
    cv2.circle(img, (100, 100), 25, (60, 210, 240), -1)
    d = Detection(class_id=0, class_name="PET", conf=0.9,
                  x1=75, y1=75, x2=125, y2=125)
    assert M.aplicar_a_deteccion(d, img, 30.0)
    assert d.revisar is False


def _trozo_del_medio_de_algo_largo():
    """Una barra brillante larga, con la caja puesta en un trozo del medio.

    Es la forma del fallo real, y la unica que llega al recorte: no hay dos
    bultos, asi que el watershed no ve dos nucleos y no separa nada. La mascara
    sigue la barra hasta donde alcance el recorte de trabajo -- que es la caja
    mas medio lado mayor por banda --, y de ahi salia el tope de 2.1x que se vio
    en el informe.
    """
    img = np.full((260, 480, 3), 18, np.uint8)
    cv2.rectangle(img, (30, 114), (450, 126), (60, 210, 240), -1)
    return img, (200, 105, 240, 135)


def test_una_mascara_alargada_que_se_fuga_queda_acotada_por_la_caja():
    """La otra mitad, y la que faltaba.

    Una fuga -- la particula mas una cresta brillante que sale de la caja -- es
    larga y delgada, igual que una fibra, asi que la elongacion no las
    distingue. Cuando una mascara alargada dejaba de recortarse, la fuga no
    tenia tope: en un informe salieron tres fichas de seis midiendo entre 1.9 y
    2.1 veces la diagonal de su caja, una de ellas de 6.4 mm, por encima del
    limite de 5 mm con que se define un microplastico.

    Sobre 98 particulas anotadas a mano ninguna pasa de 1.41 veces la diagonal
    de su caja, asi que 1.6 es holgado y sigue siendo un tope.
    """
    img, caja = _trozo_del_medio_de_algo_largo()
    mascara = M.segmentar(img, *caja)
    assert mascara is not None
    m = M.medir(mascara)
    assert m.ok
    diagonal = float(np.hypot(caja[2] - caja[0], caja[3] - caja[1]))
    assert m.largo_px < 1.6 * diagonal, (
        f"la mascara mide {m.largo_px:.0f} px, {m.largo_px / diagonal:.1f}x la "
        f"diagonal de su caja ({diagonal:.0f} px): se fugo a la vecina")


def test_la_fuga_se_declara_aunque_se_haya_acotado():
    """Recortar la fuga no la vuelve un dato bueno.

    Lo que queda medido es el trozo de dentro de la caja, que puede quedarse
    corto si la particula seguia de verdad. El informe tiene que poder decirlo,
    asi que la Detection sale marcada."""
    from polyx.model.yolo_wrap import Detection

    img, caja = _trozo_del_medio_de_algo_largo()
    d = Detection(class_id=0, class_name="PET", conf=0.9,
                  x1=caja[0], y1=caja[1], x2=caja[2], y2=caja[3])
    assert M.aplicar_a_deteccion(d, img, 30.0)
    assert d.revisar is True, "una fuga acotada sigue siendo una fuga"
    assert d.aviso_forma
