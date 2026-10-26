"""Formas sintéticas de talla conocida, compartidas por las pruebas.

Son la referencia contra la que se mide `morfologia`: se dibujan con una
longitud que se sabe de antemano, así que el error del método es comprobable
sin depender de ninguna anotación humana.

Un detalle que costó una medición: un trazo dibujado con ``polylines`` tiene los
extremos REDONDEADOS, de modo que la figura mide la línea central **más un radio
de trazo en cada punta**, es decir el grosor entero. Tomar la línea central como
verdad hacía aparecer un error del 12 % que no existía. Por eso las funciones de
aquí devuelven ya la longitud con las tapas incluidas.
"""

from __future__ import annotations

import cv2
import numpy as np

LADO = 300


def _lienzo() -> np.ndarray:
    return np.zeros((LADO, LADO), np.uint8)


def recta(largo: int = 200, grosor: int = 10):
    """Barra recta horizontal. (mascara, largo real)"""
    m = _lienzo()
    x0 = (LADO - largo) // 2
    y0 = LADO // 2
    cv2.rectangle(m, (x0, y0), (x0 + largo, y0 + grosor), 255, -1)
    return m, float(largo)


def recta_girada(grados: float = 45.0, largo: int = 200, grosor: int = 10):
    """La misma barra, girada. Su talla no debe cambiar con el giro.

    Se dibuja rotando las CUATRO ESQUINAS y rellenando el polígono, en vez de
    rotar la imagen con ``warpAffine``. Warp interpola, y al binarizar después
    los píxeles a medias el rectángulo se ensancha: media docena de grados de
    giro bastaban para que la barra midiera 210 px en vez de 200, un 5 % que era
    del generador de la figura y no del método que se quiere probar.
    """
    a = np.deg2rad(grados)
    cx, cy = LADO / 2.0, LADO / 2.0
    hl, hg = largo / 2.0, grosor / 2.0
    esquinas = np.array([(-hl, -hg), (hl, -hg), (hl, hg), (-hl, hg)])
    giro = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = (esquinas @ giro.T + (cx, cy)).round().astype(np.int32)
    m = _lienzo()
    cv2.fillPoly(m, [pts], 255)
    return m, float(largo)


def arco(grados: float, radio: int = 70, grosor: int = 10):
    """Fibra curvada como arco de circunferencia. Es el caso donde la cuerda
    subestima y tiene que entrar el camino geodésico."""
    m = _lienzo()
    ang = np.deg2rad(np.linspace(-grados / 2, grados / 2, 400))
    pts = np.c_[LADO // 2 + radio * np.cos(ang), LADO // 2 + radio * np.sin(ang)].astype(np.int32)
    cv2.polylines(m, [pts], False, 255, grosor)
    return m, float(radio * np.deg2rad(grados) + grosor)


def circulo(diametro: int = 100):
    """Fragmento compacto. No tiene rectángulo equivalente: P² < 16A."""
    m = _lienzo()
    cv2.circle(m, (LADO // 2, LADO // 2), diametro // 2, 255, -1)
    return m, float(diametro)


def recta_dentada(largo: int = 200, grosor: int = 12):
    """Barra recta con el borde mordido, que infla el perímetro sin alargarla.

    Es el contraejemplo del modelo de rectángulo: su largo no cambia, pero P sí,
    y como en ese modelo L vale aproximadamente P/2, la talla se dispara.
    """
    m = _lienzo()
    x0, y0 = (LADO - largo) // 2, LADO // 2
    cv2.rectangle(m, (x0, y0), (x0 + largo, y0 + grosor), 255, -1)
    for i in range(x0, x0 + largo, 4):
        cv2.circle(m, (i, y0), 3, 0, -1)
        cv2.circle(m, (i + 2, y0 + grosor), 3, 255, -1)
    return m, float(largo)


def grumo_con_muesca(diametro: int = 100):
    """Fragmento compacto con una concavidad.

    El camino geodésico no puede atravesar la muesca y la rodea, de modo que sin
    la condición de delgadez devolvería un largo mayor que la propia extensión
    de la partícula.
    """
    m = _lienzo()
    c = LADO // 2
    cv2.circle(m, (c, c), diametro // 2, 255, -1)
    cv2.ellipse(
        m, (c, c + diametro // 2 + 5), (diametro // 2 - 4, diametro // 2 - 8), 0, 0, 360, 0, -1
    )
    return m, float(diametro)


# (nombre, funcion, tolerancia en % sobre la talla conocida)
#
# Las tolerancias no son un numero redondo elegido a ojo: salen de medir. El
# metodo da 0.8 % de error mediano y 4.7 % en el peor caso sobre este conjunto,
# asi que un 6 % deja margen para variaciones de version de OpenCV sin dejar
# pasar una regresion de verdad. El arco de 180 grados es el mas exigente.
CASOS = [
    ("recta", lambda: recta(), 6.0),  # noqa: PLW0108
    ("recta girada 45", lambda: recta_girada(45), 6.0),
    ("arco 60", lambda: arco(60), 6.0),
    ("arco 120", lambda: arco(120), 6.0),
    ("arco 180", lambda: arco(180), 6.0),
    ("circulo", lambda: circulo(), 6.0),  # noqa: PLW0108
    ("recta dentada", lambda: recta_dentada(), 6.0),  # noqa: PLW0108
    ("grumo con muesca", lambda: grumo_con_muesca(), 6.0),  # noqa: PLW0108
]
