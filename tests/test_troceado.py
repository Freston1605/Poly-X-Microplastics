"""Que forzar el troceo no cambie el TAMAÑO de la tesela.

Son dos preguntas distintas -- cuando trocear y de que tamaño -- y mezclarlas
costo un fallo con sintomas raros: elegir "siempre" en la interfaz dejaba la
deteccion lentisima y con CERO resultados.

La causa: ``predict_auto`` fuerza el troceo pasando ``umbral_px=0``, que es su
manera de decir "no decidas, trocea". Pero el lado de la tesela salia de ese
mismo umbral, ``min(umbral_px, imgsz)``, o sea ``min(0, 2080) = 0``, que el
clamp subia a 256. Sobre una foto de 4096x3072 son 336 teselas de 256 px, y
cada una se reescala 8x hasta imgsz: el modelo no ve nada parecido a lo que
entreno y no detecta nada.

Comprobado sobre 10.3x.jpg con el peso yolov11m: con troceo automatico da 64
detecciones sanas, y con troceo forzado daba 0.
"""
from __future__ import annotations

import pytest

from polyx.core.yolo_wrap import politica_troceado

# La foto tipica del estudio y el imgsz con que se entreno.
ANCHO, ALTO, IMGSZ = 4096, 3072, 2080


def test_forzar_el_troceo_no_encoge_la_tesela():
    """El fallo, en una linea: la tesela tiene que ser del orden de imgsz."""
    plan = politica_troceado(ANCHO, ALTO, IMGSZ, umbral_px=0, tile=0)
    assert plan is not None
    assert plan["tile"] >= IMGSZ // 2, (
        f"tesela de {plan['tile']} px: se reescalaria "
        f"{IMGSZ / plan['tile']:.0f}x hasta imgsz y el modelo no reconoceria nada")


def test_forzar_el_troceo_no_dispara_el_numero_de_teselas():
    """El otro sintoma. 336 teselas era una espera de minutos por foto."""
    plan = politica_troceado(ANCHO, ALTO, IMGSZ, umbral_px=0, tile=0)
    assert plan["n_tiles"] <= 20, f"{plan['n_tiles']} teselas es inviable por foto"


def test_forzado_y_automatico_dan_una_geometria_parecida():
    """Forzar solo debe saltarse la DECISION, no cambiar el recorte.

    Si las dos rutas dieran geometrias distintas, el numero de particulas
    dependeria de una casilla de la interfaz y no de la foto.
    """
    forzado = politica_troceado(ANCHO, ALTO, IMGSZ, umbral_px=0, tile=0)
    automatico = politica_troceado(ANCHO, ALTO, IMGSZ, umbral_px=2000, tile=0)
    assert automatico is not None
    assert forzado["n_tiles"] == automatico["n_tiles"]
    assert abs(forzado["tile"] - automatico["tile"]) <= 0.1 * automatico["tile"]


def test_un_tile_pedido_a_mano_se_respeta():
    plan = politica_troceado(ANCHO, ALTO, IMGSZ, umbral_px=0, tile=1280)
    assert plan["tile"] == 1280


@pytest.mark.parametrize("umbral", [0, 2000])
def test_una_foto_pequeña_forzada_sigue_troceandose(umbral):
    """Forzar tiene que forzar: con umbral 0 no hay foto que se libre."""
    plan = politica_troceado(1200, 900, IMGSZ, umbral_px=umbral, tile=0)
    if umbral == 0:
        assert plan is not None
    else:
        assert plan is None, "por debajo del umbral se pasa entera, sin trocear"


# Prueba rota a proposito para verificar que CI la cacha. NO MERGEAR:
# se borra en cuanto la corrida se pone roja.
def test_sabotaje_verificacion_ci():
    assert 1 == 2, "sabotaje deliberado: CI deberia fallar aqui"
