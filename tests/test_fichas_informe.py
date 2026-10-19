"""Que la muestra de particulas del informe enseñe lo que hay que comprobar.

Las fichas son el unico sitio del informe donde se puede VER si la medicion
se hizo bien. Que particulas entran ahi no es un detalle de presentacion:
decide que puede auditarse y que no.

Antes entraban las primeras por orden de archivo, y con eso la particula
mayor podia no salir y las fibras no salian nunca -- son ~1% del material.
Estas pruebas fijan el reparto para que no se pierda al editar el informe.
"""
from __future__ import annotations

from polyx.gui.report_html import choose_records


class _Det:
    """Lo minimo que mira elegir_fichas de una Detection."""

    def __init__(self, largo_um, morfotipo):
        self.length_um = largo_um
        self.morphotype = morfotipo

    def __repr__(self):  # para que un fallo se lea
        return f"<{self.morphotype} {self.length_um}>"


def _lote(n_fibras: int, n_otras: int) -> list:
    """Fibras de 100,200,... y particulas de 1000,2000,... (tallas distintas
    y separadas para que el orden por tamaño sea inequivoco al comprobarlo)."""
    formas = [(_Det((i + 1) * 100.0, "fibra"), f"f{i}.jpg") for i in range(n_fibras)]
    formas += [(_Det((i + 1) * 1000.0, "fragmento"), f"p{i}.jpg") for i in range(n_otras)]
    return formas


def _cuenta(elegidas):
    fib = sum(1 for d, _ in elegidas if d.morphotype == "fibra")
    return fib, len(elegidas) - fib


# ── El reparto pedido ──
def test_con_material_de_sobra_reparte_mitad_y_mitad():
    fib, otras = _cuenta(choose_records(_lote(20, 20), 12))
    assert (fib, otras) == (6, 6)


def test_entra_la_mayor_de_cada_morfotipo():
    """Es la razon de ser del reparto: sobre la mayor se apoya cualquier
    afirmacion de talla maxima, y hay una por tipo."""
    elegidas = choose_records(_lote(20, 20), 12)
    mayor_fibra = max(d.length_um for d, _ in elegidas if d.morphotype == "fibra")
    mayor_otra = max(d.length_um for d, _ in elegidas if d.morphotype != "fibra")
    assert mayor_fibra == 20 * 100.0
    assert mayor_otra == 20 * 1000.0


def test_dentro_de_cada_grupo_manda_el_tamaño():
    elegidas = choose_records(_lote(20, 20), 12)
    fibras = [d.length_um for d, _ in elegidas if d.morphotype == "fibra"]
    assert fibras == sorted(fibras, reverse=True)
    assert min(fibras) == 15 * 100.0, "deben ser las 6 mayores, no unas cualesquiera"


# ── Lotes reales: casi nunca hay fibras suficientes ──
def test_sin_fibras_se_llena_con_particulas():
    """El caso normal de este material. No debe devolver una muestra corta."""
    fib, otras = _cuenta(choose_records(_lote(0, 30), 12))
    assert (fib, otras) == (0, 12)


def test_con_pocas_fibras_entran_todas_y_el_resto_se_completa():
    """Dos fibras en el lote: las dos salen, y las 10 restantes son particulas.
    Si esto se rompe, las fibras vuelven a desaparecer del informe."""
    fib, otras = _cuenta(choose_records(_lote(2, 30), 12))
    assert (fib, otras) == (2, 10)


def test_con_pocas_particulas_el_hueco_lo_ocupan_las_fibras():
    fib, otras = _cuenta(choose_records(_lote(30, 3), 12))
    assert (fib, otras) == (9, 3)


# ── Bordes ──
def test_un_lote_mas_chico_que_el_cupo_entra_entero():
    elegidas = choose_records(_lote(2, 3), 12)
    assert len(elegidas) == 5


def test_lote_vacio_o_cupo_cero_no_revientan():
    assert choose_records([], 12) == []
    assert choose_records(_lote(5, 5), 0) == []


def test_nunca_devuelve_mas_de_lo_pedido():
    for cupo in (1, 2, 5, 12, 40):
        assert len(choose_records(_lote(50, 50), cupo)) <= cupo


def test_no_repite_la_misma_particula():
    """El relleno de huecos podria colar dos veces la misma si se compara mal."""
    elegidas = choose_records(_lote(4, 4), 12)
    ids = [id(d) for d, _ in elegidas]
    assert len(ids) == len(set(ids))


def test_sin_talla_medida_no_se_cae():
    """Sin calibracion, largo_um queda en None. La ficha sigue siendo util
    -- la mascara se ve igual -- asi que no debe descartarse ni reventar."""
    formas = [(_Det(None, "fragmento"), "a.jpg"), (_Det(None, "fibra"), "b.jpg")]
    assert len(choose_records(formas, 12)) == 2
