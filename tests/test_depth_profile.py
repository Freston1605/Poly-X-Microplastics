"""La tendencia con la profundidad, contra casos de respuesta conocida.

El tramo no es una carpeta cualquiera: es una variable ORDENADA, y por eso se
puede preguntar si el conteo o la talla CRECEN con la profundidad, no solo si
los grupos difieren. Esa afirmacion va a un paper, asi que la estadistica se
comprueba contra series construidas: una monotona, una plana y una con ruido.

La correlacion se calcula sobre TRAMOS y no sobre particulas. Dos particulas de
la misma placa no son observaciones independientes de la profundidad; contarlas
como tales daria p diminutos sin ninguna base, que es el error clasico de este
tipo de analisis (pseudorreplicacion).
"""

from __future__ import annotations

import numpy as np
import pytest

from polyx.core import stats


@pytest.fixture(autouse=True)
def _en_español():
    """Fija el idioma para las pruebas que miran el TEXTO de la frase.

    Sin esto la prueba depende de lo que el usuario tenga guardado en
    ~/.polyx_idioma.json, y falla en una maquina configurada en ingles sin que
    haya nada roto. Ya paso.
    """
    import importlib
    import os

    previous = os.environ.get("POLYX_LOCALE")
    os.environ["POLYX_LOCALE"] = "es"
    import polyx.core.i18n as i18n

    importlib.reload(i18n)
    importlib.reload(stats)
    yield
    if previous is None:
        os.environ.pop("POLYX_LOCALE", None)
    else:
        os.environ["POLYX_LOCALE"] = previous
    importlib.reload(i18n)
    importlib.reload(stats)


# ── Spearman ──────────────────────────────────────────────────────────


def test_una_serie_que_crece_da_correlacion_uno():
    assert stats._spearman([1, 2, 3, 4, 5], [2, 4, 9, 11, 30]) == pytest.approx(1.0)


def test_una_serie_que_decrece_da_menos_uno():
    assert stats._spearman([1, 2, 3, 4, 5], [30, 11, 9, 4, 2]) == pytest.approx(-1.0)


def test_es_de_RANGOS_y_no_le_afecta_una_cola_larga():
    """El motivo de no usar Pearson.

    El conteo de particulas tiene unas pocas placas con cientos, y con Pearson
    esa cola manda sobre el resultado. Con rangos, un valor extremo pesa lo
    mismo que cualquier otro: las dos series de abajo tienen el mismo orden y
    tienen que dar la misma correlacion.
    """
    suave = stats._spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    with_tail = stats._spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5000])
    assert suave == pytest.approx(with_tail)


def test_sin_variacion_no_hay_correlacion():
    """Una serie constante no correlaciona con nada; devolver 0 seria mentir."""
    assert stats._spearman([1, 2, 3, 4], [7, 7, 7, 7]) is None


def test_con_menos_de_tres_tramos_no_se_calcula():
    assert stats._spearman([1, 2], [4, 9]) is None


# ── p por permutacion ─────────────────────────────────────────────────


def test_una_tendencia_perfecta_sale_significativa():
    p = stats._p_permutation(list(range(1, 9)), [1, 2, 3, 4, 5, 6, 7, 8])
    assert p is not None and p < 0.05


def test_una_serie_sin_orden_no_sale_significativa():
    rng = np.random.default_rng(7)
    x = list(range(1, 13))
    y = list(rng.normal(size=12))
    p = stats._p_permutation(x, y)
    assert p is not None and p > 0.05


def test_el_p_nunca_es_cero():
    """En una prueba por permutacion, p = 0 no existe.

    De ahi el +1 en numerador y denominador. Un cero en un paper se lee como
    imposible, y lo que hubo fue un numero finito de barajadas.
    """
    p = stats._p_permutation(list(range(1, 11)), list(range(1, 11)), n_perm=200)
    assert p is not None and p > 0


def test_el_mismo_lote_da_el_mismo_p():
    """La semilla es fija: un informe tiene que ser reproducible."""
    x, y = [1, 2, 3, 4, 5, 6], [3, 1, 4, 1, 5, 9]
    assert stats._p_permutation(x, y, n_perm=500) == stats._p_permutation(x, y, n_perm=500)


def test_con_pocos_tramos_una_tendencia_debil_no_se_declara():
    """La proteccion que importa para el paper.

    Con 4 tramos casi nada puede ser significativo, y el texto no debe afirmar
    una tendencia. Si esto fallara, el informe estaria declarando hallazgos que
    los datos no sostienen.
    """
    text = stats._text_trend([1, 2, 3, 4], [10, 8, 12, 11], "el conteo")
    assert "No se detecta tendencia" in text
    assert "n = 4 tramos" in text


# ── La frase que sale en el informe ───────────────────────────────────


def test_la_frase_declara_rho_p_y_n():
    """Las tres cifras juntas o la afirmacion no se puede juzgar."""
    text = stats._text_trend(list(range(1, 10)), list(range(1, 10)), "el conteo")
    assert "ρ = +1.00" in text
    assert "p = " in text or "p < " in text
    assert "n = 9 tramos" in text


def test_la_frase_dice_en_que_sentido_va():
    rising = stats._text_trend(list(range(1, 10)), list(range(1, 10)), "el conteo")
    falling = stats._text_trend(list(range(1, 10)), list(range(9, 0, -1)), "el conteo")
    assert "crece" in rising and "decrece" not in rising
    assert "decrece" in falling
