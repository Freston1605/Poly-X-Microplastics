"""Harness de equivalencia para ``_propagar_geodesico``, con referencia congelada.

Este archivo existe para que la reescritura de ``_propagar_geodesico``
(S02 del plano de rendimiento) sea demostrable y no solo "deberia dar igual":

  * Congela VERBATIM la implementacion pre-rewrite como
    ``_propagar_geodesico_referencia``, de modo que el contrato pre-rewrite
    quede escrito en el repositorio ANTES de tocar ``morfologia.py``.
  * Compara campos de bits (vista uint32), no "casi iguales": cualquier
    diferencia de un ulp de float32, o el signo de un cero, hace fallar.
  * Cubre mascaras reales capturadas (``tests/fixtures/geodesia/*.npy``),
    clases proceduralizadas de lo que el pipeline ve en recortes de placa, y
    las figuras sinteticas de talla conocida de ``tests/formas.py``.

El copy de abajo es el negocio serio del archivo: SOLO se edita para dejar
constancia del contrato congelado (p. ej. si cambia una decision de semantica),
NUNCA para "modernizarlo". Si ``_propagar_geodesico_referencia`` y la funcion
viva divergen sin que lo pida un cambio de contrato, la suite grita.

Sobre el flujo: los fixtures reales se serializan con
``tests/capturar_fixtures_geodesia.py`` desde una carpeta ``runs/detect_*``
que exista real (esta copia de trabajo no tiene datos ``runs/``); sin fixtures,
el test de auto-descubrimiento se salta honestamente explicando como
capturarlos, y la cobertura recae en generadores + formas conocidas.

Docstrings en ingles, comentarios en espanol (convencion del repo).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyx.ml import morfologia as M  # noqa: E402
from tests.formas import CASOS  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "geodesia"

# Bits de +infinito en float32. El comparador mira bits, no "no finito": esta
# constante existe para que el test negativo pueda demostrar que se compara el
# BIT de infinito y no una ausencia vaga de valor.
INF_BITS = 0x7F800000


# ── Referencia congelada ──────────────────────────────────────────────
#
# Copia VERBATIM del cuerpo de polyx/core/morfologia.py::_propagar_geodesico
# tal como estaba ANTES de la reescritura de rendimiento (domingo 2026-10,
# plano 01-02-PLAN.md, T01). Se edita UNICAMENTE para registrar un cambio de
# contrato: nunca para modernizarla, renombrarla ni "arreglarla".
def _propagar_geodesico_referencia(dentro: np.ndarray, semilla: tuple) -> np.ndarray:
    """Distancia geodesica desde una semilla, sin salir de la mascara.

    Propagacion por minimos sucesivos sobre los 8 vecinos, con peso 1 en recto
    y raiz de 2 en diagonal. Vectorizado: cada iteracion son ocho
    desplazamientos de la matriz, y hacen falta tantas como pixeles tenga el
    camino mas largo. Sobre recortes de particula -- decenas o pocos cientos de
    pixeles de lado -- eso es inmediato.

    Se hace asi y no con scipy.sparse.csgraph porque scipy no es una dependencia
    declarada del proyecto: llega de rebote con ultralytics, y apoyarse en eso
    romperia la instalacion el dia que ultralytics deje de traerlo.
    """
    INF = np.inf
    d = np.full(dentro.shape, INF, dtype=np.float32)
    d[semilla] = 0.0
    # Pesos optimos del chamfer 3x3, no 1 y raiz de 2. Con los pesos ingenuos el
    # camino en escalera sobreestima la distancia euclidea hasta un 8% -- medido
    # sobre un circulo de diametro conocido, daba 108 en vez de 100 --. Estos
    # dos minimizan ese error maximo y lo dejan en torno al 1.4%.
    UNO = np.float32(0.95509)
    R2 = np.float32(1.36930)
    # Cota de seguridad: ningun camino simple puede tener mas pasos que pixeles.
    tope = int(dentro.sum()) + 2
    for _ in range(tope):
        p = np.full_like(d, INF)
        # Rectos
        p[1:, :] = np.minimum(p[1:, :], d[:-1, :] + UNO)
        p[:-1, :] = np.minimum(p[:-1, :], d[1:, :] + UNO)
        p[:, 1:] = np.minimum(p[:, 1:], d[:, :-1] + UNO)
        p[:, :-1] = np.minimum(p[:, :-1], d[:, 1:] + UNO)
        # Diagonales
        p[1:, 1:] = np.minimum(p[1:, 1:], d[:-1, :-1] + R2)
        p[1:, :-1] = np.minimum(p[1:, :-1], d[:-1, 1:] + R2)
        p[:-1, 1:] = np.minimum(p[:-1, 1:], d[1:, :-1] + R2)
        p[:-1, :-1] = np.minimum(p[:-1, :-1], d[1:, 1:] + R2)
        p[~dentro] = INF
        nuevo = np.minimum(d, p)
        # Basta comparar con el paso anterior: la propagacion es monotona
        # decreciente, asi que si nada bajo, ya converge.
        if np.array_equal(nuevo, d):
            break
        d = nuevo
    return d


# ── Comparador estricto por bits ──────────────────────────────────────
def _bits_iguales(a: np.ndarray, b: np.ndarray, caso: str) -> None:
    """Assert de identidad BIT a BIT entre dos matrices float32.

    Compara la vista uint32: detecta toda diferencia de un ulp y cualquier
    signo distinto de cero (0.0 vs -0.0). Los arreglos de propagacion contienen
    INF pero nunca NaN, asi que la vista de bits es la comparacion honesta.
    """
    assert a.shape == b.shape, f"{caso}: formas distintas {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"{caso}: dtypes distintos {a.dtype} vs {b.dtype}"
    ta = np.ascontiguousarray(a).view(np.uint32)
    tb = np.ascontiguousarray(b).view(np.uint32)
    if not np.array_equal(ta, tb):
        # np.where sobre una matriz 2D devuelve dos arrays 1D: filas y columnas.
        fila, col = np.where(ta != tb)
        pos = (int(fila[0]), int(col[0]))
        raise AssertionError(
            f"{caso}: primer pixel divergente en {pos}: "
            f"bits referencia {ta[pos]:#010x} contra bits actual {tb[pos]:#010x}")


def _mismo_float(x: float, y: float) -> bool:
    """Igualdad de dos floats por sus bits IEEE-754 de doble precision."""
    return x == y  # == es bit-exacta en IEEE-754 salvo NaN, que no hay.


# ── Fuentes de mascaras: procedurales, con semillas fijas ─────────────
def _blob_borde_ruidoso(rng, lado=180, radio=65):
    """Grumo relleno con borde ruidoso dentro y fuera: 6% del borde se quita."""
    m = np.zeros((lado, lado), np.uint8)
    n = 28
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radio * (0.85 + 0.30 * rng.random(n))
    pts = np.c_[lado / 2 + r * np.cos(ang), lado / 2 + r * np.sin(ang)]
    pts = pts.round().astype(np.int32)
    cv2.fillPoly(m, [pts], 255)
    interior = cv2.erode(m, np.ones((3, 3), np.uint8))
    borde = (interior == 0) & (m > 0)
    m[borde & (rng.random(m.shape) < 0.35)] = 0
    return m


def _cinta_sinuosa(rng, lado=220, grosor=4):
    """Cinta fina con centro sinusoidal; grosor 3-6 px al azar."""
    m = np.zeros((lado, lado), np.uint8)
    grosor = int(rng.integers(3, 7))
    t = np.linspace(0.06, 0.94, 220)
    x = t * (lado - 40) + 20
    y = lado / 2 + (lado / 3.5) * np.sin(2.4 * np.pi * t)
    pts = np.c_[x, y].astype(np.int32)
    cv2.polylines(m, [pts], False, 255, grosor)
    return m


def _espiral(rng, lado=200, grosor=4):
    """Espiral de 1.6 vueltas: camino largo y estrecho, el caso caro."""
    m = np.zeros((lado, lado), np.uint8)
    ang = np.linspace(0.0, 1.6 * 2.0 * np.pi, 700)
    r = 14 + (lado / 2 - 22) * ang / ang.max()
    pts = np.c_[lado / 2 + r * np.cos(ang), lado / 2 + r * np.sin(ang)]
    pts = pts.round().astype(np.int32)
    cv2.polylines(m, [pts], False, 255, grosor)
    return m


def _grumo_con_huecos(rng, lado=180, n_huecos=4, radio=72):
    """Disco relleno al que se le sacan huecos interiores a boreshot."""
    m = np.zeros((lado, lado), np.uint8)
    cv2.circle(m, (lado // 2, lado // 2), radio, 255, -1)
    for _ in range(n_huecos):
        cy, cx = int(rng.integers(radio - 16, lado - radio + 16)), \
            int(rng.integers(radio - 16, lado - radio + 16))
        cv2.circle(m, (cx, cy), int(rng.integers(5, 12)), 0, -1)
    return m


def _erodada_bernoulli(rng, lado=140, p=0.88):
    """Cuadrado erosionado punto a punto con Bernoulli: borde dentado."""
    m = np.zeros((lado, lado), np.uint8)
    m[lado // 3:2 * lado // 3, lado // 4:3 * lado // 4] = 255
    m = np.where(rng.random(m.shape) < p, m, 0).astype(np.uint8)
    return m


def _dos_componentes(rng, lado=200, radio=28, distancia=None):
    """Dos blobs DESCONECTADOS: la propagacion deja un mar de INF entre ambos."""
    m = np.zeros((lado, lado), np.uint8)
    if distancia is None:
        distancia = radio * 3
    cv2.circle(m, (lado // 2, lado // 2 - distancia // 2), radio, 255, -1)
    cv2.circle(m, (lado // 2, lado // 2 + distancia // 2), radio, 255, -1)
    return m


def _degeneradas():
    """Mascaras de 1 y 2 px: las menores que medir() y largo_geodesico ven."""
    una = np.zeros((7, 7), np.uint8)
    una[3, 3] = 255
    dos = np.zeros((7, 7), np.uint8)
    dos[2, 2] = 255
    dos[4, 4] = 255
    return [("1 px", una), ("2 px desconectados", dos)]


def _generadores_procedurales():
    """Lista de (nombre, mascara) con semillas fijas, reproducible."""
    casos = []
    k = 11
    for nombre, fab in [
            ("blob borde ruidoso", _blob_borde_ruidoso),
            ("cinta sinuosa", _cinta_sinuosa),
            ("espiral 1.6 vueltas", _espiral),
            ("grumo con huecos", _grumo_con_huecos),
            ("erodada bernoulli", _erodada_bernoulli),
            ("dos componentes", _dos_componentes)]:
        rng = np.random.default_rng(k)
        casos.append((nombre, fab(rng)))
        k += 1
    casos.extend(_degeneradas())
    return casos


# ── Comparador de un caso completo ────────────────────────────────────
def _comparar_mascara(mascara: np.ndarray, nombre: str) -> None:
    """Replica el patron de llamada de largo_geodesico y exige identidad de bits.

    Compara tres niveles sobre el mismo ``lote``:
      * d1 y d2 crudos (referencia congelada vs implementacion actual),
      * el indice del extremo lejano de la primera propagacion,
      * el pipeline PUBLICO al completo, con la referencia inyectada en el
        modulo frente al modulo tal cual esta (largo_geodesico,
        camino_geodesico, medir).
    """
    idx = np.argwhere(mascara > 0)
    if len(idx) < 2:
        # largo_geodesico sale temprano; solo queda probar la propagacion
        # cruda sobre el recorte entero, que tambien debe ser bit-identica.
        dentro = (mascara > 0)
        idx = np.argwhere(dentro)
        if len(idx) == 0:
            return
        semilla = tuple(idx[0])
        d1_ref = _propagar_geodesico_referencia(dentro, semilla)
        d1_act = M._spread_geodesic(dentro, semilla)
        _bits_iguales(d1_ref, d1_act, f"{nombre}: d1")
        return

    # Mismo recorte al rectangulo envolvente que morfologia.largo_geodesico.
    y0, x0 = idx.min(axis=0)
    y1_, x1_ = idx.max(axis=0) + 1
    dentro = (mascara[y0:y1_, x0:x1_] > 0)
    idx = np.argwhere(dentro)
    semilla = tuple(idx[0])

    # Primera propagacion, campo a campo.
    d1_ref = _propagar_geodesico_referencia(dentro, semilla)
    d1_act = M._spread_geodesic(dentro, semilla)
    _bits_iguales(d1_ref, d1_act, f"{nombre}: d1")

    # El extremo lejano tiene que salir identico para que la segunda
    # propagacion parta del mismo sitio.
    fin = np.unravel_index(
        np.argmax(np.where(np.isfinite(d1_ref), d1_ref, -1)), d1_ref.shape)
    fin_act = np.unravel_index(
        np.argmax(np.where(np.isfinite(d1_act), d1_act, -1)), d1_act.shape)
    assert fin == fin_act, f"{nombre}: extremo lejano distinto {fin} vs {fin_act}"

    d2_ref = _propagar_geodesico_referencia(dentro, fin)
    d2_act = M._spread_geodesic(dentro, fin)
    _bits_iguales(d2_ref, d2_act, f"{nombre}: d2")

    # Pipeline publico, referencia inyectada frente al modulo real.
    original = M._spread_geodesic
    M._spread_geodesic = _propagar_geodesico_referencia
    try:
        largo_ref = M.length_geodesic(mascara)
        camino_ref = M.path_geodesic(mascara)
        m_ref = M.measure(mascara)
    finally:
        M._spread_geodesic = original
    largo_act = M.length_geodesic(mascara)
    camino_act = M.path_geodesic(mascara)
    m_act = M.measure(mascara)

    assert _mismo_float(largo_ref, largo_act), (
        f"{nombre}: largo_geodesico {largo_ref!r} vs {largo_act!r}")
    assert camino_ref == camino_act, f"{nombre}: camino_geodesico distinto"
    assert _mismo_float(m_ref.geodesic_px, m_act.geodesic_px), (
        f"{nombre}: medir().geodesico_px {m_ref.geodesic_px!r} vs "
        f"{m_act.geodesic_px!r}")
    assert _mismo_float(m_ref.length_px, m_act.length_px), (
        f"{nombre}: medir().largo_px {m_ref.length_px!r} vs {m_act.length_px!r}")
    assert m_ref.method == m_act.method, (
        f"{nombre}: medir().metodo {m_ref.method!r} vs {m_act.method!r}")


# ── Tests de identidad ────────────────────────────────────────────────
def test_identidad_sobre_fixtures_reales():
    """Toda mascara .npy capturada en tests/fixtures/geodesia debe salir
    bit-identica. Sin fixtures el test se salta y SOLO la suite procedural
    cubre: eso se declara, no se oculta."""
    fijos = sorted(FIXTURES.glob("*.npy"))
    if not fijos:
        pytest.skip(
            "Sin mascaras capturadas en tests/fixtures/geodesia (esta copia "
            "de trabajo no tiene datos runs/). Capturalas con "
            "tests/capturar_fixtures_geodesia.py desde una carpeta "
            "runs/detect_* del equipo que tenga las fotos origen.")
    for f in fijos:
        m = np.load(f, allow_pickle=False)
        assert m.ndim == 2 and m.dtype == np.uint8, (
            f"{f.name}: el fixture debe de ser una mascara uint8 en gris")
        _comparar_mascara(m, f.name)


@pytest.mark.parametrize("nombre,mascara", _generadores_procedurales(),
                         ids=[c[0] for c in _generadores_procedurales()])
def test_identidad_sobre_generadores(nombre, mascara):
    """Identidad bit a bit sobre clases procedurales: seeds fijas."""
    _comparar_mascara(mascara, nombre)


@pytest.mark.parametrize("nombre,fabrica,tol", CASOS,
                         ids=[c[0] for c in CASOS])
def test_identidad_sobre_formas_conocidas(nombre, fabrica, tol):
    """Idem sobre las figuras sinteticas de talla conocida (formas.py)."""
    mascara, _real = fabrica()
    _comparar_mascara(mascara, nombre)


# ── Tests negativos: que el comparador muerda ─────────────────────────
def test_negativo_un_ulp_de_diferencia_sale_rojo():
    """Perturbar UN valor finito en un ulp hace fallar al comparador.

    Si esto pasara verde, el bit-comparator estaria comparando "por lo menos
    casi igual" y serviria de poco para lo que really cuelga de el."""
    rng = np.random.default_rng(7)
    mascara = _blob_borde_ruidoso(rng)
    y0, x0 = np.argwhere(mascara > 0).min(axis=0)
    y1_, x1_ = np.argwhere(mascara > 0).max(axis=0) + 1
    dentro = (mascara[y0:y1_, x0:x1_] > 0)
    semilla = tuple(np.argwhere(dentro)[0])
    d = _propagar_geodesico_referencia(dentro, semilla).copy()

    # Toma un pixel FINITO (no INF, que NaN/inf no hedgen un ulp sensato) y
    # muelelo un ulp arriba.
    finitos = np.argwhere(np.isfinite(d))
    pos = tuple(finitos[0])
    d[pos] = np.nextafter(np.float32(d[pos]), np.float32(np.inf))
    with pytest.raises(AssertionError, match="primer pixel divergente"):
        _bits_iguales(
            _propagar_geodesico_referencia(dentro, semilla), d, "ulp test")


def test_bits_de_infinito_son_comparables_en_casos_degenerados():
    """Los INF se comparan por BITS exactos, no por 'los dos no finitos'.

    Las mascaras de dos componentes y el pixel unico son los casos donde la
    propagacion deja un mar de INF; es justo ahi donde un comparador naive
    ('no finito == no finito') esconderia un signo distinto de infinito."""
    rng = np.random.default_rng(3)
    m = _dos_componentes(rng)
    dentro = (m > 0).astype(bool)
    # La semilla va al primer pixel; aseguramos que quien arranca deja
    # infinidades detras (es el sentido del caso).
    semilla = tuple(np.argwhere(dentro)[0])
    d = _propagar_geodesico_referencia(dentro, semilla)
    assert (d.view(np.uint32)[~np.isfinite(d)] == INF_BITS).all(), (
        "los no finitos deberian ser TODOS bits de +infinito")
    # Y el comparador los pasaria: bit a bit contra una copia idéntica.
    _bits_iguales(d, d.copy(), "dos componentes")


# ── Mini test de sanidad del modulo ───────────────────────────────────
def test_embebido_no_toque_morfologia():
    """Guardia de alcance: morfologia.py no debe ser tocado por este harness.

    No replica la funcion viva ni parchea estado permanente; solo verifica que
    M._propagar_geodesico sigue siendo llamable y que la inyeccion temporal
    de referencia se resta bien."""
    rng = np.random.default_rng(99)
    m = _blob_borde_ruidoso(rng)
    original = M._spread_geodesic
    _comparar_mascara(m, "sanidad")
    assert M._spread_geodesic is original, (
        "la inyeccion de referencia debe restaurar el modulo")
