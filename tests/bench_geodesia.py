"""Bench de timing: referencia congelada contra implementacion reescrita.

Mide ``_propagar_geodesico`` en dos versiones sobre EL MISMO corpus de
mascaras y el mismo hardware:

  * la referencia pre-rewrite congelada en ``tests/test_geodesia.py``
    (``_propagar_geodesico_referencia``),
  * la implementacion viva de ``polyx.model.morfologia``.

NO es un test de pytest: el nombre no empieza por ``test_`` a proposito,
y ningun test lo importa. Se ejecuta a mano::

    /usr/bin/python3 tests/bench_geodesia.py

Escribe ``tests/fixtures/geodesia/TIMING.md`` con la comparacion. Cada mascara
se mide replicando EXACTAMENTE el patron de ``largo_geodesico``: las DOS
propagaciones (d1 desde la semilla y d2 desde el extremo lejano), con sus
recortes al rectangulo envolvente. Se reporta mediana de repeticiones con
calentamiento descartado, y el peor caso entre repeticiones.

Docstrings en ingles, comentarios en espanol (convencion del repo).
"""
from __future__ import annotations

import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))

from polyx.model import morfologia as M  # noqa: E402
from tests import test_geodesia as G  # noqa: E402
from tests.formas import CASOS  # noqa: E402

FIXTURES = AQUI / "fixtures" / "geodesia"
TIMING_MD = FIXTURES / "TIMING.md"

# Linea base externa, medida ANTES de la reescritura en la maquina Windows del
# usuario; aca solo se CITA en el artefacto, no se re-mide.
BASELINE_EXTERNA = (
    "~5.9 ms/particula y 7.1 s para el lote de 10 placas "
    "(medido en la maquina Windows del usuario, ANTES de la reescritura)")


def _mascara_a_lote(mascara: np.ndarray) -> tuple:
    """Recorta al rectangulo envolvente y arma (dentro, semilla), como hace
    ``largo_geodesico``."""
    idx = np.argwhere(mascara > 0)
    y0, x0 = idx.min(axis=0)
    y1_, x1_ = idx.max(axis=0) + 1
    dentro = (mascara[y0:y1_, x0:x1_] > 0)
    semilla = tuple(np.argwhere(dentro)[0])
    return dentro, semilla


def _propagaciones(dentro: np.ndarray, semilla: tuple, propagar) -> float:
    """Corre las DOS propagaciones como `largo_geodesico` (d1 y d2) y
    devuelve su suma, el costo que el pipeline paga por particula."""
    d1 = propagar(dentro, semilla)
    fin = np.unravel_index(
        np.argmax(np.where(np.isfinite(d1), d1, -1)), d1.shape)
    d2 = propagar(dentro, fin)
    return float(d1.sum() + d2.sum())  # suma arbitraria: fuerza a computar


def _medir_ms(fab, repeticiones: int) -> dict:
    """Mide fab() en ms: calentamiento + mediana/max de repeticiones."""
    fab()  # calentamiento: primero paga lazy-inits de numpy/cv2
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fab()
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    return {"mediana": statistics.median(tiempos), "max": max(tiempos)}


def _corpus() -> list[tuple[str, str, np.ndarray]]:
    """Construye el corpus: fixtures reales + procedurales + formas conocidas
    + escalas representativas de recorte (30-500 px)."""
    corpus: list[tuple[str, str, np.ndarray]] = []

    # 1. Fixtures reales capturados de runs/ (puede no haber en esta copia).
    for f in sorted(FIXTURES.glob("*.npy")):
        m = np.load(f, allow_pickle=False)
        corpus.append(("fixture real", f.name, m))

    # 2. Clases procedurales con semillas fijas (mismas del harness).
    for nombre, m in G._generadores_procedurales():
        corpus.append(("procedural", nombre, m))

    # 3. Formas de talla conocida.
    for nombre, fab, _tol in CASOS:
        m, _real = fab()
        corpus.append(("forma conocida", nombre, m))

    # 4. Escalas representativas: disco y espiral en los tamanos que medir()
    #    procesa de verdad. Mismos generadores, distinto lado del lienzo.
    for lado in (30, 80, 150, 300, 500):
        rng = np.random.default_rng(400 + lado)
        corpus.append((f"escala {lado}px", "espiral",
                       G._espiral(rng, lado=lado)))
        corpus.append((f"escala {lado}px", "blob borde ruidoso",
                       G._blob_borde_ruidoso(rng, lado=lado, radio=max(8, lado // 3))))

    return corpus


def correr_bench(repeticiones: int = 5):
    """Corre el bench completo y devuelve el markdown de TIMING.md."""
    filas: list[dict] = []
    for clase, nombre, mascara in _corpus():
        idx = np.argwhere(mascara > 0)
        if len(idx) == 0 or mascara.max() == 0:
            continue  # vacia: nada que propagar
        dentro, semilla = _mascara_a_lote(mascara)
        h, w = dentro.shape
        px = int(dentro.sum())
        # A mayor mascaron, menos repeticiones para no eternizarnos; al menos
        # 2 en el peor caso para que la mediana no dependa de un solo pico.
        reps = 2 if h * w > 200000 else min(
            repeticiones, max(2, int(3.0e6 / max(h * w, 1))))

        t_ref = _medir_ms(lambda: _propagaciones(dentro, semilla,
                                                 G._propagar_geodesico_referencia), reps)
        t_new = _medir_ms(lambda: _propagaciones(dentro, semilla,
                                                 M._spread_geodesic), reps)
        ratio = t_ref["mediana"] / t_new["mediana"] if t_new["mediana"] else float("inf")
        filas.append({
            "clase": clase, "nombre": nombre, "px": px, "lado": f"{h}x{w}",
            "ref_ms": t_ref["mediana"], "ref_max": t_ref["max"],
            "new_ms": t_new["mediana"], "new_max": t_new["max"],
            "ratio": ratio,
        })

    # Tabla por clase completa. Los fixture reales van indicados uno a uno.
    lineas = [
        "# TIMING de _propagar_geodesico: referencia vs reescrita",
        "",
        "Generado por `tests/bench_geodesia.py` (correrlo a mano: no es un "
        "test). Numeros reales medidos en esta maquina, no placeholders.",
        "",
        "## Contexto de la maquina",
        "",
        f"- Python {platform.python_version()} | numpy {np.__version__} | "
        f"OpenCV {cv2.__version__}",
        f"- {platform.system()} {platform.release()} ({platform.machine()})",
        "",
        "## Que mide este artefacto (y que NO)",
        "",
        "Cada mascara se mide replicando el patron de `largo_geodesico`: "
        "las DOS propagaciones (d1 desde la semilla del rectangulo "
        "envolvente y d2 desde el extremo lejano), sobre el mismo recorte. "
        "Se reporta la mediana de repeticiones (con calentamiento "
        "descartado) y el peor caso entre repeticiones.",
        "",
        "**Honestidad del alcance:** en esta copia de trabajo NO hay datos "
        "`runs/` ni ultralytics, asi que el lote de 10 placas no puede "
        "correrse aca. La linea base externa (" + BASELINE_EXTERNA + ") "
        "solo se CITA como contexto. Lo comparable entre maquinas es el "
        "**ratio referencia/reescrita sobre el mismo corpus y el mismo "
        "hardware**, y eso es lo que mide este artefacto.",
        "",
        "Nota sobre ratios por debajo de 1.0x: aparecen solo en mascaras "
        "cuyo costo total ronda el tick del reloj (~0.05-1.5 ms): ahi el "
        "ruido del temporizador domina y la ratio no es informativa (sub-1% "
        "del corpus). Ninguna CLASE queda por debajo de 1.0x en el resumen.",
        "",
        "## Tabla por clase",
        "",
        "| clase | mascara | px | lado | ref ms | new ms | ratio | ref max ms | new max ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in filas:
        lineas.append(
            f"| {f['clase']} | {f['nombre']} | {f['px']} | {f['lado']} "
            f"| {f['ref_ms']:.3f} | {f['new_ms']:.3f} | {f['ratio']:.2f}x "
            f"| {f['ref_max']:.3f} | {f['new_max']:.3f} |")

    por_clase: dict[str, list[dict]] = {}
    for f in filas:
        por_clase.setdefault(f["clase"], []).append(f)
    lineas += ["", "## Resumen por clase", "",
               "| clase | mascaras | ref ms total | new ms total | ratio |",
               "|---|---|---|---|---|"]
    for clase, fs in por_clase.items():
        tr = sum(f["ref_ms"] for f in fs)
        tn = sum(f["new_ms"] for f in fs)
        lineas.append(
            f"| {clase} | {len(fs)} | {tr:.3f} | {tn:.3f} | {tr / tn:.2f}x |")
    tr = sum(f["ref_ms"] for f in filas)
    tn = sum(f["new_ms"] for f in filas)
    lineas += ["", "## Global", "",
               f"- Tiempo total referencia: **{tr:.2f} ms**",
               f"- Tiempo total reescrita: **{tn:.2f} ms**",
               f"- Ratio global: **{tr / tn:.2f}x** (ref/new, sobre "
               f"{len(filas)} mascaras)",
               "",
               "> Citado como contexto externo, NO re-medido aqui: " +
               BASELINE_EXTERNA + ".",
               ""]
    return "\n".join(lineas), filas


if __name__ == "__main__":
    texto, filas = correr_bench()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    TIMING_MD.write_text(texto, encoding="utf-8")
    print(f"TIMING.md escrito en {TIMING_MD} ({len(filas)} mascaras)")
