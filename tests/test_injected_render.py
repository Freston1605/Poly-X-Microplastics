"""Hostile input must die on the real pipeline render (T06).

The escaping from T05 (``_esc = html.escape(str(v), quote=False)`` at every
sink in ``polyx/core/report_html.py``) has to be proven on the real RENDER,
not only on the internal functions: this test builds the hostile state from
``tests/report_fixtures.py``, passes it through ``generate_report`` (the same
seam the application uses) with EVERY section, and checks that:

  1. NEITHER of the two reference injections reaches the HTML alive: no raw
     ``<img src=x onerror=alert(N)>`` or ``<script>`` in the document.
  2. The injection IS there, escaped as visible text (``<`` becomes
     ``&lt;``), so the user sees the data someone tried to smuggle in instead
     of a silence that hides it.
  3. The hostile text reappears ONCE AND ONLY ONCE per sink: each known site
     is located by its structural context (the table/heading mark that
     contains it) and the per-sink counts add up to the total occurrences of
     the escaped string; if any occurrence does not match a documented sink,
     the sum does not close and the test fails.

The two reference injections separate sinks without ambiguity (P001): the
model ALIAS carries ``alert(1)`` and appears in 5 sinks (header, methods
table, calibration figure text, count table and comparison caption); the FILE
NAME carries ``alert(2)`` and appears in 5 sinks (count table, per-image
section h3, 2 rows of the particle card and 1 row of the size table). The name
deliberately has no ``/``: ``pathlib`` would split it into components and the
opening ``<`` would be lost from ``Path.name`` (P001).

The language is pinned to 'en' before i18n is imported (env -> saved preference
-> system locale); without the pin the counts in this test depend on the
machine. The structural anchors of the sinks (``td``, ``strong``, ``h3``, the
guillemets) come from the report templates themselves, not from user data, and
are stable with the language pinned: if a ``report_html`` reform moves a
template mark, this test pinpoints exactly which sink changed.
"""

from __future__ import annotations

import importlib
import logging
import os
from html import escape as _hesc
from pathlib import Path

import pytest

# Inyecciones de referencia: misma pareja de cadenas que define
# tests/report_fixtures.py (P001). Cada una vive en un grupo de sumideros
# distinto y por eso llevan alert(1)/alert(2) distintos.
ALIAS_HOSTIL: str = "<img src=x onerror=alert(1)>"
NOMBRE_FOTO_HOSTIL: str = "<img src=x onerror=alert(2)>.png"

# La forma que esas cadenas toman en el HTML cuando el escape de T05 las
# pasa por _esc (html.escape con quote=False): < > & se vuelven entidades y
# las comillas quedan igual (P002: el contenido del informe no va entre
# atributos, escapar comillas romperia la identidad de bytes con nombres
# benignos con apostrofes).
ALIAS_ESCAPADO: str = _hesc(ALIAS_HOSTIL, quote=False)
NOMBRE_ESCAPADO: str = _hesc(NOMBRE_FOTO_HOSTIL, quote=False)

# Sumidero del alias: (nombre descriptivo, ventana estructural, apariciones
# esperadas). La ventana es "el texto de plantilla que rodea la cadena
# escapada" y saca del informe (no del dato) la parte que distingue cada
# sitio. Con el fixture de 1 modelo x 1 imagen cada uno aporta exactamente 1.
SUMIDEROS_ALIAS: list[tuple[str, str, int]] = [
    ("encabezado (linea de modelos)", f"{ALIAS_ESCAPADO}</p>", 1),
    ("tabla de metodos (modelos cargados)", f"<td>{ALIAS_ESCAPADO}</td></tr>", 1),
    ("texto de la figura de calibracion", f"\u00ab{ALIAS_ESCAPADO}\u00bb", 1),
    ("tabla de conteo (fila por modelo)", f"<td>{ALIAS_ESCAPADO}</td><td class='r'>", 1),
    ("caption de comparacion", f"<strong>{ALIAS_ESCAPADO}</strong>", 1),
]

# Sumideros del nombre de archivo. La ficha de articulo renderiza la misma
# celda 2 veces (2 filas de estadisticas por imagen, con marca identica),
# por eso espera 2; los demas sitios aportan 1 con el fixture de 1 imagen.
SUMIDEROS_NOMBRE: list[tuple[str, str, int]] = [
    (
        "ficha de articulo (filas de estadisticas)",
        f"<td style='font-size:9pt'>{NOMBRE_ESCAPADO}</td></tr>",
        2,
    ),
    (
        "tabla de tamano (fila por imagen)",
        f"<td style='font-size:9pt'>{NOMBRE_ESCAPADO}</td><td>1</td>",
        1,
    ),
    ("tabla de conteo (fila por imagen)", f"<td>{NOMBRE_ESCAPADO}</td><td class='r'>", 1),
    ("titulo h3 de la seccion por imagen", f"<h3>{NOMBRE_ESCAPADO}</h3>", 1),
]


@pytest.fixture(autouse=True)
def _idioma_y_logs_muertos(tmp_path):
    """Force 'en' and revert the pipeline logger to a NullHandler.

    The language is pinned to 'en' before i18n is imported; without
    POLYX_LOCALE the counts in this test depend on the machine. The
    'polyx.detector' logger is cached by logger_para with a RotatingFileHandler
    pointed at carpeta_logs(); without POLYX_LOG_DIR the suite would write to
    the real HOME and vary with test order. POLYX_LOG_DIR is set to tmp_path
    before the first logger_para. The fixture states use no QObjects and emit
    no events, so nothing is lost by silencing it.
    """
    import polyx.core.i18n as i18n

    previous_locale = os.environ.get("POLYX_LOCALE")
    previous_log_dir = os.environ.get("POLYX_LOG_DIR")
    os.environ["POLYX_LOCALE"] = "en"
    os.environ["POLYX_LOG_DIR"] = str(tmp_path / "logs")

    importlib.reload(i18n)
    import polyx.gui.report_html as rh

    importlib.reload(rh)
    import polyx.core.observability as obs

    logger = obs.logger_for("detector")
    handlers_previous = logger.handlers[:]
    logger.handlers = [logging.NullHandler()]

    yield

    logger.handlers = handlers_previous
    if previous_locale is None:
        os.environ.pop("POLYX_LOCALE", None)
    else:
        os.environ["POLYX_LOCALE"] = previous_locale
    if previous_log_dir is None:
        os.environ.pop("POLYX_LOG_DIR", None)
    else:
        os.environ["POLYX_LOG_DIR"] = previous_log_dir
    importlib.reload(i18n)
    importlib.reload(rh)


def _render_hostil(tmp_path: Path) -> str:
    """Render the hostile state through the real pipeline, return the HTML.

    Usa generate_report (el mismo seam que la aplicacion y que T03-T04) con
    el conjunto completo de secciones: la prueba es sobre el render, no
    sobre fixtures recortados.
    """
    import polyx.gui.report_html as rh
    from tests.report_fixtures import estado_hostil

    st = estado_hostil(directorio=tmp_path / "fotos")
    salida = tmp_path / "informe.html"
    rh.generate_report(st, salida, sections=list(rh.IDS_SECTIONS))
    return salida.read_text(encoding="utf-8")


def test_la_inyeccion_no_llega_viva_al_html(tmp_path):
    """Ninguna de las dos cadenas hostiles aparece cruda en el documento.

    El ataque de referencia (<img src=x onerror=alert(N)>) y un <script>
    generico son las marcas que la inyeccion activa dejaria: si cualquiera
    aparece tal cual, el escape fallo y el assert nombra la cadena exacta
    que se colo.
    """
    html_text = _render_hostil(tmp_path)
    for cruda in (ALIAS_HOSTIL, NOMBRE_FOTO_HOSTIL, "<script>"):
        assert cruda not in html_text, (
            f"la inyeccion llego viva al HTML: {cruda!r} aparece cruda; "
            f"el escape de report_html no cubrio algun sumidero"
        )


def test_la_inyeccion_esta_como_texto_escapado(tmp_path):
    """El dato hostil SI esta en el informe, pero convertido a texto.

    Inerte no significa silencio: el usuario debe ver el nombre de
    archivo/alias tal como se le cargo, con < > convertidos en entidades.
    Si el escape desapareciese, la cadena cruda la caza
    test_la_inyeccion_no_llega_viva_al_html; si un sumidero se tragara el
    dato por completo, el conteo por sumidero de
    test_cada_sumidero_hostil_aparece_una_vez_y_solo_una lo delimita.
    """
    html_text = _render_hostil(tmp_path)
    assert "&lt;img src=x" in html_text, (
        "el HTML no lleva la cadena hostil escapada: el dato hostil se perdio "
        "del informe, no solo se neutralizo"
    )
    for marcador in ("alert(1)", "alert(2)"):
        assert marcador in html_text, (
            f"la inyeccion escapada ({marcador}) no esta en el HTML como texto visible"
        )


def test_cada_sumidero_hostil_reaparece_una_vez_por_sumidero(tmp_path):
    """La cadena hostil reaparece una y solo una vez por sumidero.

    Cada sumidero documentado se cuenta por su ventana estructural con el
    conteo esperado (1, o 2 cuando la plantilla renderiza la misma celda en
    dos filas). Despues se comprueba que la suma de los conteos agota el
    total de apariciones de la cadena escapada: si hay mas ocurrencias que
    sumideros reconocidos, algo nuevo esta llegando sin pasar por los sitios
    documentados y hay que mapearlo.
    """
    html_text = _render_hostil(tmp_path)

    for cadena_escape, sumideros, total in (
        ("alias", SUMIDEROS_ALIAS, html_text.count(ALIAS_ESCAPADO)),
        ("nombre de archivo", SUMIDEROS_NOMBRE, html_text.count(NOMBRE_ESCAPADO)),
    ):
        suma = 0
        for nombre, ventana, esperado in sumideros:
            n = html_text.count(ventana)
            assert n == esperado, (
                f"sumidero {cadena_escape} '{nombre}': {n} apariciones "
                f"(esperadas {esperado}); la plantilla de ese sumidero mutó "
                f"o un sitio nuevo esta inyectando la misma ventana"
            )
            suma += n
        assert suma == total, (
            f"cadena hostil '{cadena_escape}': la suma de sumideros "
            f"({suma}) no agota el total de apariciones ({total}); "
            f"hay ocurrencias sin sumidero documentado"
        )
