"""Inyeccion muerta sobre el render del pipeline real (T06).

El escape de T05 (_esc = html.escape(str(v), quote=False) en cada sumidero
de polyx/core/report_html.py) tiene que demostrarse sobre el RENDER, no solo
sobre las funciones internas: este test construye el estado hostil de
tests/fixtures_informe.py, lo pasa por generate_report (el mismo seam que
usa la aplicacion) con TODAS las secciones, y comprueba que:

  1. NINGUNA de las dos inyecciones de referencia llega viva al HTML: no
     existe <img src=x onerror=alert(N)> crudo ni <script> en el documento.
  2. La inyeccion SI esta, escapada como texto visible (el < se convierte en
     &lt;), para que el usuario vea el dato que se quiso colar en vez de un
     silencio que lo oculte.
  3. El texto hostil reaparece UNA Y SOLO UNA vez por sumidero: cada sitio
     conocido se ubica por su contexto estructural (la marca de
     tabla/heading que lo contiene) y la suma de los conteos por sumidero
     agota el total de apariciones de la cadena escapada; si alguna
     ocurrencia no cuadra con un sumidero documentado, la suma no cierra y
     el test avisa.

Las dos inyecciones de referencia distinguen sumideros sin ambiguedad (P001):
el ALIAS del modelo lleva alert(1) y aparece en 5 sumideros (encabezado,
tabla de metodos, texto de la figura de calibracion, tabla de conteo y
caption de comparacion); el NOMBRE DE ARCHIVO lleva alert(2) y aparece en 5
sumideros (tabla de conteo, titulo h3 de la seccion por imagen, 2 filas de
la ficha de articulo y 1 fila de la tabla de tamano). El nombre no lleva '/'
a proposito: pathlib la partiria en componentes y el '<' de apertura se
perderia de Path.name (P001).

El idioma se fija a 'en' con la misma mecanica del fixture autouse de
tests/test_render_reporte_benigno.py: polyx.core.i18n cachea el idioma al
importar (env -> ~/.polyx_idioma.json -> locale del sistema) y sin forzarlo
los conteos dependen de la maquina. Los anclajes estructurales de los
sumideros (td, strong, h3, comillas «») salen de las plantillas del informe
mismo, no de datos de usuarios, y con el idioma fijado son estables: si una
reforma de report_html mueve una marca de plantilla, este test delimita
exactamente que sumidero muto.
"""
from __future__ import annotations

import importlib
import logging
import os
from html import escape as _hesc
from pathlib import Path

import pytest

# Inyecciones de referencia: misma pareja de cadenas que define
# tests/fixtures_informe.py (P001). Cada una vive en un grupo de sumideros
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
    ("tabla de conteo (fila por modelo)",
     f"<td>{ALIAS_ESCAPADO}</td><td class='r'>", 1),
    ("caption de comparacion", f"<strong>{ALIAS_ESCAPADO}</strong>", 1),
]

# Sumideros del nombre de archivo. La ficha de articulo renderiza la misma
# celda 2 veces (2 filas de estadisticas por imagen, con marca identica),
# por eso espera 2; los demas sitios aportan 1 con el fixture de 1 imagen.
SUMIDEROS_NOMBRE: list[tuple[str, str, int]] = [
    ("ficha de articulo (filas de estadisticas)",
     f"<td style='font-size:9pt'>{NOMBRE_ESCAPADO}</td></tr>", 2),
    ("tabla de tamano (fila por imagen)",
     f"<td style='font-size:9pt'>{NOMBRE_ESCAPADO}</td><td>1</td>", 1),
    ("tabla de conteo (fila por imagen)",
     f"<td>{NOMBRE_ESCAPADO}</td><td class='r'>", 1),
    ("titulo h3 de la seccion por imagen", f"<h3>{NOMBRE_ESCAPADO}</h3>", 1),
]


@pytest.fixture(autouse=True)
def _idioma_y_logs_muertos(tmp_path):
    """Fuerza el idioma 'en' y revierte el logger del pipeline a NullHandler.

    Misma mecanica que tests/test_render_reporte_benigno.py: i18n cachea el
    idioma al importar y sin POLYX_IDIOMA los conteos de este test son
    maquina-dependientes. El logger 'polyx.detector' queda cacheado por
    logger_para con un RotatingFileHandler a carpeta_logs(); si no se
    controla, la suite escribiria en el HOME real y variaria segun el orden
    de los tests. Se fija POLYX_LOG_DIR a tmp_path antes del primer
    logger_para. Los estados de fixture no usan QObjects y no emiten
    eventos: no se pierde nada al silenciar.
    """
    import polyx.core.i18n as i18n

    previo_idioma = os.environ.get("POLYX_IDIOMA")
    previo_log_dir = os.environ.get("POLYX_LOG_DIR")
    os.environ["POLYX_IDIOMA"] = "en"
    os.environ["POLYX_LOG_DIR"] = str(tmp_path / "logs")

    importlib.reload(i18n)
    import polyx.core.report_html as rh
    importlib.reload(rh)
    import polyx.core.observabilidad as obs

    logger = obs.logger_para("detector")
    handlers_previos = logger.handlers[:]
    logger.handlers = [logging.NullHandler()]

    yield

    logger.handlers = handlers_previos
    if previo_idioma is None:
        os.environ.pop("POLYX_IDIOMA", None)
    else:
        os.environ["POLYX_IDIOMA"] = previo_idioma
    if previo_log_dir is None:
        os.environ.pop("POLYX_LOG_DIR", None)
    else:
        os.environ["POLYX_LOG_DIR"] = previo_log_dir
    importlib.reload(i18n)
    importlib.reload(rh)


def _render_hostil(tmp_path: Path) -> str:
    """Render the hostile state through the real pipeline, return the HTML.

    Usa generate_report (el mismo seam que la aplicacion y que T03-T04) con
    el conjunto completo de secciones: la prueba es sobre el render, no
    sobre fixtures recortados.
    """
    import polyx.core.report_html as rh
    from tests.fixtures_informe import estado_hostil

    st = estado_hostil(directorio=tmp_path / "fotos")
    salida = tmp_path / "informe.html"
    rh.generate_report(st, salida, secciones=list(rh.IDS_SECCIONES))
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
            f"el escape de report_html no cubrio algun sumidero")


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
        "del informe, no solo se neutralizo")
    for marcador in ("alert(1)", "alert(2)"):
        assert marcador in html_text, (
            f"la inyeccion escapada ({marcador}) no esta en el HTML como "
            f"texto visible")


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
            ("nombre de archivo", SUMIDEROS_NOMBRE,
             html_text.count(NOMBRE_ESCAPADO))):
        suma = 0
        for nombre, ventana, esperado in sumideros:
            n = html_text.count(ventana)
            assert n == esperado, (
                f"sumidero {cadena_escape} '{nombre}': {n} apariciones "
                f"(esperadas {esperado}); la plantilla de ese sumidero mutó "
                f"o un sitio nuevo esta inyectando la misma ventana")
            suma += n
        assert suma == total, (
            f"cadena hostil '{cadena_escape}': la suma de sumideros "
            f"({suma}) no agota el total de apariciones ({total}); "
            f"hay ocurrencias sin sumidero documentado")
