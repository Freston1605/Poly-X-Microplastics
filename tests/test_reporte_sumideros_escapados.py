# tests/test_reporte_sumideros_escapados.py
"""Test unitario por sumidero de escape del informe (T07).

T06 demuestra por el render completo que la inyeccion queda muerta; cada sumidero
merece ademas su prueba AISLADA, para que ante cualquier regresion el fallo diga
exactamente QUE sumidero muto y no obligue a re-leer el informe entero. Este archivo
ejercita la funcion de seccion o el bloque interno directamente (sin render completo)
con una entrada hostil por sumidero, y en cada assert comprueba:

  1. que el HTML devuelto NO contiene el markup crudo de la entrada hostil
     (``<img src=x onerror=alert(N)>`` ni ``<script>``), y
  2. que SI contiene su forma escapada por _esc (html.escape, quote=False):
     ``&lt;img src=x onerror=alert(N)&gt;``.

La pareja hostil de referencia viene de tests/fixtures_informe.py (P001): el ALIAS
del modelo lleva alert(1) y el NOMBRE de archivo alert(2), sin '/' embebida en el
nombre para que la cadena completa sobreviva a ``Path.name``. Para el CLASS_NAME
de particula (sumidero de fichas) se usa una tercera cadena con alert(3) y para el
NOMBRE DE CLASE de _tabla_conteo otra con alert(4); el numero distingue sumideros
al asertar igual que lo hace el alert(1)/alert(2) del par de P001.

Los conteos exactos por sumidero (no hace falta que T06 los cierre aqui a lo
largo de todo el render) salen de la estructura del propio codigo: si un fix
futuro agrega una interpolacion hostil que no este en esta lista, el conteo
exacto no cierra y la prueba apunta el sumidero nuevo en el mensaje. Cada seccion
del impl se renderiza via ``_generate_report_impl(secciones=[...])`` (el seam de
generate_report con cronometro en report_html.py ~2776 queda INTACTO: estas
pruebas no pasan por el), con el ancho de datos minimo que hace que esa seccion
tenga cuerpo vacio no lo caiga el ensamblador ("vivas") y el sink nunca salga a
impresion.

Estacion, tramo y placa (sumideros listados en T05 dentro de _seccion_conteo y
la galeria) NO llevan prueba hostil dedicada: el texto que puede llegarles esta
acotado por el regex de la nomenclatura (_RE_PLACA solo acepta digitos y 'x'),
de modo que no hay cadena hostil que atraviese ``_muestra_de`` hasta ellos. Su
_escape es blindaje por capas, verificado por el test de identidad de bytes del
render benigno (tests/test_render_reporte_benigno.py), no por inyeccion.

El idioma se fija a 'en' y el logger del pipeline a NullHandler con la misma
mecanica de tests/test_render_reporte_benigno.py: polyx.model.i18n cachea el
idioma al importar y sin POLYX_IDIOMA los anclajes de este archivo dependen de
la maquina; el logger 'polyx.detector' queda cacheado por logger_para con un
RotatingFileHandler a carpeta_logs() y sin POLYX_LOG_DIR la suite escribiria en
el HOME real. Los estados de fixture no usan QObjects y no emiten eventos: no
se pierde nada al silenciar.

La tabla de hardware de _seccion_equipo (filas_eq k/v) sale de _equipo() (informe
del sistema), no de entrada de usuario: sin inyeccion posible, su _esc es
defensivo. El sink de ENTRENAMIENTO (alias + i['base'] + i['optimizador'], que
LEEN del checkpoint .pt) si es de datos y se prueba aqui: como torch no esta
instalado en la suite, la fuente se aislÃ³ en el monkeypatch del propio
_entrenamiento_de; el sink probado es la interpolacion, no la lectura del .pt.
"""
from __future__ import annotations

import importlib
import logging
import os
from collections import Counter
from html import escape as _hesc
from pathlib import Path

import pytest

from polyx.gui.report_html import (
    _table_width,
    _table_count,
    _figure_calibration,
    _section_count,
    _section_team,
    _generate_report_impl,
)

# ── P001: par de cadenas hostil de referencia ──
ALIAS_HOSTIL = "<img src=x onerror=alert(1)>"
NOMBRE_FOTO_HOSTIL = "<img src=x onerror=alert(2)>.png"
# Sinks extra que no comparten concatenacion con el par de P001:
CLASE_HOSTIL = "<img src=x onerror=alert(3)>"
NOTA_HOSTIL = "<b>Fijkstra</b>"   # sink del nombre de clase C en nota_ausentes
TITULO_HOSTIL = "?<script>alert(0)</script>"


def _e(x: str) -> str:
    """Forma escapada de x segun _esc de report_html: html.escape(quote=False)."""
    return _hesc(x, quote=False)


ALIAS_ESCAPADO = _e(ALIAS_HOSTIL)
NOMBRE_ESCAPADO = _e(NOMBRE_FOTO_HOSTIL)
CLASE_ESCAPADA = _e(CLASE_HOSTIL)
NOTA_ESCAPADA = _e(NOTA_HOSTIL)
TITULO_ESCAPADO = _e(TITULO_HOSTIL)


@pytest.fixture(autouse=True)
def rh(tmp_path, monkeypatch):
    """Fuerza 'en' y revierte el logger del pipeline a NullHandler.

    Misma mecanica de tests/test_render_reporte_benigno.py: i18n cachea el
    idioma al importar y los anclajes de este archivo son literales de
    plantilla en el idioma del render; sin fuerza, el resultado dependia de
    la maquina. POLYX_LOG_DIR va a tmp_path antes del primer logger_para.
    """
    import polyx.model.i18n as i18n
    import polyx.model.observabilidad as obs

    previo_idioma = os.environ.get("POLYX_IDIOMA")
    previo_log_dir = os.environ.get("POLYX_LOG_DIR")
    monkeypatch.setenv("POLYX_IDIOMA", "en")
    monkeypatch.setenv("POLYX_LOG_DIR", str(tmp_path / "logs"))

    importlib.reload(i18n)
    import polyx.gui.report_html as rh
    importlib.reload(rh)

    logger = obs.logger_for("detector")
    handlers_previos = logger.handlers[:]
    logger.handlers = [logging.NullHandler()]

    yield rh

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


def _render_de(seccion: str, st, destino: Path) -> str:
    """Render parcial via _generate_report_impl (NO via generate_report).

    El ensamblador tumba una seccion pedida con cuerpo vacio, de modo que el
    HTML devuelto describe SOLO la seccion pedida (mas cabecera/indice), lo
    que aísla el conteo por sumidero. Devuelve el texto HTML.
    """
    salida = destino.parent / f"informe_{seccion}.html"
    ruta = _generate_report_impl(st, salida, sections=[seccion])
    return ruta.read_text(encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════
# A. _tabla_conteo — nombre de clase en la tabla por imagen (galeria, conteo)
# ════════════════════════════════════════════════════════════════════════

def test_tabla_conteo_escapa_el_nombre_de_clase():
    """alert(4) del sink de clase de _tabla_conteo: solo la primera celda."""
    det = Counter({CLASE_HOSTIL: 2, "PET": 1})
    html = _table_count(None, det, ["PET", CLASE_HOSTIL])

    assert CLASE_HOSTIL not in html, "markup crudo del sink de clase"
    assert "<script>" not in html
    assert f"<tr><td>{CLASE_ESCAPADA}</td><td>2</td></tr>" in html, \
        "celda de clase no escapada como texto"
    # El sink no muta la columna del modelo: la fija del lote va intacta.
    # Y sin doble escape: escapar la forma ya escapada daria &amp;lt;.
    assert _e(CLASE_ESCAPADA) not in html, "doble escape prohibido"
    assert f"<tr><td>PET</td><td>1</td></tr>" in html, "deteccion de PET se pierde"


def test_tabla_conteo_sin_doble_escape_benigno():
    """La clase BENIGNA no sale escapada: el sink solo toca los valores."""
    html = _table_count(None, Counter({"PET": 3}), ["PET"])
    assert "&lt;" not in html and "&gt;" not in html
    assert "<tr><td>PET</td><td>3</td></tr>" in html


# ════════════════════════════════════════════════════════════════════════
# B. _tabla_ancha — titulo de columna y cabeceras; etiquetas pasan tal cual (P002)
# ════════════════════════════════════════════════════════════════════════

def test_tabla_ancha_escapa_titulo_y_clases_y_mantiene_las_etiquetas_compuestas():
    """titulo_col y los nombre de clase se _esc'ean; la etiqueta de fila NO.

    P002: _tabla_ancha recibe la ETIQUETA compuesta ya sana del caller
    (_seccion_conteo) que lleva <br><span> deliberados — escaparla aca
    derompe el markup del informe. La prueba del sink de etiqueta vive a
    traves de _seccion_conteo (sumidero C), no aqui.
    """
    etiqueta_compuesta = ("<b>2.3x</b><br>"
                          "<span style='font-size:8.5pt;color:var(--ink3)'>"
                          "Taira &middot; tramo 2</span>")
    html = _table_width(
        [(etiqueta_compuesta, Counter({"PET": 1}), Counter({"PET": 2}))],
        ["PET", ALIAS_HOSTIL],
        TITULO_HOSTIL,
        there_is_a_manual=True,
    )

    assert TITULO_HOSTIL not in html, "titulo_col crudo"
    assert ALIAS_HOSTIL not in html, "clase hostil cruda"
    assert f"<th rowspan='2'>{TITULO_ESCAPADO}</th>" in html
    assert f"<th colspan='2' style='text-align:center'>{ALIAS_ESCAPADO}</th>" in html
    assert etiqueta_compuesta in html, \
        "P002: la etiqueta compuesta pasa VERBATIM por _tabla_ancha"
    # Y el call local de PET tampoco se escapa a mano:
    assert "<th colspan='2' style='text-align:center'>PET</th>" in html


# ════════════════════════════════════════════════════════════════════════
# C. _seccion_conteo — intro del modelo + fila por imagen
# ════════════════════════════════════════════════════════════════════════

def test_seccion_conteo_escapa_la_intro_del_modelo_y_la_fila_de_imagen(rh, tmp_path):
    """alert(1) escapa solo en la etiqueta intro de la seccion; alert(2)
    escapa solo en la celda de fila por imagen (_tabla_ancha)."""
    from tests.fixtures_informe import estado_hostil

    st = estado_hostil(directorio=tmp_path / "fotos")
    html = rh._section_count(st.results, st, st.active_models(), classes=["PET"])

    assert ALIAS_HOSTIL not in html
    assert NOMBRE_FOTO_HOSTIL not in html
    assert f"<strong>{ALIAS_ESCAPADO}</strong>" in html, \
        "el alias del modelo en la intro debe salir escapado"
    assert f"<td>{NOMBRE_ESCAPADO}</td><td class='r'>" in html, \
        "el nombre de archivo en la tabla de conteo debe salir escapado"
    # Y exactamente un sitio para cada uno (1 modelo x 1 imagen):
    assert html.count(ALIAS_ESCAPADO) == 1, "solo la intro de la seccion nombra al modelo"
    assert html.count(NOMBRE_ESCAPADO) == 1, "solo la fila por imagen"
    assert "&amp;lt;" not in html, "doble escape prohibido"


def test_seccion_conteo_con_nomenclatura_de_pauta_escapa_el_segmento_hostil(rh, tmp_path):
    """Con nombre parseable por _muestra_de la etiqueta compone estacion y
    tramo: ambos viven acotados por el regex, y el segmento hostil solo
    aparece via r.image_path.name escapado."""
    from tests.fixtures_informe import estado_hostil

    nombre_pauta = "2.3x__A&amp;B<img src=x onerror=alert(2)>.png"
    st = estado_hostil(directorio=tmp_path / "fotos", nombres=[nombre_pauta])
    html = rh._section_count(st.results, st, st.active_models(), classes=["PET"])

    assert nombre_pauta not in html
    assert _e(nombre_pauta) in html
    # La estacion del pauta "2.3x" es Desembocadura ("3" en _ESTACION):
    assert "Desembocadura &middot; tramo 2 &middot; placa b</span>" in html


# ════════════════════════════════════════════════════════════════════════
# D. _seccion_equipo — alias y datos del checkpoint .pt (_entrenamiento_de)
# ════════════════════════════════════════════════════════════════════════

def test_seccion_equipo_escapa_el_alias_y_lo_que_viene_del_checkpoint(rh, tmp_path, monkeypatch):
    """alert(1) en el alias del modelo y en base/optimizador del .pt.

    La lectura del .pt neceseita torch (no instalado en la suite); el sink
    probado es la interpolacion en la tabla de entrenamiento, no la lectura.
    Con falso encontrado:
      <tr><td>{alias}</td><td>{base}</td> --- <td>{optimizador}</td>
    """
    entrenamiento = {"base": ALIAS_HOSTIL, "imgsz": 640, "batch": 8, "epocas": 3,
                     "optimizador": ALIAS_HOSTIL, "lr0": "—", "amp": "—",
                     "precision": 0.9, "recall": 0.8, "map50": 0.7, "map": 0.6}
    monkeypatch.setattr(rh, "_training_of", lambda peso: entrenamiento)

    from polyx.gui.detector.state import ModelSlot
    slot = ModelSlot(alias=ALIAS_HOSTIL, path=Path("peso_falso.pt"))
    html = _section_team([slot])

    assert ALIAS_HOSTIL not in html, "sink de entrenamiento con markup crudo"
    # 3 apariciones: alias + base + optimizador (todas con la misma cadena):
    assert html.count(ALIAS_ESCAPADO) == 3, \
        "alias/base/optimizador del .pt deben salir escapados y sin mas"
    assert f"<tr><td>{ALIAS_ESCAPADO}</td><td>{ALIAS_ESCAPADO}</td>" in html
    assert f"<td>{ALIAS_ESCAPADO}</td><td class='r'>0.9000" in html


def test_seccion_equipo_sin_checkpoint_no_nombra_al_modelo():
    """Sin datos de .pt el sink de entrenamiento no se emite en absoluto."""
    from polyx.gui.detector.state import ModelSlot
    slot = ModelSlot(alias=ALIAS_HOSTIL, path=Path("vacio.pt"))
    html = _section_team([slot])
    assert ALIAS_HOSTIL not in html
    assert ALIAS_ESCAPADO not in html


# ════════════════════════════════════════════════════════════════════════
# E. _figura_calibracion — nombre de la placa en el rotulo monoespaciado
# ════════════════════════════════════════════════════════════════════════

def test_figura_calibracion_escapa_el_nombre_de_la_placa(rh, tmp_path):
    """alert(2) del sink de ''nombre de placa'' de _figura_calibracion.

    La clave de calibracion ES el nombre de archivo; el fix lo renderiza en
    la brecha monoespaciada del rotulo, escapado."""
    from polyx.model.calibracion import Calibration, ORIGIN_PLATE
    from tests.fixtures_informe import estado_hostil

    st = estado_hostil(directorio=tmp_path / "fotos")
    st.calibrations[NOMBRE_FOTO_HOSTIL] = Calibration(
        um_por_px=30.0, origin=ORIGIN_PLATE,
        cx=160.0, cy=130.0, radius_px=22.0, n_points=12)

    html = _figure_calibration(st)
    assert NOMBRE_FOTO_HOSTIL not in html
    assert ALIAS_HOSTIL not in html, "la figura nombraria al alias del sink"
    # La composicion del rotulo parte el nombre y la cuenta en dos lineas
    # del codigo fuente: el ancla cierra en la coma del nombre.
    assert f"({_e(NOMBRE_FOTO_HOSTIL)}," in html, \
        "el nombre de la placa en el rotulo monoespaciado"
    assert html.count(_e(NOMBRE_FOTO_HOSTIL)) == 1


# ════════════════════════════════════════════════════════════════════════
# F..M. Bloques del render parcial via _generate_report_impl(secciones=[...])
# ════════════════════════════════════════════════════════════════════════

def test_metodos_declara_el_alias_escapado_en_los_tres_sitios(rh, tmp_path):
    """Cabecera Models: + fila Modelos cargados + blockquote «{modelos}».

    Conteo exacto = 3: cabecera global 1, methods_rows 1, methods_para 1."""
    from tests.fixtures_informe import estado_hostil

    st = estado_hostil(directorio=tmp_path / "fotos")
    html = _render_de("metodos", st, tmp_path)

    assert ALIAS_HOSTIL not in html
    assert NOMBRE_FOTO_HOSTIL not in html
    assert f"<strong>Models:</strong> {ALIAS_ESCAPADO}</p>" in html
    assert f"<td>{ALIAS_ESCAPADO}</td></tr>" in html
    assert f"YOLO model \u00ab{ALIAS_ESCAPADO}\u00bb" in html
    assert html.count(ALIAS_ESCAPADO) == 3


def test_forma_muestra_el_nombre_escapado_en_fichas_y_recuento(rh, tmp_path):
    """tabla mayor/menor ``font-size:9pt`` x2 + recuento por foto x1.

    Conteo exacto = 3 en la seccion de forma."""
    from tests.fixtures_informe import estado_hostil

    st = estado_hostil(directorio=tmp_path / "fotos")
    html = _render_de("forma", st, tmp_path)

    assert NOMBRE_FOTO_HOSTIL not in html
    assert ALIAS_HOSTIL not in html, "la seccion de forma no nombra alias"
    fila_ficha = f"<td style='font-size:9pt'>{NOMBRE_ESCAPADO}</td></tr>"
    fila_recuento = (f"<td style='font-size:9pt'>{NOMBRE_ESCAPADO}</td><td>1</td>")
    assert html.count(NOMBRE_ESCAPADO) == 3, "fichas mayor/menor + recuento"
    assert html.count(fila_ficha) == 2
    assert html.count(fila_recuento) == 1


def test_forma_sufre_el_nombre_en_el_recuento_por_foto(rh, tmp_path):
    """En el recuento de STRINGS de datos crudos: el nombre ``barra`` no vive."""
    from tests.fixtures_informe import estado_hostil

    st = estado_hostil(directorio=tmp_path / "fotos")
    html = _render_de("forma", st, tmp_path)
    # El class_name literal 'PET' queda intacto. El markup crudo de la clase
    # no cruza: los valores sirven antes de llegar al html.
    assert "<td>PET</td>" in html
    assert CLASE_HOSTIL not in html


def test_fichas_escapan_la_clase_y_el_titulo_de_la_imagen(rh, tmp_path):
    """alert(3) del sink de clase de fichas + alert(2) del h3 de imagen."""
    from tests.fixtures_informe import estado_hostil

    st = estado_hostil(directorio=tmp_path / "fotos")
    for rs in st.results.values():
        for r in rs:
            r.predictions[0].class_name = CLASE_HOSTIL
    html = _render_de("fichas", st, tmp_path)

    assert CLASE_HOSTIL not in html
    assert NOMBRE_FOTO_HOSTIL not in html
    assert f"#{r.predictions[0].number} · {CLASE_ESCAPADA} ·" in html
    assert f"<h3>{NOMBRE_ESCAPADO}</h3>" in html
    assert html.count(CLASE_ESCAPADA) == 1
    assert html.count(NOMBRE_ESCAPADO) == 1


def test_resultados_no_suelta_texto_hostil_fotos_solamente(rh, tmp_path):
    """Los sumideros de la seccion de resultados (figuras) son IMAGENES.

    El eje y leyenda del histograma entran en los PNG (raster): el escape
    alli es el _esc de matplotlib, no html. La seccion HTML no debe soltar
    ninguna cadena hostil cruda (ni escapada: sin sinks HTML)."""
    from tests.fixtures_informe import estado_hostil

    st = estado_hostil(directorio=tmp_path / "fotos")
    for rs in st.results.values():
        for r in rs:
            r.predictions[0].class_name = CLASE_HOSTIL
    html = _render_de("resultados", st, tmp_path)

    assert NOMBRE_FOTO_HOSTIL not in html
    assert ALIAS_HOSTIL not in html
    assert CLASE_HOSTIL not in html
    # Y no hay escaped caso: la seccion solo lleva base64/data URI:
    assert NOMBRE_ESCAPADO not in html
    assert CLASE_ESCAPADA not in html


def test_galeria_escapa_el_alias_y_el_nombre_de_la_imagen(rh, tmp_path):
    """alert(1) x2 (figcaption + tag modelo) + alert(2) x1 (meta strong).

    La galeria ahi* no levanta si el ImageResult no lleva bytes pred/annot;
    los sub-successes pred_portas escriben PNG de la misma sintetica."""
    import cv2
    from tests.fixtures_informe import estado_hostil, foto_sintetica

    st = estado_hostil(directorio=tmp_path / "fotos")
    for rs in st.results.values():
        for r in rs:
            ok, buf = cv2.imencode(".png", foto_sintetica())
            assert ok
            r.pred_path = r.image_path.parent / "pred.png"
            r.pred_path.write_bytes(buf.tobytes())
    html = _render_de("galeria", st, tmp_path)

    assert ALIAS_HOSTIL not in html
    assert NOMBRE_FOTO_HOSTIL not in html
    assert f"Model prediction · {ALIAS_ESCAPADO}" in html   # figcaption
    assert f"modelo: {ALIAS_ESCAPADO}</span>" in html       # tag
    assert f"<strong>{NOMBRE_ESCAPADO}</strong>" in html    # meta
    # 3 = cabecera global (Models:) + figcaption + tag del meta
    assert html.count(ALIAS_ESCAPADO) == 3
    assert html.count(NOMBRE_ESCAPADO) == 1
    # Y la 'img' captada del dato de la foto es data URI, no markup crudo:
    assert "<img src=x" not in html


def test_errores_declara_la_matriz_y_la_nota_de_ausentes_escapados(rh, tmp_path):
    """Con GT: caption de la matriz (active[0].alias) y nota_ausentes (nombres
    de clase del checkpoint cargado)."""
    from tests.fixtures_informe import deteccion_medida, estado_hostil, foto_sintetica

    class _NombresFicticios:
        """Complemento minimo del modelo: solo el dict ``names`` que lee el impl."""
        names = {0: "PET", 5: CLASE_HOSTIL}

    st = estado_hostil(directorio=tmp_path / "fotos")
    # GT con clase ajena (id 3) para que entreste miscls y sirva de datos;
    # el name del nombre ausente (id 5, del checkpoint) es quien salta
    # a la nota_ausente con escape.
    for rs in st.results.values():
        for r in rs:
            gt = deteccion_medida(foto_sintetica())
            gt.class_id = 3
            gt.class_name = "GT_NEG"
            r.gt = [gt]
            r.has_gt = True
    st.model_slots[0].loaded = _NombresFicticios()

    html = _render_de("errores", st, tmp_path)

    assert CLASE_HOSTIL not in html
    assert ALIAS_HOSTIL not in html
    assert f"(main model: {ALIAS_ESCAPADO}," in html      # caption de la matriz
    assert f"No row for {CLASE_ESCAPADA}:" in html         # nota_ausentes
    assert html.count(CLASE_ESCAPADA) == 1
    assert html.count(ALIAS_ESCAPADO) == 2                 # cabecera + caption


def test_comparacion_escapa_el_alias_en_el_veredicto_y_en_la_cabecera(rh, tmp_path):
    """alert(1) x3: veredicto estricto (Best performance), th de comparacion
    tabla veredicto localizable... &gt; arriba y abajo."""
    from tests.fixtures_informe import deteccion_medida, estado_hostil, foto_sintetica

    st = estado_hostil(directorio=tmp_path / "fotos",
                        alias=[ALIAS_HOSTIL, "m2"])
    for rs in st.results.values():
        for r in rs:
            gt = deteccion_medida(foto_sintetica())
            r.gt = [gt]
            r.has_gt = True
    html = _render_de("comparacion", st, tmp_path)

    assert ALIAS_HOSTIL not in html
    assert NOMBRE_FOTO_HOSTIL not in html
    assert f"Best performance: {ALIAS_ESCAPADO}</strong>" in html   # veredicto estricto
    assert f"<th>{ALIAS_ESCAPADO}<br>" in html                      # cabecera de la tabla
    assert f"<td>{NOMBRE_ESCAPADO}</td>" in html                    # fila por imagen
    assert html.count(ALIAS_ESCAPADO) == 3
    assert html.count(NOMBRE_ESCAPADO) == 1


def test_comparacion_sin_gt_no_nombra_ganador_ni_alias(rh, tmp_path):
    """Sin GT, el veredicto estricto no existe: el sink no se emite."""
    from tests.fixtures_informe import estado_hostil

    st = estado_hostil(directorio=tmp_path / "fotos")
    html = _render_de("comparacion", st, tmp_path)
    # Sin GT el veredicto no se emite: la unica aparicion del alias es la
    # cabecera global (Models:), nunca un ganador:
    assert html.count(ALIAS_ESCAPADO) == 1, "sin GT el sink del veredicto nace muerto"
    assert "Best performance" not in html
    assert "Mejor" not in html
