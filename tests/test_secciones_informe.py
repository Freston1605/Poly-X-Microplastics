"""Llamada directa a cada constructor de seccion del informe (M004/S02).

El slice S02 extrajo todos los cuerpos inline de ``_generate_report_impl``
(polyx/gui/report_html.py) a funciones de nivel de modulo que devuelven el
cuerpo de su seccion SIN el ``<h2>`` (el ensamblador pone titulo y numero).
Este archivo cierra el contrato del slice: cada constructor es importable y
llamable por su cuenta, con los mismos precomputos que usa el impl, y su
salida para el estado benigno coincide con lo que el baseline benigno
(tests/fixtures/baseline_informe/benigna/informe.html) realmente contiene.
Ese baseline es el gate actual, recapturado en M006/S05 en ingles.

El contrato de idioma es el de tests/test_render_reporte_benigno.py
(POLYX_IDIOMA=en): el baseline se capturo en ingles y cualquier aserto sobre
texto del cuerpo necesita el mismo idioma, fijado y recargado en el fixture
autouse -- ver la NOTA DE IDIOMA de ese modulo para la procedencia completa.

Los cuerpos vacios NO son fallos: para el lote benigno (una foto, un modelo,
sin ground truth, sin calibraciones, sin tramo de profundidad en el nombre)
los constructores de errores/veredicto/galeria/calibracion/profundidad y
talla_carpetas devuelven '' -- y el ensamblador suelta esas secciones, que
es exactamente por lo que el baseline no tiene sus <h2>.
"""
from __future__ import annotations

import importlib
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

# Baseline benigno de M006/S05: solo lectura en la suite; se recaptura con
# tools/audit/capturar_fixtures_informe.py solo ante un cambio consciente del
# informe, actualizando sha256, pin y README juntos.
BASELINE_HTML = Path(__file__).resolve().parent / "fixtures" / "baseline_informe" \
    / "benigna" / "informe.html"

# Las figuras del baseline son PNG base64 de matplotlib, cuyos bytes cambian
# con el renderer/fuentes del SO incluso a la misma version pinneada. Los
# tests que comparan un CUERPO DE SECCION aislado contra el baseline completo
# normalizan las figuras a un marcador CONSTANTE (<FIG>): el cuerpo aislado
# renumera su marcador ordinal desde 1 pero el baseline los numera desde su
# posicion global, asi que un marcador ordinal no casa entre ambos. La
# posicion de cada figura dentro del informe completo la cubre el gate de
# tests/test_render_reporte_benigno.py (renders de TODAS las secciones
# juntas, mismos ordinales en ambos lados); aqui solo se fija estructura.
import re as _re

_PNG_URI_RE_SECTION = _re.compile(r"data:image/png;base64,[A-Za-z0-9+/=]+")


def _sin_payload_figuras(html: str) -> str:
    """Reemplaza cada data-URI de PNG por el marcador constante <FIG>."""
    return _PNG_URI_RE_SECTION.sub("data:image/png;base64,<FIG>", html)

from tests.test_render_reporte_benigno import _normalizar_figuras  # noqa: E402,F401

# El baseline ya normalizado con marcador constante (solo separa estructura):
# unico lugar que lee el archivo. Los cuerpos de seccion se comparan contra
# esta variante, cada uno normalizado con la misma funcion.
BASELINE_TXT = _sin_payload_figuras(BASELINE_HTML.read_text(encoding="utf-8"))

# Constructores extraidos en S02/T01-T07 mas los que ya eran de modulo.
CONSTRUCTORES = (
    "_section_summary", "_section_errors", "_section_models",
    "_section_verdict", "_section_comparison", "_section_gallery",
    "_section_results", "_section_methods", "_section_references",
    "_section_calibration", "_section_shape", "_section_records",
    "_section_count", "section_depth",
)


@pytest.fixture(autouse=True)
def _idioma_del_baseline(tmp_path):
    """Fuerza el idioma del baseline (en) y recarga i18n/report_html.

    Misma receta que el fixture de tests/test_render_reporte_benigno.py:
    polyx.core.i18n cachea el idioma al importar, y los cuerpos que aqui se
    comparan con texto del baseline (en ingles) saldrian en otro idioma sin
    esta recarga. Se restaura el entorno al salir para no contaminar a otros
    tests. POLYX_LOG_DIR ya lo aisla el fixture autouse de tests/conftest.py
    (decision D007); aqui no se llama a generate_report, asi que no hay
    logger que silenciar.
    """
    import polyx.core.i18n as i18n

    previo_idioma = os.environ.get("POLYX_IDIOMA")
    os.environ["POLYX_IDIOMA"] = "en"

    importlib.reload(i18n)
    import polyx.gui.report_html as rh
    importlib.reload(rh)

    yield

    if previo_idioma is None:
        os.environ.pop("POLYX_IDIOMA", None)
    else:
        os.environ["POLYX_IDIOMA"] = previo_idioma
    importlib.reload(i18n)
    importlib.reload(rh)


def _precomputados(st) -> SimpleNamespace:
    """Mirror the precomputation block of _generate_report_impl.

    Los constructores reciben estos valores ya calculados (el impl no
    recompone nada); si el impl cambia sus precomputos, este espejo hay que
    moverlo con el. Devuelve un namespace con los mismos nombres que usan
    las llamadas de abajo.
    """
    import polyx.gui.report_html as rh

    resultados = dict(st.results)
    active = st.active_models()
    all_results = [r for rs in resultados.values() for r in rs]
    total_imgs = len({r.image_path for r in all_results})
    total_dets = sum(len(r.predictions) for r in all_results)
    confs = [p.conf for r in all_results for p in r.predictions]
    sizes = [p.diam_um for r in all_results for p in r.predictions if p.diam_um]
    avg_conf = sum(confs) / len(confs) if confs else 0
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    any_gt = any(r.has_gt for r in all_results)
    clases_lote = rh._classes_views(all_results)
    per_class: Counter = Counter()
    for r in all_results:
        for p in r.predictions:
            per_class[p.class_name] += 1
    return SimpleNamespace(
        resultados=resultados, active=active, all_results=all_results,
        total_imgs=total_imgs, total_dets=total_dets, confs=confs,
        sizes=sizes, avg_conf=avg_conf, avg_size=avg_size, any_gt=any_gt,
        clases_lote=clases_lote, per_class=per_class)


def _estado_benigno(tmp_path) -> tuple:
    """Estado benigno + sus precomputos, con la receta del render del baseline."""
    from tests.fixtures_informe import estado_benigno

    st = estado_benigno(directorio=tmp_path / "fotos")
    return st, _precomputados(st)


def test_los_constructores_son_de_nivel_de_modulo():
    """Cada constructor extraido en S02 existe y es llamable.

    Sin esta sonda, un renombre accidental en el modulo se manifestaria como
    AttributeError dentro de un test con nombre confuso; aqui el fallo
    nombra exactamente al simbolo que falta.
    """
    import polyx.gui.report_html as rh

    for nombre in CONSTRUCTORES:
        assert callable(getattr(rh, nombre)), f"falta {nombre}"


def test_resumen_devuelve_cuerpo_sin_h2(tmp_path):
    """Resumen: cuerpo no vacio, str, sin <h2>, y literal en el baseline."""
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    cuerpo = rh._section_summary(pre.total_imgs, pre.total_dets,
                                 pre.avg_conf, pre.avg_size,
                                 pre.active, pre.any_gt)
    assert isinstance(cuerpo, str) and cuerpo
    assert "<h2" not in cuerpo
    # El ensamblador inserta el cuerpo tal cual cuando no lleva {NUM}.
    assert _sin_payload_figuras(cuerpo) in BASELINE_TXT


def test_errores_queda_vacio_sin_ground_truth(tmp_path):
    """Errores: el lote benigno no tiene GT, el cuerpo legitimo es ''."""
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    cuerpo = rh._section_errors(st, pre.resultados, pre.active, pre.any_gt)
    assert cuerpo == ""
    # Y el baseline lo confirma: la seccion no aparece en el documento.
    assert "<h2 id='errors'>" not in BASELINE_TXT


def test_modelos_nombra_al_alias(tmp_path):
    """Resumen por modelo: no vacio, menciona el alias benigno, sin <h2>."""
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    cuerpo = rh._section_models(st, pre.resultados, pre.all_results, pre.any_gt)
    assert isinstance(cuerpo, str) and cuerpo
    assert "<h2" not in cuerpo
    assert "m1" in cuerpo  # alias de tests/fixtures_informe.ALIAS_BENIGNO
    assert _sin_payload_figuras(cuerpo) in BASELINE_TXT


def test_veredicto_sin_modelos_que_comparar_queda_vacio(tmp_path):
    """Veredicto: con un solo modelo no hay barrido que reportar -> ''."""
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    assert rh._section_verdict(pre.resultados, st) == ""


def test_comparacion_devuelve_cuerpo_sin_h2(tmp_path):
    """Comparacion foto por foto: cuerpo str no vacio y sin <h2>."""
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    cuerpo = rh._section_comparison(st, pre.resultados)
    assert isinstance(cuerpo, str) and cuerpo
    assert "<h2" not in cuerpo
    assert _sin_payload_figuras(cuerpo) in BASELINE_TXT


def test_galeria_sin_bytes_de_prediccion_queda_vacia(tmp_path):
    """Galeria: el fixture no publica bytes anotados, el cuerpo es ''.

    _seccion_galeria solo incrusta imagenes cuando el ImageResult trae las
    anotaciones renderizadas; el estado benigno no las lleva, y el baseline
    (donde 'galeria' no aparece) lo confirma.
    """
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    cuerpo = rh._section_gallery(pre.resultados, st, pre.clases_lote,
                                 max_gallery=60, include_gallery=True)
    assert cuerpo == ""
    assert "<h2 id='gallery'>" not in BASELINE_TXT


def test_resultados_lleva_las_figuras_agregadas(tmp_path):
    """Resultados: cuerpo con figuras embebidas (base64), sin <h2>."""
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    cuerpo = rh._section_results(dict(pre.per_class), pre.confs, pre.sizes)
    assert isinstance(cuerpo, str) and cuerpo
    assert "<h2" not in cuerpo
    assert "base64," in cuerpo
    assert _sin_payload_figuras(cuerpo) in BASELINE_TXT


def test_metodos_devuelve_tres_partes_con_placeholder_num(tmp_path):
    """Metodos: (tabla, equipo, parrafo) no vacios, sin <h2>, con {NUM}."""
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    partes = rh._section_methods(st, pre.active, pre.all_results,
                                 pre.total_imgs, pre.total_dets, pre.any_gt)
    assert isinstance(partes, tuple) and len(partes) == 3
    assert all(isinstance(p, str) and p for p in partes)
    combinado = "".join(partes)
    assert "<h2" not in combinado
    # La tabla de metodos usa sub-encabezados numerados: el placeholder se
    # sustituye en el ensamblador, nunca aqui.
    assert rh.NUM in combinado


def test_referencias_con_y_sin_bandera(tmp_path):
    """Referencias: con la bandera incluye el <ol> de papers; sin ella, ''."""
    import polyx.gui.report_html as rh

    cuerpo = rh._section_references(include_refs=True)
    assert isinstance(cuerpo, str) and cuerpo
    assert "<h2" not in cuerpo
    assert "<ol" in cuerpo
    assert _sin_payload_figuras(cuerpo) in BASELINE_TXT
    assert rh._section_references(include_refs=False) == ""


def test_calibracion_vacia_sin_calibraciones(tmp_path):
    """Calibracion: el estado benigno no lleva calibraciones -> ''.

    Caso frontera declarado en el plan: la seccion se omite cuando no hay
    nada que reportar, y el baseline (sin <h2 id='calib'>) lo confirma.
    """
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    assert rh._section_calibration(st) == ""
    assert "<h2 id='calib'>" not in BASELINE_TXT


def test_conteo_por_muestra_con_placeholder_num(tmp_path):
    """Conteo: cuerpo no vacio con sub-encabezado {NUM} y sin <h2>."""
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    cuerpo = rh._section_count(pre.resultados, st, pre.active,
                                pre.clases_lote)
    assert isinstance(cuerpo, str) and cuerpo
    assert "<h2" not in cuerpo
    assert rh.NUM in cuerpo


def test_profundidad_vacia_sin_tramo_en_los_nombres(tmp_path):
    """Profundidad: los nombres benignos no llevan tramo de profundidad -> ''."""
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    formas, n_aro, n_sin = rh._lot_forms(st, pre.all_results)
    assert formas, "el lote benigno debe tener al menos una particula medida"
    assert rh.section_depth(formas, pre.all_results) == ""


def test_forma_devuelve_cuerpo_y_talla_carpetas_vacio(tmp_path):
    """Forma: cuerpo grande con placeholders {NUM}.N; talla_carpetas ''.

    Con una sola carpeta no hay comparacion entre carpetas ni entre fotos
    dentro de carpeta, asi que el segundo cuerpo legitimo es '' (el baseline
    no trae la seccion).
    """
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    formas, n_aro, n_sin = rh._lot_forms(st, pre.all_results)
    forma_html, por_carpeta_html = rh._section_shape(st, formas, n_aro, n_sin)
    assert isinstance(forma_html, str) and forma_html
    assert "<h2" not in forma_html
    # Sub-encabezados numerados de la seccion: {NUM}.1, {NUM}.4, ...
    assert rh.NUM in forma_html
    assert isinstance(por_carpeta_html, str)
    assert por_carpeta_html == ""


def test_fichas_tarjetas_y_tope_de_muestra(tmp_path):
    """Fichas: tarjetas con imagen base64 por foto; con tope 0 queda ''.

    Casos frontera del cap: sin formas no hay fichas, y max_fichas=0 apaga
    la seccion (mismo guard que tenia el bloque inline).
    """
    import polyx.gui.report_html as rh

    st, pre = _estado_benigno(tmp_path)
    formas, _aro, _sin = rh._lot_forms(st, pre.all_results)
    cuerpo = rh._section_records(st, formas, max_records=12)
    assert isinstance(cuerpo, str) and cuerpo
    assert "<h2" not in cuerpo
    assert "base64," in cuerpo
    # Agrupado por imagen: el nombre de la foto llega escapado en un <h3>.
    assert "<h3>placa_muestra_a.png</h3>" in cuerpo
    assert _sin_payload_figuras(cuerpo) in BASELINE_TXT
    # Fronteras: sin formas o con tope 0, vacio (y el ensamblador la suelta).
    assert rh._section_records(st, [], max_records=12) == ""
    assert rh._section_records(st, formas, max_records=0) == ""


def test_generar_impl_centrada_en_secciones():
    """Structural gate: the impl is <=150 lines and reads as an ordered section list.

    Contrato del slice S03/T02: tras hoistear el precalculo del lote (T01,
    _metricas_de_lote) y el esqueleto del documento (T02, _documento_html),
    el impl debe quedar como la lista ordenada de llamadas a constructores
    que alimenta el mapeo cuerpos, de a lo sumo 150 lineas. Sin este gate,
    un bloque de calculo inline nuevo haria crecer el impl de nuevo sin que
    nadie lo note hasta que alguien intente leerlo.

    Mide con ast sobre el codigo fuente (no sobre el modulo vivo) el span
    real de la funcion, y lee SECCIONES del modulo importado para que una
    seccion nueva sin constructor en el impl falle aqui y no en produccion.
    """
    import ast

    import polyx.gui.report_html as rh

    fuente = Path(rh.__file__).resolve().read_text(encoding="utf-8")
    impl = next(
        n for n in ast.walk(ast.parse(fuente))
        if isinstance(n, ast.FunctionDef) and n.name == "_generate_report_impl")

    lineas = impl.end_lineno - impl.lineno + 1
    assert lineas <= 150, (
        f"_generate_report_impl crecio a {lineas} lineas (tope 150): "
        f"un bloque de calculo volvio a crecer inline y hay que hoistearlo "
        f"a un helper de nivel de modulo")

    # Toda llamada directa dentro del impl, por nombre de la funcion llamada.
    llamadas = {n.func.id for n in ast.walk(impl)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    # id de seccion -> constructores que producen su cuerpo en el impl.
    # talla_carpetas sale de la misma llamada que forma (segundo retorno) y
    # comparacion junta el veredicto con la comparacion foto por foto.
    constructor_de = {
        "resumen": ("_section_summary",),
        "metodos": ("_section_methods",),
        "calibracion": ("_section_calibration",),
        "forma": ("_section_shape",),
        "talla_carpetas": ("_section_shape",),
        "profundidad": ("section_depth",),
        "fichas": ("_section_records",),
        "resultados": ("_section_results",),
        "modelos": ("_section_models",),
        "errores": ("_section_errors",),
        "comparacion": ("_section_verdict", "_section_comparison"),
        "galeria": ("_section_gallery",),
        "conteo": ("_section_count",),
        "referencias": ("_section_references",),
    }
    ids_secciones = {sid for sid, _titulo in rh.SECTIONS}
    assert ids_secciones == set(constructor_de), (
        f"SECCIONES y constructor_de desincronizados: sin constructor "
        f"{sorted(ids_secciones - set(constructor_de))}, "
        f"fantasmas {sorted(set(constructor_de) - ids_secciones)}")

    faltantes = {sid: [c for c in cs if c not in llamadas]
                 for sid, cs in constructor_de.items()
                 if any(c not in llamadas for c in cs)}
    assert not faltantes, (
        f"el impl no llama a los constructores de {faltantes}: la seccion "
        f"quedo sin cuerpo o su llamada salio del impl")
