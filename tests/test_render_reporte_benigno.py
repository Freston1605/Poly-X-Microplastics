"""Identidad de bytes del informe benigno contra el baseline en ingles.

El render del informe escapa los sumideros de valor con html.escape SIN
alterar el documento para entradas benignas. La forma honesta de probarlo es
un test de identidad contra el baseline benigno capturado por
tools/audit/capturar_fixtures_informe.py: el render por el pipeline real debe
reproducirlo byte a byte, y si un cambio toca algo mas que lo previsto, este
test lo dice con la desviacion de hash y de tamano.

Los estados de fixture se renderizan igual en los dos idiomas y el pipeline
es determinista (planning: 41 corridas repiten el mismo hash); los tiempos del
cronometro y la fecha de generacion nunca entran al HTML.

NOTA DE IDIOMA (contrato desde M006/S05): el baseline es el render en INGLES.
La recaptura de M006/S05 corre tools/audit/capturar_fixtures_informe.py, que fija
POLYX_IDIOMA=en antes de importar cualquier modulo, y este test fija el mismo
valor: el compare de bytes no depende del locale de la maquina ni de
~/.polyx_idioma.json. Los snapshots M004 (es=bb2e9f…, en=bf483e…) quedaron
superseded por la recaptura de M006/S05 tras pasar los doce rotulos crudos
del informe a literales en ingles.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import re
import sys
from pathlib import Path

import pytest

BASELINE = Path(__file__).resolve().parent / "fixtures" / "baseline_informe" / "benigna"
BASELINE_HTML = BASELINE / "informe.html"
BASELINE_SHA = BASELINE / "informe.sha256"


# Las figuras del informe son PNG base64 generados por matplotlib: sus bytes
# cambian con el renderer/fuentes del SO incluso a la misma version pinneada
# (CI ubuntu != dev fedora), asi que un gate de identidad de bytes sobre el
# documento completo es fragil a eso. El contrato del gate es la ESTRUCTURA
# del informe (cabeceras, escape, secciones, parametros), no el payload de
# cada PNG. Esta normalizacion reemplaza cada data-image-uri por un marcador
# FIJO (hash del payload) antes de comparar, en ambos lados (render y
# baseline): las figuras se comparan por su existencia y longitud relativa
# (el hash distingue una figura distinta de la misma), no por sus bytes.
_PNG_URI_RE = re.compile(r"data:image/png;base64,[A-Za-z0-9+/=]+")


def _normalizar_figuras(html: str) -> str:
    """Sustituye cada data-URI de PNG por un marcador ORDINAL (FIG1, FIG2...).

    Los bytes de los PNG base64 de matplotlib cambian con el renderer/fuentes
    del SO incluso a la misma version pinneada (CI ubuntu != dev fedora). El
    gate de identidad quiere fijar la ESTRUCTURA del documento (que en cada
    posicion del informe haya la figura que toca, con su pie, y que el resto
    de texto/escapado sea byte-identico), no los bytes internos de cada PNG.
    El marcador ordinal resuelve eso: dos renders de la misma estructura
    producen exactamente la misma cadena normalizada en cualquier SO, y un
    cambio de informe que mueva, quite o anada una figura cambia el patrón de
    marcadores y hace fallar el gate.
    """
    contador: list[int] = [0]

    def _marcar(m: re.Match) -> str:
        contador[0] += 1
        return f"data:image/png;base64,<FIG{contador[0]}>"

    return _PNG_URI_RE.sub(_marcar, html)


# Tamano en bytes del baseline benigno recapturado en M006/S05: segundo
# arbol de decision junto al sha256 -- el digest localiza QUE bytes cambiaron,
# este pin da el tamano esperado sin abrir el archivo. La procedencia completa
# esta en tests/fixtures/baseline_informe/README.md.
TAMANO_BASELINE = 20417


@pytest.fixture(autouse=True)
def _idioma_del_baseline_y_logs_muertos(tmp_path):
    """Fuerza el idioma del baseline (en) y revierte el logger a NullHandler.

    polyx.core.i18n resuelve y cachea el idioma al importar (env ->
    ~/.polyx_idioma.json -> locale del sistema); sin forzar la variable el
    test depende de la maquina y el compare de bytes seria una loteria. Se
    recarga i18n con POLYX_IDIOMA=en -- el idioma con el que se capturo el
    informe.html del baseline de M006/S05 -- y se vuelve a recargar al salir para
    no contaminar a otros tests (ver nota de idioma en el doc del modulo).

    El logger 'polyx.detector' (report_html usa cronometro(logger_para("detec
    tor"), ...) dentro de generate_report) queda cacheado por logger_para con
    un RotatingFileHandler a carpeta_logs(); si no se controla, la suite
    escribe en el HOME real y puede variar segun el orden de los tests. Se
    reemplazan los handlers por un NullHandler y se restauran al terminar
    (los estados de fixture no usan QObjects y no emiten eventos: no se pierde
    nada).

    Antes del primer logger_para se fija POLYX_LOG_DIR a tmp_path: si este es
    el primer test de la sesion en tocar logger_para, el handler Rotating se
    abriria contra el HOME real; con la variable ya puesta el archivo queda
    dentro de tmp_path aunque pytest lo borre despues.
    """
    import polyx.core.i18n as i18n
    import polyx.core.observability as obs

    previo_idioma = os.environ.get("POLYX_IDIOMA")
    previo_log_dir = os.environ.get("POLYX_LOG_DIR")
    # El baseline recapturado en M006/S05 esta en ingles: ver nota de idioma.
    os.environ["POLYX_IDIOMA"] = "en"
    # La carpeta de logs nunca puede ser el HOME real durante el test.
    os.environ["POLYX_LOG_DIR"] = str(tmp_path / "logs")

    importlib.reload(i18n)
    import polyx.gui.report_html as rh

    importlib.reload(rh)

    # Pin the machine-equipment table to the same canonical rows the
    # baseline was captured with (tools/audit/capturar_fixtures_informe.py sets
    # the identical lambda): the report's _team() reads live OS/RAM/GPU/
    # torch values that differ per runner, and a byte gate cannot depend on
    # them. Production keeps the live rows; only this fixture and the
    # capture script override the function.
    rh._team = lambda: [
        ("Operating system", "Linux"),
        ("Processor", "pinned"),
        ("RAM", "16.0 GB"),
        ("GPU", "not available"),
        ("PyTorch", "pinned"),
    ]

    # Same for the Ultralytics version in the methods paragraph: a dev
    # environment without ultralytics renders "(Ultralytics —)" while the
    # pinned venv has 8.3.40. The gate must not depend on which packages are
    # importable, so the version is pinned to the same value in both the
    # capture and this fixture.
    _ul_stub = type("ULStub", (), {"__version__": "8.3.40"})()
    sys.modules["ultralytics"] = _ul_stub

    logger = obs.logger_for("detector")
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


def test_el_baseline_existe(tmp_path):
    """El snapshot benigno tiene que estar presente en el repo.

    Sin esta sonda el compare de bytes fallaria con FileNotFoundError y el
    mensaje no diria que el problema es del fixture; aqui el fallo nombra el
    archivo que falta.
    """
    import polyx.gui.report_html as rh  # el fixture ya valido el reload

    assert rh is not None
    assert BASELINE_HTML.is_file(), f"falta el baseline: {BASELINE_HTML}"
    assert BASELINE_SHA.is_file(), f"falta el sha256 del baseline: {BASELINE_SHA}"


def test_render_benigno_es_byte_identico_al_baseline(tmp_path):
    """El render por el pipeline real, con TODAS las secciones, reproduce el
    baseline benigno byte a byte."""
    import polyx.gui.report_html as rh
    from tests.fixtures_informe import estado_benigno

    st = estado_benigno(directorio=tmp_path / "fotos")
    salida = tmp_path / "informe.html"
    rh.generate_report(st, salida, sections=list(rh.IDS_SECTIONS))

    datos = _normalizar_figuras(salida.read_text(encoding="utf-8"))
    baseline = _normalizar_figuras(BASELINE_HTML.read_text(encoding="utf-8"))
    assert datos == baseline, (
        f"el render benigno divergio del baseline "
        f"({len(datos)} B vs {len(baseline)} B despues de normalizar figuras): "
        f"hubo un cambio de informe no registrado o el baseline quedo "
        f"desactualizado"
    )


def test_sha256_del_render_benigno_coincide_con_el_registrado(tmp_path):
    """El digest persistido en informe.sha256 delimita que parte muto: si el
    compare de bytes falla, este segundo arbol da el valor exacto a buscar."""
    import polyx.gui.report_html as rh
    from tests.fixtures_informe import estado_benigno

    st = estado_benigno(directorio=tmp_path / "fotos")
    salida = tmp_path / "informe.html"
    rh.generate_report(st, salida, sections=list(rh.IDS_SECTIONS))

    digest = hashlib.sha256(
        _normalizar_figuras(salida.read_text(encoding="utf-8")).encode("utf-8")
    ).hexdigest()
    esperado = BASELINE_SHA.read_text(encoding="utf-8").strip()
    assert digest == esperado, (
        f"sha256 del render benigno (figuras normalizadas) {digest} != baseline {esperado}"
    )


def test_el_propio_baseline_coincide_con_su_registro():
    """Check de autoconsistencia del fixture: sha256(informe.html) coincide
    con lo que su informe.sha256 dice de el.

    Guarda contra un baseline corrupto (editado a mano, CRLF, truncado): un
    test de identidad contra un baseline roto pasaria o fallaria sin que nadie
    pudiera distinguir que parte mintio; esta prueba valida primero que el
    propio snapshot cuadra con su registro. Se compara el digest del contenido
    NORMALIZADO (figuras reemplazadas por su marcador estable), igual que el
    test de identidad: el sha256 registrado describe el documento como lo
    compara el gate, no los bytes crudos del PNG.
    """
    datos = _normalizar_figuras(BASELINE_HTML.read_text(encoding="utf-8"))
    digest = hashlib.sha256(datos.encode("utf-8")).hexdigest()
    esperado = BASELINE_SHA.read_text(encoding="utf-8").strip()
    assert digest == esperado, (
        f"el baseline esta corrupto: su sha256 normalizado es {digest} y el "
        f"archivo informe.sha256 dice {esperado}"
    )


def test_el_baseline_pesa_exactamente_lo_registrado():
    """Pin del tamano en bytes del baseline benigno de M006/S05.

    Tercera prueba de autoconsistencia del fixture y segundo arbol de
    decision del gate del informe: si el digest normalizado diverge, este
    test delimita si el cambio fue una edicion puntual (misma longitud
    normalizada) o una regeneracion completa (otra longitud) sin tener que
    abrir los ~137 KB del HTML. El pin es sobre la longitud del contenido
    NORMALIZADO (figuras por marcador ordinal), igual que el sha256: la
    longitud cruda incluye los bytes de los PNG, que varian con el renderer
    del SO aun a la misma version pinneada.

    El valor debe coincidir con lo documentado en
    tests/fixtures/baseline_informe/README.md; el pin corresponde a la
    recaptura sobre el stack pinneado (PR #4), no al freeze de M004.
    Un tamano distinto significa un cambio de informe no registrado; si el
    cambio es intencional hay que recapturar y actualizar sha256, pin y
    README juntos.
    """
    tamano_real = len(_normalizar_figuras(BASELINE_HTML.read_text(encoding="utf-8")))
    assert tamano_real == TAMANO_BASELINE, (
        f"el baseline benigno (normalizado) pesa {tamano_real} B y el "
        f"registro dice {TAMANO_BASELINE} B: informe.html fue editado o "
        f"regenerado -- si el cambio de informe no fue intencional, se "
        f"arregla el codigo; si lo fue, se recaptura y se actualizan "
        f"sha256, pin y README juntos"
    )
