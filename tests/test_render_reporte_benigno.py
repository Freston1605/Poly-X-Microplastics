"""Identidad de bytes del informe benigno contra el baseline en ingles.

El render del informe escapa los sumideros de valor con html.escape SIN
alterar el documento para entradas benignas. La forma honesta de probarlo es
un test de identidad contra el baseline benigno capturado por
scripts/capturar_fixtures_informe.py: el render por el pipeline real debe
reproducirlo byte a byte, y si un cambio toca algo mas que lo previsto, este
test lo dice con la desviacion de hash y de tamano.

Los estados de fixture se renderizan igual en los dos idiomas y el pipeline
es determinista (planning: 41 corridas repiten el mismo hash); los tiempos del
cronometro y la fecha de generacion nunca entran al HTML.

NOTA DE IDIOMA (contrato desde M006/S05): el baseline es el render en INGLES.
La recaptura de M006/S05 corre scripts/capturar_fixtures_informe.py, que fija
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
from pathlib import Path

import pytest

BASELINE = Path(__file__).resolve().parent / "fixtures" / "baseline_informe" \
    / "benigna"
BASELINE_HTML = BASELINE / "informe.html"
BASELINE_SHA = BASELINE / "informe.sha256"

# Tamano en bytes del baseline benigno recapturado en M006/S05: segundo
# arbol de decision junto al sha256 -- el digest localiza QUE bytes cambiaron,
# este pin da el tamano esperado sin abrir el archivo. La procedencia completa
# esta en tests/fixtures/baseline_informe/README.md.
TAMANO_BASELINE = 139503


@pytest.fixture(autouse=True)
def _idioma_del_baseline_y_logs_muertos(tmp_path):
    """Fuerza el idioma del baseline (en) y revierte el logger a NullHandler.

    polyx.ml.i18n resuelve y cachea el idioma al importar (env ->
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
    import polyx.ml.i18n as i18n
    import polyx.ml.observabilidad as obs

    previo_idioma = os.environ.get("POLYX_IDIOMA")
    previo_log_dir = os.environ.get("POLYX_LOG_DIR")
    # El baseline recapturado en M006/S05 esta en ingles: ver nota de idioma.
    os.environ["POLYX_IDIOMA"] = "en"
    # La carpeta de logs nunca puede ser el HOME real durante el test.
    os.environ["POLYX_LOG_DIR"] = str(tmp_path / "logs")

    importlib.reload(i18n)
    import polyx.gui.report_html as rh
    importlib.reload(rh)

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

    datos = salida.read_bytes()
    baseline = BASELINE_HTML.read_bytes()
    assert datos == baseline, (
        f"el render benigno divergio del baseline "
        f"({len(datos)} B vs {len(baseline)} B): hubo un cambio de informe no "
        f"registrado o el baseline quedo desactualizado")


def test_sha256_del_render_benigno_coincide_con_el_registrado(tmp_path):
    """El digest persistido en informe.sha256 delimita que parte muto: si el
    compare de bytes falla, este segundo arbol da el valor exacto a buscar."""
    import polyx.gui.report_html as rh
    from tests.fixtures_informe import estado_benigno

    st = estado_benigno(directorio=tmp_path / "fotos")
    salida = tmp_path / "informe.html"
    rh.generate_report(st, salida, sections=list(rh.IDS_SECTIONS))

    digest = hashlib.sha256(salida.read_bytes()).hexdigest()
    esperado = BASELINE_SHA.read_text(encoding="utf-8").strip()
    assert digest == esperado, (
        f"sha256 del render benigno {digest} != baseline {esperado}")


def test_el_propio_baseline_coincide_con_su_registro():
    """Check de autoconsistencia del fixture: sha256(informe.html) coincide
    con lo que su informe.sha256 dice de el.

    Guarda contra un baseline corrupto (editado a mano, CRLF, truncado): un
    test de identidad contra un baseline roto pasaria o fallaria sin que nadie
    pudiera distinguir que parte mintio; esta prueba valida primero que el
    propio snapshot cuadra con su registro.
    """
    datos = BASELINE_HTML.read_bytes()
    digest = hashlib.sha256(datos).hexdigest()
    esperado = BASELINE_SHA.read_text(encoding="utf-8").strip()
    assert digest == esperado, (
        f"el baseline esta corrupto: su sha256 es {digest} y el archivo "
        f"informe.sha256 dice {esperado}")


def test_el_baseline_pesa_exactamente_lo_registrado():
    """Pin del tamano en bytes del baseline benigno de M006/S05.

    Tercera prueba de autoconsistencia del fixture y segundo arbol de
    decision del gate del informe: si el sha256 diverge, este test delimita
    si el cambio fue una edicion puntual (mismo tamano) o una regeneracion
    completa (otro tamano) sin tener que abrir los 139 KB del HTML.

    El valor debe coincidir con lo documentado en
    tests/fixtures/baseline_informe/README.md; el pin corresponde a la
    recaptura de M006/S05, no al freeze de M004. Un tamano distinto significa
    un cambio de informe no registrado; si el cambio es intencional hay que
    recapturar y actualizar sha256, pin y README juntos.
    """
    tamano_real = len(BASELINE_HTML.read_bytes())
    assert tamano_real == TAMANO_BASELINE, (
        f"el baseline benigno pesa {tamano_real} B y el registro de M006/S05 "
        f"dice {TAMANO_BASELINE} B: informe.html fue editado o regenerado -- "
        f"si el cambio de informe no fue intencional, se arregla el codigo; "
        f"si lo fue, se recaptura y se actualizan sha256, pin y README juntos")
