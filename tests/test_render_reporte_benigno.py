"""Identidad de bytes del informe benigno contra el baseline pre-fix (T04).

El slice M001/S01 exige que el render del informe escape los sumideros de
valor con html.escape SIN alterar el documento para entradas benignas. La unica
forma honesta de probar eso es un test de identidad que exista ANTES del fix:
corre verde sobre el codigo actual (renderiza igual que el baseline capturado
por scripts/capturar_fixtures_informe.py, T03) y debe seguir en verde cuando el
escape aterrice en polyx/core/report_html.py (T05). Si el cambio toca algo mas
que la inyeccion, este test lo dice con la desviacion de hash.

Los estados de fixture se renderizan igual en los dos idiomas y el pipeline
es determinista (planning: 41 corridas repiten el mismo hash); los tiempos del
cronometro y la fecha de generacion nunca entran al HTML.

NOTA DE IDIOMA (desviacion del plan, decidida y verificada en T04): el baseline
de T03 se capturo con la resolucion DEFAULT de i18n (sin POLYX_IDIOMA), que en
la maquina de captura cayo en ingles via locale.getlocale(); el archivo
informe.html guardado esta en ingles. Un fixture que fuerza espanol diverge
por idioma del propio baseline (verificado: es=bb2e9f…, en=bf483e…), y NO
forzar el idioma seria maquina-dependiente (otra maquina con locale vacio o
~/.polyx_idioma.json caeria en es). Se fija entonces POLYX_IDIOMA=en
explicitamente: el mismo idioma del baseline, ahora por contrato en vez de
por azar de la maquina de captura.
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


@pytest.fixture(autouse=True)
def _idioma_del_baseline_y_logs_muertos(tmp_path):
    """Fuerza el idioma del baseline (en) y revierte el logger a NullHandler.

    polyx.core.i18n resuelve y cachea el idioma al importar (env ->
    ~/.polyx_idioma.json -> locale del sistema); sin forzar la variable el
    test depende de la maquina y el compare de bytes seria una loteria. Se
    recarga i18n con POLYX_IDIOMA=en -- el idioma que de verdad tiene el
    informe.html del baseline de T03 -- y se vuelve a recargar al salir para
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
    import polyx.core.observabilidad as obs

    previo_idioma = os.environ.get("POLYX_IDIOMA")
    previo_log_dir = os.environ.get("POLYX_LOG_DIR")
    # El baseline capturado en T03 esta en ingles: ver nota de idioma.
    os.environ["POLYX_IDIOMA"] = "en"
    # La carpeta de logs nunca puede ser el HOME real durante el test.
    os.environ["POLYX_LOG_DIR"] = str(tmp_path / "logs")

    importlib.reload(i18n)
    import polyx.core.report_html as rh
    importlib.reload(rh)

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


def test_el_baseline_existe(tmp_path):
    """El snapshot pre-fix (T03) tiene que estar presente en el repo.

    Sin esta sonda el compare de bytes fallaria con FileNotFoundError y el
    mensaje no diria que el problema es del fixture; aqui el fallo nombra el
    archivo que falta.
    """
    import polyx.core.report_html as rh  # el fixture ya valido el reload
    assert rh is not None
    assert BASELINE_HTML.is_file(), f"falta el baseline: {BASELINE_HTML}"
    assert BASELINE_SHA.is_file(), f"falta el sha256 del baseline: {BASELINE_SHA}"


def test_render_benigno_es_byte_identico_al_baseline(tmp_path):
    """El render por el pipeline real, con TODAS las secciones, reproduce el
    baseline pre-fix byte a byte."""
    import polyx.core.report_html as rh
    from tests.fixtures_informe import estado_benigno

    st = estado_benigno(directorio=tmp_path / "fotos")
    salida = tmp_path / "informe.html"
    rh.generate_report(st, salida, secciones=list(rh.IDS_SECCIONES))

    datos = salida.read_bytes()
    baseline = BASELINE_HTML.read_bytes()
    assert datos == baseline, (
        f"el render benigno divergio del baseline pre-fix "
        f"({len(datos)} B vs {len(baseline)} B): el cambio de escape toco "
        f"algo del documento para entradas benignas")


def test_sha256_del_render_benigno_coincide_con_el_registrado(tmp_path):
    """El digest persistido en informe.sha256 delimita que parte muto: si el
    compare de bytes falla, este segundo arbol da el valor exacto a buscar."""
    import polyx.core.report_html as rh
    from tests.fixtures_informe import estado_benigno

    st = estado_benigno(directorio=tmp_path / "fotos")
    salida = tmp_path / "informe.html"
    rh.generate_report(st, salida, secciones=list(rh.IDS_SECCIONES))

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
