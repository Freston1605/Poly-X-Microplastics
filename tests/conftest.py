"""Configuracion compartida de la suite.

El idioma del programa se resuelve una vez a partir del entorno (ver
``polyx.core.i18n``) y se cachea en el modulo, asi que varios tests que no
fuercen idioma propio --p. ej. ``test_asignacion`` lee texto del informe--
acaban dependiendo del locale de la maquina donde corra pytest. Con un
locale en ingles el informe sale en ingles y la prueba falla; en CI el
locale suele estar vacio y pasa por casualidad.

Aqui fijamos espanol como idioma de la suite, explicito y reproducible. Las
pruebas que si prueban idiomas --``test_informe_bilingue``-- fijan el suyo
mediante ``POLYX_IDIOMA`` directamente, y ``setdefault`` respeta lo que alguien
pida a mano.

Aislamiento de observabilidad y logs (Decision D007):
Varios modulos de prueba (como ``test_asignacion`` o ``test_informe_bilingue``)
invocan el pipeline real de generacion de reportes (``report_html.generate_report``),
el cual instrumenta eventos a traves de ``logger_para("detector")``. Sin una variable
de entorno ``POLYX_LOG_DIR`` explicita, estos loggers escriben en el directorio
real del usuario (``~/.polyx/logs/``). Para garantizar que la ejecucion de la suite
de pruebas sea completamente hermetica y nunca contamine ``~/.polyx/logs/``,
definimos un fixture autouse que aisla ``POLYX_LOG_DIR`` en un directorio temporal
por test (``tmp_path / "_polyx_logs"``), limpia variables de entorno ambientales
(``POLYX_LOG``, ``POLYX_LOG_TOTAL_CAP_MB``) y cierra y reinicia los loggers y el
run_id en setup y teardown.
"""
import os

import pytest

os.environ.setdefault("POLYX_IDIOMA", "es")


@pytest.fixture(autouse=True)
def _aislar_logs_suite(tmp_path, monkeypatch):
    """Aisla POLYX_LOG_DIR a tmp_path / '_polyx_logs' para toda la suite.

    Evita que generate_report y modulos de prueba escriban en ~/.polyx/logs/.
    No crea el directorio con .mkdir() para no colisionar con tests que
    crean tmp_path / 'logs'. Cierra loggers y limpia run_id antes y despues
    de cada prueba.
    """
    monkeypatch.setenv("POLYX_LOG_DIR", str(tmp_path / "_polyx_logs"))
    monkeypatch.delenv("POLYX_LOG", raising=False)
    monkeypatch.delenv("POLYX_LOG_TOTAL_CAP_MB", raising=False)

    import polyx.core.observabilidad as ob

    ob._cerrar_loggers()
    ob.clear_run_id()
    yield
    ob._cerrar_loggers()
    ob.clear_run_id()

