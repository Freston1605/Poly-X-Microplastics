"""Que el sistema de log que nunca puede tumbar nada, siga siendo util.

El observabilidad es infraestructura invisible si falla: estas pruebas fijan
el sobre necesario para diagnosticar a las 2 AM (ts/level/modulo/pid/thread/
run_id), la correlacion por run por hilo, el cronometro con su par start/
finish y su error con traceback, y sobre todo el modo de fallo a prueba de
todo: sin carpeta de log el programa sigue parecido a como estaba.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _leer_jsonl(carpeta: Path, modulo: str) -> list[dict]:
    ruta = carpeta / f"{modulo}.jsonl"
    assert ruta.exists(), "el log no se escribio donde se pidio"
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    carpeta = tmp_path / "logs"
    monkeypatch.setenv("POLYX_LOG_DIR", str(carpeta))
    carpeta.mkdir()  # la poda crea archivos antes que el primer logger
    # limpiar cache de loggers: cada prueba tiene su carpeta propria
    import polyx.core.observability as ob

    monkeypatch.delenv("POLYX_LOG_TOTAL_CAP_MB", raising=False)
    ob._close_loggers()
    monkeypatch.delenv("POLYX_LOG", raising=False)
    ob._run_local.__dict__.pop("run_id", None)
    yield carpeta
    ob._close_loggers()


def test_sobre_minimo_con_campo_fijos(log_dir):
    from polyx.core import observability as ob

    log = ob.logger_for("prueba")
    ob.event(log, "prueba.ping", placas=3)
    recs = _leer_jsonl(log_dir, "prueba")
    assert len(recs) == 1
    rec = recs[0]
    for fijo in ("ts", "level", "module", "pid", "thread", "event"):
        assert fijo in rec, f"el sobre debe llevar {fijo}"
    assert rec["event"] == "prueba.ping"  # clave de evento, no mensaje
    assert rec["data"] == {"placas": 3}
    assert rec["run_id"] is None  # sin run, sin run_id
    assert "polyx.prueba" in rec["module"] or rec["module"].endswith("prueba")


def test_run_id_correlaciona_por_hilo(log_dir):
    from polyx.core import observability as ob

    log = ob.logger_for("prueba")
    ob.set_run_id("runA")
    ob.event(log, "prueba.a")

    # otro hilo NO hereda el run_id del principal
    def _otro():
        ob.event(log, "prueba.a")

    h = threading.Thread(target=_otro)
    h.start()
    h.join()
    recs = _leer_jsonl(log_dir, "prueba")
    assert recs[0]["run_id"] == "runA"
    assert recs[1]["run_id"] is None
    ob.clear_run_id()
    assert ob.run_id_current() is None


def test_cronometro_emite_par_y_ms(log_dir):
    import time

    from polyx.core import observability as ob

    log = ob.logger_for("prueba")
    ob.set_run_id("runB")
    with ob.timer(log, "prueba.cosa", placas=2):
        time.sleep(0.01)
    recs = _leer_jsonl(log_dir, "prueba")
    eventos = [r["event"] for r in recs]
    assert eventos == ["prueba.cosa.start", "prueba.cosa.finish"]
    fin = recs[-1]
    assert fin["ms"] >= 10 and fin["run_id"] == "runB"


def test_cronometro_en_error_relaza_y_ground_trasbazo(log_dir):
    from polyx.core import observability as ob

    log = ob.logger_for("prueba")
    with pytest.raises(ValueError, match="boom"):
        with ob.timer(log, "prueba.fallo", imagen="mala.png"):
            raise ValueError("boom")
    recs = _leer_jsonl(log_dir, "prueba")
    assert recs[-1]["event"] == "prueba.fallo.error"
    assert recs[-1]["level"] == "ERROR"
    assert "boom" in "".join(recs[-1]["exc"])


def test_payload_no_serializable_no_tumba(log_dir):
    import numpy as np

    from polyx.core import observability as ob

    log = ob.logger_for("prueba")
    ob.event(log, "prueba.raro", mapa=np.zeros(3))
    recs = _leer_jsonl(log_dir, "prueba")
    assert recs[0]["data"]["mapa"].startswith("[")


def test_sin_carpeta_el_plan_no_revienta(tmp_path, monkeypatch):
    from polyx.core import observability as ob

    ob._loggers.clear()
    destino_archivo = tmp_path / "archivo"
    destino_archivo.write_text("x")
    monkeypatch.setenv("POLYX_LOG_DIR", str(destino_archivo / "logs"))
    # POLYX_LOG_DIR apunta DENTRO de un archivo: la carpeta de log no puede existir
    log = ob.logger_for("prueba_apagada")
    assert not [h for h in log.handlers if hasattr(h, "baseFilename")]
    ob.event(log, "prueba.que_se_ignora")  # no puede reventar
    ob._loggers.clear()


def test_nivel_por_entorno(tmp_path, monkeypatch):
    from polyx.core import observability as ob

    monkeypatch.setenv("POLYX_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("POLYX_LOG", "error")
    ob._loggers.clear()
    log = ob.logger_for("prueba_nivel")
    assert log.getEffectiveLevel() == logging.ERROR
    ob._loggers.clear()


# ── Poda por tamano TOTAL del directorio (techo 200 MB, viejo primero) ──


def _crear_falso(carpeta: Path, nombre: str, bytes_: int, mtime: float) -> Path:
    """Archivo falso de logs con mtime fijado (la poda ordena por mtime)."""
    ruta = carpeta / nombre
    ruta.write_bytes(b"x" * bytes_)
    os.utime(ruta, (mtime, mtime))
    return ruta


def _total_carpeta(carpeta: Path) -> int:
    return sum(os.path.getsize(p) for p in carpeta.iterdir())


def test_poda_borra_mas_viejo_primero(log_dir, monkeypatch):
    from polyx.core import observability as ob

    monkeypatch.setattr(ob, "SIZE_TOTAL", 900_000)  # techo pequeño de prueba
    t = 1_000_000.0
    for i in range(5):  # 5 archivos de 300 KB = 1.5 MB
        _crear_falso(log_dir, f"modulo.jsonl.{i + 1}", 300_000, t + i * 10)
    ob.prune(folder=log_dir)
    sobrantes = sorted(p.name for p in log_dir.iterdir())
    # hay que soltar 600 KB: caen los dos más viejos (.1 y .2), quedan 900 KB
    assert sobrantes == ["modulo.jsonl.3", "modulo.jsonl.4", "modulo.jsonl.5"]
    assert _total_carpeta(log_dir) <= 900_000


def test_poda_no_borra_archivo_activo(log_dir, monkeypatch):
    from polyx.core import observability as ob

    monkeypatch.setattr(ob, "SIZE_TOTAL", 900_000)
    t = 2_000_000.0
    # el MAS viejo tambien es el activo: ~300 KB de JSONL valido simulado
    linea = b'{"viejo":true}\n'
    activo = log_dir / "a.jsonl"
    activo.write_bytes(linea * 20_000)  # 300_000 bytes parseables
    os.utime(activo, (t, t))
    _crear_falso(log_dir, "b.jsonl", 300_000, t + 10)
    _crear_falso(log_dir, "c.jsonl", 300_000, t + 20)
    _crear_falso(log_dir, "d.jsonl", 400_000, t + 30)  # total 1.3 MB
    # abrir el logger de "a" deja a.jsonl como destino activo (handler abierto)
    log = ob.logger_for("a")
    ob.event(log, "a.ping")  # el activo recibe escritura real
    ob.prune(folder=log_dir)  # y una poda directa del operador encima
    nombres = {p.name for p in log_dir.iterdir()}
    assert "a.jsonl" in nombres, "la poda borro el archivo activo"
    assert "b.jsonl" not in nombres  # los no activos sí salen, viejo primero
    assert "c.jsonl" not in nombres
    assert "d.jsonl" in nombres  # con b+c (600 KB) fuera queda bajo techo
    assert _total_carpeta(log_dir) <= 900_000
    assert activo.exists()
    recs = _leer_jsonl(log_dir, "a")  # el log activo sigue utilizándose
    assert recs[-1]["event"] == "a.ping"


def test_poda_mantiene_directorio_acotado_lote_simulado(log_dir, monkeypatch):
    from polyx.core import observability as ob

    techo = 900_000
    monkeypatch.setattr(ob, "SIZE_TOTAL", techo)
    t = 3_000_000.0
    for i in range(40):  # uso simulado largo: 40 llegadas nuevas
        _crear_falso(log_dir, f"placa_{i:03}.jsonl", 300_000, t + i * 10)
        ob.prune(folder=log_dir)  # cada arranque de modulo poda
        total = _total_carpeta(log_dir)
        assert total <= techo, f"iteracion {i}: el directorio acumulo {total}"
    sobrantes = list(log_dir.iterdir())
    assert 1 <= len(sobrantes) <= 4  # acotado: caben ~3 archivos de 300 KB
    assert _total_carpeta(log_dir) <= techo


def test_rotacion_5mb_x3(log_dir):
    """Rotacion real por tamano: >5 MB fuerza rollover y COPIAS techa el set.

    maxBytes es fijo al crear el handler: la rotacion se prueba con escrituras
    reales, no con archivos falsos.
    """
    from polyx.core import observability as ob

    log = ob.logger_for("prueba_rot")
    bloque = "x" * (2 * 1024 * 1024)  # registros de ~2 MB
    for _ in range(8):  # ~17 MB escritos en disco
        ob.event(log, "prueba.rot", text=bloque)
    assert (log_dir / "prueba_rot.jsonl.1").exists(), "no hubo rollover"
    assert (log_dir / "prueba_rot.jsonl").exists()
    backups = [n for n in os.listdir(log_dir) if n.startswith("prueba_rot.jsonl.")]
    assert len(backups) <= ob.COPIES, (
        f"los backups crecieron sin techo: {len(backups)} > {ob.COPIES}"
    )
    assert _total_carpeta(log_dir) <= ob.SIZE_TOTAL  # la poda no toco nada aqui


# ── Fallo de escritura TARDIO: el handler vivo se rompe a mitad de corrida ──


def test_fallo_de_escritura_posterior_no_revienta(log_dir, monkeypatch):
    """El handler abrio bien su archivo, y DESPUES el stream se rompe.

    El fallo de creacion ya esta cubierto (NullHandler); aqui se corrompe el
    stream del handler vivo en memoria para que cada emit explote debajo de
    ``evento()``/``cronometro()``: la corrida tiene que seguir igual.
    """
    from polyx.core import observability as ob

    log = ob.logger_for("prueba_late_fail")
    handlers = [h for h in log.handlers if hasattr(h, "baseFilename")]
    assert handlers, "el logger debio abrir su archivo antes de romperse"

    class _StreamRoto:
        """Stream que explota en cada operacion, como un disco lleno."""

        def write(self, *a, **k):
            raise OSError("disco lleno")

        def flush(self):
            raise OSError("disco lleno")

    roto = _StreamRoto()
    for h in handlers:
        monkeypatch.setattr(h, "stream", roto)

    ob.event(log, "prueba.despues", x=1)  # no reventa
    assert ob.run_id_current() is None  # el hilo sigue intacto

    ejecutado: list[bool] = []
    with ob.timer(log, "prueba.late"):  # el cuerpo corre igual
        ejecutado.append(True)
    assert ejecutado == [True], "el bloque del cronometro no debio saltarse"


def test_techo_poda_por_defecto_y_entorno(log_dir, monkeypatch):
    from polyx.core import observability as ob

    # Por defecto (sin variable) es TAMANO_TOTAL = 200 MB
    assert ob._ceiling_pruning_bytes() == ob.SIZE_TOTAL == 200 * 1024 * 1024

    # Entorno con 1.5 MB
    monkeypatch.setenv("POLYX_LOG_TOTAL_CAP_MB", "1.5")
    assert ob._ceiling_pruning_bytes() == int(1.5 * 1024 * 1024)

    # Entorno con 0.5 MB (524,288 bytes) y poda real
    monkeypatch.setenv("POLYX_LOG_TOTAL_CAP_MB", "0.5")
    t = 4_000_000.0
    for i in range(5):
        _crear_falso(log_dir, f"modulo.jsonl.{i + 1}", 300_000, t + i * 10)
    ob.prune(folder=log_dir)
    sobrantes = sorted(p.name for p in log_dir.iterdir())
    assert sobrantes == ["modulo.jsonl.5"]
    assert _total_carpeta(log_dir) <= 524_288


def test_techo_poda_entorno_invalido_cae_al_defecto(log_dir, monkeypatch):
    from polyx.core import observability as ob

    valores_invalidos = ["", "   ", "abc", "0", "-5", "nan", "inf", "-inf", "1e-12"]
    for val in valores_invalidos:
        monkeypatch.setenv("POLYX_LOG_TOTAL_CAP_MB", val)
        assert ob._ceiling_pruning_bytes() == ob.SIZE_TOTAL

    monkeypatch.setenv("POLYX_LOG_TOTAL_CAP_MB", "invalido")
    _crear_falso(log_dir, "m1.jsonl", 300_000, 100.0)
    _crear_falso(log_dir, "m2.jsonl", 300_000, 200.0)
    ob.prune(folder=log_dir)
    sobrantes = sorted(p.name for p in log_dir.iterdir())
    assert sobrantes == ["m1.jsonl", "m2.jsonl"]


def test_cambio_de_carpeta_cierra_handler_anterior(tmp_path, monkeypatch):
    from polyx.core import observability as ob

    dir_a = tmp_path / "logs_a"
    dir_b = tmp_path / "logs_b"

    monkeypatch.setenv("POLYX_LOG_DIR", str(dir_a))
    ob._close_loggers()
    log_a = ob.logger_for("detector")
    ob.event(log_a, "run.a")

    ob._loggers.clear()
    monkeypatch.setenv("POLYX_LOG_DIR", str(dir_b))
    log_b = ob.logger_for("detector")
    ob.event(log_b, "run.b")

    handlers_b = [h for h in log_b.handlers if hasattr(h, "baseFilename")]
    assert len(handlers_b) == 1
    assert Path(handlers_b[0].baseFilename) == dir_b / "detector.jsonl"

    recs_a = [r["event"] for r in _leer_jsonl(dir_a, "detector")]
    recs_b = [r["event"] for r in _leer_jsonl(dir_b, "detector")]
    assert recs_a == ["run.a"]
    assert recs_b == ["run.b"]


def test_conftest_aisla_polyx_log_dir_fuera_de_home(tmp_path):
    from polyx.core import observability as ob

    assert ob.folder_logs() == tmp_path / "_polyx_logs"
    assert Path.home() / ".polyx" / "logs" != ob.folder_logs()


def test_generate_report_escribe_en_log_dir_aislado(tmp_path):
    import cv2
    import numpy as np

    from polyx.core.detect.protocol import Instance as Detection
    from polyx.gui import report_html
    from polyx.gui.detector.state import DetectorState, ImageResult, ModelSlot

    img = np.full((260, 320, 3), 18, np.uint8)
    cv2.circle(img, (160, 130), 20, (60, 210, 240), -1)
    is_ok, buf = cv2.imencode(".png", img)
    assert is_ok
    ruta = tmp_path / "foto.png"
    ruta.write_bytes(buf.tobytes())

    d = Detection(class_id=0, class_name="PET", conf=0.90, x1=140, y1=110, x2=180, y2=150)
    st = DetectorState()
    st.model_slots[0] = ModelSlot(alias="m1", path=tmp_path / "fake.pt")
    st.results = {
        0: [ImageResult(image_path=ruta, model_idx=0, predictions=[d], gt=[], has_gt=False)]
    }

    salida = tmp_path / "out.html"
    report_html.generate_report(st, salida, sections=["resumen"])

    log_file = tmp_path / "_polyx_logs" / "detector.jsonl"
    assert log_file.is_file()
    eventos = [r["event"] for r in _leer_jsonl(tmp_path / "_polyx_logs", "detector")]
    assert "report.start" in eventos
    assert "report.finish" in eventos
