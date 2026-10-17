"""Que el arranque del launcher deje rastro JSONL sin poder tumbar la ventana.

El launcher era la unica de las cuatro rutas instrumentadas del hito que no
emiti­a nada: abrir el programa y abrir un modulo no ocupaba espacio en los
logs. Estas pruebas fijan los dos eventos nuevos (bootstrap.finish en
app.jsonl al arrancar el programa, module.launch / module.launch.error al
lanzar cada modulo) y el modo de fallo a prueba de todo de siempre: si el
log no se puede escribir, la ventana abre igual que siempre.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polyx.model.observabilidad as ob   # noqa: E402  (despues del path hack)


def _leer_jsonl(carpeta: Path, modulo: str) -> list[dict]:
    ruta = carpeta / f"{modulo}.jsonl"
    assert ruta.exists(), "el log no se escribio donde se pidio"
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Carpeta de logs propia por prueba y cache de loggers limpia.

    QT_QPA_PLATFORM=offscreen se fija ANTES de importar polyx.gui.launcher (el
    import lo hacen las pruebas, despues del fixture), para que Qt no necesite
    pantalla aunque algo se inicialice al importar. No se crea QApplication:
    solo se prueban funciones a nivel de modulo.
    """
    carpeta = tmp_path / "logs"
    monkeypatch.setenv("POLYX_LOG_DIR", str(carpeta))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("POLYX_LOG", raising=False)
    ob._loggers.clear()
    ob._run_local.__dict__.pop("run_id", None)
    carpeta.mkdir()      # la poda del primer logger mira archivos existentes
    yield carpeta
    ob._loggers.clear()
    ob._run_local.__dict__.pop("run_id", None)


def test_bootstrap_registra_evento(log_dir):
    import polyx.gui.launcher as launcher
    launcher._registrar_arranque()
    recs = _leer_jsonl(log_dir, "app")
    assert len(recs) == 1, "el arranque deja exactamente un rastro"
    rec = recs[0]
    for fijo in ("ts", "level", "module", "event", "run_id"):
        assert fijo in rec, f"el sobre debe llevar {fijo}"
    assert rec["event"] == "bootstrap.finish"
    assert rec["module"].endswith("app")
    assert rec["data"]["version"] == launcher.__version__
    assert rec["data"]["python"]                 # rama de python, p.ej "3.12.4"
    assert rec["data"]["sistema"]                # nombre_sistema(), no vacio


def test_bootstrap_no_revienta_sin_carpeta(tmp_path, monkeypatch):
    """POLYX_LOG_DIR dentro de un archivo: el arranque sigue sin lanzar."""
    import polyx.gui.launcher as launcher
    ob._loggers.clear()
    destino_archivo = tmp_path / "archivo"
    destino_archivo.write_text("x")
    monkeypatch.setenv("POLYX_LOG_DIR", str(destino_archivo / "logs"))
    launcher._registrar_arranque()      # no puede lanzar
    ob._loggers.clear()


def test_lanzar_modulo_registra_module_launch(log_dir, monkeypatch):
    """Popen con el MISMO argv y cwd de siempre + module.launch en app.jsonl."""
    import polyx.gui.launcher as launcher
    llamadas = []

    class FakePopen:
        def __init__(self, args, **kwargs):
            llamadas.append((args, kwargs))

    monkeypatch.setattr(launcher.subprocess, "Popen", FakePopen)
    assert launcher._spawn_module("polyx.gui.detector") is True
    assert len(llamadas) == 1
    args, kwargs = llamadas[0]
    assert args == [sys.executable, "-m", "polyx.gui.detector"], \
        "no se puede cambiar como se lanzan los procesos"
    assert kwargs["cwd"] == str(Path(launcher.__file__).resolve().parents[2])
    recs = _leer_jsonl(log_dir, "app")
    eventos = [r for r in recs if r["event"] == "module.launch"]
    assert len(eventos) == 1
    assert eventos[0]["data"] == {"modulo": "polyx.gui.detector"}


def test_lanzar_modulo_inexistente_devuelve_false(log_dir, monkeypatch,):
    """find_spec None: False y nada escrito; la ventana decide el aviso."""
    import polyx.gui.launcher as launcher
    lanzadas = []

    class FakePopen:
        def __init__(self, args, **kwargs):
            lanzadas.append(args)

    monkeypatch.setattr(launcher.subprocess, "Popen", FakePopen)
    assert launcher._spawn_module("polyx.modulo_que_no_existe") is False
    assert lanzadas == []                # jamas llega a Popen
    ruta_log = log_dir / "app.jsonl"
    if ruta_log.exists():                # logger 'app' pudo arrancar
        recs = [json.loads(l) for l in ruta_log.read_text().splitlines() if l.strip()]
        assert recs == []                # y no emitio rastro ninguno


def test_lanzar_modulo_con_error_emite_y_no_revienta(log_dir, monkeypatch):
    """Popen que revienta: _spawn_module devuelve False y deja el rastro."""
    import polyx.gui.launcher as launcher

    def reventar(*args, **kwargs):
        raise OSError("no hay ejecutable")

    monkeypatch.setattr(launcher.subprocess, "Popen", reventar)
    # Contrato elegido: _spawn_module NUNCA lanza (True/False), porque el
    # modulo Python lo llama la ventana; el error queda solo como evento.
    assert launcher._spawn_module("polyx.gui.detector") is False
    recs = _leer_jsonl(log_dir, "app")
    errores = [r for r in recs if r["event"] == "module.launch.error"]
    assert len(errores) == 1
    assert errores[0]["level"] == "WARNING"
    assert errores[0]["data"]["modulo"] == "polyx.gui.detector"
    assert "no hay ejecutable" in errores[0]["data"]["error"]
