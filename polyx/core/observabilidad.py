"""Log estructurado para diagnosticar un incidente sin consola delante.

Poly-X puede correr bajo pythonw: ``sys.stdout`` y ``sys.stderr`` quedan en
``None`` y se redirigen a os.devnull (ver ``polyx/__init__.py``). Todo lo que
hoy se diagnostica a mano vive repartido entre dialogs Qt y artifacts del run;
si un hilo de fondo revienta, el traceback se pierde sin rastro.

Aqui se centralizan las tres piezas que hacen usable el diagnostico a las 2 AM:

 - LOGS: un JSONL por modulo en ``~/.polyx/logs/`` (o ``POLYX_LOG_DIR``), con
   rotacion por tamano (5 MB x 3 copias por modulo) y poda del total del
   directorio con techo de ``TAMANO_TOTAL`` (200 MB): si el directorio entero
   pasa del techo se borra el archivo mas viejo primero (los backups rotados
   incluidos) y nunca el archivo activo de los loggers abiertos. El sobre
   lleva ts/level/module/pid/thread y, dentro de un run, ``run_id``. Nunca
   stderr.
 - CORRELACION: run_id en un threading.local; set_run_id / clear_run_id
   alrededor de la corrida (QThread + pools comparten el contexto por hilo).
   Los nombres de evento son verbo.objeto Y EN INGLES: convencion de la
   industria, a peticion de quien operara el sistema; los datos viajan en el
   campo ``data``.
 - CRONOMETRO: pares evento.start / evento.finish con ``ms`` de pared, y
   evento.error con traceback si la pieza peta.

El modulo no puede hacer caer el programa: si la carpeta de log no existe ni
se puede crear (HOME read-only, perfil bloqueado), el logger queda en
NullHandler y la corrida sigue igual que hoy. Un observador que no funciona
no puede ser peor que no tener observabilidad.
"""
from __future__ import annotations

import json
import logging
import os
import stat
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator, Optional

__all__ = [
    "logger_para", "evento", "cronometro", "set_run_id", "run_id_actual",
    "clear_run_id", "carpeta_logs", "evento_error", "podar", "TAMANO_TOTAL",
]

TAMANO_MAX = 5 * 1024 * 1024     # 5 MB por archivo; x 3 copias: ~20 MB por modulo
COPIAS = 3
TAMANO_TOTAL = 200 * 1024 * 1024  # techo del tamaño TOTAL de la carpeta de logs

_run_local = threading.local()
_loggers: dict[str, logging.Logger] = {}
# RLock: podar() lo toma dentro de logger_para, que ya lo tiene tomado.
_loggers_lock = threading.RLock()
# Este lock protege la creacion y el cacheo de loggers; los registros
# individuales los trata logging con su propio lock por handler.


def carpeta_logs() -> Path:
    """Carpeta de logs: ``POLYX_LOG_DIR`` si existe, si no ``~/.polyx/logs``."""
    return Path(os.environ.get("POLYX_LOG_DIR") or (Path.home() / ".polyx" / "logs"))


def _nivel_desde_entorno() -> int:
    mapping = {"debug": logging.DEBUG, "info": logging.INFO,
               "warn": logging.WARNING, "warning": logging.WARNING,
               "error": logging.ERROR}
    return mapping.get(os.environ.get("POLYX_LOG", "info").strip().lower(),
                       logging.INFO)


def _sobre_json(record: logging.LogRecord) -> str:
    """Sobre JSON de un LogRecord: campos fijos + payload del evento."""
    sobre: dict = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")
              .replace("+00:00", "Z"),
        "level": record.levelname,
        "module": record.name,
        "pid": os.getpid(),
        "thread": record.threadName,
        "event": getattr(record, "event", record.getMessage()),
        "run_id": getattr(record, "run_id",
                          getattr(_run_local, "run_id", None)),
    }
    ms = getattr(record, "ms", None)
    if ms is not None:
        sobre["ms"] = ms
    datos = getattr(record, "data", None)
    if datos:
        try:
            json.dumps(datos)      # verificacion: serializable tal cual
            sobre["data"] = datos
        except (TypeError, ValueError):
            # Payload con objetos raros (paths, ndarrays): a string, nunca tumba.
            sobre["data"] = {k: str(v) for k, v in datos.items()}
    if record.exc_info:
        import traceback
        sobre["exc"] = traceback.format_exception(*record.exc_info)
    return json.dumps(sobre, default=str, ensure_ascii=False)


class _SobreFormateador(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            return _sobre_json(record)
        except Exception:
            # Un log que rompe no puede tumbar la pieza que lo usa.
            return json.dumps({"level": record.levelname, "module": record.name,
                               "event": "log.format_error"}, ensure_ascii=False)


def logger_para(modulo: str) -> logging.Logger:
    """Logger file-only para un modulo, con rotacion de tamano.

    Si la carpeta no se puede crear (HOME bloqueado), devuelve el logger con
    NullHandler: el programa queda tan como estaba, sin escritor.
    """
    with _loggers_lock:
        ya = _loggers.get(modulo)
        if ya is not None:
            return ya

        logger = logging.getLogger(f"polyx.{modulo}")
        logger.setLevel(_nivel_desde_entorno())
        logger.propagate = False
        try:
            carpeta = carpeta_logs()
            carpeta.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                carpeta / f"{modulo}.jsonl",
                maxBytes=TAMANO_MAX, backupCount=COPIAS, encoding="utf-8")
            handler.setFormatter(_SobreFormateador())
            logger.addHandler(handler)
            # Cada arranque de modulo chequea que el directorio siga acotado.
            # El archivo recien creado se protege explicitamente: todavia no
            # esta cacheado en _loggers cuando la poda corre.
            podar(carpeta, activo=carpeta / f"{modulo}.jsonl")
        except OSError:
            logger.addHandler(logging.NullHandler())
        _loggers[modulo] = logger
        return logger


def podar(carpeta: Path | None = None, activo: Path | str | None = None) -> None:
    """Poda el directorio de logs por tamano TOTAL: techo ``TAMANO_TOTAL``.

    Suma el tamano de los archivos del directorio (con ``lstat``: un symlink
    cuenta solo su enlace, nunca el destino, y no se sigue fuera de la
    carpeta) y, si pasa del techo, borra el archivo mas viejo primero por
    mtime hasta quedar debajo. Los backups rotados entran en la lista igual
    que los log activos. Nunca se borra el destino de escritura activo: los
    ``baseFilename`` de los handlers abiertos en cache mas ``activo``, que es
    el archivo del logger que acaba de arrancar. Nunca lanza: borrar log no
    puede tumbar nada, filosofia de ``evento()``.
    """
    try:
        carpeta = carpeta_logs() if carpeta is None else Path(carpeta)
        if not carpeta.is_dir():
            return
        activos: set[str] = set()
        if activo is not None:
            activos.add(os.path.normcase(os.path.abspath(str(activo))))
        with _loggers_lock:
            for lg in list(_loggers.values()):
                for h in lg.handlers:
                    base = getattr(h, "baseFilename", None)
                    if base:
                        activos.add(os.path.normcase(os.path.abspath(base)))
        total = 0
        candidatos: list[tuple[float, str, int, str]] = []  # (mtime, nombre, bytes, ruta)
        for entrada in carpeta.iterdir():
            try:
                st = entrada.lstat()
            except OSError:
                continue                 # desaparecio mientras listabamos
            if stat.S_ISDIR(st.st_mode):
                continue
            total += st.st_size
            ruta = os.path.normcase(os.path.abspath(str(entrada)))
            if ruta not in activos:
                candidatos.append((st.st_mtime, entrada.name, st.st_size, ruta))
        if total <= TAMANO_TOTAL:
            return
        candidatos.sort()                # mas viejo primero (mtime, nombre)
        for _mtime, _nombre, tamano, ruta in candidatos:
            if total <= TAMANO_TOTAL:
                break
            try:
                os.unlink(ruta)
            except OSError:
                continue                 # no se pudo; se sigue con el proximo
            total -= tamano
    except Exception:
        # La poda es el mejor esfuerzo de un observador: nada de aqui abajo
        # merece tumbar una corrida real.
        pass


# ── Correlacion por run ──
def set_run_id(run_id: str) -> None:
    """Marca el run_id de este hilo (thread-local; cada QThread el suyo)."""
    _run_local.run_id = run_id or None


def run_id_actual() -> Optional[str]:
    return getattr(_run_local, "run_id", None)


def clear_run_id() -> None:
    _run_local.run_id = None


def evento_error(logger: logging.Logger, nombre: str, exc: BaseException,
                 ms: Optional[int] = None, **datos) -> None:
    """Evento a nivel ERROR con traceback completo.

    Existe para lo que hoy se pierde: un hilo de fondo que revienta bajo
    pythonw no deja rastro por consola. Nunca lanza.
    """
    try:
        logger._log(logging.ERROR, nombre, (), exc_info=exc,
                    extra={"event": nombre, "data": datos, "ms": ms})
    except Exception:
        pass


def evento(logger: logging.Logger, nombre: str, level: int = logging.INFO,
           ms: Optional[int] = None, **datos) -> None:
    """Emite un evento verbo.objeto con payload en ``datos``.

    Nunca lanza: un fallo del sistema de log no puede parar el analisis.
    """
    try:
        logger._log(level, nombre, (),
                    extra={"event": nombre, "data": datos, "ms": ms})
    except Exception:
        pass


@contextmanager
def cronometro(logger: logging.Logger, nombre: str, level: int = logging.INFO,
               **datos) -> Iterator[None]:
    """Context manager que emite ``nombre.start`` / ``nombre.finish`` con ms.

    Si el bloque lanza, emite ``nombre.error`` a ERROR con traceback y
    RELANZA: registrar para observabilidad no captura el error de nadie.
    """
    evento(logger, f"{nombre}.start", level=level, **datos)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        ms = int((time.perf_counter() - t0) * 1000)
        try:
            logger._log(logging.ERROR, f"{nombre}.error", (), exc_info=exc,
                        extra={"event": f"{nombre}.error", "data": datos,
                               "ms": ms})
        except Exception:
            pass
        raise
    ms = int((time.perf_counter() - t0) * 1000)
    evento(logger, f"{nombre}.finish", level=level, ms=ms, **datos)
