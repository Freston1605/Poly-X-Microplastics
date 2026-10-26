"""Structured logging to diagnose an incident with no console in front.

Poly-X can run under pythonw: ``sys.stdout`` and ``sys.stderr`` stay ``None``
and are redirected to os.devnull (see ``polyx/__init__.py``). Everything
diagnosed by hand today lives scattered between Qt dialogs and run artifacts;
if a background thread blows up, the traceback is lost without a trace.

This module centralizes the three pieces that make diagnostics usable at
2 AM:

 - LOGS: one JSONL per module in ``~/.polyx/logs/`` (or ``POLYX_LOG_DIR``),
   with size rotation (5 MB x 3 copies per module) and pruning of the whole
   directory with a cap of ``SIZE_TOTAL`` (200 MB, configurable via
   ``POLYX_LOG_TOTAL_CAP_MB``): when the whole directory goes past the cap
   the oldest file is deleted first (rotated backups included) and never the
   active file of the open loggers. The envelope carries
   ts/level/module/pid/thread and, inside a run, ``run_id``. Never stderr.
 - CORRELATION: run_id on a threading.local; set_run_id / clear_run_id
   around the run (QThread + pools share the context per thread). Event
   names are verb.object AND IN ENGLISH: industry convention, at the request
   of whoever operates the system; the data travels in the ``data`` field.
 - STOPWATCH: evento.start / evento.finish pairs with wall ``ms``, and
   evento.error with the traceback when a piece crashes.

The module cannot bring the program down: if the log folder cannot be
created (read-only HOME, locked profile), the logger falls back to
NullHandler and the run continues exactly as today. An observer that does
not work must not be worse than having no observability."""

from __future__ import annotations

import json
import logging
import math
import os
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__ = [
    "SIZE_TOTAL",
    "clear_run_id",
    "event",
    "event_error",
    "folder_logs",
    "logger_for",
    "prune",
    "run_id_current",
    "set_run_id",
    "timer",
]

SIZE_MAX = 5 * 1024 * 1024  # 5 MB per file; x 3 copies: ~20 MB per module
COPIES = 3
SIZE_TOTAL = 200 * 1024 * 1024  # cap on the TOTAL size of the logs folder

_run_local = threading.local()
_loggers: dict[str, logging.Logger] = {}
# RLock: `podar()` acquires it inside `logger_para()`, which already holds it.
_loggers_lock = threading.RLock()
# This lock protects logger creation and caching; individual records
# are handled by logging with its own per-handler lock.


def folder_logs() -> Path:
    """Logs folder: ``POLYX_LOG_DIR`` if set, otherwise ``~/.polyx/logs``."""
    return Path(os.environ.get("POLYX_LOG_DIR") or (Path.home() / ".polyx" / "logs"))


def _level_from_environment() -> int:
    mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    return mapping.get(os.environ.get("POLYX_LOG", "info").strip().lower(), logging.INFO)


def _about_json(record: logging.LogRecord) -> str:
    """From a LogRecord to JSON: fixed fields + event payload."""
    about: dict = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": record.levelname,
        "module": record.name,
        "pid": os.getpid(),
        "thread": record.threadName,
        "event": getattr(record, "event", record.getMessage()),
        "run_id": getattr(record, "run_id", getattr(_run_local, "run_id", None)),
    }
    ms = getattr(record, "ms", None)
    if ms is not None:
        about["ms"] = ms
    data = getattr(record, "data", None)
    if data:
        try:
            json.dumps(data)  # check: serialisable as is
            about["data"] = data
        except (TypeError, ValueError):
            # Payload with unusual objects (paths, ndarrays): turned into a string, never crashes.
            about["data"] = {k: str(v) for k, v in data.items()}
    if record.exc_info:
        import traceback

        about["exc"] = traceback.format_exception(*record.exc_info)
    return json.dumps(about, default=str, ensure_ascii=False)


class _AboutFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            return _about_json(record)
        except Exception:  # noqa: BLE001
            # A log that breaks cannot bring down the piece that uses it.
            return json.dumps(
                {"level": record.levelname, "module": record.name, "event": "log.format_error"},
                ensure_ascii=False,
            )


def _ceiling_pruning_bytes() -> int:
    """Returns the maximum cap of the logs folder in bytes.

    Reads ``POLYX_LOG_TOTAL_CAP_MB`` from the environment. If it is
    undefined, empty, non-numeric, infinite/NaN or <= 0, it falls back to
    ``SIZE_TOTAL`` (read dynamically from the module to respect test
    configurations).
    """
    raw = os.environ.get("POLYX_LOG_TOTAL_CAP_MB", "").strip()
    if not raw:
        return SIZE_TOTAL
    try:
        mb = float(raw)
    except ValueError:
        return SIZE_TOTAL
    if not math.isfinite(mb) or mb <= 0:
        return SIZE_TOTAL
    ceiling = int(mb * 1024 * 1024)
    return ceiling if ceiling > 0 else SIZE_TOTAL


def _close_loggers() -> None:
    """Closes and removes the handlers of every logger and empties the _loggers
    cache.

        Never raises: useful to isolate tests or reconfigure log directories.
    """
    try:
        with _loggers_lock:
            loggers_to_close = list(_loggers.values())
            try:
                for name, lg in getattr(logging.Logger.manager, "loggerDict", {}).items():
                    if isinstance(lg, logging.Logger) and (  # noqa: SIM102
                        name == "polyx" or name.startswith("polyx.")
                    ):
                        if lg not in loggers_to_close:
                            loggers_to_close.append(lg)
            except Exception:  # noqa: BLE001, S110
                pass
            for lg in loggers_to_close:
                for h in list(lg.handlers):
                    try:  # noqa: SIM105
                        lg.removeHandler(h)
                    except Exception:  # noqa: BLE001, S110
                        pass
                    try:  # noqa: SIM105
                        h.close()
                    except Exception:  # noqa: BLE001, S110
                        pass
            _loggers.clear()
    except Exception:  # noqa: BLE001, S110
        pass


def logger_for(module: str) -> logging.Logger:
    """File-only logger for one module, with size rotation.

    If the folder cannot be created (HOME locked), it returns the logger with
    NullHandler: the program stays as it was, with no writer.
    """
    with _loggers_lock:
        ya = _loggers.get(module)
        if ya is not None:
            return ya

        logger = logging.getLogger(f"polyx.{module}")
        logger.setLevel(_level_from_environment())
        logger.propagate = False

        for h in list(logger.handlers):
            try:  # noqa: SIM105
                logger.removeHandler(h)
            except Exception:  # noqa: BLE001, S110
                pass
            try:  # noqa: SIM105
                h.close()
            except Exception:  # noqa: BLE001, S110
                pass

        try:
            folder = folder_logs()
            folder.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                folder / f"{module}.jsonl", maxBytes=SIZE_MAX, backupCount=COPIES, encoding="utf-8"
            )
            handler.setFormatter(_AboutFormatter())
            logger.addHandler(handler)
            # Every module start checks that the directory is still within its cap.
            # The newly created file is explicitly protected: it is not yet
            # cached in `_loggers` when pruning runs.
            prune(folder, active_2=folder / f"{module}.jsonl")
        except OSError:
            logger.addHandler(logging.NullHandler())
        _loggers[module] = logger
        return logger


def prune(folder: Path | None = None, active_2: Path | str | None = None) -> None:
    """Prunes the logs directory by TOTAL size: cap ``SIZE_TOTAL`` (or ``POLYX_LOG_TOTAL_CAP_MB``).

    It adds up the size of the files in the directory (with ``lstat``: a
    symlink counts only its link, never the target, and nothing outside the
    folder is followed) and, if it goes past the cap, deletes the oldest file
    first by mtime until it is back under. Rotated backups enter the list the
    same as active logs. The active write target is never deleted: the
    ``baseFilename`` of the open handlers kept in cache plus ``activo``, the
    file of the logger that just started. The 200 MB cap (``SIZE_TOTAL``)
    can be overridden through the environment variable
    ``POLYX_LOG_TOTAL_CAP_MB``. Never raises: deleting logs cannot bring
    anything down, the ``evento()`` philosophy.
    """
    try:
        folder = folder_logs() if folder is None else Path(folder)
        if not folder.is_dir():
            return
        active_3: set[str] = set()
        if active_2 is not None:
            active_3.add(os.path.normcase(os.path.abspath(str(active_2))))
        with _loggers_lock:
            for lg in list(_loggers.values()):
                for h in lg.handlers:
                    base = getattr(h, "baseFilename", None)
                    if base:
                        active_3.add(os.path.normcase(os.path.abspath(base)))
        total = 0
        candidates_2: list[tuple[float, str, int, str]] = []  # (mtime, name, bytes, path)
        for input_value in folder.iterdir():
            try:
                st = input_value.lstat()
            except OSError:
                continue  # disappeared while we were listing
            if stat.S_ISDIR(st.st_mode):
                continue
            total += st.st_size
            path_2 = os.path.normcase(os.path.abspath(str(input_value)))
            if path_2 not in active_3:
                candidates_2.append((st.st_mtime, input_value.name, st.st_size, path_2))
        ceiling = _ceiling_pruning_bytes()
        if total <= ceiling:
            return
        candidates_2.sort()  # oldest first (mtime, name)
        for _mtime, _name, size, path_2 in candidates_2:
            if total <= ceiling:
                break
            try:
                os.unlink(path_2)
            except OSError:
                continue  # could not be removed; continue with the next one
            total -= size
    except Exception:  # noqa: BLE001, S110
        # Pruning is an observer's best effort: nothing below here
        # deserves to bring down a real run.
        pass


# ── Per-run correlation ──
def set_run_id(run_id: str) -> None:
    """Sets the run_id of this thread (thread-local; each QThread its own)."""
    _run_local.run_id = run_id or None


def run_id_current() -> str | None:
    return getattr(_run_local, "run_id", None)


def clear_run_id() -> None:
    _run_local.run_id = None


def event_error(
    logger: logging.Logger, name_2: str, exc: BaseException, ms: int | None = None, **data
) -> None:
    """ERROR-level event with a full traceback.

    It exists for what gets lost today: a background thread that blows up
    under pythonw leaves no console trace. Never raises.
    """
    try:  # noqa: SIM105
        logger._log(
            logging.ERROR, name_2, (), exc_info=exc, extra={"event": name_2, "data": data, "ms": ms}
        )
    except Exception:  # noqa: BLE001, S110
        pass


def event(
    logger: logging.Logger, name_2: str, level: int = logging.INFO, ms: int | None = None, **data
) -> None:
    """Emits a verb.object event with the payload in ``datos``.

    Never raises: a logging-system failure cannot stop the analysis.
    """
    try:  # noqa: SIM105
        logger._log(level, name_2, (), extra={"event": name_2, "data": data, "ms": ms})
    except Exception:  # noqa: BLE001, S110
        pass


@contextmanager
def timer(logger: logging.Logger, name_2: str, level: int = logging.INFO, **data) -> Iterator[None]:
    """Context manager that emits ``nombre.start`` / ``nombre.finish`` with ms.

    If the block raises, it emits ``nombre.error`` at ERROR with the
    traceback and RERAISES: logging for observability does not capture
    anyone's error.
    """
    event(logger, f"{name_2}.start", level=level, **data)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        ms = int((time.perf_counter() - t0) * 1000)
        try:  # noqa: SIM105
            logger._log(
                logging.ERROR,
                f"{name_2}.error",
                (),
                exc_info=exc,
                extra={"event": f"{name_2}.error", "data": data, "ms": ms},
            )
        except Exception:  # noqa: BLE001, S110
            pass
        raise
    ms = int((time.perf_counter() - t0) * 1000)
    event(logger, f"{name_2}.finish", level=level, ms=ms, **data)
