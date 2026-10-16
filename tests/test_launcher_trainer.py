"""Launcher → trainer subprocess integration tests.

The launcher is the entry point that spawns the trainer subprocess via
``_spawn_module('polyx.trainer')``.  These tests exercise the full
launcher-to-trainer wiring chain — ``open_trainer`` → ``_launch_module`` →
``_spawn_module`` → ``subprocess.Popen`` — with Popen mocked so no real
process is ever spawned.

All tests run under an offscreen QApplication and rely on the conftest
autouse fixture (``_aislar_logs_suite``) to isolate ``POLYX_LOG_DIR`` to
``tmp_path / "_polyx_logs"`` per test, so ``app.jsonl`` lands there.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest

# Offscreen platform BEFORE importing polyx.launcher (which pulls in Qt).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import polyx.launcher as launcher                               # noqa: E402
from polyx.launcher import LauncherWindow                        # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────

def _leer_app_jsonl(tmp_path: Path) -> list[dict]:
    """Read app.jsonl records from the isolated log directory."""
    ruta = tmp_path / "_polyx_logs" / "app.jsonl"
    if not ruta.exists():
        return []
    return [
        json.loads(linea)
        for linea in ruta.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]


class _FakePopen:
    """Mock Popen that records calls and does nothing."""

    llamadas: list[tuple] = []

    def __init__(self, args, **kwargs):
        _FakePopen.llamadas.append((args, kwargs))


@pytest.fixture(autouse=True)
def _reset_fake_popen():
    """Reset the fake Popen call log before each test."""
    _FakePopen.llamadas = []
    yield


@pytest.fixture(scope="module")
def qapp():
    """One QApplication per module: PySide6 does not allow recreation."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ── 1. Success path: _spawn_module("polyx.trainer") ─────────────────

class TestSpawnModuleTrainerExitoso:
    def test_spawn_module_trainer_exitoso(self, tmp_path, monkeypatch):
        """_spawn_module('polyx.trainer') returns True and emits module.launch."""
        monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)

        resultado = launcher._spawn_module("polyx.trainer")

        assert resultado is True
        # Popen was called with the right argv
        assert len(_FakePopen.llamadas) == 1
        args, kwargs = _FakePopen.llamadas[0]
        assert args == [sys.executable, "-m", "polyx.trainer"]
        assert kwargs["cwd"] == str(Path(launcher.__file__).resolve().parents[1])

    def test_spawn_module_emite_module_launch_en_app_jsonl(
        self, tmp_path, monkeypatch
    ):
        """The module.launch record has the English core schema."""
        monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)

        launcher._spawn_module("polyx.trainer")

        eventos = _leer_app_jsonl(tmp_path)
        launches = [e for e in eventos if e["event"] == "module.launch"]
        assert len(launches) == 1
        rec = launches[0]
        # English core schema: ts, level, module, event, run_id
        assert isinstance(rec["ts"], str) and rec["ts"].endswith("Z")
        assert rec["level"] == "INFO"
        assert rec["module"] == "polyx.app"
        assert "run_id" in rec
        assert rec["data"]["modulo"] == "polyx.trainer"


# ── 2. Module-not-found path ────────────────────────────────────────

class TestSpawnModuleModuloInexistente:
    def test_spawn_module_modulo_inexistente(self, tmp_path, monkeypatch):
        """find_spec returns None → False, no log line written."""
        monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)

        resultado = launcher._spawn_module("polyx.no_existe_modulo")

        assert resultado is False
        # Popen was never called
        assert _FakePopen.llamadas == []
        # No log line written
        eventos = _leer_app_jsonl(tmp_path)
        assert eventos == []


# ── 3. Popen failure path ───────────────────────────────────────────

class TestSpawnModulePopenFalla:
    def test_spawn_module_popen_falla(self, tmp_path, monkeypatch):
        """Popen raises OSError → False + module.launch.error at WARNING."""
        def reventar(*args, **kwargs):
            raise OSError("no hay ejecutable")

        monkeypatch.setattr(launcher.subprocess, "Popen", reventar)

        resultado = launcher._spawn_module("polyx.trainer")

        assert resultado is False

    def test_spawn_module_error_emite_module_launch_error(
        self, tmp_path, monkeypatch
    ):
        """Error-path record: level=WARNING, event=module.launch.error, non-empty error."""
        def reventar(*args, **kwargs):
            raise OSError("no hay ejecutable")

        monkeypatch.setattr(launcher.subprocess, "Popen", reventar)

        launcher._spawn_module("polyx.trainer")

        eventos = _leer_app_jsonl(tmp_path)
        errores = [e for e in eventos if e["event"] == "module.launch.error"]
        assert len(errores) == 1
        rec = errores[0]
        assert rec["level"] == "WARNING"
        assert rec["data"]["modulo"] == "polyx.trainer"
        assert "no hay ejecutable" in rec["data"]["error"]


# ── 4. open_trainer → _launch_module wiring ─────────────────────────

class TestOpenTrainerWiring:
    def test_open_trainer_lanza_modulo_trainer(self, qapp, monkeypatch):
        """open_trainer() calls _launch_module with 'polyx.trainer'."""
        window = LauncherWindow()
        llamadas: list[str] = []

        def capturar(dotted):
            llamadas.append(dotted)

        monkeypatch.setattr(window, "_launch_module", capturar)
        window.open_trainer()

        assert llamadas == ["polyx.trainer"]
        window.deleteLater()

    def test_launch_module_avisa_en_fallo(self, qapp, monkeypatch):
        """_launch_module shows QMessageBox.information when _spawn_module returns False."""
        window = LauncherWindow()
        avisos: list[tuple] = []

        # Monkeypatch _spawn_module at module level
        monkeypatch.setattr(launcher, "_spawn_module", lambda dotted: False)

        # Monkeypatch QMessageBox.information to capture the call
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(
            QMessageBox, "information",
            lambda parent, title, msg: avisos.append((title, msg))
        )

        window._launch_module("polyx.trainer")

        assert len(avisos) == 1
        title, msg = avisos[0]
        assert "polyx.trainer" in msg
        window.deleteLater()
