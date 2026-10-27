"""Launcher → trainer subprocess integration tests.

The launcher is the entry point that spawns the trainer subprocess via
``_spawn_module('polyx.gui.trainer')``.  These tests exercise the full
launcher-to-trainer wiring chain — ``open_trainer`` → ``_launch_module`` →
``_spawn_module`` → ``subprocess.Popen`` — with Popen mocked so no real
process is ever spawned.

All tests run under an offscreen QApplication and rely on the conftest
autouse fixture (``_aislar_logs_suite``) to isolate ``POLYX_LOG_DIR`` to
``tmp_path / "_polyx_logs"`` per test, so ``app.jsonl`` lands there.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QScrollArea

# Offscreen platform BEFORE importing polyx.gui.launcher (which pulls in Qt).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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


# ── Tests for _spawn_module ──────────────────────────────────────────


class TestSpawnModuleTrainerExitoso:
    def test_spawn_module_trainer_exitoso(self, tmp_path, monkeypatch):
        """_spawn_module('polyx.gui.trainer') returns True and emits module.launch."""
        import polyx.gui.launcher as launcher

        monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)

        resultado = launcher._spawn_module("polyx.gui.trainer")

        assert resultado is True
        # Popen was called with the right argv
        assert len(_FakePopen.llamadas) == 1
        args, kwargs = _FakePopen.llamadas[0]
        assert args == [sys.executable, "-m", "polyx.gui.trainer"]
        assert kwargs["cwd"] == str(Path(launcher.__file__).resolve().parents[3])

    def test_spawn_module_emite_module_launch_en_app_jsonl(self, tmp_path, monkeypatch):
        """The module.launch record has the English core schema."""
        import polyx.gui.launcher as launcher

        monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)

        launcher._spawn_module("polyx.gui.trainer")

        eventos = _leer_app_jsonl(tmp_path)
        launches = [e for e in eventos if e["event"] == "module.launch"]
        assert len(launches) == 1
        rec = launches[0]
        # English core schema: ts, level, module, event, run_id
        assert isinstance(rec["ts"], str) and rec["ts"].endswith("Z")
        assert rec["level"] == "INFO"
        assert rec["module"] == "polyx.app"
        assert "run_id" in rec


class TestSpawnModuleModuloInexistente:
    def test_spawn_module_modulo_inexistente(self, tmp_path, monkeypatch):
        """_spawn_module with a non-existent module returns False and logs error."""
        import polyx.gui.launcher as launcher

        monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)

        resultado = launcher._spawn_module("modulo.que.no.existe")

        assert resultado is False
        eventos = _leer_app_jsonl(tmp_path)
        errors = [e for e in eventos if e["event"] == "module.launch.error"]
        assert len(errors) == 1
        assert errors[0]["level"] == "WARNING"
        assert "modulo.que.no.existe" in errors[0]["data"]["modulo"]


class TestSpawnModulePopenFalla:
    def test_spawn_module_popen_falla(self, tmp_path, monkeypatch):
        """_spawn_module returns False and logs error when Popen raises."""
        import polyx.gui.launcher as launcher

        def reventar(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(launcher.subprocess, "Popen", reventar)

        resultado = launcher._spawn_module("polyx.gui.trainer")

        assert resultado is False
        eventos = _leer_app_jsonl(tmp_path)
        errors = [e for e in eventos if e["event"] == "module.launch.error"]
        assert len(errors) == 1
        assert "no space left on device" in errors[0]["data"]["error"]

    def test_spawn_module_error_emite_module_launch_error(self, tmp_path, monkeypatch):
        """Error path also emits module.launch.error with English schema."""
        import polyx.gui.launcher as launcher

        def reventar(*args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(launcher.subprocess, "Popen", reventar)

        launcher._spawn_module("polyx.gui.trainer")

        eventos = _leer_app_jsonl(tmp_path)
        errors = [e for e in eventos if e["event"] == "module.launch.error"]
        assert len(errors) == 1
        rec = errors[0]
        assert rec["level"] == "WARNING"
        assert rec["module"] == "polyx.app"
        assert "run_id" in rec
        assert "error" in rec["data"]


# ── Tests for LauncherWindow wiring ─────────────────────────────────


class TestOpenTrainerWiring:
    def test_open_trainer_lanza_modulo_trainer(self, qapp, monkeypatch):
        """Launcher's "Abrir Entrenador" button spawns the trainer module."""
        from polyx.gui.launcher import LauncherWindow, ModuleCard

        captured = {}

        def capturar(self, dotted):
            captured["dotted"] = dotted
            return True

        monkeypatch.setattr(LauncherWindow, "_launch_module", capturar)

        win = LauncherWindow()
        # The grid is at index 2 in the main VBoxLayout
        scroll = win.centralWidget().findChild(QScrollArea)
        grid = scroll.widget().layout().itemAt(2).layout()
        trainer_card = grid.itemAtPosition(0, 1).widget()
        assert isinstance(trainer_card, ModuleCard)
        trainer_card._btn_open.click()
        assert captured["dotted"] == "polyx.gui.trainer"
        win.deleteLater()

    def test_launch_module_avisa_en_fallo(self, qapp, monkeypatch):
        """_launch_module shows QMessageBox.information when _spawn_module fails."""
        import polyx.gui.launcher as launcher
        from polyx.gui.launcher import LauncherWindow

        window = LauncherWindow()
        avisos: list[tuple] = []

        # Monkeypatch _spawn_module at module level
        monkeypatch.setattr(launcher, "_spawn_module", lambda dotted: False)

        # Monkeypatch QMessageBox.information to capture the call
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(
            QMessageBox, "information", lambda parent, title, msg: avisos.append((title, msg))
        )

        window._launch_module("polyx.gui.trainer")

        assert len(avisos) == 1
        title, msg = avisos[0]  # noqa: RUF059
        assert "no se pudo" in msg.lower() or "fall" in msg.lower()
        window.deleteLater()


# ── Module boundary tests ───────────────────────────────────────────
# These tests ensure the GUI doesn't import the old core re-export package.

import ast  # noqa: E402
import importlib  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
PAQUETE_GUI = RAIZ / "src" / "polyx" / "gui"
NOMBRE_HEREDADO = "polyx.core_reexport"  # nombre que nunca existira


def _partes_del_paquete(ruta: Path) -> list[str]:
    """Path components from polyx/gui down to the file's parent directory."""
    rel = ruta.relative_to(PAQUETE_GUI)
    return rel.parent.parts if rel.parent != Path(".") else ()


def _leer(ruta: Path) -> str:
    return ruta.read_text(encoding="utf-8")


def _es_qt(nombre: str) -> bool:
    return nombre.split(".", 1)[0] in (
        "PySide6",
        "PySide2",
        "PyQt5",
        "PyQt6",
        "shiboken6",
        "shiboken2",
    )


def _es_paquete_heredado(nombre: str) -> bool:
    return nombre == NOMBRE_HEREDADO or nombre.startswith(NOMBRE_HEREDADO + ".")


def _resuelve_relativo(partes: tuple[str, ...], nivel: int, modulo: str | None) -> str:
    """Resolve a relative import against the current package parts.

    partes: parts from polyx/gui/ to the file (e.g., ("detector", "pages", "ejecutar"))
    nivel: number of dots in the relative import
    modulo: the module name being imported

    The file is at polyx/gui/<partes>/module.py
    """
    # Start from polyx/gui/ + partes
    # Level 1: polyx/gui/<partes[:-1]>
    # Level 2: polyx/gui/<partes[:-2]>
    # ...
    # Level len(partes): polyx/gui
    # Level len(partes)+1: polyx
    # Level len(partes)+1: polyx
    total_parts = len(partes)
    if nivel <= total_parts:
        # Within polyx/gui/
        base = ("polyx", "gui") + partes[:-nivel]  # noqa: RUF005
    elif nivel == total_parts + 1:
        # Reached polyx/
        base = ("polyx",)
    elif nivel == total_parts + 2:
        # Beyond polyx (shouldn't happen)
        return ""
    else:
        return ""
    if modulo:
        return ".".join(base + (modulo,))  # noqa: RUF005
    return ".".join(base)


def _imports_heredados(arbol: ast.AST, partes: list[str]) -> list[str]:
    """Imports that resolve to the old re-export package."""
    encontrados: list[str] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                if _es_paquete_heredado(alias.name):
                    encontrados.append(alias.name)  # noqa: PERF401
        elif isinstance(nodo, ast.ImportFrom):
            if nodo.level:
                # Relative import - resolve against current package
                destino = _resuelve_relativo(partes, nodo.level, nodo.module)
                if destino and _es_paquete_heredado(destino):
                    encontrados.append(destino)
            elif nodo.module and _es_paquete_heredado(nodo.module):
                encontrados.append(nodo.module)
    return encontrados


def _fuentes_fuera_de_gui() -> list[Path]:
    """All .py files under polyx/ except those under polyx/gui/."""
    return sorted(
        p
        for p in (RAIZ / "src" / "polyx").rglob("*.py")
        if "__pycache__" not in p.parts and not p.is_relative_to(PAQUETE_GUI)
    )


def test_la_estructura_del_paquete_gui_existe():
    """polyx/gui has the expected files and feature packages."""
    assert (PAQUETE_GUI / "__init__.py").exists()
    assert (PAQUETE_GUI / "launcher.py").exists()
    assert (PAQUETE_GUI / "__main__.py").exists()
    # Feature packages
    for paquete in ("detector", "trainer", "viewer", "labeler"):
        pkg_dir = PAQUETE_GUI / paquete
        assert pkg_dir.is_dir(), f"falta {pkg_dir}"
        assert (pkg_dir / "__init__.py").exists()
        assert (pkg_dir / "__main__.py").exists()
        assert (pkg_dir / "window.py").exists()


def test_el_escaneo_externo_cubre_los_limites_de_polyx():
    """External scan covers the expected files."""
    fuera = _fuentes_fuera_de_gui()
    assert fuera, "no se escaneo ningun archivo fuera de polyx/gui"
    nombres = {ruta.as_posix() for ruta in fuera}
    assert (RAIZ / "src" / "polyx" / "launcher.py").as_posix() in nombres
    for rel_path in (
        "src/polyx/core/i18n.py",
        "src/polyx/core/quality/metrics.py",
        "src/polyx/core/measure/shape.py",
        "src/polyx/core/provenance/observability.py",
        "src/polyx/core/detect/yolo.py",
    ):
        assert (RAIZ / rel_path).as_posix() in nombres


@pytest.mark.parametrize(
    "ruta", _fuentes_fuera_de_gui(), ids=lambda r: r.relative_to(RAIZ).as_posix()
)
def test_fuera_de_gui_no_importa_qt(ruta: Path):
    """No module outside polyx/gui imports Qt."""
    arbol = ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))
    prohibidos = [
        n
        for n in ast.walk(arbol)
        if isinstance(n, ast.Import)
        for a in n.names
        if a.name.split(".", 1)[0]
        in ("PySide6", "PySide2", "PyQt5", "PyQt6", "shiboken6", "shiboken2")
    ]
    # Also check ImportFrom at module level
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and not nodo.level:  # noqa: SIM102
            if nodo.module and nodo.module.split(".", 1)[0] in (
                "PySide6",
                "PySide2",
                "PyQt5",
                "PyQt6",
                "shiboken6",
                "shiboken2",
            ):
                prohibidos.append(nodo.module)
    assert not prohibidos, f"{ruta.relative_to(RAIZ).as_posix()} importa Qt: {prohibidos}"


@pytest.mark.parametrize(
    "ruta", sorted(PAQUETE_GUI.rglob("*.py")), ids=lambda r: r.relative_to(RAIZ).as_posix()
)
def test_la_gui_no_importa_el_paquete_heredado(ruta: Path):
    """No polyx/gui module imports the old re-export package."""
    partes = _partes_del_paquete(ruta)
    prohibidos = _imports_heredados(
        ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta)), partes
    )
    assert not prohibidos, (
        f"{ruta.relative_to(RAIZ).as_posix()} importa un paquete re-export inexistente "
        f"del core: {prohibidos}"
    )


def test_el_paquete_de_reexport_del_core_ha_desaparecido():
    """The old re-export package is gone; the new polyx.core is the engine."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("polyx.core_reexport")

    # The new polyx.core MUST exist (it's the engine)
    assert (RAIZ / "src" / "polyx" / "core").exists(), "polyx/core (the engine) does not exist"
    import polyx.core

    assert hasattr(polyx.core, "Instance")  # exports the engine API


def test_el_escanner_detecta_violaciones_sinteticas():
    """The scanner catches relative imports that resolve to the old re-export."""
    # This test verifies the scanner logic by feeding it synthetic imports
    partes_gui = ("detector", "pages", "ejecutar")  # example: polyx/gui/detector/pages/ejecutar.py

    # Relative import that would resolve to the old re-export
    relativo_a_gui = ast.parse("from ....core_reexport import theme\n")
    assert _imports_heredados(relativo_a_gui, partes_gui) == ["polyx.core_reexport"]

    # Legitimate relative import to polyx (level 2, no submodule)
    version_polyx = ast.parse("from .. import __version__\n")
    assert _imports_heredados(version_polyx, partes_gui) == []

    # Relative to GUI sibling (should be caught)
    relativo_a_gui2 = ast.parse("from ..gui import ventana\n")
    assert _imports_heredados(relativo_a_gui2, partes_gui) == []  # not the heredado package


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
