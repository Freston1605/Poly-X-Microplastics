"""Live language switching: every open window re-reads the active catalog.

The tests here assert what the user sees after flipping the language on a
running application, and that the state a researcher cares about — loaded
images, loaded models, the current run and the current selection — is still
there afterwards. They never look inside the catalog: the seam is
``set_language`` plus the visible text.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Offscreen before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
)


# ── Helpers ──────────────────────────────────────────────────────────
def _texts(widget) -> list[str]:
    """Every text a user can read inside ``widget``."""
    out: list[str] = []
    for kind in (QLabel, QPushButton, QCheckBox, QRadioButton):
        out.extend(child.text() for child in widget.findChildren(kind))
    return out


def _has_restart_promise(widget) -> bool:
    joined = " ".join(_texts(widget)).lower()
    return any(word in joined for word in ("restart", "reopen", "reinici", "reabrir"))


def _sidebar_texts(window) -> list[str]:
    return [button.text().strip() for button in window.sidebar_buttons]


def _report_page(window):
    """The detector's report page, unwrapped from its scroll area."""
    for index in range(window.stack.count()):
        page = window.stack.widget(index).widget()
        if type(page).__name__ == "ReportPage":
            return page
    raise AssertionError("report page not found")


def _models_page(window):
    for index in range(window.stack.count()):
        page = window.stack.widget(index).widget()
        if type(page).__name__ == "ModelsPage":
            return page
    raise AssertionError("models page not found")


def _results_page(window):
    for index in range(window.stack.count()):
        page = window.stack.widget(index).widget()
        if type(page).__name__ == "ResultsPage":
            return page
    raise AssertionError("results page not found")


@pytest.fixture(autouse=True)
def _restore_language(qapp):
    """Leave the suite in the language it pins and flush destroyed windows."""
    from polyx.core import i18n

    yield
    qapp.processEvents()
    i18n.carry()


# ── The switch itself ────────────────────────────────────────────────
def test_switching_language_loads_qt_catalogs(qapp):
    """Native dialogs follow the interface instead of staying English."""
    from polyx.core import i18n
    from polyx.gui import i18n_qt

    i18n_qt.activate()
    assert i18n.set_language("es")
    assert QCoreApplication.translate("QFileDialog", "Open") == "Abrir"

    assert i18n.set_language("en")
    assert QCoreApplication.translate("QFileDialog", "Open") == "Open"


def test_language_change_is_rebroadcast_to_top_level_windows(qapp):
    """Every top-level window receives Qt's own language-change event."""

    class Probe(QMainWindow):
        changes = 0

        def changeEvent(self, event):
            if event.type() == QEvent.LanguageChange:
                type(self).changes += 1
            super().changeEvent(event)

    from polyx.core import i18n
    from polyx.gui import i18n_qt

    # Start from a different language: asking for the active one is a no-op.
    i18n.set_language("en")
    i18n_qt.activate()
    probe = Probe()
    i18n.set_language("es")
    assert Probe.changes >= 1
    probe.deleteLater()


# ── Windows ──────────────────────────────────────────────────────────
def test_launcher_retranslates_without_restart(qapp):
    from polyx.core import i18n
    from polyx.gui.launcher import LauncherWindow

    i18n.set_language("es")
    window = LauncherWindow()
    assert window.btn_prefs.text() == "Preferencias"
    assert "microplásticos" in window.windowTitle()

    i18n.set_language("en")
    qapp.processEvents()

    assert window.btn_prefs.text() == "Preferences"
    assert window.windowTitle() == "Poly-X · Microplastics suite"
    window.deleteLater()


def test_preferences_applies_immediately_and_stops_promising_a_restart(qapp):
    from polyx.core import i18n
    from polyx.gui.preferencias import PreferencesDialog

    i18n.set_language("es")
    dialog = PreferencesDialog()
    assert dialog.windowTitle() == "Preferencias"

    dialog._choose_language("en")
    qapp.processEvents()

    assert i18n.language() == "en"
    assert dialog.windowTitle() == "Preferences"
    assert not _has_restart_promise(dialog)
    dialog.deleteLater()


def test_detector_retranslates_and_keeps_loaded_state(qapp, tmp_path):
    from polyx.core import i18n
    from polyx.gui.detector.window import DetectorWindow

    i18n.set_language("es")
    window = DetectorWindow()
    image = tmp_path / "placa.png"
    model = tmp_path / "model.pt"
    run = object()
    window.state.images = [image]
    window.state.model_slots[0].path = model
    window.state.model_slots[0].alias = "my-model"
    window.state.model_slots[0].alias_default = False
    window.state.run_result = run
    window.stack.setCurrentIndex(4)
    window.sidebar_buttons[4].setChecked(True)

    assert _sidebar_texts(window)[1] == "Imágenes"
    assert _report_page(window).btn_gen.text() == "Generar reporte HTML"

    i18n.set_language("en")
    qapp.processEvents()

    assert _sidebar_texts(window)[1] == "Images"
    assert _report_page(window).btn_gen.text() == "Generate HTML report"
    assert window.state.images == [image]
    assert window.state.model_slots[0].path == model
    assert window.state.model_slots[0].alias == "my-model"
    assert window.state.run_result is run
    assert window.stack.currentIndex() == 4
    assert _models_page(window).slot_cards[0].ed_alias.text() == "my-model"
    assert _models_page(window).slot_cards[0].ed_path.text() == str(model)
    window.deleteLater()


def test_trainer_retranslates_and_keeps_loaded_state(qapp, tmp_path):
    from polyx.core import i18n
    from polyx.gui.trainer.window import TrainerWindow

    i18n.set_language("es")
    window = TrainerWindow()
    dataset = tmp_path / "data.yaml"
    window.state.dataset.yaml_path = dataset
    window.state.dataset.class_names = ["PET", "PP"]
    window.state.run_dir = tmp_path / "run"
    window.stack.setCurrentIndex(5)

    assert _sidebar_texts(window)[0] == "Modelo"

    i18n.set_language("en")
    qapp.processEvents()

    assert _sidebar_texts(window)[0] == "Model"
    assert window.state.dataset.yaml_path == dataset
    assert window.state.dataset.class_names == ["PET", "PP"]
    assert window.state.run_dir == tmp_path / "run"
    assert window.stack.currentIndex() == 5
    window.deleteLater()


def test_labeler_retranslates_and_keeps_loaded_state(qapp, tmp_path):
    from polyx.core import i18n
    from polyx.gui.labeler.window import LabelerWindow

    i18n.set_language("es")
    window = LabelerWindow()
    images = [tmp_path / "a.png", tmp_path / "b.png"]
    model = object()
    window.state.images = images
    window.state.current_idx = 1
    window._pre_model = model

    assert "Cargar modelo…" in _texts(window)

    i18n.set_language("en")
    qapp.processEvents()

    assert "Load model…" in _texts(window)
    assert window.state.images == images
    assert window.state.current_idx == 1
    assert window._pre_model is model
    window.deleteLater()


def test_viewer_retranslates_and_keeps_loaded_state(qapp, tmp_path):
    from polyx.core import i18n
    from polyx.gui.viewer.window import ViewerWindow

    i18n.set_language("es")
    window = ViewerWindow()
    images = [tmp_path / "a.png"]
    model = object()
    window.state.images = images
    window.state.current_idx = 0
    window.state.um_per_px = 31.5
    window.state.model = model

    assert "Cargar modelo…" in _texts(window)

    i18n.set_language("en")
    qapp.processEvents()

    assert "Load model…" in _texts(window)
    assert window.state.images == images
    assert window.state.current_idx == 0
    assert window.state.um_per_px == 31.5
    assert window.state.model is model
    window.deleteLater()


# ── Another application ──────────────────────────────────────────────
def _pump_until(qapp, predicate, timeout_ms: int = 3000) -> bool:
    """Run the event loop until ``predicate`` holds or the budget runs out."""
    from PySide6.QtTest import QTest

    waited = 0
    while waited < timeout_ms and not predicate():
        QTest.qWait(20)
        waited += 20
    qapp.processEvents()
    return predicate()


def test_a_language_saved_by_another_process_reaches_open_windows(qapp):
    """A module process applies the preference another process saved."""
    import json

    from polyx.core import i18n, preferences
    from polyx.gui import i18n_qt
    from polyx.gui.launcher import LauncherWindow

    i18n.set_language("es")
    window = LauncherWindow()
    assert window.btn_prefs.text() == "Preferencias"
    i18n_qt.activate()

    # The launcher, in the real program, saves the choice and stays in its
    # own application: no signal crosses the boundary, only the file does.
    preferences.FILE.write_text(json.dumps({"locale": "en"}), encoding="utf-8")

    assert _pump_until(qapp, lambda: i18n.language() == "en")
    assert window.btn_prefs.text() == "Preferences"
    window.deleteLater()


def test_a_preferences_file_created_after_startup_is_still_watched(qapp):
    """A first run has no preferences file yet; the parent is watched."""
    import json

    from polyx.core import i18n, preferences
    from polyx.gui import i18n_qt

    i18n.set_language("es")
    preferences.FILE.unlink(missing_ok=True)
    i18n_qt.activate()
    assert not preferences.FILE.exists()

    preferences.FILE.write_text(json.dumps({"locale": "en"}), encoding="utf-8")

    assert _pump_until(qapp, lambda: i18n.language() == "en")


def test_applying_a_watched_language_does_not_loop(qapp):
    """Applying another process's choice never writes or broadcasts again."""
    import json

    from polyx.core import i18n, preferences
    from polyx.gui import i18n_qt

    changes: list[str] = []
    i18n.set_language("es")
    i18n_qt.activate()
    i18n.when_changing(changes.append)

    preferences.FILE.write_text(json.dumps({"locale": "en"}), encoding="utf-8")
    assert _pump_until(qapp, lambda: i18n.language() == "en")
    assert changes == ["en"]

    # The applied value is not written back, so the watcher stays quiet.
    saved = preferences.FILE.read_bytes()
    _pump_until(qapp, lambda: False, timeout_ms=300)
    assert changes == ["en"]
    assert preferences.FILE.read_bytes() == saved


# ── Reports: a view re-renders, an artifact does not ─────────────────
def _minimal_run(tmp_path):
    import cv2
    import numpy as np

    import polyx.core.measure as measure
    from polyx.core.detect.protocol import Instance as Detection
    from polyx.core.detection_run import ImageResult, ModelInfo, RunParams, RunResult

    image = np.full((260, 320, 3), 18, np.uint8)
    cv2.circle(image, (160, 130), 22, (60, 210, 240), -1)
    image_path = tmp_path / "placa.png"
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    image_path.write_bytes(buffer.tobytes())

    detection = Detection(class_id=0, class_name="PET", conf=0.9, x1=138, y1=108, x2=182, y2=152)
    measure.apply_to_detection(detection, image, 30.0)
    detection.number = 1

    return RunResult(
        run_dir=tmp_path / "run",
        images=[image_path],
        models=[ModelInfo(alias="m1", path=Path("fake.pt"))],
        params=RunParams(),
        results={
            0: [
                ImageResult(
                    image_path=image_path,
                    model_idx=0,
                    predictions=[detection],
                    gt=[],
                    has_gt=False,
                )
            ]
        },
    )


def test_saved_report_is_never_rewritten_by_the_switch(qapp, tmp_path):
    from polyx.core import i18n
    from polyx.gui.report_html import generate_report

    run = _minimal_run(tmp_path)
    i18n.set_language("es")
    report = tmp_path / "informe_deteccion.html"
    generate_report(run, report, sections=["resumen"])
    before = report.read_bytes()

    from polyx.gui.detector.window import DetectorWindow

    window = DetectorWindow()
    window.state.run_result = run
    i18n.set_language("en")
    qapp.processEvents()

    assert report.read_bytes() == before
    window.deleteLater()


def test_displayed_report_view_rerenders_in_the_new_language(qapp, tmp_path):
    """The in-app report surface re-reads the catalog instead of freezing text."""
    from polyx.core import i18n
    from polyx.gui.detector.window import DetectorWindow
    from polyx.gui.report_html import generate_report

    run = _minimal_run(tmp_path)
    i18n.set_language("es")

    window = DetectorWindow()
    window.state.run_result = run
    page = _report_page(window)
    results = _results_page(window)
    assert page.btn_gen.text() == "Generar reporte HTML"
    assert "Informe de detección" in _texts(page)
    assert results.kpi_imgs.lbl_title.text() == "Imágenes"

    i18n.set_language("en")
    qapp.processEvents()

    page = _report_page(window)
    results = _results_page(window)
    assert page.btn_gen.text() == "Generate HTML report"
    assert "Detection report" in _texts(page)
    assert results.kpi_imgs.lbl_title.text() == "Images"

    # Rendering again after the switch produces the new language.
    after = tmp_path / "informe_en.html"
    generate_report(run, after, sections=["resumen"])
    html = after.read_text(encoding="utf-8")
    assert "Summary" in html
    assert "Resumen" not in html
    window.deleteLater()
