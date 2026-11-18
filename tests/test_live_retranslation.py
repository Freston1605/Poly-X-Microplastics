"""Live language switching: every open window re-reads the active catalog.

The tests here assert what the user sees after flipping the language on a
running application, and that the state a researcher cares about — loaded
images, loaded models, the current run and the current selection — is still
there afterwards. They never look inside the catalog: the seam is
``set_language`` plus the visible text.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

# Offscreen before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt, QThread, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
)

from tests.conftest import destroy_windows


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


def _manual_gt_page(window):
    for index in range(window.stack.count()):
        page = window.stack.widget(index).widget()
        if type(page).__name__ == "GTManualPage":
            return page
    raise AssertionError("manual GT page not found")


def _write_image(path: Path) -> Path:
    """A real PNG: the Viewer decodes the file, so a path alone is not loaded."""
    import cv2
    import numpy as np

    ok, buffer = cv2.imencode(".png", np.full((240, 320, 3), 24, np.uint8))
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def _wheel_over(canvas, steps: int = 1) -> None:
    """Zoom with the wheel, over the middle of the canvas, as a user does."""
    position = QPointF(canvas.width() / 2, canvas.height() / 2)
    for _step in range(steps):
        event = QWheelEvent(
            position,
            canvas.mapToGlobal(position),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(canvas, event)


def _drag(canvas, dx: float = 40.0, dy: float = 30.0) -> None:
    """Scroll the canvas with a middle-button drag, as a user does."""
    start = QPointF(canvas.width() / 2, canvas.height() / 2)
    end = QPointF(start.x() + dx, start.y() + dy)
    for kind, position, button, buttons in (
        (QMouseEvent.MouseButtonPress, start, Qt.MiddleButton, Qt.MiddleButton),
        (QMouseEvent.MouseMove, end, Qt.NoButton, Qt.MiddleButton),
        (QMouseEvent.MouseButtonRelease, end, Qt.MiddleButton, Qt.NoButton),
    ):
        event = QMouseEvent(
            kind, position, canvas.mapToGlobal(position), button, buttons, Qt.NoModifier
        )
        QApplication.sendEvent(canvas, event)


def _detection_result(image_path: Path, model_idx: int):
    """One detected particle stored against a real model slot."""
    from polyx.core.detect.protocol import Instance
    from polyx.core.detection_run import ImageResult

    prediction = Instance(class_id=0, class_name="PET", conf=0.875, x1=6, y1=7, x2=40, y2=44)
    return ImageResult(
        image_path=image_path,
        model_idx=model_idx,
        predictions=[prediction],
        gt=[],
        has_gt=False,
    )


@pytest.fixture(autouse=True)
def _restore_language(qapp):
    """Leave the suite in the language it pins, and no window behind.

    Every case starts from the top-level widgets it inherited: anything a
    previous case created has been destroyed and its deferred delete flushed,
    so a broadcast cannot reach a window that is already being torn down.
    """
    from polyx.core import i18n

    inherited = set(qapp.topLevelWidgets())
    yield
    destroy_windows([widget for widget in qapp.topLevelWidgets() if widget not in inherited], qapp)
    i18n.carry()


@pytest.fixture(autouse=True)
def _no_update_check(no_update_check):
    """Keep the launcher's background GitHub check out of this module.

    It fires 1.2 s after a launcher is built — inside a wait — and its thread
    can outlive a window the case destroyed. It has nothing to do with live
    retranslation, and its timer is another race this module does not test.
    Every case here may build a launcher, so the shared fixture is applied to
    all of them instead of listed in each signature.
    """


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


# ── A window that closes mid-change ──────────────────────────────────
def test_destroying_a_window_before_its_queued_rebuild_survives_the_pump(
    qapp, capfd, selected_language
):
    """A language event must outlive the window it reached without a crash.

    The rebuild is queued, so the window can be gone by the time it runs.
    This case destroys one window between the event and the rebuild, flushes
    the destruction by hand (no wall-clock race decides the outcome) and
    then pumps: the process must survive, nothing may be written to stderr,
    and the other window must still show the new language.

    The aborted CI run (43793135897, job 133055865448) could not be
    reproduced deterministically on PySide6 6.7.2 when this case was added:
    the per-window ``is_alive`` guard already returned before a destroyed
    window touched Qt, and an unguarded rebuild printed a ``RuntimeError``
    instead of aborting. The case guards the contract; it does not reproduce
    that signal.
    """
    import shiboken6

    from polyx.core import i18n
    from polyx.gui import i18n_qt
    from polyx.gui.launcher import LauncherWindow

    selected_language("es")
    i18n_qt.activate()
    doomed = LauncherWindow()
    survivor = LauncherWindow()
    assert doomed.btn_prefs.text() == "Preferencias"
    assert survivor.btn_prefs.text() == "Preferencias"

    # Synchronous: both windows queue their rebuild during this call.
    i18n.set_language("en")

    doomed.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not shiboken6.isValid(doomed)

    capfd.readouterr()  # Ignore anything written before the race.
    _wait_until(
        qapp,
        lambda: survivor.btn_prefs.text() == "Preferences",
        expected="en",
        what="the surviving launcher to show the new language",
    )

    captured = capfd.readouterr()
    assert "Traceback" not in captured.err
    assert "Fatal Python error" not in captured.err
    survivor.deleteLater()


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
    window.state.model_slots[0].alias_generated = False
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


def test_generated_alias_keeps_its_run_folder_name_across_a_language_switch(qapp):
    """#124: the card translates its label; the alias it writes does not."""
    from polyx.core import i18n
    from polyx.gui.detector.window import DetectorWindow

    i18n.set_language("es")
    window = DetectorWindow()
    assert window.state.model_slots[0].alias == "model_1"
    card = _models_page(window).slot_cards[0]
    assert card.ed_alias.text() == "model_1"
    assert "Modelo 1" in _texts(card)

    i18n.set_language("en")
    qapp.processEvents()

    card = _models_page(window).slot_cards[0]
    assert card.ed_alias.text() == "model_1", "the run folder name is not interface prose"
    assert "Model 1" in _texts(card)
    assert window.state.model_slots[0].alias == "model_1"
    assert window.state.model_slots[0].alias_generated is True
    window.deleteLater()


def test_detector_kpis_keep_the_no_data_placeholder_after_a_switch(qapp):
    """A rebuilt page must not turn "no data yet" into a measured zero."""
    from polyx.core import i18n
    from polyx.gui.detector.window import DetectorWindow

    i18n.set_language("es")
    window = DetectorWindow()
    assert _results_page(window).kpi_imgs.lbl_value.text() == "—"

    i18n.set_language("en")
    qapp.processEvents()

    results = _results_page(window)
    assert results.kpi_imgs.lbl_value.text() == "—"
    assert results.kpi_dets.lbl_value.text() == "—"
    assert results.kpi_tp.lbl_value.text() == "—"
    assert results.kpi_f1.lbl_value.text() == "—"
    window.deleteLater()


def test_labeler_keeps_the_no_data_placeholder_after_a_switch(qapp):
    """With no images loaded, the box list still says "no data", not "none"."""
    from polyx.core import i18n
    from polyx.gui.labeler.window import LabelerWindow

    i18n.set_language("es")
    window = LabelerWindow()
    assert window.lbl_box_info.text() == "—"

    i18n.set_language("en")
    qapp.processEvents()

    assert window.lbl_box_info.text() == "—"
    window.deleteLater()


def test_destroying_a_trainer_window_while_a_probe_runs_survives(
    qapp, monkeypatch, selected_language
):
    """A probe thread must outlive the page and window that started it.

    The trainer starts two probes shortly after it opens. The probe body is
    held open here so the race is deterministic instead of depending on how
    fast torch answers; the case then destroys the window and drops it, which
    is what a language change plus a closed module does in the real program.
    The window is a page tree, so the probe is one attribute of a page that
    the teardown destroys.

    What is proven: a ``QThread`` object destroyed while its thread still runs
    aborts the interpreter with ``QThread: Destroyed while thread is still
    running`` and SIGABRT — the same signature as the failed CI run (a
    thread inside the event pump, the step exiting 250). That signature was
    reproduced locally by dropping the last reference to a running ``QThread``
    (exit 134). What is not proven: that this abort, and not something else in
    the runner, produced that CI event — the pre-fix tree ran this file and
    the case above without aborting here, so the evidence is the signature,
    not the incident. The case guards the contract; it does not replay CI.
    """
    import gc
    import threading
    import weakref

    from polyx.gui.trainer.window import TrainerWindow

    release = threading.Event()
    began = threading.Event()
    # Weak references: holding the probe here would keep the QThread alive and
    # hide the very destruction this case exists to catch.
    probes: list[weakref.ReferenceType] = []

    def hold(probe) -> None:
        probes.append(weakref.ref(probe))
        began.set()
        release.wait(5)

    monkeypatch.setattr("polyx.gui.trainer.pages.train._HwProbeWorker.run", hold)
    monkeypatch.setattr("polyx.gui.trainer.pages.parameters._GpuDetectWorker.run", hold)

    selected_language("es")
    window = TrainerWindow()
    _wait_until(qapp, began.is_set, what="a trainer hardware probe to be running")

    window.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    del window
    gc.collect()
    qapp.processEvents()

    # Reaching this line is the assertion: the interpreter survived. The
    # probes are still alive and finish on their own once released.
    assert probes, "the probe recorded no weak reference to itself"
    release.set()
    _wait_until(
        qapp,
        lambda: all(ref() is None or ref().isFinished() for ref in probes),
        what="the trainer hardware probes to finish",
    )


class _BlockingRunner(QThread):
    """A TrainerRunner stand-in whose thread runs until the case releases it."""

    epoch_metrics = Signal(object)
    log_line = Signal(str)
    progress = Signal(int, int)
    finished_ok = Signal(str)
    aborted = Signal()
    failed = Signal(str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self):
        self.started.set()
        self.release.wait(10)


def test_closing_the_trainer_window_during_a_run_keeps_the_runner_alive(qapp, monkeypatch, capfd):
    """Closing the Trainer must not drop the only reference to a running runner.

    The page owns the runner in the normal case, so closing the window would
    destroy the page and the QThread with it. The lifetime registry owns it
    until ``finished`` instead. Late progress and completion signals must also
    miss the destroyed page and window: those handlers are bound slots now.
    """
    import gc
    import weakref

    from polyx.gui.trainer.pages import train as train_module
    from polyx.gui.trainer.pages.parameters import ParametersPage
    from polyx.gui.trainer.pages.train import TrainPage
    from polyx.gui.trainer.window import TrainerWindow

    # The hardware probes have their own close-during-probe case. Do not let
    # their timers start threads here; this case isolates the runner.
    monkeypatch.setattr(TrainPage, "_refresh_hw", lambda self: None)
    monkeypatch.setattr(ParametersPage, "_detect_gpu", lambda self: None)
    monkeypatch.setattr(train_module, "validate_training_config", lambda state: (True, []))
    monkeypatch.setattr(train_module, "TrainerRunner", _BlockingRunner)

    window = TrainerWindow()
    page = next(
        window.stack.widget(index).widget()
        for index in range(window.stack.count())
        if isinstance(window.stack.widget(index).widget(), TrainPage)
    )
    page._start()
    runner = page.runner
    assert isinstance(runner, _BlockingRunner)
    _wait_until(qapp, runner.started.is_set, what="the training runner to start")

    state = page.state
    release = runner.release
    runner_ref = weakref.ref(runner)
    del runner

    try:
        window.close()
        window.deleteLater()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        del page
        del window
        gc.collect()
        qapp.processEvents()

        kept = runner_ref()
        assert kept is not None, "the runner died with the page that started it"
        assert kept.isRunning(), "the runner stopped when the window closed"

        # The window is gone; a late result must not reach it or its page.
        kept.progress.emit(1, 10)
        state.train_finished.emit("best.pt")
        qapp.processEvents()
    finally:
        release.set()
        _wait_until(
            qapp,
            lambda: (ref := runner_ref()) is None or ref.isFinished(),
            what="the training runner to finish",
        )
        qapp.processEvents()

    captured = capfd.readouterr()
    assert "Traceback" not in captured.err
    assert "QThread: Destroyed" not in captured.err
    assert "Fatal Python error" not in captured.err


class _BlockingEvalWorker(QThread):
    log_line = Signal(str)
    metrics_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self):
        self.started.set()
        self.release.wait(10)


def test_a_late_worker_failure_after_page_destruction_is_a_no_op(
    qapp, tmp_path, monkeypatch, capfd
):
    """A failure dialog must not be raised by a page that no longer exists."""
    import gc
    import weakref

    from PySide6.QtWidgets import QMessageBox

    from polyx.gui.trainer.pages import evaluate as evaluate_module
    from polyx.gui.trainer.pages.evaluate import EvaluatePage
    from polyx.gui.trainer.state import TrainerState

    dialogs: list[str] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args: dialogs.append("critical"))
    monkeypatch.setattr(evaluate_module, "_EvalWorker", _BlockingEvalWorker)

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"x")
    data = tmp_path / "data.yaml"
    data.write_text("path: .\n", encoding="utf-8")
    page = EvaluatePage(TrainerState())
    page.ed_w.setText(str(weights))
    page.ed_y.setText(str(data))
    page._run()
    worker = page.worker
    assert isinstance(worker, _BlockingEvalWorker)
    _wait_until(qapp, worker.started.is_set, what="the evaluate worker to start")

    release = worker.release
    worker_ref = weakref.ref(worker)
    del worker

    try:
        page.deleteLater()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        del page
        gc.collect()
        qapp.processEvents()

        kept = worker_ref()
        assert kept is not None, "the registry dropped a worker that is still running"
        kept.failed.emit("boom")
        qapp.processEvents()
        assert dialogs == [], "a destroyed page received a late failure dialog"
    finally:
        release.set()
        _wait_until(
            qapp,
            lambda: (ref := worker_ref()) is None or ref.isFinished(),
            what="the evaluate worker to finish",
        )
        qapp.processEvents()

    captured = capfd.readouterr()
    assert "Traceback" not in captured.err


class _BlockingExportWorker(QThread):
    log_line = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self):
        self.started.set()
        self.release.wait(10)


def test_a_late_export_result_after_page_destruction_is_a_no_op(qapp, tmp_path, monkeypatch, capfd):
    """Neither an export result nor its failure may touch a destroyed page."""
    import gc
    import weakref

    from PySide6.QtWidgets import QMessageBox

    from polyx.gui.trainer.pages import export as export_module
    from polyx.gui.trainer.pages.export import ExportPage
    from polyx.gui.trainer.state import TrainerState

    dialogs: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: dialogs.append("information"))
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args: dialogs.append("critical"))
    monkeypatch.setattr(export_module, "_ExportWorker", _BlockingExportWorker)

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"x")
    page = ExportPage(TrainerState())
    page.ed_w.setText(str(weights))
    page._run()
    worker = page.worker
    assert isinstance(worker, _BlockingExportWorker)
    _wait_until(qapp, worker.started.is_set, what="the export worker to start")

    release = worker.release
    worker_ref = weakref.ref(worker)
    del worker

    try:
        page.deleteLater()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        del page
        gc.collect()
        qapp.processEvents()

        kept = worker_ref()
        assert kept is not None, "the registry dropped a worker that is still running"
        kept.finished_ok.emit(str(tmp_path / "exported.pt"))
        kept.failed.emit("boom")
        qapp.processEvents()
        assert dialogs == [], "a destroyed page received a late export result"
    finally:
        release.set()
        _wait_until(
            qapp,
            lambda: (ref := worker_ref()) is None or ref.isFinished(),
            what="the export worker to finish",
        )
        qapp.processEvents()

    captured = capfd.readouterr()
    assert "Traceback" not in captured.err


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
    assert window.img_list.currentRow() == 1
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


def test_viewer_keeps_the_loaded_image_visible_across_a_switch(qapp, tmp_path):
    """The new canvas must draw the image the state still points at.

    The rebuild destroys the canvas and creates an empty one. Re-reading
    ``state.current_image`` on the window labels is not enough: the canvas
    only gets its pixmap through ``state.image_loaded``, so a window that
    skips that signal shows the "Open an image" placeholder after the switch.
    """
    from polyx.core import i18n
    from polyx.gui.viewer.window import ViewerWindow

    image = _write_image(tmp_path / "placa.png")
    earlier = _write_image(tmp_path / "otra.png")
    i18n.set_language("es")
    window = ViewerWindow()
    window.resize(1200, 800)
    window.show()
    window.state.images = [earlier, image]
    window.state.current_idx = 1
    window.state.image_loaded.emit(image)
    qapp.processEvents()
    assert window.canvas.has_image()
    assert window.lbl_nav.text() == "2 / 2"

    i18n.set_language("en")
    qapp.processEvents()

    assert window.state.images == [earlier, image]
    assert window.state.current_idx == 1
    assert window.lbl_nav.text() == "2 / 2"
    assert window.canvas.has_image(), "the rebuilt canvas lost the loaded image"
    window.deleteLater()


def test_viewer_keeps_zoom_and_scroll_across_a_switch(qapp, tmp_path):
    """Zoom and pan are the user's framing work; a rebuild must not refit.

    The gestures are real wheel and middle-drag events, and the state is read
    back through the canvas' own public view state. The rebuilt canvas also
    receives a resize event from the layout, which used to refit and discard
    the restored view, so the assertion happens after the event pump.
    """
    from polyx.core import i18n
    from polyx.gui.viewer.window import ViewerWindow

    image = _write_image(tmp_path / "placa.png")
    i18n.set_language("es")
    window = ViewerWindow()
    window.resize(1200, 800)
    window.show()
    window.state.images = [image]
    window.state.image_loaded.emit(image)
    qapp.processEvents()

    fitted = window.canvas.view_state()
    _wheel_over(window.canvas, steps=2)
    _drag(window.canvas)
    before = window.canvas.view_state()
    assert before[0] > fitted[0], "the wheel did not zoom"
    assert before[1] != fitted[1], "the drag did not scroll"

    i18n.set_language("en")
    qapp.processEvents()

    after = window.canvas.view_state()
    assert after[0] == pytest.approx(before[0])
    assert after[1] == before[1]
    window.deleteLater()


def test_detector_keeps_the_loaded_slots_and_detection_set_across_a_switch(qapp, tmp_path):
    """The slots the user loaded and the run's detections survive the rebuild.

    The Detector has no separate "selected slot" field: the selected slots
    are the ones that hold a path and a loaded backend, so both a non-default
    slot and its backend are exercised here. The detection set is the run's
    per-slot results, which the rebuilt Results page must keep showing.
    """
    from polyx.core import i18n
    from polyx.gui.detector.window import DetectorWindow

    images = [_write_image(tmp_path / "a.png"), _write_image(tmp_path / "b.png")]
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    reference_model = object()
    candidate_model = object()

    i18n.set_language("es")
    window = DetectorWindow()
    state = window.state
    state.images = images
    state.gt_image = images[1]
    state.model_slots[0].path = reference_path
    state.model_slots[0].alias = "reference"
    state.model_slots[0].alias_generated = False
    state.model_slots[0].loaded = reference_model
    state.model_slots[2].path = candidate_path
    state.model_slots[2].alias = "candidate"
    state.model_slots[2].alias_generated = False
    state.model_slots[2].loaded = candidate_model
    detections = {
        0: [_detection_result(images[0], 0)],
        2: [_detection_result(images[1], 2)],
    }
    state.results = detections
    window.stack.setCurrentIndex(5)
    window.sidebar_buttons[5].setChecked(True)

    i18n.set_language("en")
    qapp.processEvents()

    assert state.model_slots[0].path == reference_path
    assert state.model_slots[0].alias == "reference"
    assert state.model_slots[0].loaded is reference_model
    assert state.model_slots[2].path == candidate_path
    assert state.model_slots[2].alias == "candidate"
    assert state.model_slots[2].loaded is candidate_model
    assert state.results is detections
    assert state.results[2][0].predictions[0].class_name == "PET"
    assert window.stack.currentIndex() == 5
    assert state.gt_image == images[1]
    manual = _manual_gt_page(window)
    assert manual.list.currentRow() == 1
    assert manual.lbl_image_name.text() == images[1].name

    results = _results_page(window)
    assert results.table.rowCount() == 2
    assert {results.table.item(row, 0).text() for row in range(2)} == {
        "reference",
        "candidate",
    }
    window.deleteLater()


# ── One rebuild helper for the windows that share the pattern ────────
def test_the_rebuild_helper_captures_before_rebuilding_and_restores_after(qapp):
    from polyx.gui import i18n_qt

    events: list[object] = []
    window = QMainWindow()
    i18n_qt.rebuild_keeping_view(
        window,
        capture=lambda: events.append("capture") or {"page": 3},
        rebuild=lambda: events.append("rebuild"),
        restore=lambda view: events.append(("restore", view["page"])),
    )
    assert events == ["capture", "rebuild", ("restore", 3)]
    window.deleteLater()


def test_the_rebuild_helper_skips_a_window_that_is_already_gone(qapp):
    """The helper owns the liveness check every window used to repeat."""
    import shiboken6

    from polyx.gui import i18n_qt

    touched: list[str] = []
    window = QMainWindow()
    window.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not shiboken6.isValid(window)

    i18n_qt.rebuild_keeping_view(
        window,
        capture=lambda: touched.append("capture"),
        rebuild=lambda: touched.append("rebuild"),
        restore=lambda view: touched.append("restore"),
    )
    assert touched == []


def test_labeller_and_viewer_rebuild_through_the_shared_helper(qapp, monkeypatch):
    """Two windows share the helper; the worker-aware ones are exceptions."""
    from polyx.core import i18n
    from polyx.gui import i18n_qt
    from polyx.gui.labeler.window import LabelerWindow
    from polyx.gui.viewer.window import ViewerWindow

    calls: list[str] = []
    original = i18n_qt.rebuild_keeping_view

    def recorded(window, **kwargs):
        calls.append(type(window).__name__)
        return original(window, **kwargs)

    monkeypatch.setattr(i18n_qt, "rebuild_keeping_view", recorded)
    i18n.set_language("es")
    windows = [LabelerWindow(), ViewerWindow()]
    for window in windows:
        window.retranslate()
    assert calls == ["LabelerWindow", "ViewerWindow"]
    for window in windows:
        window.deleteLater()


# ── A failure a maintainer can read ──────────────────────────────────
def test_a_wait_that_cannot_succeed_reports_the_state_it_saw(qapp, selected_language):
    """A timeout is an assertion that carries the state a maintainer needs.

    The predicate is made false on purpose: the point is that the failure
    itself names the elapsed time, the two languages, the preferences bytes
    and the watcher, so a CI log is enough to diagnose the next failure.

    The message is also proof that the timeout path is an ordinary failure.
    Forcing it in a scratch run of this module's wait prints the fields below
    and pytest exits 1 ("1 failed"), not 134 (SIGABRT) — the point of the
    diagnostics is that the next red build reports instead of aborting.
    """
    import json

    from polyx.core import preferences
    from polyx.gui import i18n_qt

    selected_language("es")
    i18n_qt.activate()
    # A value the watcher has nothing to do with: the active language must stay
    # Spanish while the wait expects English, so the diagnostic shows both.
    preferences.FILE.write_text(json.dumps({"locale": "es"}), encoding="utf-8")

    started = time.monotonic()
    with pytest.raises(AssertionError) as caught:
        _wait_until(qapp, lambda: False, expected="en", timeout_s=0.3, what="nothing to change")
    elapsed = time.monotonic() - started

    message = str(caught.value)
    assert "elapsed: " in message
    assert "active language: 'es'" in message
    assert "expected language: 'en'" in message
    assert json.dumps({"locale": "es"}) in message
    assert "watcher: file=" in message
    assert "parent=" in message
    # The budget was respected and the failure was an assertion, not a hang.
    assert elapsed >= 0.3
    assert elapsed < 5


# ── The wait primitive ───────────────────────────────────────────────
#: Generous on purpose: a hosted runner shares its CPU between the quality job
#: and a test job, both holding torch and Qt resident. The deadline is a safety
#: net, not something a passing case spends: the wait returns on the event that
#: satisfies it, however early that is.
DEFAULT_WAIT_SECONDS = 15.0

#: How often the running loop re-checks the predicate. Not a budget: the
#: monotonic deadline decides when to give up, this only decides how promptly
#: a state reached without a notification is noticed.
_POLL_INTERVAL_MS = 20

#: Deliberately short: this is the window for the case that asserts *nothing*
#: changes, so waiting longer proves nothing extra while an echo write or a
#: queued rebuild would already have surfaced within a few hundred
#: milliseconds. It is a fixed window on purpose, not a scaled-down budget.
QUIET_SECONDS = 0.3


def _watcher_state(path: Path) -> str:
    """What the preferences watcher is observing, for a timeout message."""
    from polyx.gui import i18n_qt

    watcher = i18n_qt._watcher
    if watcher is None:
        return "no watcher (activate() never ran)"
    files = {Path(entry) for entry in watcher.files()}
    directories = {Path(entry) for entry in watcher.directories()}
    return f"file={path in files}, parent={path.parent in directories}"


def _timeout_report(what: str, elapsed: float, expected: str | None) -> str:
    """The state a maintainer needs to diagnose a wait that never finished."""
    from polyx.core import i18n, preferences

    target = Path(preferences.FILE)
    if target.is_file():
        try:
            saved = repr(target.read_text(encoding="utf-8"))
        except OSError as error:  # the file vanished between stat and read
            saved = f"unreadable ({error})"
        preferences_state = f"{target}: {saved}"
    else:
        preferences_state = f"{target}: absent"
    languages = f"  active language: {i18n.language()!r}\n"
    if expected is not None:
        languages += f"  expected language: {expected!r}\n"
    return (
        f"Live retranslation timed out after {elapsed:.2f} s waiting for {what}.\n"
        f"  elapsed: {elapsed:.2f} s\n"
        f"{languages}"
        f"  preferences file {preferences_state}\n"
        f"  watcher: {_watcher_state(target)}"
    )


def _wait_until(
    qapp,
    predicate,
    *,
    expected: str | None = None,
    timeout_s: float = DEFAULT_WAIT_SECONDS,
    what: str = "the language change",
    expect_timeout: bool = False,
) -> float:
    """Process events until ``predicate`` holds, bounded by a monotonic clock.

    The event loop, not a count of sleeps, decides the outcome: a timer asks
    the predicate after the queued work has run, so a rebuild scheduled by the
    language-change notification is seen as soon as it lands. Only the
    monotonic deadline ends the wait, and a loaded machine makes each check
    later instead of changing the answer.

    ``expected`` is the language the wait wanted, when that is what it was
    waiting for; a wait on a probe or on anything else leaves it out instead of
    reporting a language it never looked at. ``expect_timeout`` makes the
    deadline the success condition: that is the "nothing must change" case,
    which runs the same loop for its whole window and asserts nothing about
    what it saw.

    Returns the elapsed seconds. On timeout raises ``AssertionError`` with the
    elapsed time, the languages, the preferences bytes and the watcher state,
    so the failure is diagnosable from the log. Never aborts the interpreter.
    """
    from PySide6.QtCore import QEventLoop, QTimer

    started = time.monotonic()
    deadline = started + timeout_s
    loop = QEventLoop()

    def poll() -> None:
        if time.monotonic() >= deadline or (not expect_timeout and predicate()):
            loop.quit()

    ticker = QTimer()
    ticker.setInterval(_POLL_INTERVAL_MS)
    ticker.timeout.connect(poll)
    ticker.start()
    try:
        loop.exec()
    finally:
        ticker.stop()

    qapp.processEvents()
    elapsed = time.monotonic() - started
    if expect_timeout or predicate():
        return elapsed
    raise AssertionError(_timeout_report(what, elapsed, expected))


def _stay_quiet(qapp, seconds: float = QUIET_SECONDS) -> float:
    """Run the loop for a deliberate short window, then return.

    The same loop as :func:`_wait_until`, with the deadline as the only exit:
    the point is that nothing happens, so there is no predicate to satisfy.
    """
    return _wait_until(
        qapp,
        lambda: False,
        timeout_s=seconds,
        what="nothing to happen",
        expect_timeout=True,
    )


# ── Another application ──────────────────────────────────────────────
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

    # Wait on what the user sees, not only on the active locale: the window's
    # rebuild is queued behind the notification, so the locale can already be
    # 'en' while the button still reads in Spanish.
    _wait_until(
        qapp,
        lambda: window.btn_prefs.text() == "Preferences",
        expected="en",
        what="the launcher to show the language saved by another process",
    )
    assert i18n.language() == "en"
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

    _wait_until(
        qapp,
        lambda: i18n.language() == "en",
        expected="en",
        what="the watched language saved in a new preferences file",
    )


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
    _wait_until(
        qapp,
        lambda: i18n.language() == "en",
        expected="en",
        what="the language saved by another process",
    )
    assert changes == ["en"]

    # The applied value is not written back, so the watcher stays quiet. The
    # window is deliberately short: this case asserts nothing happens, and an
    # echo write would already have surfaced within a few hundred milliseconds.
    saved = preferences.FILE.read_bytes()
    _stay_quiet(qapp)
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
    generate_report(run, report, sections=["summary"])
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
    generate_report(run, after, sections=["summary"])
    html = after.read_text(encoding="utf-8")
    assert "Summary" in html
    assert "Resumen" not in html
    window.deleteLater()
