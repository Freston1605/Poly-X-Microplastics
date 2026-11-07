"""Every top-level window drops the previous language when the catalog changes.

This is the live-switch gate the spec asks for: instead of one assertion per
window, flip the language on a running application and walk the whole widget
tree. The reference is a window built fresh in the target language, so the
test catches a label, a button, a combo entry, a table header, a tab title, a
placeholder or a tooltip that was built once and never re-read.

The seam is ``i18n.set_language`` plus the text the widgets hold; nothing here
looks inside a catalog or at how a string was wrapped. Strings that read the
same in both languages (class names, units, numbers, file paths) are not
evidence either way, so the comparison is built from the texts that only exist
in the Spanish build: those are the previous language's text, and none of them
may survive the switch.
"""

from __future__ import annotations

import os

import pytest

# Offscreen before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QWidget,
)

WINDOW_NAMES = (
    "LauncherWindow",
    "DetectorWindow",
    "TrainerWindow",
    "LabelerWindow",
    "ViewerWindow",
    "PreferencesDialog",
)


def _window_classes() -> dict[str, type]:
    """Import lazily: the suite must stay importable without a display."""
    from polyx.gui.detector.window import DetectorWindow
    from polyx.gui.labeler.window import LabelerWindow
    from polyx.gui.launcher import LauncherWindow
    from polyx.gui.preferencias import PreferencesDialog
    from polyx.gui.trainer.window import TrainerWindow
    from polyx.gui.viewer.window import ViewerWindow

    return {
        "LauncherWindow": LauncherWindow,
        "DetectorWindow": DetectorWindow,
        "TrainerWindow": TrainerWindow,
        "LabelerWindow": LabelerWindow,
        "ViewerWindow": ViewerWindow,
        "PreferencesDialog": PreferencesDialog,
    }


def _build_windows() -> dict[str, QWidget]:
    return {name: cls() for name, cls in _window_classes().items()}


def _destroy(windows: dict[str, QWidget], qapp) -> None:
    for window in windows.values():
        window.deleteLater()
    qapp.processEvents()


def _widget_texts(widget: QWidget) -> list[str]:
    """Every piece of user-visible text a single widget owns."""
    texts: list[str] = []
    if isinstance(widget, QLabel):
        texts.append(widget.text())
    if isinstance(widget, QAbstractButton):
        texts.append(widget.text())
    if isinstance(widget, QGroupBox):
        texts.append(widget.title())
    if isinstance(widget, QLineEdit):
        texts.append(widget.text())
        texts.append(widget.placeholderText())
    if isinstance(widget, QComboBox):
        texts.extend(widget.itemText(index) for index in range(widget.count()))
    if isinstance(widget, QTableWidget):
        for column in range(widget.columnCount()):
            header = widget.horizontalHeaderItem(column)
            texts.append(header.text() if header is not None else "")
        for row in range(widget.rowCount()):
            header = widget.verticalHeaderItem(row)
            texts.append(header.text() if header is not None else "")
    if isinstance(widget, QTabWidget):
        texts.extend(widget.tabText(index) for index in range(widget.count()))
    if isinstance(widget, QListWidget):
        texts.extend(
            widget.item(index).text()
            for index in range(widget.count())
            if widget.item(index) is not None
        )
    if isinstance(widget, (QPlainTextEdit, QTextEdit)):
        texts.append(widget.placeholderText())
    texts.append(widget.toolTip())
    return [text for text in texts if text]


def _window_texts(window: QWidget) -> set[str]:
    """Every user-visible text inside one top-level window."""
    texts = {window.windowTitle(), window.toolTip()}
    for widget in window.findChildren(QWidget):
        texts.update(_widget_texts(widget))
    for action in window.findChildren(QAction):
        texts.add(action.text())
    return {text for text in texts if text}


def _all_texts(windows: dict[str, QWidget]) -> dict[str, set[str]]:
    return {name: _window_texts(window) for name, window in windows.items()}


@pytest.fixture(autouse=True)
def _restore_language(qapp):
    """Leave the suite in the language it pins, and flush destroyed windows."""
    from polyx.core import i18n

    previous = i18n.language()
    yield
    qapp.processEvents()
    i18n.set_language(previous)


@pytest.fixture
def _no_update_check(monkeypatch):
    """The launcher schedules a background update check; keep the test offline."""
    monkeypatch.setattr("polyx.gui.launcher.SearchUpdates.lookfor", lambda self: None)


def test_no_visible_text_survives_a_language_switch(qapp, _no_update_check):
    from polyx.core import i18n
    from polyx.gui import i18n_qt

    i18n_qt.activate()

    # Reference build: the same windows, freshly built in English.
    assert i18n.set_language("en")
    english_windows = _build_windows()
    try:
        english = _all_texts(english_windows)
    finally:
        _destroy(english_windows, qapp)

    # The application runs in Spanish, then the language changes underneath it.
    assert i18n.set_language("es")
    windows = _build_windows()
    try:
        spanish = _all_texts(windows)
        _assert_walk_is_not_vacuous(windows)
        assert i18n.set_language("en")
        qapp.processEvents()
        after = _all_texts(windows)

        # Anti-vacuity: every window must have carried text that only exists
        # in Spanish, or the walk is not exercising its translated strings.
        spanish_only = {name: spanish[name] - english[name] for name in WINDOW_NAMES}
        empty = sorted(name for name, texts in spanish_only.items() if not texts)
        assert not empty, f"no Spanish-only text was collected from: {empty}"

        survivors = {name: sorted(after[name] & spanish_only[name]) for name in WINDOW_NAMES}
        leftovers = {name: texts for name, texts in survivors.items() if texts}
        assert not leftovers, "text from the previous language survived the switch: " + "; ".join(
            f"{name}: {texts[:5]}" for name, texts in leftovers.items()
        )
    finally:
        _destroy(windows, qapp)


def _assert_walk_is_not_vacuous(windows: dict[str, QWidget]) -> None:
    """The tree must carry combos and table headers, not just labels.

    Without this probe the main test could pass by walking only labels, and a
    combo entry or a table header built once would go unnoticed.
    """
    kinds = set()
    for window in windows.values():
        for widget in window.findChildren(QWidget):
            if isinstance(widget, QComboBox) and widget.count():
                kinds.add("combo")
            if isinstance(widget, QTableWidget) and widget.columnCount():
                kinds.add("table")
            if isinstance(widget, QTabWidget) and widget.count():
                kinds.add("tab")
            if isinstance(widget, QLineEdit) and widget.placeholderText():
                kinds.add("placeholder")
    assert {"combo", "table"} <= kinds, f"the walk missed widget kinds: {kinds}"
