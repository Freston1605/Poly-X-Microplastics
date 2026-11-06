"""Qt's own catalogs and the language-change event.

Poly-X owns its strings through :mod:`polyx.core.i18n`; Qt owns the ones it
draws itself: file dialogs, message-box buttons, context menus and the
standard shortcuts. Both have to move together, or a Spanish window opens an
English file dialog and the language appears to change mid-action.

Installing a :class:`QTranslator` is not enough on its own. Qt announces a
language change to top-level widgets with ``QEvent.LanguageChange``, and a
window that does not receive it keeps the text it built at startup. This
module installs the catalog, re-broadcasts the event to every top-level
window and lets each window decide how to rebuild its own text.

``polyx.core.i18n`` must stay Qt-free, so the bridge lives here: importing
this module registers it as a language-change subscriber.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QLibraryInfo, QTranslator
from PySide6.QtWidgets import QApplication

from polyx.core import i18n

_log = logging.getLogger(__name__)

_translator: QTranslator | None = None
_installed_tag: str | None = None


def translations_dir() -> Path:
    """Directory where the PySide6 wheel ships Qt's compiled catalogs."""
    return Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))


def is_alive(widget) -> bool:
    """Whether the Qt object behind ``widget`` still exists.

    A window scheduled for deletion can still receive the language-change
    event; rebuilding it afterwards would touch a destroyed object.
    """
    import shiboken6

    return shiboken6.isValid(widget)


def install_qt_translations(tag: str) -> bool:
    """Load ``qtbase_<tag>`` for the interface, replacing the previous one.

    Returns whether a catalog was found. English is the source language, so
    no translator is needed for it: removing the previous one is enough and
    native dialogs fall back to English.
    """
    global _translator, _installed_tag  # noqa: PLW0603
    app = QApplication.instance()
    if app is None:
        return False

    if _translator is not None:
        app.removeTranslator(_translator)
        _translator = None

    if tag == "en":
        _installed_tag = tag
        return False

    directory = str(translations_dir())
    translator = QTranslator(app)
    loaded = translator.load(f"qtbase_{tag}", directory)
    if not loaded:
        # A regional tag may have no catalog of its own; the language one
        # (es_AR -> es) is the correct fallback.
        language_part = tag.split("_", 1)[0]
        if language_part != tag:
            loaded = translator.load(f"qtbase_{language_part}", directory)
    if loaded:
        app.installTranslator(translator)
        _translator = translator
    else:
        _log.info("Qt has no catalog for %r; native dialogs stay in English", tag)
    _installed_tag = tag
    return loaded


def broadcast_language_change() -> None:
    """Send ``QEvent.LanguageChange`` to every top-level window.

    Child widgets are the responsibility of their window: Qt does not
    propagate the event downwards, and a window knows which of its labels
    are data and which are interface text.
    """
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        QCoreApplication.sendEvent(widget, QEvent(QEvent.LanguageChange))


def _on_language_change(tag: str) -> None:
    install_qt_translations(tag)
    broadcast_language_change()


def activate() -> None:
    """Make the current language active for Qt without broadcasting.

    Called by every window when it is created, so a module launched in
    Spanish has Spanish native dialogs from its first frame. It is cheap and
    idempotent; it also re-registers the subscriber when a test has reloaded
    the translation module.
    """
    i18n.when_changing(_on_language_change)
    tag = i18n.language()
    if _installed_tag != tag:
        install_qt_translations(tag)


activate()
