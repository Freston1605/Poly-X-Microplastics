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

The modules are separate processes, so the language change also has to
cross a process boundary. The launcher saves the choice in the shared
preferences file and every process watches that file: a module already
open applies the saved locale without being restarted. The watcher never
writes the value back (see :func:`polyx.core.i18n.set_language`), or the
processes would keep answering each other.

``polyx.core.i18n`` must stay Qt-free, so the bridge lives here: importing
this module registers it as a language-change subscriber.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QFileSystemWatcher,
    QLibraryInfo,
    QTranslator,
)
from PySide6.QtWidgets import QApplication

from polyx.core import i18n, preferences

_log = logging.getLogger(__name__)

_translator: QTranslator | None = None
_installed_tag: str | None = None
_watcher: QFileSystemWatcher | None = None
_watched_file: Path | None = None
_watched_signature: tuple[int, int] | None = None


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


# ── A choice made elsewhere ─────────────────────────────────────────
def _signature(path: Path) -> tuple[int, int] | None:
    """What identifies the file's current contents, or ``None`` if absent."""
    try:
        status = path.stat()
    except OSError:
        return None
    return (status.st_mtime_ns, status.st_size)


def _apply_saved_language() -> None:
    """Apply the locale another process saved, without saving it again.

    A change written by the launcher reaches the modules only through the
    file: there is no signal between processes. Applying it with
    ``persist=False`` is what keeps the watcher from echoing the value back
    and turning one change into a storm of rewrites.
    """
    saved = preferences.peek("locale")
    if not isinstance(saved, str) or not saved.strip():
        return
    resolved = i18n.resolve(saved)
    if resolved is None or resolved == i18n.language():
        return
    i18n.set_language(resolved, persist=False)


def _on_preferences_changed(path: str) -> None:
    """React to the file changing, wherever the write came from."""
    global _watched_signature  # noqa: PLW0603
    target = Path(preferences.FILE)
    changed = Path(path)
    if changed != target and changed != target.parent:
        return

    # A file replaced rather than edited loses its watch; put it back so
    # the next change is seen too.
    if _watcher is not None and target.is_file() and str(target) not in _watcher.files():
        _watcher.addPath(str(target))

    signature = _signature(target)
    if signature == _watched_signature:
        return
    _watched_signature = signature
    _apply_saved_language()


def _watch_preferences() -> None:
    """Follow the shared preferences file written by every Poly-X process.

    The file may not exist yet and may be rewritten in place, so the parent
    directory is watched as well and the file is re-added on every event.
    Calling it again for another window is cheap and only reconfigures the
    watcher when the preferences path itself has changed.
    """
    global _watcher, _watched_file, _watched_signature  # noqa: PLW0603
    app = QApplication.instance()
    if app is None:
        return

    target = Path(preferences.FILE)
    if _watcher is None:
        # Not parented to the application on purpose: the module keeps the
        # reference, and destroying it with the application can leave a
        # pending timer behind when a watched directory is gone.
        _watcher = QFileSystemWatcher()
        _watcher.fileChanged.connect(_on_preferences_changed)
        _watcher.directoryChanged.connect(_on_preferences_changed)
    elif _watched_file == target:
        if target.is_file() and str(target) not in _watcher.files():
            _watcher.addPath(str(target))
        return
    else:
        watched = _watcher.files() + _watcher.directories()
        if watched:
            _watcher.removePaths(watched)

    _watched_file = target
    _watched_signature = _signature(target)
    if target.parent.is_dir():
        _watcher.addPath(str(target.parent))
    if target.is_file():
        _watcher.addPath(str(target))


def activate() -> None:
    """Make the current language active for Qt without broadcasting.

    Called by every window when it is created, so a module launched in
    Spanish has Spanish native dialogs from its first frame. It is cheap and
    idempotent; it also re-registers the subscriber when a test has reloaded
    the translation module, and starts watching the shared preferences file
    so a language chosen in another process reaches this one.
    """
    i18n.when_changing(_on_language_change)
    tag = i18n.language()
    if _installed_tag != tag:
        install_qt_translations(tag)
    _watch_preferences()


activate()
