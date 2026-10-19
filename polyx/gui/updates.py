"""Notice of a new version available on GitHub.

The check runs on a separate thread and never blocks the interface: if
there is no internet, if GitHub answers slowly or if the installation
did not come from a ZIP, the launcher simply shows nothing. An update
notice does not justify delaying the startup or bothering with an
error.

Applying the update is done by an external script, different per system
(`actualizar.bat` on Windows, `actualizar_macOS.command` on Mac); here
it is only detected and announced. The installed-version mark is the
SHA of the latest commit, saved in `.polyx_version` by the installer
and by the updater itself.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.request
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from polyx.model.paths import ROOT
from polyx.model.plataforma import name_updater
from polyx.model.observabilidad import logger_for, event

REPO = "CrissFerrada/Poly-X-Microplastics"
BRANCH = "main"
API = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"
FILE_VERSION = ROOT / ".polyx_version"
# The name depends on the system: each one needs its own shell to
# unzip and overwrite. See `plataforma.nombre_actualizador()`.
UPDATER = ROOT / name_updater()

# Short on purpose: if GitHub does not reply quickly, it is not worth insisting.
TIMEOUT_S = 6


def version_installed() -> str:
    """Return the commit SHA recorded at install time, or '' if unknown."""
    try:
        return FILE_VERSION.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def version_remote() -> str:
    """Return the latest commit SHA on the tracked branch, or '' on failure."""
    req = urllib.request.Request(API, headers={"User-Agent": "PolyX-Updater"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return (json.load(r).get("sha") or "").strip()
    except Exception:
        # No internet, GitHub down, proxy, rate limit... none of this deserves
        # interrupting the user: it stays silent and shows no notice.
        return ""


class SearchUpdates(QObject):
    """Checks in the background whether there is a new version on GitHub."""

    there_is_a_new_version = Signal(str)  # Remote SHA, abbreviated

    def lookfor(self) -> None:
        """Launches the check without blocking; emits the signal only when needed."""
        thread = threading.Thread(target=self._work, daemon=True)
        thread.start()

    def _work(self) -> None:
        log = logger_for("updates")
        local = version_installed()
        # Without a local marker there is no baseline. It happens in development
        # installs (git clone): there the notice is unnecessary, it is updated with git.
        if not local:
            event(log, "update.skipped", reason="sin_version_instalada")
            return
        remote = version_remote()
        if not remote:
            # `version_remota` swallows every error: here it is recorded AT LEAST
            # that the check could not resolve, with a generic reason.
            event(log, "update.error", level=logging.WARNING,
                   reason="sin_respuesta_remota", local=local)
            return
        if remote != local:
            event(log, "update.available", remoto=remote[:7], local=local)
            self.there_is_a_new_version.emit(remote[:7])
        else:
            event(log, "update.ok", local=local)


def can_update() -> bool:
    """True when the updater that applies the changes exists."""
    return UPDATER.exists()
