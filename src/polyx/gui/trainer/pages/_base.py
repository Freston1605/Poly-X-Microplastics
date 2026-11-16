"""Base page of the Trainer (header with icon+title, not a number)."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from polyx.gui import iconos
from polyx.gui import theme as T

from ..state import TrainerState

#: Threads whose run loop has not finished: they outlive the page that
#: started them. The set is the only owner, so a worker is released when its
#: ``finished`` signal arrives, not when the page dies.
#:
#: Known limit: the set is unbounded and it does not stop a thread. A worker
#: that never returns — a driver that hangs on a dead GPU, for instance — is
#: held until the interpreter exits, and Qt may then destroy it while it still
#: runs: the same abort, moved to shutdown. There is no safe bounded cleanup
#: here (terminating an arbitrary thread can corrupt the driver's state), so
#: stop such a worker where it is written instead of expecting this set to
#: reap it.
_live_threads: set[QThread] = set()


def keep_thread_alive(worker: QThread) -> None:
    """Start ``worker`` and keep it alive until its thread finishes.

    A language change rebuilds the Trainer's pages and the window can be
    closed while a job is still running. Qt aborts the process when a
    ``QThread`` object is destroyed with its thread still running, so the
    worker is owned here instead of by the page; a result the old page no
    longer listens to is dropped by Qt's own disconnection rule.

    The new live-retranslation teardown destroys a window with a probe in
    flight, and only this registry decides when that probe's wrapper is
    released: without it, the moment depends on when the last reference to a
    running thread is dropped. It closes a hazard that predates the test
    change, not a defect the test introduced.
    """
    _live_threads.add(worker)
    worker.finished.connect(lambda: _live_threads.discard(worker))
    worker.start()


class TrainerPage(QWidget):
    """Page with a manual-style header: large icon + title + subtitle."""

    PAGE_ICON: str = "diana"
    PAGE_TITLE: str = ""
    PAGE_DESCRIPTION: str = ""

    def __init__(self, state: TrainerState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setStyleSheet(f"QWidget {{ background: {T.BG_SOFT}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(38, 28, 38, 32)
        outer.setSpacing(18)

        # Step header (same as the manual: icon + large title)
        head = QHBoxLayout()
        head.setSpacing(14)
        # PAGE_ICON is the name of a drawing in polyx/gui/iconos.py, not an emoji.
        icon_lbl = QLabel()
        icon_lbl.setPixmap(iconos.pixmap(self.PAGE_ICON, 30, T.ACCENT_TX, 1.9))
        icon_lbl.setStyleSheet("border: none;")
        head.addWidget(icon_lbl, 0, Qt.AlignVCenter)
        title_lbl = QLabel(str(self.PAGE_TITLE))
        title_lbl.setStyleSheet(f"color: {T.INK}; font-size: 22pt; font-weight: 600; border: none;")
        head.addWidget(title_lbl, 1)
        outer.addLayout(head)

        if self.PAGE_DESCRIPTION:
            desc = QLabel(str(self.PAGE_DESCRIPTION))
            desc.setWordWrap(True)
            desc.setStyleSheet(
                f"color: {T.INK3}; font-size: 10pt; border: none; margin-left: 52px;"
            )
            outer.addWidget(desc)

        self.body = QVBoxLayout()
        self.body.setSpacing(16)
        outer.addLayout(self.body)
        outer.addStretch(1)

    def card(self, title: str, icon: str = "") -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(20, 16, 20, 18)
        lay.setSpacing(12)
        if title:
            row = QHBoxLayout()
            row.setSpacing(8)
            if icon:
                # `icon` is the name of a drawing in polyx/gui/iconos.py, not an
                # emoji: this way the header looks the same on Windows and macOS and
                # takes the theme color instead of coming with its own.
                ic = QLabel()
                ic.setPixmap(iconos.pixmap(icon, 17, T.INK2, 1.85))
                ic.setStyleSheet("border: none;")
                row.addWidget(ic)
            t = QLabel(title)
            t.setStyleSheet(f"color: {T.INK}; font-size: 13pt; font-weight: 600; border: none;")
            row.addWidget(t)
            row.addStretch(1)
            lay.addLayout(row)
        return frame, lay
