"""Qt-side pieces the two annotation shells share.

The editing domain itself is Qt-free and lives in :mod:`polyx.core.editing`;
what remains here is the presentation the shells must agree on but that
cannot live in the core: the cursor a resize handle shows. One map, so the
Labeller and the Manual GT canvas point the same way at the same handle.
"""

from __future__ import annotations

from PySide6.QtCore import Qt

from polyx.core.editing import Handle

HANDLE_CURSORS: dict[Handle, Qt.CursorShape] = {
    Handle.TOP_LEFT: Qt.SizeFDiagCursor,
    Handle.BOTTOM_RIGHT: Qt.SizeFDiagCursor,
    Handle.TOP_RIGHT: Qt.SizeBDiagCursor,
    Handle.BOTTOM_LEFT: Qt.SizeBDiagCursor,
    Handle.TOP: Qt.SizeVerCursor,
    Handle.BOTTOM: Qt.SizeVerCursor,
    Handle.LEFT: Qt.SizeHorCursor,
    Handle.RIGHT: Qt.SizeHorCursor,
}


def cursor_for_handle(handle: Handle, default: Qt.CursorShape = Qt.CrossCursor):
    """The cursor for ``handle``, or ``default`` when there is no handle."""
    return HANDLE_CURSORS.get(handle, default)
