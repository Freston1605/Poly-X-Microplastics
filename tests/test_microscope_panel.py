"""Microscope panel class colours go through ``theme.class_qcolor``.

Ticket #39: the panel used to index the class palette directly, so an
unknown class raised ``KeyError`` while painting. Unknown classes now get
the documented fallback colour, and known classes keep their palette hex.
"""

from __future__ import annotations

import os

# Offscreen platform before importing anything that pulls in Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from polyx.gui import theme as T
from polyx.gui.microscope_panel import MicroscopePanel


def test_class_qcolor_unknown_class_falls_back():
    """The helper maps an unknown class to the documented fallback colour."""
    assert T.class_qcolor("ZZZ").name() == T.FALLBACK_COLOR_HEX


@pytest.mark.parametrize("name", sorted(T.CLASS_COLOR_HEX))
def test_class_qcolor_known_class_keeps_palette_colour(name):
    """Known classes still resolve to the exact palette colour."""
    assert T.class_qcolor(name).name() == T.CLASS_COLOR_HEX[name]


def test_unknown_class_renders_with_fallback_instead_of_raising(qapp, monkeypatch):
    """An unknown class paints the fallback colour instead of raising KeyError."""
    monkeypatch.setattr(T, "animations", lambda: False)
    panel = MicroscopePanel()
    panel.resize(320, 320)
    for part in panel._particles:
        part["clase"] = "ZZZ"

    pixmap = panel.grab()

    assert not pixmap.isNull()
