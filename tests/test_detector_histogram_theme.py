"""The detector size histogram chrome follows the active theme.

The bars are domain data, but the surface around them is interface chrome: on
a dark theme it must not stay a white card. No image is decoded — renderer and
font differences make pixel assertions machine-dependent — so this test pins
the values the histogram paints with by replacing the theme tokens.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from polyx.gui import theme as T
from polyx.gui.detector.pages.results import histogram_backgrounds


def test_histogram_backgrounds_read_the_theme_tokens(monkeypatch) -> None:
    monkeypatch.setattr(T, "BG_SOFT", "#010203")
    monkeypatch.setattr(T, "BG", "#040506")

    assert histogram_backgrounds() == ("#010203", "#040506")
