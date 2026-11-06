"""The detector size histogram chrome follows the active theme.

The bars are domain data, but the surface around them is interface chrome: on
a dark theme it must not stay a white card. No image is decoded — renderer and
font differences make pixel assertions machine-dependent — so these tests pin
the values the histogram paints with.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from polyx.gui import theme as T
from polyx.gui.detector.pages import results
from polyx.gui.detector.pages.results import histogram_backgrounds


@pytest.mark.parametrize("theme_name", sorted(T.PALETTES))
def test_histogram_backgrounds_follow_the_active_theme(theme_name: str) -> None:
    original = T.theme()
    try:
        T.set_theme(theme_name)
        palette = T.PALETTES[theme_name]
        assert histogram_backgrounds() == (palette["BG_SOFT"], palette["BG"])
    finally:
        T.set_theme(original)


def test_histogram_source_has_no_literal_surface_colours() -> None:
    source = Path(results.__file__).read_text(encoding="utf-8")
    assert "#f6f8fa" not in source
    assert "#ffffff" not in source
    assert "T.BG_SOFT" in source
    assert "T.BG" in source
