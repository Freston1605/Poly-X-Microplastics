"""The palette-fallback guardrail catches literal fallbacks.

A guardrail that silently under-fires is worse than none: the old grep only
matched variable keys, so ``CLASS_COLOR_HEX.get("PET", "#e3342f")`` passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import check_palette_fallbacks as guard  # noqa: E402


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_literal_key_and_literal_fallback_are_flagged(tmp_path):
    path = _write(tmp_path, 'X = CLASS_COLOR_HEX.get("PET", "#e3342f")\n')
    assert guard.violations_in(path)


def test_variable_key_and_literal_fallback_are_flagged(tmp_path):
    path = _write(tmp_path, 'X = T.CLASS_COLOR_HEX.get(name, "#888")\n')
    assert guard.violations_in(path)


def test_shared_fallback_constant_is_allowed(tmp_path):
    path = _write(tmp_path, "X = T.CLASS_COLOR_HEX.get(name, T.FALLBACK_COLOR_HEX)\n")
    assert guard.violations_in(path) == []


def test_allowlist_names_real_files():
    # A dead exemption is how a guardrail quietly stops covering anything.
    for relative in guard.ALLOWED:
        assert (guard.ROOT / relative).is_file(), relative
