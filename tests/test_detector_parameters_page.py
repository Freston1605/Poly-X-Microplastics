"""Ticket #127: clearing the device field restores the resolved default.

``RunParams.device`` resolves through ``device_available`` (CUDA → MPS → CPU)
so the same build runs on a GPU workstation and on a CPU-only laptop. The
Parameters page used to hard-code ``"0"`` when the field was empty, which
bypassed that resolution and picked a CUDA card that a CPU-only machine does
not have.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PySide6.QtWidgets import QLabel  # noqa: E402

from polyx.gui.detector.pages import parameters  # noqa: E402
from polyx.gui.detector.pages.parameters import ParametersPage  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402


def _commit_device(page: ParametersPage, text: str) -> None:
    """Type a device and finish editing, as a user leaving the field does."""
    page.ed_device.setText(text)
    page.ed_device.editingFinished.emit()


def test_clearing_the_device_field_restores_the_resolved_default(qapp, monkeypatch):
    # A CPU-only machine: the resolver must never return a CUDA index.
    monkeypatch.setattr(parameters, "device_available", lambda: "cpu")
    state = DetectorState()
    state.params.device = "0"  # what a GPU machine had saved
    page = ParametersPage(state)

    _commit_device(page, "")

    assert state.params.device == "cpu"
    assert state.params.device != "0"


def test_a_typed_device_is_saved_unchanged(qapp, monkeypatch):
    monkeypatch.setattr(parameters, "device_available", lambda: "cpu")
    state = DetectorState()
    page = ParametersPage(state)

    _commit_device(page, "mps")

    assert state.params.device == "mps"


def test_the_device_hint_still_explains_the_device(qapp):
    page = ParametersPage(DetectorState())

    hints = [label.text() for label in page.findChildren(QLabel)]

    assert any("cpu" in text.lower() for text in hints), (
        "the Parameters page must explain what the device field accepts"
    )
