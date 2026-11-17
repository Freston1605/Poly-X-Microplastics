"""Ticket #150: the Viewer asks ``polyx.core.platform`` for its device.

The GPU checkbox keeps its meaning -- off is CPU, on is the best accelerator
this machine has -- but the answer is no longer derived inside the Viewer.
These tests replace the platform resolver and the model with fakes: no GPU, no
display and no real torch are involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")


class _RecordingModel:
    """Stands in for ``YoloModel``: records the call and returns no boxes."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._error = error

    def predict_auto(self, *_args, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return []

    @property
    def device(self) -> str:
        assert len(self.calls) == 1, "the Viewer must run inference exactly once"
        return self.calls[0]["device"]


def _detect_with(
    qapp,
    monkeypatch,
    resolved: str,
    *,
    gpu_checked: bool,
    rocm: bool | None = None,
):
    """Build a Viewer, resolve ``resolved`` and return what inference got.

    Returns the fake model and the status lines the window produced, so a
    test can assert both the device and what the user reads.
    ``rocm`` replaces the platform's ROCm probe: ``description_device`` names
    the card from it, and a real torch must never be needed.
    """
    from polyx.core import platform
    from polyx.gui.viewer.window import ViewerWindow

    monkeypatch.setattr(platform, "device_available", lambda: resolved)
    if rocm is not None:
        monkeypatch.setattr(platform, "is_rocm_available", lambda: rocm)

    window = ViewerWindow()
    model = _RecordingModel()
    window.state.model = model
    window.state.images = [Path("plate.png")]
    window.state.current_idx = 0
    window.chk_gpu.setChecked(gpu_checked)
    statuses: list[str] = []
    monkeypatch.setattr(window, "_set_status", lambda text, color=None: statuses.append(text))
    try:
        window._detect()
    finally:
        window.close()
    return model, statuses


@pytest.mark.parametrize("resolved", ["0", "mps", "cpu"])
def test_checkbox_on_uses_the_device_the_platform_resolves(resolved, qapp, monkeypatch):
    model, _statuses = _detect_with(qapp, monkeypatch, resolved, gpu_checked=True)

    assert model.device == resolved


@pytest.mark.parametrize("resolved", ["0", "mps"])
def test_checkbox_off_uses_cpu_whatever_hardware_exists(resolved, qapp, monkeypatch):
    """Off is an explicit CPU request, not "use the accelerator if you can"."""
    model, _statuses = _detect_with(qapp, monkeypatch, resolved, gpu_checked=False)

    assert model.device == "cpu"


def test_the_status_line_names_the_device_that_runs_inference(qapp, monkeypatch):
    model, statuses = _detect_with(qapp, monkeypatch, "mps", gpu_checked=True)

    assert model.device == "mps"
    assert "Metal Performance Shaders" in statuses[0]


def test_a_cuda_run_is_named_as_an_nvidia_gpu(qapp, monkeypatch):
    model, statuses = _detect_with(qapp, monkeypatch, "0", gpu_checked=True, rocm=False)

    assert model.device == "0"
    assert "NVIDIA" in statuses[0]
    assert "#0" in statuses[0]


def test_a_rocm_run_is_named_as_an_amd_gpu(qapp, monkeypatch):
    """ROCm also answers '0'; the description must not call it NVIDIA."""
    model, statuses = _detect_with(qapp, monkeypatch, "0", gpu_checked=True, rocm=True)

    assert model.device == "0"
    assert "AMD" in statuses[0]
    assert "ROCm" in statuses[0]
    assert "NVIDIA" not in statuses[0]


def test_a_cpu_run_says_it_runs_on_the_processor(qapp, monkeypatch):
    model, statuses = _detect_with(qapp, monkeypatch, "0", gpu_checked=False)

    assert model.device == "cpu"
    assert "CPU" in statuses[0]


def test_an_out_of_memory_run_leaves_the_fallback_to_the_model(
    qapp, monkeypatch, selected_language
):
    """The Viewer keeps its message and does not retry by itself.

    Retrying at smaller sizes is ``YoloModel``'s ladder; the Viewer must only
    report what the model could not do.
    """
    from PySide6.QtWidgets import QMessageBox

    from polyx.core import platform
    from polyx.gui.viewer.window import ViewerWindow

    selected_language("en")
    monkeypatch.setattr(platform, "device_available", lambda: "0")
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text))
    window = ViewerWindow()
    model = _RecordingModel(error=RuntimeError("CUDA out of memory"))
    window.state.model = model
    window.state.images = [Path("plate.png")]
    window.state.current_idx = 0
    window.chk_gpu.setChecked(True)
    try:
        window._detect()
    finally:
        window.close()

    assert len(model.calls) == 1, "the Viewer must not run its own retry ladder"
    assert "auto_fallback" not in model.calls[0], "the model's ladder stays in charge"
    assert "Out of GPU memory at imgsz 2080" in warnings[0]
    assert "untick" in warnings[0]
