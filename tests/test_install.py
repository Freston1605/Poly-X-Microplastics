"""Smoke tests for the cross-platform installer helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "install"))

import gpu  # noqa: E402
import versions  # noqa: E402


def test_versions_pins() -> None:
    assert versions.TORCH_VERSION
    assert versions.TORCHVISION_VERSION
    assert versions.REQUIREMENTS == "requirements.txt"


def test_gpu_detect_cpu_when_no_tools() -> None:
    with mock.patch("shutil.which", return_value=None):
        info = gpu.detect_gpu()
    assert info["backend"] == "cpu"


def test_gpu_detect_cuda(monkeypatch) -> None:
    def fake_which(name):
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    def fake_run(cmd, **kw):
        class R:
            returncode = 0
            stdout = "8.6\n" if "compute_cap" in " ".join(cmd) else "RTX 3080\n"
            stderr = ""

        return R()

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)
    info = gpu.detect_gpu()
    assert info["backend"] == "cuda"
    assert info["cc_major"] == 8


def test_bootstrap_imports() -> None:
    import bootstrap

    assert hasattr(bootstrap, "main")
    assert hasattr(bootstrap, "_detect_gpu")
