"""GPU detection for the installer (no torch required)."""

from __future__ import annotations

import shutil
import subprocess


def _run(cmd, **kw):
    return subprocess.run(cmd, check=False, **kw)  # noqa: S603


def detect_gpu() -> dict:
    """Return dict: backend ('cuda'|'rocm'|'cpu'), cc, cc_major, name."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            r = _run(
                [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode == 0 and r.stdout.strip():
                cc = r.stdout.strip().splitlines()[0].strip()
                major = int(cc.split(".")[0]) if "." in cc else int(cc)
                r2 = _run(
                    [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                name = (
                    r2.stdout.strip().splitlines()[0].strip()
                    if (r2.returncode == 0 and r2.stdout.strip())
                    else "NVIDIA GPU"
                )
                return {"backend": "cuda", "cc": cc, "cc_major": major, "name": name}
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            r = _run([nvidia_smi], capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                return {"backend": "cuda", "cc": None, "cc_major": None, "name": "NVIDIA GPU"}
        except Exception:  # noqa: BLE001, S110
            pass

    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        try:
            r = _run([rocm_smi], capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                return {"backend": "rocm", "cc": None, "cc_major": None, "name": "AMD GPU (ROCm)"}
        except Exception:  # noqa: BLE001, S110
            pass

    return {"backend": "cpu", "cc": None, "cc_major": None, "name": None}
