"""What changes between Windows, macOS and Linux, in a single place.

Poly-X was born on Windows and took that for granted in eleven spots of the
code: ``os.startfile`` to open a folder or a report, ``cmd /c start`` to
launch the updater, and ``device="0"`` (CUDA) as the default device. None of
the three exists on a Mac.

Those differences are concentrated here so the rest of the program does not
have to know which system it is running on. The alternative -- an
``if sys.platform`` in every spot -- is a guaranteed way for one of them to
miss an update within a month and fail on a single platform.

ABOUT APPLE SILICON
-------------------
Macs with an M1 chip or later carry **MPS** (Metal Performance Shaders),
PyTorch's way to use the integrated GPU. It is not CUDA and it is not named
the same, but for YOLO inference it gives a real speedup over CPU. An Intel
Mac has no MPS and stays on CPU.

That is why the default device cannot be a constant: it is decided by looking
at what is available, in this order -- CUDA, MPS, CPU."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

# ── System identification ──
# Computed once: `sys.platform` does not change mid-run, and
# querying it on every call only makes the code noisier.
IS_WINDOWS = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


def name_system() -> str:
    """Readable system name, for the report and diagnostics."""
    if IS_MAC:
        # `platform.mac_ver()` gives "14.5"; `platform.release()` would give the
        # Darwin kernel number (23.5.0), which tells nobody anything.
        version = platform.mac_ver()[0] or platform.release()
        return f"macOS {version} ({architecture_mac() or platform.machine()})"
    return f"{platform.system()} {platform.release()}"


def architecture_mac() -> str:
    """'Apple Silicon', 'Intel' or '' if it is not a Mac.

    It matters because it decides whether GPU acceleration is available: MPS
    exists on Apple Silicon and not on Intel.
    """
    if not IS_MAC:
        return ""
    machine_2 = platform.machine().lower()
    if machine_2 in ("arm64", "aarch64"):
        return "Apple Silicon"
    return "Intel"


def open_in_system(path_2: str | Path) -> bool:
    """Opens a file or folder with its corresponding application.

    Cross-platform equivalent of ``os.startfile``, which only exists on
    Windows. It is used to open a run folder, an HTML report in the browser
    or an annotated image in the system viewer.

    **Never raises an exception.** Returns whether it succeeded. A folder not
    opening is an annoyance; taking down a freshly finished analysis is not.

    On macOS and Linux the process is started and not waited on: ``open``
    returns right away, but ``xdg-open`` on some desktops stays alive while
    the application is open, and waiting for it would freeze the interface.
    """
    path_2 = Path(path_2)
    if not path_2.exists():
        return False
    try:
        if IS_WINDOWS:
            os.startfile(str(path_2))  # type: ignore[attr-defined]  # noqa: S606
            return True
        command = ["open"] if IS_MAC else ["xdg-open"]
        subprocess.Popen(  # noqa: S603
            [*command, str(path_2)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, AttributeError, subprocess.SubprocessError):
        return False


# ── Compute device ──

DEVICE_CUDA = "0"
DEVICE_ROCM = "0"
DEVICE_HIP = "0"
DEVICE_AMD = "0"
DEVICE_MPS = "mps"
DEVICE_CPU = "cpu"


def _torch_module():
    try:
        import torch

        return torch
    except Exception:  # noqa: BLE001
        return None


def is_cuda_available() -> bool:
    """Whether an NVIDIA GPU is usable through CUDA."""
    torch = _torch_module()
    if torch is None:
        return False
    try:
        if not torch.cuda.is_available():
            return False
    except Exception:  # noqa: BLE001
        return False
    try:
        version_obj = getattr(torch, "version", None)
        return not bool(getattr(version_obj, "hip", None) if version_obj is not None else None)
    except Exception:  # noqa: BLE001
        return True


def is_rocm_available() -> bool:
    """Whether an AMD GPU is usable through ROCm (HIP)."""
    torch = _torch_module()
    if torch is None:
        return False
    try:
        if not torch.cuda.is_available():
            return False
    except Exception:  # noqa: BLE001
        return False
    try:
        version_obj = getattr(torch, "version", None)
        return bool(getattr(version_obj, "hip", None) if version_obj is not None else None)
    except Exception:  # noqa: BLE001
        return False


cuda_available = is_cuda_available
rocm_available = is_rocm_available
hip_available = is_rocm_available
amd_available = is_rocm_available


def accelerator_backend() -> str:
    """Name of the accelerator that ``device_available`` will pick."""
    if is_rocm_available():
        return "rocm"
    if is_cuda_available():
        return "cuda"
    torch = _torch_module()
    if torch is not None:
        try:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001, S110
            pass
    return "cpu"


backend = accelerator_backend
compute_backend = accelerator_backend
torch_backend = accelerator_backend


def device_available() -> str:
    """The best device this machine can use: '0', 'mps' or 'cpu'."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return DEVICE_CPU
    try:
        if torch.cuda.is_available():
            return DEVICE_ROCM if is_rocm_available() else DEVICE_CUDA
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return DEVICE_MPS
    except Exception:  # noqa: BLE001, S110
        pass
    return DEVICE_CPU


def label_devices() -> str:
    """Text for the device selector label in the interface.

    On Windows/Linux with NVIDIA the card index is shown; on a Mac that '0'
    means nothing and offering it only leads to error.
    """
    if IS_MAC:
        return (
            "Dispositivo (mps/cpu):"
            if architecture_mac() == "Apple Silicon"
            else "Dispositivo (cpu):"
        )
    if is_rocm_available():
        return "Dispositivo (0/rocm/cpu):"
    return "Dispositivo (0/cpu):"


def description_device(device: str) -> str:
    """Short phrase explaining what that device is, for the report."""
    d = (device or "").strip().lower()
    if d in ("mps",):  # noqa: FURB171
        return "GPU integrada del Mac (Metal Performance Shaders)"
    if d in ("cpu", "-1"):
        return "procesador (CPU)"
    if d.isdigit():
        if is_rocm_available():
            return f"GPU AMD #{d} (ROCm)"
        return f"GPU NVIDIA #{d} (CUDA)"
    if d in ("rocm", "hip", "amd"):
        return "GPU AMD (ROCm)"
    if d == "cuda":
        return "GPU NVIDIA (CUDA)"
    return device or "—"


# ── Updater ──
def set_identity_app(app_id: str) -> bool:
    """Declares to the system that this process is Poly-X, not Python.

    Windows groups taskbar buttons by **AppUserModelID**, not by the
    executable. A process launched as
    ``pythonw.exe -m polyx.gui.detector`` that does not declare its own
    inherits Python's, and the shell gives it the Python icon even when the
    window has its own. It also makes pinning the window to the taskbar pin
    Python.

    It must be called **before the first window is created**: the taskbar
    button is built with whatever identifier is in place at that moment, and
    changing it later has no effect until it is reopened.

    On macOS and Linux the concept does not exist and there is nothing to
    do: there the window icon is enough. Returns whether it got set.
    """
    if not IS_WINDOWS:
        return False
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return True
    except (AttributeError, OSError):
        # Windows Server without a complete shell32, or an old version. It is not
        # a reason not to start: only the taskbar icon is lost.
        return False


# The script that applies the update differs per system because it
# has to talk to each one's package manager and shell. The
# name is resolved here so `updates.py` and `launcher.py` do not repeat it.


def name_updater() -> str:
    if IS_WINDOWS:
        return "actualizar.bat"
    if IS_MAC:
        return "actualizar_macOS.command"
    return "actualizar_linux.sh"


def launch_updater(path_2: str | Path) -> bool:
    """Starts the updater in its own window and returns control.

    In a separate window on purpose: the updater overwrites the program
    files, including the one that launched it, so Poly-X must be able to
    close while it keeps running.
    """
    path_2 = Path(path_2)
    if not path_2.exists():
        return False
    try:
        if IS_WINDOWS:
            # 'start' with an empty title: without that "" the first quoted
            # argument is taken as the window title and the script
            # never gets to run.
            subprocess.Popen(  # noqa: S603
                ["cmd", "/c", "start", "", str(path_2)],  # noqa: S607
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return True
        if IS_MAC:
            # Terminal.app so the user can see the download progress.
            # Running it in the background would leave the application apparently
            # hung for several minutes without saying anything.
            subprocess.Popen(  # noqa: S603
                ["open", "-a", "Terminal", str(path_2)],  # noqa: S607
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        subprocess.Popen(  # noqa: S603
            ["/bin/bash", str(path_2)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


# ── Per-user data directory ──
def user_data_dir() -> Path:
    """The per-user writable home of models, runs and datasets.

    Windows uses ``%LOCALAPPDATA%``, macOS ``~/Library/Application Support``
    and the rest ``XDG_DATA_HOME`` (defaulting to ``~/.local/share``).
    ``POLYX_HOME`` overrides all of them, which is what tests and portable
    installs use.
    """
    override = os.environ.get("POLYX_HOME")
    if override:
        return Path(override).expanduser()
    if IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif IS_MAC:
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "polyx"


def environment_python(root: Path | None = None) -> Path | None:
    """Path to the .venv python, which hangs off a different folder per OS.

    Windows puts it in ``.venv/Scripts/python.exe`` and the rest of the world
    in ``.venv/bin/python``. Returns None if no environment has been created
    yet.
    """
    from polyx.core.paths import ROOT

    base = Path(root) if root else ROOT
    candidates_2 = [base / ".venv" / "Scripts" / "python.exe", base / ".venv" / "bin" / "python"]
    for c in candidates_2:
        if c.is_file():
            return c
    return None
