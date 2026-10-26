"""Standard project paths.

Bundled (read-only) assets live next to the package; user-writable data
(models cache, runs, datasets) live in a per-user directory so the package
can be installed read-only (wheels, site-packages).
"""
from pathlib import Path
from typing import Optional
import os
import sys

# src/polyx/core/paths.py -> parents[0]=core, [1]=polyx, [2]=src, [3]=project root
ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = ROOT / "assets"

def _user_data_dir() -> Path:
    override = os.environ.get("POLYX_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "polyx"

USER_DATA_DIR = _user_data_dir()
MODELS_DIR = USER_DATA_DIR / "models"
RUNS_DIR = USER_DATA_DIR / "runs"
DATA_DIR = USER_DATA_DIR / "data"

DEFAULT_MODEL = MODELS_DIR / "bestdetectormedium.pt"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def discover_default_model() -> Optional[Path]:
    """Return DEFAULT_MODEL if it exists on disk, else None.

    Falls back to a bundled model next to the package (dev checkout).
    """
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    bundled = ROOT / "models" / "bestdetectormedium.pt"
    if bundled.is_file():
        return bundled
    return None

def ensure_dirs():
    for d in (MODELS_DIR, RUNS_DIR, DATA_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)