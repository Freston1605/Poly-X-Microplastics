"""Rutas estándar del proyecto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR  = ROOT / "models"
RUNS_DIR    = ROOT / "runs"
DATA_DIR    = ROOT / "data"
ASSETS_DIR  = ROOT / "polyx" / "assets"

DEFAULT_MODEL = ROOT / "bestdetectormedium.pt"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def ensure_dirs():
    for d in (MODELS_DIR, RUNS_DIR, DATA_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
