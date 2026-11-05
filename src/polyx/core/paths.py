"""Standard project paths.

Bundled (read-only) assets live next to the package; ALL user-generated
data — the models cache, detector runs, training runs and the review
dialog's correction dataset — lives in a per-user directory (ADR-0001),
so the package can be installed read-only (wheels, site-packages) and
the data survives reinstalls. The OS-dependent part of that directory
lives in :mod:`polyx.core.platform`; the ``POLYX_HOME`` environment
variable overrides it (portable installs).

``ROOT`` is the source-checkout root: meaningful when the package runs
from a source tree — the bundled-model fallback, the launcher's manual
and readme links — and meaningless under an installed package, where
nothing may write.
"""

from pathlib import Path

from polyx.core.platform import user_data_dir

# src/polyx/core/paths.py -> parents[0]=core, [1]=polyx, [2]=src, [3]=project root
ROOT = Path(__file__).resolve().parents[3]
# Bundled, read-only assets ship inside the package (src/polyx/assets) so
# they travel with the wheel; the repo-root assets/ directory is gone.
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

USER_DATA_DIR = user_data_dir()
MODELS_DIR = USER_DATA_DIR / "models"
DATA_DIR = USER_DATA_DIR / "data"

# ADR-0001: every run a user generates lands here — detector batches,
# training runs and the review dialog's correction dataset alike. The
# repo-root runs/ and runs_train/ folders of older versions are legacy;
# nothing is migrated and nothing is written there anymore.
RUNS_DIR = USER_DATA_DIR / "runs"
RUNS_TRAIN_DIR = USER_DATA_DIR / "runs_train"
DATASET_CORRECTIONS_DIR = USER_DATA_DIR / "dataset_correcciones"

DEFAULT_MODEL = MODELS_DIR / "bestdetectormedium.pt"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def discover_default_model() -> Path | None:
    """Return DEFAULT_MODEL if it exists on disk, else None.

    Falls back to a bundled model next to the package (dev checkout).
    """
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    bundled = ROOT / "models" / "bestdetectormedium.pt"
    if bundled.is_file():
        return bundled
    return None
