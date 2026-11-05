"""Tests for the project paths — where Poly-X writes its data.

The rule is ADR-0001: ALL user-generated run data — detector runs,
training runs, the review dialog's correction dataset — lives in the
per-user data directory, never next to the code. ``POLYX_HOME``
overrides where that directory physically is (portable installs); the
rule itself does not change with how the package was installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
_SRC = _RAIZ / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core import paths  # noqa: E402


def test_run_data_lives_in_the_user_data_dir():
    assert paths.RUNS_DIR == paths.USER_DATA_DIR / "runs"
    assert paths.RUNS_TRAIN_DIR == paths.USER_DATA_DIR / "runs_train"
    assert paths.DATASET_CORRECTIONS_DIR == paths.USER_DATA_DIR / "dataset_correcciones"


def test_user_data_is_never_inside_the_source_tree():
    # The property that makes wheel installs safe: no matter how the
    # package was installed, the data directory is not a child of the
    # source checkout, so nothing ever tries to write next to site-packages.
    assert paths.ROOT not in paths.RUNS_DIR.parents
    assert paths.ROOT not in paths.RUNS_TRAIN_DIR.parents
    assert paths.ROOT not in paths.DATASET_CORRECTIONS_DIR.parents


def test_the_ghost_dir_maker_is_gone():
    # ensure_dirs() had zero callers and manufactured user-data folders
    # nothing used. Its deletion is pinned so it is not re-added casually.
    assert not hasattr(paths, "ensure_dirs")
