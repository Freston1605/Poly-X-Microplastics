"""Compatibility shim for the old headless run-folder loader.

Pre-canonical headless folders used to be rebuilt into a Qt ``DetectorState``
just so the report could consume them. The canonical report now speaks
``RunResult``, so this module keeps only the old public name and delegates to
the Qt-free core reader. It must never grow a second implementation.
"""

from __future__ import annotations

from pathlib import Path

from polyx.core.detection_run import RunResult, load_legacy_run


def build_state_from_run(run_dir: Path, image_dir: Path | None = None) -> RunResult:
    """Load an old headless run folder as the canonical ``RunResult``.

    Kept for callers that still import the historical name; the returned value
    is the same ``RunResult`` the report consumes for new runs.
    """
    return load_legacy_run(Path(run_dir), image_dir=Path(image_dir) if image_dir else None)


__all__ = ["build_state_from_run"]
