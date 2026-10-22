"""Identifies with which code, which versions and which weights a result was
produced.

A run folder stores the parameters of the run, but that is not enough to
reproduce it: two different ``best.pt`` share the same name, ultralytics
changes the AP computation between versions and Poly-X itself changes
between commits. Without writing it down, within a year there is no way to
prove which weights produced a table of the paper.

Everything here is informational and must not take down a run: if git is not
installed or the .pt is on a disk that got disconnected, the gap is left
blank and the run continues."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from .paths import ROOT


def sha256_file(path_2: str | Path, _block: int = 1 << 20) -> str:
    """Fingerprint of the file, read in blocks.

    In blocks rather than all at once because a YOLO medium .pt weighs 52 MB
    and up to three can be loaded at the same time. Empty string if it
    cannot be read.
    """
    try:
        h = hashlib.sha256()
        with open(path_2, "rb") as f:
            for piece in iter(lambda: f.read(_block), b""):
                h.update(piece)
        return h.hexdigest()
    except OSError:
        return ""


def _version_de(module: str) -> str:
    """Version of an already-imported package, without forcing its import.

    ``sys.modules`` is checked on purpose: asking for torch here would load
    it in full (seconds and VRAM) in a run that may never use it.
    """
    mod = sys.modules.get(module)
    return str(getattr(mod, "__version__", "")) if mod is not None else ""


def versions() -> Dict[str, str]:
    """Version of Poly-X and of the libraries that decide the numerical result."""
    from .. import __version__ as version_polyx

    return {
        "polyx": version_polyx,
        "python": platform.python_version(),
        "torch": _version_de("torch"),
        "ultralytics": _version_de("ultralytics"),
        "opencv": _version_de("cv2"),
        "numpy": _version_de("numpy"),
        "so": f"{platform.system()} {platform.release()}",
    }


def commit_git(root: Optional[Path] = None) -> Dict[str, str]:
    """Current commit of the repository and whether the tree had uncommitted
changes.

    ``sucio`` matters as much as the hash: a run made on modified,
    uncommitted code is not reproducible even with the commit recorded.
    """
    root = Path(root) if root is not None else ROOT
    output = {"commit": "", "sucio": ""}

    def _git(*args: str) -> Optional[str]:
        try:
            r = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, timeout=5,
                # No black console when running under pythonw.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    commit = _git("rev-parse", "--short", "HEAD")
    if commit is None:
        return output
    output["commit"] = commit
    state = _git("status", "--porcelain")
    output["sucio"] = "si" if state else "no"
    return output


def provenance(root: Optional[Path] = None) -> Dict[str, object]:
    """Complete block to dump into the summary of a run."""
    data: Dict[str, object] = {"versiones": versions()}
    data.update(commit_git(root))
    return data
