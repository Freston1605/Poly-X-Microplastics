"""Pinned versions shared by the installer and the updater."""

import re
from pathlib import Path

TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"
REQUIREMENTS = "requirements.txt"
PYTHON_MIN = (3, 11)

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s#;]+)")


def pinned_version(package: str, requirements: Path) -> str | None:
    """The exact version ``requirements`` pins for ``package``, if any.

    The requirements file is the desktop manifest, so reading it is what keeps
    the installer and the user-facing dependency set from drifting apart.
    """
    wanted = package.lower().replace("_", "-")
    for line in requirements.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line.strip())
        if match and match.group(1).lower().replace("_", "-") == wanted:
            return match.group(2)
    return None
