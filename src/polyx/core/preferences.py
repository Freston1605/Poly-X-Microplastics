"""Per-user preferences: one JSON file for theme, motion and language.

The three settings used to live in separate dotfiles, each with its own
loader and its own failure mode. Reading and writing them through one
owner keeps the file from being clobbered when two features save at the
same time, and gives the migration from the old files a single place to
happen.

The old files are read once and left untouched on disk: deleting a user's
file to migrate it is how a preference gets lost when the migration itself
fails.
"""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

FILE = Path.home() / ".polyx_preferences.json"

#: Old file -> {old key: new key}. Read once, on the first load.
_LEGACY: tuple[tuple[Path, dict[str, str]], ...] = (
    (Path.home() / ".polyx_tema.json", {"tema": "theme", "animaciones": "animations"}),
    (Path.home() / ".polyx_theme.json", {"tema": "theme", "animaciones": "animations"}),
    (Path.home() / ".polyx_idioma.json", {"idioma": "locale"}),
)


def _read_raw() -> dict[str, Any]:
    """The file as it is on disk. Absent or corrupt reads as an empty mapping."""
    try:
        data = json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_raw(data: dict[str, Any]) -> None:
    # A read-only home directory cannot stop the program: the choice applies
    # for this session and is simply not remembered.
    with suppress(OSError):
        FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def read() -> dict[str, Any]:
    """Every saved preference, with the old files folded in on first use."""
    data = _read_raw()
    migrated = _migrate(data)
    if migrated:
        data.update(migrated)
        _write_raw(data)
    return data


def get(key: str, default: Any = None) -> Any:
    return read().get(key, default)


def write(**values: Any) -> None:
    """Merge ``values`` into the file; other keys are preserved."""
    data = read()
    data.update(values)
    _write_raw(data)


def _migrate(current: dict[str, Any]) -> dict[str, Any]:
    """Keys found in the old files that ``current`` does not already have."""
    found: dict[str, Any] = {}
    for path, rename in _LEGACY:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(old, dict):
            continue
        for old_key, new_key in rename.items():
            if new_key not in current and old_key in old:
                found[new_key] = old[old_key]
    return found
