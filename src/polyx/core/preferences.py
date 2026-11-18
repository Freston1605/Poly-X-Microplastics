"""Per-user preferences: one JSON file for theme, motion and language.

The three settings used to live in separate dotfiles, each with its own
loader and its own failure mode. Reading and writing them through one
owner keeps the file from being clobbered when two features save at the
same time.

The old dotfiles are not read: there is no migration and no dual-read.
They stay on disk untouched, and new code only ever uses this one file.
"""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

FILE = Path.home() / ".polyx_preferences.json"


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
    """Every saved preference, exactly as stored in the one canonical file.

    No legacy dotfile is consulted and nothing is written here; callers that
    persist a change go through :func:`write`.
    """
    return _read_raw()


def get(key: str, default: Any = None) -> Any:
    return read().get(key, default)


def peek(key: str, default: Any = None) -> Any:
    """Read one saved value without writing the file.

    A process that watches the file for another process's language change
    must not turn its own read into a write: that is how a watcher ends up
    feeding itself.
    """
    return _read_raw().get(key, default)


def write(**values: Any) -> None:
    """Merge ``values`` into the file; other keys are preserved."""
    data = read()
    data.update(values)
    _write_raw(data)
