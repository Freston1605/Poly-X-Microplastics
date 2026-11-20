"""Where compiled catalogs have to end up.

The wheel gets them through a build hook. The compilation step leaves the
development-only pseudo-locale out: a mangled catalog must never ship.
"""

from __future__ import annotations

from pathlib import Path

from tools.i18n.catalog import compile_catalogs

__all__ = [
    "compile_catalogs",
    "wheel_catalog_entries",
]


def wheel_catalog_entries(root: Path) -> dict[str, str]:
    """Source ``.mo`` path -> destination inside the wheel, for every catalog."""
    entries: dict[str, str] = {}
    for compiled in compile_catalogs(root):
        source = compiled.relative_to(root).as_posix()
        destination = compiled.relative_to(root / "src").as_posix()
        entries[source] = destination
    return entries
