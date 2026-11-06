"""Where compiled catalogs have to end up.

The wheel gets them through a build hook, the frozen application through the
PyInstaller data files. Both use the same compilation step, and both leave the
development-only pseudo-locale out: a mangled catalog must never ship.
"""

from __future__ import annotations

from pathlib import Path

from tools.i18n.catalog import DOMAIN, LOCALES, PSEUDO_TAG, compile_catalogs

__all__ = [
    "compile_catalogs",
    "frozen_locale_datas",
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


def frozen_locale_datas(root: Path) -> list[tuple[str, str]]:
    """``(source, destination)`` pairs for the PyInstaller ``datas`` table."""
    compile_catalogs(root)
    datas: list[tuple[str, str]] = []
    for catalog in sorted((root / LOCALES).glob(f"*/LC_MESSAGES/{DOMAIN}.po")):
        locale_dir = catalog.parent.parent
        tag = locale_dir.name
        if tag == PSEUDO_TAG:
            continue
        for path in sorted(locale_dir.rglob("*")):
            if path.is_file():
                destination = f"polyx/locales/{tag}/{path.relative_to(locale_dir).as_posix()}"
                datas.append((str(path), destination))
    return datas
