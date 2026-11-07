"""Babel is the extraction and compilation toolchain.

``extract`` regenerates the committed template, ``update`` merges it into
every catalog, and ``compile`` writes the ``.mo`` files the runtime reads.
The audit does not import this module: reading a catalog is a small job and
pre-commit installs no third-party package, so the gate uses ``tools.i18n.po``
instead.
"""

from __future__ import annotations

from pathlib import Path

from babel.messages.catalog import Catalog
from babel.messages.extract import DEFAULT_KEYWORDS, extract_from_dir
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po, write_po
from tools.i18n import po

SOURCE = Path("src/polyx")
LOCALES = Path("src/polyx/locales")
TEMPLATE = Path("src/polyx/locales/messages.pot")
DOMAIN = "messages"
EXTRACTOR = "tools.i18n.extract:extract"
PSEUDO_TAG = "qps_ploc"

#: ``lazy`` is a project marker; the standard gettext names come from Babel.
KEYWORDS = {**DEFAULT_KEYWORDS, "lazy": None}


def extract_catalog(root: Path) -> Catalog:
    """Walk the source tree into an in-memory Babel catalog."""
    catalog = Catalog(project="Poly-X", charset="utf-8")
    source_root = root / SOURCE
    if not source_root.is_dir():
        return catalog

    for filename, lineno, message, comments, context in extract_from_dir(
        source_root,
        method_map=[("**.py", EXTRACTOR)],
        keywords=KEYWORDS,
        comment_tags=[],
    ):
        location = (SOURCE / filename).as_posix()
        catalog.add(message, None, [(location, lineno)], auto_comments=comments, context=context)
    return catalog


def write_template(root: Path) -> Path:
    """Regenerate the committed ``.pot`` from the source tree."""
    target = root / TEMPLATE
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        write_po(handle, extract_catalog(root), width=99)
    return target


def update_catalogs(root: Path) -> list[Path]:
    """Merge the template into every shipped catalog, keeping translations."""
    template = read_po((root / TEMPLATE).open("rb"))
    updated: list[Path] = []
    for catalog_path in _catalogs(root):
        with catalog_path.open("rb") as handle:
            catalog = read_po(handle)
        catalog.update(template)
        with catalog_path.open("wb") as handle:
            write_po(handle, catalog, width=99)
        updated.append(catalog_path)
    return updated


def compile_catalogs(root: Path, *, include_pseudo: bool = False) -> list[Path]:
    """Compile every catalog next to its ``.po`` and return the ``.mo`` paths."""
    compiled: list[Path] = []
    for catalog_path in _catalogs(root, include_pseudo=include_pseudo):
        target = catalog_path.with_suffix(".mo")
        with catalog_path.open("rb") as source, target.open("wb") as destination:
            write_mo(destination, read_po(source))
        compiled.append(target)
    return compiled


def _catalogs(root: Path, *, include_pseudo: bool = False) -> list[Path]:
    found = sorted((root / LOCALES).glob(f"*/LC_MESSAGES/{DOMAIN}.po"))
    if include_pseudo:
        return found
    return [path for path in found if path.parent.parent.name != PSEUDO_TAG]


def catalog_entries(root: Path) -> dict[tuple[str | None, object], po.PoEntry]:
    """The committed template as a keyed mapping (used by tooling and tests)."""
    return {
        entry.key: entry
        for entry in po.parse_po((root / TEMPLATE).read_text(encoding="utf-8"))
        if not entry.obsolete and entry.msgid
    }
