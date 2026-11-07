"""A development-only pseudo-locale.

It exists so a developer can see by eye which strings were never extracted:
every translation is bracketed and its vowels are accented, so an English
sentence on screen is a string that bypassed ``_()``. Placeholders and HTML
tags are left intact, because mangling them would break the layout rather than
reveal it.

The catalog is generated under ``src/polyx/locales/qps_ploc``, which is
gitignored and excluded from the wheel and the frozen application. It is not
part of any gate.
"""

from __future__ import annotations

import re
from pathlib import Path

from babel.messages.catalog import Catalog
from babel.messages.pofile import read_po, write_po

LOCALES = Path("src/polyx/locales")
TEMPLATE = Path("src/polyx/locales/messages.pot")
TAG = "qps_ploc"

_ACCENTS = str.maketrans("aeiouAEIOU", "áéíóúÁÉÍÓÚ")
#: Placeholders and markup survive untouched; only prose is mangled.
_PROTECTED = re.compile(r"%\([^)]+\)[sdifr]|<[^>]+>|%%")


def generate(root: Path) -> Path:
    """Write the pseudo catalog for ``root`` and return its path."""
    template = read_po((root / TEMPLATE).open("rb"))
    catalog = Catalog(project="Poly-X", charset="utf-8")
    for message in template:
        if not message.id:
            continue
        translated = (
            tuple(mangle(part) for part in message.id)
            if isinstance(message.id, tuple)
            else mangle(message.id)
        )
        catalog.add(message.id, translated, context=message.context)

    target = root / LOCALES / TAG / "LC_MESSAGES" / "messages.po"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        write_po(handle, catalog, width=99)
    return target


def mangle(text: str) -> str:
    """Accent the prose of ``text`` and bracket the result."""
    pieces: list[str] = []
    cursor = 0
    for match in _PROTECTED.finditer(text):
        pieces.append(text[cursor : match.start()].translate(_ACCENTS))
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(text[cursor:].translate(_ACCENTS))
    return "⟦" + "".join(pieces) + "⟧"
