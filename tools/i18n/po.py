"""A standard-library reader for the gettext files the project commits.

Babel writes the template and the catalogs, but the audit runs from a
pre-commit hook whose environment installs no third-party packages. Reading a
``.po``/``.pot`` is a small, well-specified job, so the gate owns this reader
instead of taking a dependency just to look at a file.

The reader covers what this repository commits: the header entry, ``msgctxt``,
``msgid``, ``msgid_plural``, ``msgstr``, ``msgstr[n]``, ``#:`` source
locations, ``#,`` flags and ``#~`` obsolete entries. Multi-line strings are
joined and escapes are decoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    '"': '"',
    "\\": "\\",
}

_FIELD = re.compile(r"^(msgctxt|msgid_plural|msgid|msgstr)(?:\[(\d+)\])?\s+(.*)$")

#: Named ``%(name)s`` specifiers and the positional forms the audit rejects.
#: The conversion specifier may carry flags, a width and a precision
#: (``%(done)6.1f``, ``%(pct)3d``), and those are part of the same token.
_NAMED_PLACEHOLDER = re.compile(r"%\(([A-Za-z_][A-Za-z0-9_]*)\)(?:[#0 +\-]*\d*(?:\.\d+)?)[sdifr]")
_POSITIONAL_PLACEHOLDER = re.compile(r"(?<!%)%([sdifr])")


@dataclass
class PoEntry:
    """One message of a catalog, with the line it starts on."""

    msgid: str = ""
    msgid_plural: str | None = None
    msgstr: str = ""
    msgstr_plural: dict[int, str] = field(default_factory=dict)
    context: str | None = None
    locations: list[tuple[str, int]] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    lineno: int = 0
    msgstr_lineno: int = 0
    obsolete: bool = False

    @property
    def fuzzy(self) -> bool:
        """Whether gettext will ignore this translation at runtime."""
        return "fuzzy" in self.flags

    @property
    def is_plural(self) -> bool:
        """Whether the entry carries a plural form."""
        return self.msgid_plural is not None

    @property
    def key(self) -> tuple[str | None, str | tuple[str, str]]:
        """The identity gettext uses: context plus id, or the plural pair."""
        if self.msgid_plural is not None:
            return self.context, (self.msgid, self.msgid_plural)
        return self.context, self.msgid

    def translations(self) -> list[str]:
        """Every translation string, singular first."""
        if self.is_plural:
            return [self.msgstr_plural[index] for index in sorted(self.msgstr_plural)]
        return [self.msgstr]


def _unquote(raw: str) -> str:
    """Decode a PO quoted string body."""
    body = raw.strip()
    if len(body) >= 2 and body.startswith('"') and body.endswith('"'):
        body = body[1:-1]
    out: list[str] = []
    index = 0
    while index < len(body):
        character = body[index]
        if character == "\\" and index + 1 < len(body):
            following = body[index + 1]
            out.append(_ESCAPES.get(following, following))
            index += 2
            continue
        out.append(character)
        index += 1
    return "".join(out)


def _locations(raw: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for token in raw.split():
        filename, separator, line = token.rpartition(":")
        if separator and line.isdigit():
            found.append((filename, int(line)))
        elif token:
            found.append((token, 0))
    return found


def parse_po(text: str) -> list[PoEntry]:
    """Parse ``text`` into entries, header included and obsolete entries marked."""
    entries: list[PoEntry] = []
    current = PoEntry()
    seen_msgid = False
    field_name: str | None = None
    field_index: int | None = None
    obsolete = False

    def flush() -> None:
        nonlocal current, seen_msgid, field_name, field_index, obsolete
        if seen_msgid or current.msgstr or current.msgstr_plural:
            current.obsolete = obsolete
            entries.append(current)
        current = PoEntry()
        seen_msgid = False
        field_name = None
        field_index = None
        obsolete = False

    def append(value: str) -> None:
        nonlocal current
        if field_name is None:
            return
        if field_name == "msgctxt":
            current.context = (current.context or "") + value
        elif field_name == "msgid":
            current.msgid += value
        elif field_name == "msgid_plural":
            current.msgid_plural = (current.msgid_plural or "") + value
        elif field_name == "msgstr":
            current.msgstr += value
        elif field_name == "msgstr_plural" and field_index is not None:
            current.msgstr_plural[field_index] = current.msgstr_plural.get(field_index, "") + value

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue

        is_obsolete = line.startswith("#~")
        if is_obsolete:
            line = line[2:].lstrip()
            obsolete = True
        elif line.startswith("#"):
            if line.startswith("#:"):
                current.locations.extend(_locations(line[2:].strip()))
            elif line.startswith("#,"):
                current.flags.update(part.strip() for part in line[2:].split(",") if part.strip())
            continue

        if line.startswith('"'):
            append(_unquote(line))
            continue

        match = _FIELD.match(line)
        if match is None:
            continue
        name, index, rest = match.groups()
        value = _unquote(rest)

        if name in {"msgctxt", "msgid"} and (seen_msgid or current.msgstr or current.msgstr_plural):
            flush()
        if name == "msgid":
            seen_msgid = True
            if not current.lineno:
                current.lineno = number
        if name == "msgstr" and not current.msgstr_lineno:
            current.msgstr_lineno = number
        if name == "msgstr_plural" and not current.msgstr_lineno:
            current.msgstr_lineno = number

        field_name = "msgstr_plural" if name == "msgstr" and index is not None else name
        field_index = int(index) if index is not None else None
        append(value)

    flush()
    return entries


def format_placeholders(text: str) -> set[str]:
    """The ``%`` substitution keys a message expects or provides.

    Named keys are returned bare (``count``); positional forms are returned as
    ``%d``/``%s`` so a translation that drops the name is visibly different.
    """
    named = {match.group(1) for match in _NAMED_PLACEHOLDER.finditer(text)}
    positional = {f"%{match.group(1)}" for match in _POSITIONAL_PLACEHOLDER.finditer(text)}
    return named | positional


def has_braces(text: str) -> bool:
    """Whether a message carries ``str.format``-style braces."""
    return "{" in text or "}" in text
