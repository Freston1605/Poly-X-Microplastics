"""Stable model aliases and the interface label that represents them.

A run folder's ``<alias>`` directory, the ``model`` column of
``detections.csv`` and ``ModelInfo.alias`` in ``run.json`` are artifacts: they
must read the same in every language (ADR-0004). The label the user reads is
prose and *is* translated (ADR-0003). One string cannot be both, so this module
owns the two values apart:

- :func:`resolve_identity` derives a stable, filesystem-safe, unique alias.
- :func:`display_name` renders ``_("Model %(n)s")`` for an alias the resolver
  generated and the alias itself for a name somebody chose.

Safety and uniqueness live here and nowhere else: ``ModelSpec``, the engine,
the GUI and the CLI all ask this module.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from polyx.core.i18n import _
from polyx.core.platform import INVALID_FILENAME_CHARS, RESERVED_FILENAME_NAMES

#: The aliases the resolver invents, including their ``_2``, ``_3`` suffixes.
_GENERATED_ALIAS = re.compile(r"^model_(\d+)(?:_\d+)*$")


@dataclass(frozen=True)
class ModelIdentity:
    """The two values a model needs: a folder key and how it was chosen.

    ``alias`` is stable, unique and filesystem-safe. ``alias_generated`` says
    the resolver had to invent ``model_<n>`` because neither the caller nor the
    weights named the model; only then does the interface translate the label.
    """

    alias: str
    alias_generated: bool


def generated_alias(index: int) -> str:
    """The invariant placeholder for the ``index``-th model (zero-based)."""
    return f"model_{index + 1}"


def sanitize_alias(raw: str) -> str:
    """One directory name, safe on Windows and POSIX, or ``""`` when unusable.

    Whitespace is trimmed, the string NFC-normalised, invalid characters
    replaced with ``_`` and trailing dots/spaces dropped (Windows strips them
    silently, so they would make two names collide). ``.``, ``..`` and the
    Windows device names cannot name a folder and come back empty or suffixed.
    """
    text = unicodedata.normalize("NFC", str(raw)).strip()
    clean = "".join(
        "_"
        if (character in INVALID_FILENAME_CHARS or character == "\x00" or ord(character) < 32)
        else character
        for character in text
    )
    clean = clean.rstrip(" .")
    if clean in ("", ".", ".."):
        return ""
    if clean.split(".", 1)[0].strip().upper() in RESERVED_FILENAME_NAMES:
        clean += "_"
    return clean


def _unique(alias: str, taken: set[str]) -> str:
    """Suffix ``_2``, ``_3``… until no existing alias shares the name.

    Compared case-insensitively because ``run`` and ``Run`` are one folder on
    Windows; a run folder must not depend on the filesystem it was written on.
    """
    existing = {name.casefold() for name in taken}
    if alias.casefold() not in existing:
        return alias
    suffix = 2
    while f"{alias}_{suffix}".casefold() in existing:
        suffix += 1
    return f"{alias}_{suffix}"


def resolve_identity(
    *,
    typed: str,
    weights: Path | None,
    index: int,
    taken: set[str],
) -> ModelIdentity:
    """Resolve one model's stable alias from what the caller actually chose.

    Order: the typed alias, else the weights stem, else ``model_<index>``. Only
    the last one is *generated*: a stem is a name derived from the file and is
    shown verbatim, like a typed name.
    """
    text = str(typed or "").strip()
    if not text and weights is not None:
        text = Path(weights).stem
    # Only ``model_<n>`` is a generated placeholder. A stem is a name derived
    # from the weights file, so it reads verbatim like a typed alias.
    generated = not text.strip()
    alias = sanitize_alias(text)
    if not alias:
        alias = generated_alias(index)
        generated = True
    return ModelIdentity(alias=_unique(alias, taken), alias_generated=generated)


def display_name(alias: str, alias_generated: bool, index: int) -> str:
    """What the interface shows for a model: a translated label or its name.

    A generated alias is ``model_<n>`` in every language; the interface reads
    ``_("Model %(n)s")``. A chosen alias — typed, taken from the weights stem
    or recorded by an older run — is shown exactly as it is, never retranslated.
    """
    if not alias_generated:
        return alias
    match = _GENERATED_ALIAS.match(alias)
    number = int(match.group(1)) if match else index + 1
    return _("Model %(n)s") % {"n": number}


__all__ = [
    "ModelIdentity",
    "display_name",
    "generated_alias",
    "resolve_identity",
    "sanitize_alias",
]
