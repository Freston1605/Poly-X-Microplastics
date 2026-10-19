#!/usr/bin/env python3
"""LibCST codemod scaffolding to translate polyx comments to English.

Milestone M006, slice S03, task T01. This core mirrors the S02 docstring
driver (``tools/translate_docstrings.py``) but visits ``cst.Comment``
nodes instead of docstrings. It enumerates every comment in document
order and rewrites only the COMMENT tokens selected by a translation
manifest, preserving the leading hash prefix and any trailing whitespace
byte-for-byte. Because LibCST round-trips the module, code, string
literals, identifiers, docstrings and line endings stay unchanged.

CLI:
    python3 tools/translate_comments.py --dry-run polyx/model
    python3 tools/translate_comments.py --apply \
        --manifest <manifest_module> polyx/model

A manifest is a Python module defining ``COMMENT_TRANSLATIONS``:
``{file_path: {comment_index: new_body}}`` with repository-relative
posix paths and 0-based indices in document order. ``new_body`` is the
comment text WITHOUT the ``#`` (or ``#:``) prefix and without the run of
spaces/tabs that follows it; the core re-adds that prefix and preserves
any trailing whitespace. Re-running ``--apply`` with the same manifest
is a no-op once the comments already match (idempotency).

Protected directives (shebangs, coding cookies, ``# noqa``, ``# type:``,
``# pragma:``, ``# fmt:``, ``# isort:``, ``# pylint:``, ``# mypy:``,
``# nosec``, ``# %%`` cell markers) and empty/separator-only comments
are enumerated but can never be translated.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import libcst as cst

# A comment starts with one or more hashes, an optional ':' (the '#:'
# doc-comment style), and the full run of spaces/tabs that follows (D026).
# The whitespace run is part of the prefix so aligned tables keep their
# left margin.
_PREFIX = re.compile(r"^(?P<hashes>#+)(?P<colon>:?)(?P<space>[ \t]*)")

# Directives that must never be rewritten.
_SHEBANG = re.compile(r"^#!")
_CODING_COOKIE = re.compile(r"\bcoding[:=]\s*[-.\w]+", re.IGNORECASE)
_DIRECTIVE = re.compile(
    r"^#\s*(?:"
    r"noqa\b|type\s*:|pragma\s*:|fmt\s*:|isort\s*:|pylint\s*:|mypy\s*:|nosec\b"
    r")",
    re.IGNORECASE,
)
_CELL_MARKER = re.compile(r"^#\s*%%")

# Any Unicode alphanumeric (letters and digits) makes a comment prose;
# a body made only of punctuation or box-drawing characters is a
# separator line and stays untouched.
_PROSE_CHARACTER = re.compile(r"[^\W_]", re.UNICODE)


@dataclass
class CommentChange:
    """One comment rewrite (applied, or would-be under ``--dry-run``)."""

    index: int
    before: str  # body before the change, for the report
    after: str  # body after the change, for the report


def split_comment(raw: str) -> Tuple[str, str, str]:
    """Split a raw comment into ``(prefix, body, trailing_whitespace)``.

    The prefix is the leading hash run, an optional ``:`` and the FULL
    following run of spaces/tabs. The trailing whitespace is the run of
    spaces/tabs at the end of the comment. Both are preserved exactly by
    :func:`translate_source`; only ``body`` is translatable.
    """
    match = _PREFIX.match(raw)
    if match is None:
        # Defensive: Python comments always start with '#'. Treat the
        # whole value as the body so nothing is silently dropped.
        return "", raw, ""
    prefix = match.group(0)
    rest = raw[len(prefix) :]
    body = rest.rstrip(" \t")
    trailing = rest[len(body) :]
    return prefix, body, trailing


def is_protected_comment(raw: str) -> bool:
    """Return True when a raw comment must never be translated.

    Protected comments are shebangs, PEP 263 coding cookies, tool
    directives (``# noqa``, ``# type:``, ``# pragma:``, ``# fmt:``,
    ``# isort:``, ``# pylint:``, ``# mypy:``, ``# nosec``), ``# %%``
    cell markers, and empty or separator-only comments (bodies with no
    letters or digits, such as ``# ----`` or ``# =====``).
    """
    if _SHEBANG.match(raw):
        return True
    if _CODING_COOKIE.search(raw):
        return True
    if _DIRECTIVE.match(raw):
        return True
    if _CELL_MARKER.match(raw):
        return True
    _, body, _ = split_comment(raw)
    if not body.strip():
        return True
    if _PROSE_CHARACTER.search(body) is None:
        return True
    return False


def _validate_translation(index: int, raw: str, body: object) -> str:
    """Validate one manifest body and return it typed as ``str``.

    Raises ``ValueError`` when the body is not a string, contains a
    newline (a comment must stay on one line), is empty/whitespace-only
    (which would erase the comment body and leave a bare prefix), or
    targets a protected comment.
    """
    if not isinstance(body, str):
        raise ValueError(
            f"comment #{index}: translation must be a string, got {type(body).__name__}"
        )
    if "\n" in body or "\r" in body:
        raise ValueError(f"comment #{index}: translation contains a newline")
    if not body.strip():
        raise ValueError(
            f"comment #{index}: translation is empty and would erase the comment body"
        )
    if is_protected_comment(raw):
        raise ValueError(
            f"comment #{index}: protected comment cannot be translated: {raw!r}"
        )
    prefix, _, _ = split_comment(raw)
    if not prefix:
        raise ValueError(f"comment #{index}: cannot preserve the comment prefix")
    return body


class CommentTranslator(cst.CSTTransformer):
    """Translate the comments selected by an ``index -> body`` mapping.

    Indices are assigned in ``visit_Comment`` (document order) and the
    rewrites happen in ``leave_Comment`` keyed by ``id(node)``, the only
    place LibCST allows node replacement. No other node type is
    rewritten, so nothing outside COMMENT tokens can change.
    """

    def __init__(self, translations: Dict[int, str]):
        self.translations = translations
        self.index = 0
        self.changes: List[CommentChange] = []
        # (index, raw text) in document order, for inventory/list modes.
        self.enumeration: List[Tuple[int, str]] = []
        # id(node) -> index, assigned in visit and read in leave.
        self._index_by_id: Dict[int, int] = {}

    def visit_Comment(self, node: cst.Comment) -> None:
        index = self.index
        self.index += 1
        self._index_by_id[id(node)] = index
        self.enumeration.append((index, node.value))

    def leave_Comment(
        self, original_node: cst.Comment, updated_node: cst.Comment
    ) -> cst.Comment:
        index = self._index_by_id.get(id(original_node))
        if index is None or index not in self.translations:
            return updated_node
        raw = original_node.value
        body = _validate_translation(index, raw, self.translations[index])
        prefix, old_body, trailing = split_comment(raw)
        if old_body == body:
            # Already applied: a second pass reports no change (idempotency).
            return updated_node
        new_raw = f"{prefix}{body}{trailing}"
        if new_raw == raw:
            return updated_node
        self.changes.append(
            CommentChange(index=index, before=old_body, after=body)
        )
        return updated_node.with_changes(value=new_raw)


def translate_source(
    text: str, translations: Dict[int, str]
) -> Tuple[str, List[CommentChange], List[Tuple[int, str]]]:
    """Apply comment translations to a module.

    Returns ``(new_text, changes, enumeration)`` where ``enumeration``
    lists ``(index, raw_comment)`` for every COMMENT token in document
    order, whether translated or not.
    """
    module = cst.parse_module(text)
    translator = CommentTranslator(translations)
    new_module = module.visit(translator)
    enumeration = sorted(translator.enumeration, key=lambda pair: pair[0])
    for index in translations:
        if not isinstance(index, int):
            raise ValueError(
                f"comment index must be an int, got {index!r}"
            )
        if index < 0 or index >= len(enumeration):
            raise ValueError(
                f"comment #{index}: index out of range "
                f"(file has {len(enumeration)} comments)"
            )
    changes = sorted(translator.changes, key=lambda change: change.index)
    return new_module.code, changes, enumeration


def enumerate_comments(text: str) -> List[Tuple[int, str]]:
    """Return ``[(index, raw_comment)]`` for every comment in document order."""
    return translate_source(text, {})[2]


def _python_files(paths: Sequence[Path]) -> List[Path]:
    """Expand files and directories into a sorted list of ``*.py`` files."""
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        else:
            files.append(path)
    return files


def process_file(
    path: Path, translations: Dict[int, str], apply: bool
) -> List[CommentChange]:
    """Process one file; with ``apply`` rewrite in place, otherwise report only."""
    # newline="" preserves the original line endings (CRLF included) byte for byte.
    with path.open("r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    new_text, changes, _ = translate_source(text, translations)
    if apply and changes:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(new_text)
    return changes


def load_manifest(path: Path) -> Dict[str, Dict[int, str]]:
    """Load a manifest module that defines ``COMMENT_TRANSLATIONS``."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load manifest: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass consults sys.modules (MEM105).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    translations = getattr(module, "COMMENT_TRANSLATIONS", None)
    if not isinstance(translations, dict):
        raise SystemExit(f"manifest {path} does not define COMMENT_TRANSLATIONS")
    return translations


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the comment codemod CLI."""
    parser = argparse.ArgumentParser(
        description="Translate polyx comments to English through LibCST (M006/S03)."
    )
    parser.add_argument(
        "paths", nargs="+", type=Path, help="files or directories to process"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="report changes without writing"
    )
    mode.add_argument(
        "--apply", action="store_true", help="rewrite the files in place"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Python module that defines COMMENT_TRANSLATIONS",
    )
    args = parser.parse_args(argv)

    by_file: Dict[str, Dict[int, str]] = {}
    if args.manifest is not None:
        by_file = load_manifest(args.manifest)

    total = 0
    for path in _python_files(args.paths):
        key = path.as_posix()
        try:
            changes = process_file(path, by_file.get(key, {}), args.apply)
        except (OSError, UnicodeDecodeError, ValueError, cst.ParserSyntaxError) as error:
            print(f"ERROR in {key}: {error}", file=sys.stderr)
            return 1
        for change in changes:
            print(
                f"{key} #{change.index}: {change.before!r} -> {change.after!r}"
            )
        total += len(changes)
    verb = "applied" if args.apply else "detected"
    print(f"{total} change(s) {verb}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
