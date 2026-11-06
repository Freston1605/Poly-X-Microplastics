"""Babel extraction method that also sees translation calls inside f-strings.

The built-in ``python`` extractor walks the module AST but stops at a
``JoinedStr``: a call like ``f"<h3>{_('Methods')}</h3>"`` is invisible to it,
and this codebase puts most of the report's strings exactly there. A global
that is extracted for the ``.pot`` is a global that cannot be forgotten at
translation time, so this method walks every node instead.

Registered as ``polyx`` in the ``[tool.babel.extractors]`` table.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from typing import Any

Keywords = dict[str, Any]

#: Message id -> the tuple Babel expects for each writing style.
_SINGULAR = {"_", "N_", "lazy"}


def extract(
    fileobj,
    keywords: Keywords,
    comment_tags: list[str],
    options: dict[str, Any],
) -> Iterator[tuple[int, Any, list[str], str | None]]:
    """Yield ``(lineno, message, comments, context)`` for every call found."""
    source = fileobj.read()
    if isinstance(source, bytes):
        source = source.decode(options.get("encoding", "utf-8"))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    names = set(keywords) or _SINGULAR
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id not in names:
            continue
        func = node.func.id
        args = [_constant(argument) for argument in node.args]
        first = args[0] if args and isinstance(args[0], str) else None
        second = args[1] if len(args) > 1 and isinstance(args[1], str) else None
        third = args[2] if len(args) > 2 and isinstance(args[2], str) else None

        if func in _SINGULAR and first is not None:
            yield node.lineno, first, [], None
        elif func in {"ngettext", "ungettext"} and first is not None and second is not None:
            yield node.lineno, (first, second), [], None
        elif func == "pgettext" and second is not None:
            yield node.lineno, second, [], first
        elif func == "npgettext" and second is not None and third is not None:
            yield node.lineno, (second, third), [], first


def _constant(node: ast.expr) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    return None
