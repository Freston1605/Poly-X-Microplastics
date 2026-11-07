"""Babel extraction method that also sees translation calls inside f-strings.

The built-in ``python`` extractor walks the module AST but stops at a
``JoinedStr``: a call like ``f"<h3>{_('Methods')}</h3>"`` is invisible to it,
and this codebase puts most of the report's strings exactly there. A global
that is extracted for the ``.pot`` is a global that cannot be forgotten at
translation time, so this method walks every node instead.

Registered as ``polyx`` in the ``[tool.babel.extractors]`` table. The same
walk feeds :func:`iter_source_messages`, which the audit uses to compare the
committed template with the source without importing Babel.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

Keywords = dict[str, Any]

#: Singular message ids, resolved immediately or deferred.
_SINGULAR = {"_", "N_", "lazy", "gettext", "ugettext"}
#: Plural message ids.
_PLURAL = {"ngettext", "ungettext"}
#: Context-disambiguated ids.
_CONTEXTUAL = {"pgettext"}
#: Context-disambiguated plural ids.
_CONTEXTUAL_PLURAL = {"npgettext"}

#: Every function the extractor understands.
KEYWORDS = _SINGULAR | _PLURAL | _CONTEXTUAL | _CONTEXTUAL_PLURAL


@dataclass(frozen=True)
class SourceMessage:
    """A translatable message found in a source file."""

    line: int
    function: str
    message: str | tuple[str, str]
    context: str | None = None


def extract(
    fileobj,
    keywords: Keywords,
    comment_tags: list[str],
    options: dict[str, Any],
) -> Iterator[tuple[int, Any, list[str], str | None]]:
    """Yield Babel's ``(lineno, function, arguments, comments)`` protocol.

    Babel applies the keyword specification to ``arguments``: it knows that
    ``ngettext`` reads arguments 1 and 2 and that ``pgettext`` reads its
    context from argument 1. Yielding the raw positional arguments keeps that
    policy in one place, in ``[tool.babel]``'s keyword table and in the
    standard extractor defaults.
    """
    source = fileobj.read()
    if isinstance(source, bytes):
        source = source.decode(options.get("encoding", "utf-8"))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    names = set(keywords) or KEYWORDS
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = _call_name(node.func)
        if function is None or function not in names:
            continue
        yield node.lineno, function, [_constant(argument) for argument in node.args], []


def iter_source_messages(source: str) -> Iterator[SourceMessage]:
    """Yield every translatable message in ``source``.

    The audit walks the tree itself so a stale template is detected with the
    same rules the extractor uses, without a Babel import in the hook.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = _call_name(node.func)
        if function is None:
            continue
        arguments = [_constant(argument) for argument in node.args]
        message = _message(function, arguments, node.lineno)
        if message is not None:
            yield message


def _message(function: str, arguments: list[Any], line: int) -> SourceMessage | None:
    """Apply the gettext argument policy to one call, or return ``None``."""
    if function in _SINGULAR:
        if arguments and isinstance(arguments[0], str):
            return SourceMessage(line, function, arguments[0])
        return None
    if function in _PLURAL:
        if len(arguments) >= 2 and all(isinstance(arg, str) for arg in arguments[:2]):
            return SourceMessage(line, function, (arguments[0], arguments[1]))
        return None
    if function in _CONTEXTUAL:
        if len(arguments) >= 2 and isinstance(arguments[0], str) and isinstance(arguments[1], str):
            return SourceMessage(line, function, arguments[1], arguments[0])
        return None
    if function in _CONTEXTUAL_PLURAL:
        if len(arguments) >= 3 and all(isinstance(arg, str) for arg in arguments[:3]):
            return SourceMessage(line, function, (arguments[1], arguments[2]), arguments[0])
        return None
    return None


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _constant(node: ast.expr) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    return None
