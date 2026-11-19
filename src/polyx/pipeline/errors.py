"""Machine-readable pipeline failures with display-time translation.

A failure that reaches the JSONL journal must carry invariant tokens: a stable
code and its parameters. The message template stays in English for the
traceback; the command translates it only when writing to the terminal, using
the same gettext catalog as the rest of the interface (ADR-0003/0004).
"""

from __future__ import annotations

from typing import Any

from polyx.core.i18n import _


class PipelineError(ValueError):
    """A failure whose machine identity is separate from its display text."""

    def __init__(self, code: str, template: str, **parameters: Any) -> None:
        self.code = code
        self.template = template
        self.parameters = parameters
        super().__init__(template % parameters if parameters else template)

    def display(self) -> str:
        """The operator-facing message, translated in the active language."""
        translated = _(self.template)
        return translated % self.parameters if self.parameters else translated
