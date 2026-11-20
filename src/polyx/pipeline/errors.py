"""Machine-readable pipeline failures with display-time translation.

A failure that reaches the JSONL journal must carry invariant tokens: a stable
code and its parameters. The message template stays in English for the
traceback; the command translates it only when writing to the terminal, using
the same gettext catalog as the rest of the interface (ADR-0003/0004).

One boundary serves every headless entry point: :func:`report_failure` journals
the traceback under the ``*.error`` alarm convention and prints the translated
sentence, so the CLI, the campaign and the sweeper do not each invent their own
half of it.
"""

from __future__ import annotations

import sys
from typing import Any

from polyx.core.i18n import _
from polyx.core.provenance.observability import event_error, logger_for


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


def describe_failure(exc: BaseException) -> str:
    """The sentence a person reads: translated for a coded failure."""
    if isinstance(exc, PipelineError):
        return exc.display()
    return f"{type(exc).__name__}: {exc}"


def report_failure(exc: BaseException, *, stage: str) -> None:
    """Journal one failure with its traceback and print it translated.

    The journal entry is the invariant half: ``pipeline.error``, the traceback
    (whose message is the untranslated template) and, for a coded failure, its
    stable ``error_code`` and ``error_parameters``. The terminal gets the active
    language, which is the only place a translation belongs.
    """
    extra: dict[str, Any] = {"stage": stage}
    if isinstance(exc, PipelineError):
        extra["error_code"] = exc.code
        extra["error_parameters"] = exc.parameters
    event_error(logger_for("pipeline"), "pipeline.error", exc, **extra)
    print(f"ERROR: {describe_failure(exc)}", file=sys.stderr)
