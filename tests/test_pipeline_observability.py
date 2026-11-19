"""Headless pipeline failures reach the shared JSONL diagnostic journal."""

from __future__ import annotations

import os
from pathlib import Path

from polyx.pipeline.command import main


def test_pipeline_errors_keep_an_invariant_traceback_and_translate_at_display() -> None:
    """The journal gets the code and English template; the terminal gets prose."""
    from polyx.core.i18n import N_, set_language
    from polyx.pipeline.errors import PipelineError

    error = PipelineError("campaign.example", N_("Audit"))
    assert str(error) == "Audit"
    try:
        set_language("es")
        assert error.display() == "Auditoría"
    finally:
        set_language("en")


def test_a_failed_command_logs_an_error_event_with_a_traceback() -> None:
    """The terminal line is not the only record of a failed pipeline run."""
    assert main([]) == 1  # output_dir is required

    log = Path(os.environ["POLYX_LOG_DIR"]) / "pipeline.jsonl"
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    assert '"event": "pipeline.error"' in text
    assert '"level": "ERROR"' in text
    assert "ValueError" in text
