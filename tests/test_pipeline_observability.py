"""Headless pipeline failures reach the shared JSONL diagnostic journal."""

from __future__ import annotations

import os
from pathlib import Path

from polyx.pipeline.command import main


def test_a_failed_command_logs_an_error_event_with_a_traceback() -> None:
    """The terminal line is not the only record of a failed pipeline run."""
    assert main([]) == 1  # output_dir is required

    log = Path(os.environ["POLYX_LOG_DIR"]) / "pipeline.jsonl"
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    assert '"event": "pipeline.command_error"' in text
    assert '"level": "ERROR"' in text
    assert "ValueError" in text
