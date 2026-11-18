"""Updater events carry English data keys and stable English reason tokens.

The check itself runs in a background thread, but ``SearchUpdates._work`` is
the single place that emits the update.* events. Calling it directly with the
version lookups stubbed exercises the real observability seam: the JSONL
records on disk, not a private helper or an internal constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _records(tmp_path: Path) -> list[dict]:
    path = tmp_path / "_polyx_logs" / "updates.jsonl"
    assert path.is_file()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def updates(monkeypatch):
    from polyx.gui import updates as module

    monkeypatch.setattr(module, "version_installed", lambda: "abc1234")
    monkeypatch.setattr(module, "version_remote", lambda: "abc1234")
    return module


def test_skipped_without_installed_version_uses_english_reason(
    qapp, tmp_path, monkeypatch, updates
):
    monkeypatch.setattr(updates, "version_installed", lambda: "")

    updates.SearchUpdates()._work()

    record = _records(tmp_path)[0]
    assert record["event"] == "update.skipped"
    assert record["data"] == {"reason": "no_installed_version"}


def test_error_without_remote_response_uses_english_reason(qapp, tmp_path, monkeypatch, updates):
    monkeypatch.setattr(updates, "version_remote", lambda: "")

    updates.SearchUpdates()._work()

    record = _records(tmp_path)[0]
    assert record["event"] == "update.error"
    assert record["level"] == "WARNING"
    assert record["data"]["reason"] == "no_remote_response"
    assert record["data"]["local"] == "abc1234"


def test_available_uses_english_remote_key(qapp, tmp_path, monkeypatch, updates):
    monkeypatch.setattr(updates, "version_remote", lambda: "deadbeefcafe")

    updates.SearchUpdates()._work()

    record = _records(tmp_path)[0]
    assert record["event"] == "update.available"
    assert record["data"] == {"remote": "deadbee", "local": "abc1234"}


def test_update_events_keep_envelope_and_english_data(qapp, tmp_path, monkeypatch, updates):
    monkeypatch.setattr(updates, "version_remote", lambda: "deadbeefcafe")

    updates.SearchUpdates()._work()

    for record in _records(tmp_path):
        assert {"ts", "level", "module", "pid", "thread", "event", "data"} <= set(record)
        assert record["module"] == "polyx.updates"
        assert not (
            {"remoto", "sin_respuesta_remota", "sin_version_instalada"} & set(record["data"])
        )
