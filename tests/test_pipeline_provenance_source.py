"""Training provenance names the source even inside a source-less container."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import build_tiny_dataset
from tests.test_pipeline_local_tracer import _boxes_prediction, _install_fakes


def test_provenance_uses_the_stamped_commit_when_git_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container sees POLYX_SOURCE_SHA instead of an empty git answer."""
    from polyx.pipeline.command import main

    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    _install_fakes(monkeypatch, [_boxes_prediction()])
    monkeypatch.setenv("POLYX_SOURCE_SHA", "abc1234")
    monkeypatch.setattr(
        "polyx.pipeline.execution.commit_git", lambda _root: {"commit": "", "dirty": False}
    )
    output = tmp_path / "executions" / "container"
    assert main([f"data_yaml={data_yaml}", f"output_dir={output}", "protocol.epochs=1"]) == 0

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["git"]["commit"] == "abc1234"
    assert provenance["git"]["source"] == "image"
