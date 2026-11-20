"""Best-epoch selection and patience, as Lightning's callbacks implement them.

The benchmark needs two thresholds on one score: the kept checkpoint is any
strictly better score, while patience is measured against ``min_delta``. This
suite drives both through the command seam on the real Lightning loop with only
the Ultralytics wrapper doubled, so it asserts the wiring rather than a private
helper. The proof is on disk: which epoch the published metrics name, which
native snapshot ``best.pt`` carries, and how many epochs ran before the loop
stopped itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_pipeline_local_tracer import (
    _boxes_prediction,
    _empty_prediction,
    _install_fakes,
    _read,
    _same_weights,
    build_tiny_dataset,
)


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(list(overrides))


def _partial_prediction() -> list[dict]:
    """Two of the three validation instances: with-class F1 = 0.8."""
    return _boxes_prediction()[:2]


def test_a_sub_delta_improvement_replaces_best_without_resetting_patience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two thresholds stay independent, which is why both callbacks exist.

    Epoch 0 scores 0.8 and epoch 1 scores 1.0. With ``min_delta=0.5`` the gain
    is smaller than the delta, so patience expires; but the score is strictly
    better, so ``best.pt`` is epoch 1. A single hand-rolled threshold could
    not do both.
    """
    output = tmp_path / "executions" / "two-thresholds"
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    _install_fakes(
        monkeypatch,
        [
            _partial_prediction(),
            _boxes_prediction(),
            # The publication pass re-scores the kept epoch.
            _boxes_prediction(),
        ],
    )

    code = _command(
        f"data_yaml={data_yaml}",
        f"output_dir={output}",
        "protocol.epochs=6",
        "protocol.early_stopping.patience=1",
        "protocol.early_stopping.min_delta=0.5",
        "protocol.amp.enabled=false",
    )

    assert code == 0
    state = _read(output / "training_state.json")
    metrics = _read(output / "metrics.json")

    # Exactly one non-improving record (against the delta) and then the stop.
    assert state["early_stopping"]["stopped_early"] is True
    assert state["early_stopping"]["wait_count"] == 1
    assert state["epochs_completed"] == 2

    # The checkpoint callback used the strict score, not the delta.
    assert metrics["selected_epoch"] == 1
    assert state["early_stopping"]["best_epoch"] == 1
    assert state["early_stopping"]["best_score"] == pytest.approx(1.0)
    assert _same_weights(
        output / "checkpoints" / "best.pt", output / "checkpoints" / "epoch_001.pt"
    )


def test_patience_survives_a_preemption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The callback's wait count is in the Lightning checkpoint, not the process.

    Two non-improving epochs happen before the job dies. After the resume only
    one more is needed to reach ``patience=3``; a run that restarted its
    patience bookkeeping would keep training to ``max_epochs``.
    """
    output = tmp_path / "executions" / "patience-resume"
    data_yaml = build_tiny_dataset(tmp_path / "dataset")
    overrides = [
        f"data_yaml={data_yaml}",
        f"output_dir={output}",
        "protocol.epochs=6",
        "protocol.early_stopping.patience=3",
        "protocol.amp.enabled=false",
    ]

    _install_fakes(
        monkeypatch,
        [
            _boxes_prediction(),
            _empty_prediction(),
            _empty_prediction(),
            _empty_prediction(),
        ],
        # Preempt during epoch 3's validation: epochs 0-2 completed.
        interrupt_after=4,
    )
    assert _command(*overrides) == 1
    progress = _read(output / "progress.json")
    assert progress["epochs_completed"] == 3
    assert progress["best_epoch"] == 0

    _install_fakes(
        monkeypatch,
        [
            # Epoch 3 is the third non-improving record: patience expires.
            _empty_prediction(),
            # The publication pass re-scores the kept (first) epoch.
            _boxes_prediction(),
        ],
    )
    assert _command(*overrides, "resume=true") == 0

    state = _read(output / "training_state.json")
    metrics = _read(output / "metrics.json")
    assert state["resumed"] is True
    assert state["epochs_completed"] == 4, "patience did not survive the resume"
    assert state["early_stopping"]["stopped_early"] is True
    assert state["early_stopping"]["best_epoch"] == 0
    assert metrics["selected_epoch"] == 0
    assert json.loads((output / "progress.json").read_text())["best_epoch"] == 0
