"""The kept checkpoint is the best score; patience is the delta comparison."""

from __future__ import annotations

from pathlib import Path

from polyx.pipeline.training import EarlyStoppingState


def _state(**overrides: object) -> EarlyStoppingState:
    config: dict = {"patience": 2, "min_delta": 0.001}
    config.update(overrides)
    return EarlyStoppingState(config)


def test_an_improvement_below_the_delta_still_becomes_the_kept_checkpoint(
    tmp_path: Path,
) -> None:
    """A gain smaller than min_delta is a better model, not a discardable one."""
    state = _state()
    first = tmp_path / "epoch_000.pt"
    second = tmp_path / "epoch_001.pt"
    first.write_text("weights-0", encoding="utf-8")
    second.write_text("weights-1", encoding="utf-8")

    assert state.update(0.9000, 0, first).improved
    decision = state.update(0.9005, 1, second)

    assert decision.improved, "a strictly better score was not kept as the best"
    assert state.best_score == 0.9005
    assert state.best_epoch == 1
    assert state.best_checkpoint == second
    restored = state.restore(tmp_path / "best.pt")
    assert restored == tmp_path / "best.pt"
    assert restored.read_text(encoding="utf-8") == "weights-1"


def test_patience_ignores_the_same_marginal_improvements(tmp_path: Path) -> None:
    """Epochs that only gain less than the delta still exhaust patience."""
    state = _state(patience=2)

    assert not state.update(0.9000, 0, tmp_path / "0.pt").stop
    assert not state.update(0.9005, 1, tmp_path / "1.pt").stop
    stopped = state.update(0.9009, 2, tmp_path / "2.pt")

    assert stopped.stop
    assert stopped.improved, "the last epoch also set a new best score"
    assert state.stopped_early
    assert state.best_score == 0.9009


def test_a_gain_at_or_above_the_delta_resets_patience(tmp_path: Path) -> None:
    state = _state(patience=2)

    state.update(0.9000, 0, tmp_path / "0.pt")
    state.update(0.9001, 1, tmp_path / "1.pt")
    reset = state.update(0.9100, 2, tmp_path / "2.pt")

    assert reset.improved and not reset.stop
    assert state.non_improving == 0
    assert state.patience_reference == 0.9100


def test_the_patience_reference_survives_a_snapshot_round_trip(tmp_path: Path) -> None:
    """A resumed run measures patience from where the interrupted one stopped."""
    state = _state()
    state.update(0.9000, 0, tmp_path / "0.pt")
    state.update(0.9005, 1, tmp_path / "1.pt")
    snapshot = state.to_dict(epochs_completed=2)

    assert snapshot["patience_reference"] == 0.9000

    resumed = _state()
    resumed.load_state(snapshot)
    decision = resumed.update(0.9006, 2, tmp_path / "2.pt")

    assert resumed.best_epoch == 2
    assert resumed.non_improving == 2
    assert decision.stop, "patience was reset by a gain below the delta"


def test_a_snapshot_without_a_patience_reference_falls_back_to_the_best_score(
    tmp_path: Path,
) -> None:
    state = _state()
    state.load_state(
        {
            "best_score": 0.75,
            "best_epoch": 4,
            "best_checkpoint": str(tmp_path / "4.pt"),
            "non_improving": 1,
        }
    )

    assert state.patience_reference == 0.75
    assert state.best_epoch == 4
