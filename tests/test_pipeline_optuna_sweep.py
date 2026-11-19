"""Optuna sweep driven only through the established headless command.

The same doubles as the local tracer replace Lightning and Ultralytics at the
import boundary; Optuna, Hydra composition, the shared evaluator, the SQLite
study and the artifact writer are the real code under test. Every assertion
reads a file the command wrote or the value it returned.
"""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest
import yaml

from tests.test_pipeline_local_tracer import (
    _boxes_prediction,
    _empty_prediction,
    _install_fakes,
)

ROOT = Path(__file__).resolve().parents[1]
SEARCH_KEYS = (
    "protocol.learning_rate",
    "protocol.momentum",
    "protocol.weight_decay",
)


def _wrong_class_prediction() -> list[dict]:
    """Correct boxes with shifted categories: localisation yes, class no."""
    predictions = _boxes_prediction()
    for prediction in predictions:
        prediction["category_id"] = (prediction["category_id"] + 1) % 3
    return predictions


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _command(*overrides: str) -> int:
    from polyx.pipeline.command import main

    return main(["--sweep", *overrides])


@pytest.fixture
def work_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Work and scratch roots supplied through the existing layout contract."""
    work = tmp_path / "work"
    scratch = tmp_path / "scratch"
    work.mkdir()
    scratch.mkdir()
    monkeypatch.setenv("POLYX_HPC_WORK", str(work))
    monkeypatch.setenv("POLYX_HPC_SCRATCH", str(scratch))
    return work


def test_sweep_config_declares_maximize_seeded_tpe_and_protocol_only_space(
    tmp_path: Path, tiny_dataset: Path
) -> None:
    """The sweeper node is the one the plugin consumes; no second config exists."""
    from polyx.pipeline.sweep import load_sweep_config

    config = load_sweep_config([f"data_yaml={tiny_dataset}", f"output_dir={tmp_path / 'out'}"])

    assert config.direction == "maximize"
    assert config.sampler_target == "optuna.samplers.TPESampler"
    assert config.sampler_seed == config.execution.protocol.seed
    assert config.n_trials <= 5, "the declared trial count must stay small"
    assert config.n_jobs == 1, "one trial at a time on the single GPU"
    assert config.pruner == "none", "cross-process pruning is not attempted"
    assert set(config.search_space) == set(SEARCH_KEYS)
    for key in config.search_space:
        assert key.startswith("protocol."), key
        assert key.split(".", 1)[1] not in {"split", "seed", "augmentation_subset"}
        assert "evaluator" not in key


def test_small_sweep_writes_study_saves_best_and_uses_shared_objective(
    tmp_path: Path, tiny_dataset: Path, work_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three trials through the command: artifacts, study, score and fixed fields."""
    _install_fakes(
        monkeypatch,
        [_wrong_class_prediction(), _empty_prediction(), _boxes_prediction()],
    )
    output = work_env / "experiments" / "sweep"

    code = _command(
        f"data_yaml={tiny_dataset}",
        f"output_dir={output}",
        "protocol.epochs=1",
        "hydra.sweeper.n_trials=3",
    )

    assert code == 0
    study_path = work_env / "studies" / "polyx-optuna.db"
    assert study_path.is_file(), "the SQLite study must live on the work filesystem"

    # Every trial is a normal training execution with shared-evaluator metrics.
    values: dict[int, float] = {}
    for index in range(3):
        trial_dir = output / "trials" / f"trial_{index:03d}"
        metrics = _read(trial_dir / "metrics.json")
        assert metrics["selection"]["metric"] == "with_class_score"
        assert (
            metrics["selection"]["criterion"] == metrics["figures"]["with_class_score"]["criterion"]
        )
        values[index] = float(metrics["selection"]["value"])
        protocol = yaml.safe_load(
            (trial_dir / "resolved_protocol.yaml").read_text(encoding="utf-8")
        )
        assert protocol["split"] == "val"
        assert protocol["seed"] == 42
        assert protocol["augmentation_subset"] == ["hsv", "flip_lr"]
        assert (trial_dir / "checkpoints" / "best.pt").is_file()

    # The objective is the with-class score, not localisation or mAP: the first
    # trial finds every box but with the wrong polymer and scores zero. The
    # core criterion also keeps misclassified boxes out of localisation's
    # true-positive numerator.
    first = _read(output / "trials" / "trial_000" / "metrics.json")
    assert first["figures"]["localisation_score"]["value"] == pytest.approx(0.0)
    assert first["selection"]["value"] == pytest.approx(0.0)

    best = _read(output / "sweep_best.json")
    winner = max(values, key=values.get)
    assert best["best_value"] == pytest.approx(values[winner])
    assert best["metric"] == "with_class_score"
    assert best["criterion"] == first["selection"]["criterion"]
    winning_config = yaml.safe_load(
        (output / "trials" / f"trial_{winner:03d}" / "config.yaml").read_text(encoding="utf-8")
    )
    assert best["best_params"] == {
        key: winning_config["protocol"][key.split(".", 1)[1]] for key in SEARCH_KEYS
    }
    assert best["best_trial"] == winner
    assert best["total_trials"] == 3
    assert best["trials_run"] == 3
    assert best["reused_study"] is False
    assert best["pruner"] == "none"

    study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    assert all(not trial.intermediate_values for trial in study.trials), "no pruning was attempted"
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert len(study.trials) == 3
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    for key in SEARCH_KEYS:
        assert key in study.best_trial.params


def test_resumed_sweep_reuses_the_existing_study(
    tmp_path: Path, tiny_dataset: Path, work_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second invocation continues the same study instead of replacing it."""
    overrides = (
        f"data_yaml={tiny_dataset}",
        f"output_dir={work_env / 'experiments' / 'resumed'}",
        "protocol.epochs=1",
        "hydra.sweeper.n_trials=2",
    )
    _install_fakes(monkeypatch, [_wrong_class_prediction(), _empty_prediction()])
    assert _command(*overrides) == 0
    study_path = work_env / "studies" / "polyx-optuna.db"
    first_numbers = {
        trial.number
        for trial in optuna.load_study(
            study_name="polyx-optuna", storage=f"sqlite:///{study_path}"
        ).trials
    }

    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    assert _command(*overrides) == 0

    study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    assert {trial.number for trial in study.trials} > first_numbers
    assert len(study.trials) == 4
    best = _read(work_env / "experiments" / "resumed" / "sweep_best.json")
    assert best["reused_study"] is True
    assert best["trials_run"] == 2
    assert best["total_trials"] == 4
    assert best["best_value"] == pytest.approx(study.best_value)
    for index in range(4):
        assert (
            work_env / "experiments" / "resumed" / "trials" / f"trial_{index:03d}" / "metrics.json"
        ).is_file()


def test_glossary_and_adr_ship_with_the_sweep() -> None:
    """The vocabulary and the decision are part of the behaviour, not later."""
    glossary = (ROOT / "GLOSSARY.md").read_text(encoding="utf-8")
    for term in ("Study", "Trial", "Sweeper", "Objective"):
        assert f"## {term}\n" in glossary, f"glossary is missing {term}"

    adr = ROOT / "docs" / "adr" / "0010-optuna-objective.md"
    assert adr.is_file()
    text = adr.read_text(encoding="utf-8")
    assert "# ADR-0010" in text
    assert "with-class score" in text
    assert "no cross-process pruning" in text.lower()
