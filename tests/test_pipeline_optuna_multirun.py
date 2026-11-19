"""The real Hydra Optuna sweeper multirun, through the supported command.

The in-process driver is convenient for a fast local check, but the path the
spec names is the plugin: Hydra launches one job per sampled trial, the study
lives in SQLite on the work filesystem, and each trial writes a normal
execution. These tests drive that path with real Hydra and Optuna; only the
model frameworks are doubles. A post-multirun step folds the plugin's result
into ``sweep_best.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import optuna
import pytest
import yaml

from tests.test_pipeline_local_tracer import _boxes_prediction, _install_fakes


@pytest.fixture
def work_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Work and scratch roots, so the study path is the work layout one."""
    work = tmp_path / "work"
    scratch = tmp_path / "scratch"
    work.mkdir()
    scratch.mkdir()
    monkeypatch.setenv("POLYX_HPC_WORK", str(work))
    monkeypatch.setenv("POLYX_HPC_SCRATCH", str(scratch))
    return work


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_multirun(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    data_yaml: Path,
    *,
    n_trials: int,
) -> None:
    from polyx.pipeline import sweep_app

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_app",
            "--multirun",
            f"output_dir={output}",
            f"data_yaml={data_yaml}",
            "protocol.epochs=1",
            f"hydra.sweeper.n_trials={n_trials}",
        ],
    )
    sweep_app.objective()


def test_the_real_sweeper_writes_a_sqlite_study_and_one_directory_per_trial(
    tmp_path: Path,
    tiny_dataset: Path,
    work_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The path the spec names: plugin trials, SQLite study, separate outputs."""
    output = work_env / "experiments" / "sweep"
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])

    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=2)

    study_path = work_env / "studies" / "polyx-optuna.db"
    assert study_path.is_file(), "the study must live on the work filesystem"
    study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    assert len(study.trials) == 2
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    trial_directories = sorted((output / "trials").glob("trial_*"))
    assert len(trial_directories) == 2, "each trial needs its own execution directory"
    assert len({path.name for path in trial_directories}) == 2
    for directory in trial_directories:
        metrics = _read(directory / "metrics.json")
        assert metrics["selection"]["metric"] == "with_class_score"
        protocol = yaml.safe_load((directory / "resolved_protocol.yaml").read_text())
        assert protocol["split"] == "val"
        assert protocol["augmentation_subset"] == ["hsv", "flip_lr"]

    # The declared space is the protocol-only one in `params` syntax.
    from polyx.pipeline.sweep import load_sweep_config

    config = load_sweep_config([f"data_yaml={tiny_dataset}", f"output_dir={output}"])
    assert config.sampler_target == "optuna.samplers.TPESampler"
    assert set(config.search_space) == {
        "protocol.learning_rate",
        "protocol.momentum",
        "protocol.weight_decay",
    }
    assert config.storage == f"sqlite:///{study_path}"

    # The post-multirun step writes the best artifact next to the sweep.
    from polyx.pipeline import sweep_best

    assert sweep_best.main(["--output-dir", str(output)]) == 0
    best = _read(output / "sweep_best.json")
    assert best["study_name"] == "polyx-optuna"
    assert best["best_value"] == pytest.approx(study.best_value)
    assert best["best_params"] == study.best_trial.params
    assert best["best_trial"] == study.best_trial.number
    assert best["total_trials"] == 2
    assert best["metric"] == "with_class_score"
    assert (
        best["criterion"]
        == _read(output / "trials" / f"trial_{study.best_trial.number:03d}" / "metrics.json")[
            "selection"
        ]["criterion"]
    )
    assert Path(best["best_output_dir"]).is_dir()


def test_a_preempted_trial_does_not_reuse_a_completed_trial_directory(
    tmp_path: Path,
    tiny_dataset: Path,
    work_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RUNNING zombie must not make the resubmitted job write trial_000."""
    output = work_env / "experiments" / "preempted"
    study_path = work_env / "studies" / "polyx-optuna.db"
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=2)
    completed = {
        path.name: (path / "metrics.json").read_bytes()
        for path in (output / "trials").glob("trial_*")
    }

    # A time limit killed a trial after Optuna had stored it as RUNNING.
    study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    study.ask()

    _install_fakes(monkeypatch, [_boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=1)

    for name, payload in completed.items():
        assert (output / "trials" / name / "metrics.json").read_bytes() == payload
    final_study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    last_number = max(trial.number for trial in final_study.trials)
    assert last_number == 3  # trial 2 is the zombie, trial 3 is the new job
    assert (output / "trials" / f"trial_{last_number:03d}" / "metrics.json").is_file()
    index = json.loads((output / "trials" / "index.json").read_text(encoding="utf-8"))
    assert str(last_number) in index


def test_a_resumed_multirun_reuses_the_study_and_never_overwrites_a_trial(
    tmp_path: Path,
    tiny_dataset: Path,
    work_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second invocation adds trials to the same study and new directories."""
    output = work_env / "experiments" / "resumed"
    study_path = work_env / "studies" / "polyx-optuna.db"

    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=2)
    first_metrics = {
        path.name: (path / "metrics.json").read_bytes()
        for path in (output / "trials").glob("trial_*")
    }

    _install_fakes(monkeypatch, [_boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=1)

    study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    assert len(study.trials) == 3, "the second invocation must extend the study"
    directories = {path.name: path for path in (output / "trials").glob("trial_*")}
    assert len(directories) == 3, "the resumed trial must not reuse a directory"
    for name, payload in first_metrics.items():
        assert directories[name].joinpath("metrics.json").read_bytes() == payload

    from polyx.pipeline import sweep_best

    assert sweep_best.main(["--output-dir", str(output)]) == 0
    best = _read(output / "sweep_best.json")
    assert best["total_trials"] == 3
    assert best["best_value"] == pytest.approx(study.best_value)
