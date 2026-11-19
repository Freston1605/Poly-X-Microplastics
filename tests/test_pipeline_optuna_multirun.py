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


def test_a_zombie_without_recorded_parameters_is_left_untouched(
    tmp_path: Path,
    tiny_dataset: Path,
    work_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trial Optuna never sampled parameters for cannot be relaunched."""
    output = work_env / "experiments" / "preempted"
    study_path = work_env / "studies" / "polyx-optuna.db"
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=2)
    completed = {
        path.name: (path / "metrics.json").read_bytes()
        for path in (output / "trials").glob("trial_*")
    }

    # A time limit killed a trial before it sampled anything: no record, no
    # parameters, no execution directory to continue.
    study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    zombie = int(study.ask().number)

    _install_fakes(monkeypatch, [_boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=1)

    for name, payload in completed.items():
        assert (output / "trials" / name / "metrics.json").read_bytes() == payload
    final_study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    assert final_study.trials[zombie].state == optuna.trial.TrialState.RUNNING
    newest = max(final_study.trials, key=lambda trial: trial.number)
    assert newest.number > zombie
    assert newest.state == optuna.trial.TrialState.COMPLETE


def test_each_trial_records_its_study_id_and_the_study_identity(
    tmp_path: Path,
    tiny_dataset: Path,
    work_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record names the trial and the exact study used for export."""
    output = work_env / "experiments" / "identity"
    _install_fakes(monkeypatch, [_boxes_prediction(), _boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=2)

    study = yaml.safe_load((output / "sweep_config.yaml").read_text(encoding="utf-8"))
    assert study["study_name"] == "polyx-optuna"
    assert study["storage"].endswith("studies/polyx-optuna.db")
    for directory in sorted((output / "trials").glob("trial_*")):
        record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
        assert record["trial_id"] == int(directory.name.removeprefix("trial_"))
        assert record["status"] == "completed"
        assert record["execution"] == str(directory)


def test_a_trial_with_progress_is_reused_by_its_id(tmp_path: Path) -> None:
    """A recorded trial with a checkpoint resumes in the same directory."""
    from polyx.pipeline.sweep_app import _trial_output_dir

    trial = tmp_path / "trials" / "trial_003"
    trial.mkdir(parents=True)
    (trial / "progress.json").write_text("{}", encoding="utf-8")
    assert _trial_output_dir(tmp_path, 3) == trial


def test_a_running_record_with_progress_is_pending_for_recovery(tmp_path: Path) -> None:
    """Resubmission discovers the interrupted trial from its persisted record."""
    from polyx.pipeline.optuna_sweeper import PolyxOptunaSweeper

    trial = tmp_path / "trials" / "trial_002"
    trial.mkdir(parents=True)
    (trial / "record.json").write_text(
        json.dumps({"trial_id": 2, "status": "running"}), encoding="utf-8"
    )
    (trial / "progress.json").write_text("{}", encoding="utf-8")
    assert PolyxOptunaSweeper._pending_records(tmp_path) == [2]


def _complete_execution(directory: Path, *, trial_id: int, study: str, value: float) -> None:
    """Write the artifacts the exporter requires of a finished trial."""
    from polyx.pipeline.sweep_record import write_trial_record

    directory.mkdir(parents=True, exist_ok=True)
    write_trial_record(
        trial_id=trial_id,
        parameters={},
        execution=directory,
        study_name=study,
        storage="sqlite:///unused.db",
        status="completed",
    )
    (directory / "completion.json").write_text('{"status": "completed"}\n', encoding="utf-8")
    (directory / "metrics.json").write_text(
        json.dumps({"selection": {"metric": "with_class_score", "value": value}}),
        encoding="utf-8",
    )
    (directory / "checkpoints").mkdir(exist_ok=True)
    (directory / "checkpoints" / "best.pt").write_bytes(b"weights")


def test_export_resolves_the_winner_by_trial_number_not_parameters(
    tmp_path: Path,
) -> None:
    """Two directories with the same config still resolve by study id."""
    from polyx.pipeline.sweep_best import _find_best_directory

    trials = tmp_path / "trials"
    winner = trials / "trial_001"
    other = trials / "trial_099"
    _complete_execution(winner, trial_id=1, study="polyx-optuna", value=1.0)
    _complete_execution(other, trial_id=99, study="polyx-optuna", value=1.0)
    assert _find_best_directory(trials, 1, "polyx-optuna") == winner


def test_export_refuses_an_incomplete_or_foreign_winner(tmp_path: Path) -> None:
    """A directory without its promise is missing, not a best result."""
    from polyx.pipeline.sweep_best import _find_best_directory

    trials = tmp_path / "trials"
    directory = trials / "trial_001"
    _complete_execution(directory, trial_id=1, study="polyx-optuna", value=1.0)
    assert _find_best_directory(trials, 1, "polyx-optuna") == directory

    (directory / "completion.json").unlink()
    assert _find_best_directory(trials, 1, "polyx-optuna") is None

    _complete_execution(directory, trial_id=1, study="another-study", value=1.0)
    assert _find_best_directory(trials, 1, "polyx-optuna") is None

    _complete_execution(directory, trial_id=2, study="polyx-optuna", value=1.0)
    assert _find_best_directory(trials, 1, "polyx-optuna") is None


def test_export_reports_a_missing_winner_instead_of_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing winner directory fails; it never falls back to parameters."""
    from polyx.pipeline import sweep_best

    root = tmp_path / "sweep-root"
    (root / "sweep").mkdir(parents=True)
    (root / "sweep" / "optimization_results.yaml").write_text(
        "best_value: 1.0\nbest_params: {}\n", encoding="utf-8"
    )
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    import optuna

    study = optuna.create_study(study_name="polyx-optuna", storage=storage, direction="maximize")
    study.ask()
    study.tell(0, 1.0)
    (root / "sweep_config.yaml").write_text(
        f"study_name: polyx-optuna\nstorage: {storage}\n", encoding="utf-8"
    )
    (root / "trials" / "trial_000").mkdir(parents=True)
    assert sweep_best.main(["--output-dir", str(root)]) == 1
    assert "trial 0" in capsys.readouterr().err


def _running_trial(study_path: Path) -> int:
    """Create the RUNNING trial an interrupted job would leave behind."""
    study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    trial = study.ask()
    trial.suggest_categorical("protocol.learning_rate", [0.001, 0.01, 0.05])
    trial.suggest_categorical("protocol.momentum", [0.9, 0.937])
    trial.suggest_categorical("protocol.weight_decay", [0.0001, 0.0005])
    return int(trial.number)


def _write_running_record(directory: Path, trial_id: int) -> None:
    from polyx.pipeline.sweep_record import write_trial_record

    directory.mkdir(parents=True, exist_ok=True)
    write_trial_record(
        trial_id=trial_id,
        parameters={},
        execution=directory,
        study_name="polyx-optuna",
        storage="sqlite:///unused.db",
    )


def test_resubmission_recovers_the_interrupted_trial_itself(
    tmp_path: Path,
    tiny_dataset: Path,
    work_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RUNNING trial is relaunched by its own id, never abandoned."""
    output = work_env / "experiments" / "recovered"
    study_path = work_env / "studies" / "polyx-optuna.db"
    _install_fakes(monkeypatch, [_boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=1)
    first_metrics = (output / "trials" / "trial_000" / "metrics.json").read_bytes()

    interrupted = _running_trial(study_path)
    directory = output / "trials" / f"trial_{interrupted:03d}"
    _write_running_record(directory, interrupted)

    _install_fakes(monkeypatch, [_boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=1)

    study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    assert len(study.trials) == 2, "recovery must not ask for a third trial"
    assert study.trials[interrupted].state == optuna.trial.TrialState.COMPLETE
    record = _read(directory / "record.json")
    assert record["status"] == "completed"
    assert (directory / "completion.json").is_file()
    # The already finished trial is untouched.
    assert (output / "trials" / "trial_000" / "metrics.json").read_bytes() == first_metrics


def test_a_finished_execution_whose_score_was_never_told_is_reconciled(
    tmp_path: Path,
    tiny_dataset: Path,
    work_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The crash window after publication is closed by the execution's own metrics."""
    output = work_env / "experiments" / "reconcile"
    study_path = work_env / "studies" / "polyx-optuna.db"
    _install_fakes(monkeypatch, [_boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=1)
    completed = output / "trials" / "trial_000"

    interrupted = _running_trial(study_path)
    directory = output / "trials" / f"trial_{interrupted:03d}"
    for name in ("metrics.json", "completion.json", "provenance.json", "training_state.json"):
        (directory).mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes((completed / name).read_bytes())
    _write_running_record(directory, interrupted)

    _install_fakes(monkeypatch, [_boxes_prediction()])
    _run_multirun(monkeypatch, output, tiny_dataset, n_trials=1)

    study = optuna.load_study(study_name="polyx-optuna", storage=f"sqlite:///{study_path}")
    assert study.trials[interrupted].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[interrupted].value == pytest.approx(
        _read(directory / "metrics.json")["selection"]["value"]
    )


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


def test_a_dead_runners_lease_is_stolen_and_a_live_one_is_honoured(tmp_path: Path) -> None:
    """A killed job must not block recovery; a live runner must."""
    import os

    from polyx.pipeline.sweep_record import acquire_trial_lease, release_trial_lease

    execution = tmp_path / "trials" / "trial_001"
    execution.mkdir(parents=True)
    lease = acquire_trial_lease(execution)
    assert lease is not None
    assert acquire_trial_lease(execution) is None, "a live owner keeps its lease"
    release_trial_lease(lease)

    lease = tmp_path / "trials" / ".locks" / "trial_001.lock"
    lease.write_text("999999999", encoding="utf-8")
    stolen = acquire_trial_lease(execution)
    assert stolen is not None, "a dead owner's lease is stolen"
    assert stolen.read_text(encoding="utf-8").strip() == str(os.getpid())
