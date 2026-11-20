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

    # The resolved space and the fixed fields travel with the study.
    declared = yaml.safe_load((output / "sweep_config.yaml").read_text(encoding="utf-8"))
    assert declared["direction"] == "maximize"
    assert set(declared["params"]) == {
        "protocol.learning_rate",
        "protocol.momentum",
        "protocol.weight_decay",
    }
    assert declared["fixed"]["split"] == "val"
    assert declared["fixed"]["seed"] == 42
    assert declared["fixed"]["augmentation_subset"] == ["hsv", "flip_lr"]
    assert declared["fixed"]["objective"] == "with_class_score"
    assert declared["storage"] == f"sqlite:///{study_path}"

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
        json.dumps(
            {
                "trial_id": 2,
                "status": "running",
                "study_name": "polyx-optuna",
                "storage": "sqlite:///study.db",
            }
        ),
        encoding="utf-8",
    )
    (trial / "progress.json").write_text("{}", encoding="utf-8")
    assert PolyxOptunaSweeper._pending_records(tmp_path, "polyx-optuna", "sqlite:///study.db") == [
        2
    ]
    # The same number in another database is a different trial's artifact.
    assert PolyxOptunaSweeper._pending_records(tmp_path, "polyx-optuna", "sqlite:///other.db") == []


def _complete_execution(
    directory: Path,
    *,
    trial_id: int,
    study: str,
    value: float,
    storage: str = "sqlite:///unused.db",
) -> None:
    """Write the artifacts the exporter requires of a finished trial."""
    from polyx.pipeline.sweep_record import write_trial_record

    directory.mkdir(parents=True, exist_ok=True)
    write_trial_record(
        trial_id=trial_id,
        parameters={},
        execution=directory,
        study_name=study,
        storage=storage,
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
    assert _find_best_directory(trials, 1, "polyx-optuna", "sqlite:///unused.db") == winner


def test_export_refuses_an_incomplete_or_foreign_winner(tmp_path: Path) -> None:
    """A directory without its promise is missing, not a best result."""
    from polyx.pipeline.sweep_best import _find_best_directory

    trials = tmp_path / "trials"
    directory = trials / "trial_001"
    storage = "sqlite:///unused.db"
    _complete_execution(directory, trial_id=1, study="polyx-optuna", value=1.0)
    assert _find_best_directory(trials, 1, "polyx-optuna", storage) == directory

    (directory / "completion.json").unlink()
    assert _find_best_directory(trials, 1, "polyx-optuna", storage) is None

    _complete_execution(directory, trial_id=1, study="another-study", value=1.0)
    assert _find_best_directory(trials, 1, "polyx-optuna", storage) is None

    _complete_execution(directory, trial_id=2, study="polyx-optuna", value=1.0)
    assert _find_best_directory(trials, 1, "polyx-optuna", storage) is None

    # Same study name and trial number, different database: a different trial.
    _complete_execution(
        directory,
        trial_id=1,
        study="polyx-optuna",
        value=1.0,
        storage="sqlite:///other.db",
    )
    assert _find_best_directory(trials, 1, "polyx-optuna", storage) is None


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


def _write_running_record(directory: Path, trial_id: int, storage: str) -> None:
    from polyx.pipeline.sweep_record import write_trial_record

    directory.mkdir(parents=True, exist_ok=True)
    write_trial_record(
        trial_id=trial_id,
        parameters={},
        execution=directory,
        study_name="polyx-optuna",
        storage=storage,
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
    _write_running_record(directory, interrupted, f"sqlite:///{study_path}")

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
    _write_running_record(directory, interrupted, f"sqlite:///{study_path}")

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
    import json
    import os

    from polyx.core.platform import hostname
    from polyx.pipeline.sweep_record import acquire_trial_lease, release_trial_lease

    execution = tmp_path / "trials" / "trial_001"
    execution.mkdir(parents=True)
    lease = acquire_trial_lease(execution)
    assert lease is not None
    # The owner is written with the lease, so no runner can ever observe an
    # empty lock and conclude that its owner died.
    holder = json.loads(lease.read_text(encoding="utf-8"))
    assert holder["pid"] == os.getpid()
    assert holder["host"] == hostname()
    assert acquire_trial_lease(execution) is None, "a live owner keeps its lease"
    release_trial_lease(lease)

    lease = tmp_path / "trials" / ".locks" / "trial_001.lock"
    lease.write_text(json.dumps({"host": hostname(), "pid": 999999999}), encoding="utf-8")
    stolen = acquire_trial_lease(execution)
    assert stolen is not None, "a dead owner's lease is stolen"
    assert json.loads(stolen.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_a_failed_recovery_fails_the_sweep_instead_of_consuming_the_budget(
    tmp_path: Path,
) -> None:
    """A recovery that cannot finish must not report a successful sweep.

    Before this, a launcher failure told the trial ``FAIL``, counted it as
    recovered and left the multirun with no completed trial while exiting
    normally: the caller saw success and no result.
    """
    from types import SimpleNamespace

    from polyx.pipeline.optuna_sweeper import PolyxOptunaSweeper

    trial = SimpleNamespace(
        number=1, params={"protocol.momentum": 0.9}, state=optuna.trial.TrialState.RUNNING
    )
    told: list[object] = []

    def _tell(number: int, **kwargs: object) -> None:
        told.append((number, kwargs))
        trial.state = optuna.trial.TrialState.FAIL

    study = SimpleNamespace(get_trials=lambda **_kw: [trial], tell=_tell)

    class _Failed:
        @property
        def return_value(self) -> float:
            raise RuntimeError("resume failed")

    sweeper = PolyxOptunaSweeper.__new__(PolyxOptunaSweeper)
    sweeper.config = SimpleNamespace(output_dir=str(tmp_path))
    sweeper.study_name = "polyx-optuna"
    sweeper.storage = "sqlite:///study.db"
    sweeper.search_space_distributions = None
    sweeper.launcher = SimpleNamespace(launch=lambda *_a, **_kw: [_Failed()])
    sweeper.job_idx = 0
    sweeper.n_trials = 1
    sweeper._load_study = lambda: study
    sweeper._process_searchspace_config = lambda: None
    sweeper._parse_sweeper_params_config = list
    sweeper._trial_by_number = lambda _study, number: trial if number == 1 else None
    sweeper._configure_trials = lambda trials, distributions, fixed: [[] for _ in trials]
    (tmp_path / "trials" / "trial_001").mkdir(parents=True)

    from polyx.pipeline.errors import PipelineError

    with pytest.raises(PipelineError, match=r"interrupted trial"):
        sweeper.sweep([])
    assert told and told[0][0] == 1


def test_a_lease_from_another_node_is_judged_by_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runner on another node cannot be probed, so progress is the heartbeat."""
    import json
    import os
    import time

    from polyx.pipeline.sweep_record import acquire_trial_lease, release_trial_lease

    execution = tmp_path / "trials" / "trial_002"
    execution.mkdir(parents=True)
    lease = tmp_path / "trials" / ".locks" / "trial_002.lock"
    lease.parent.mkdir(parents=True, exist_ok=True)
    lease.write_text(
        json.dumps({"host": "another-node", "pid": 1, "created": time.time()}),
        encoding="utf-8",
    )
    # Fresh: an epoch completed here a moment ago, so the owner is working.
    assert acquire_trial_lease(execution) is None

    # Silent for longer than the time limit: the owner is gone.
    monkeypatch.setenv("POLYX_LEASE_TTL", "0.001")
    old = time.time() - 60
    os.utime(lease, (old, old))
    stolen = acquire_trial_lease(execution)
    assert stolen is not None
    assert json.loads(stolen.read_text(encoding="utf-8"))["host"] != "another-node"
    release_trial_lease(stolen)


def _run_multirun_with(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    data_yaml: Path,
    *extra: str,
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
            *extra,
        ],
    )
    sweep_app.objective()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            "protocol.split=choice(val,train)",
            "search space may only declare protocol hyperparameters",
        ),
        (
            "protocol.seed=choice(1,2)",
            "search space may only declare protocol hyperparameters",
        ),
        (
            "protocol.early_stopping.patience=range(1,5)",
            "search space may only declare protocol hyperparameters",
        ),
        (
            "hydra.sweeper.direction=minimize",
            "the sweep direction must be 'maximize'",
        ),
    ],
)
def test_a_sweep_cannot_search_the_fixed_benchmark_fields(
    tmp_path: Path,
    tiny_dataset: Path,
    work_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    override: str,
    message: str,
) -> None:
    """Split, seed and the evaluator stay fixed; the objective is maximized."""
    output = work_env / "experiments" / "refused"
    _install_fakes(monkeypatch, [_boxes_prediction()])

    with pytest.raises(BaseException):  # noqa: B017 - Hydra converts the failure
        _run_multirun_with(monkeypatch, output, tiny_dataset, override)

    captured = capsys.readouterr()
    assert message in captured.err + captured.out
    assert not (output / "trials").exists(), "a refused sweep trained a trial"


def test_the_glossary_and_the_adr_ship_with_the_sweep() -> None:
    """The vocabulary and the decision are part of the behaviour, not later."""
    root = Path(__file__).resolve().parents[1]
    glossary = (root / "GLOSSARY.md").read_text(encoding="utf-8")
    for term in ("Study", "Trial", "Sweeper", "Objective"):
        assert f"## {term}\n" in glossary, f"glossary is missing {term}"

    adr = root / "docs" / "adr" / "0010-optuna-objective.md"
    assert adr.is_file()
    text = adr.read_text(encoding="utf-8")
    assert "# ADR-0010" in text
    assert "with-class score" in text
    assert "no cross-process pruning" in text.lower()
