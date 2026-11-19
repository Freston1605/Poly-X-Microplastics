"""Hydra Optuna sweeper that names the real study trial in every job.

The stock plugin acquires a trial with ``study.ask()`` and launches its job,
but the job configuration never says which trial it is. Reconstructing that
identity from parameters or Hydra's job index is ambiguous after a preemption.
This thin subclass injects the trial's own number as a job override, so the
objective persists and recovers an execution keyed by the study's identity.

A trial killed by a time limit stays ``RUNNING`` in Optuna. Before asking for
new trials, ``sweep`` relaunches those trials from their persisted record: a
trial with a progress snapshot continues its checkpoint, one that never wrote
a checkpoint retries the same parameters, and one whose execution finished
but whose score never reached Optuna is told from its own metrics.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

import optuna
from hydra_plugins.hydra_optuna_sweeper import _impl
from omegaconf import OmegaConf
from optuna.trial import Trial

from polyx.pipeline.sweep_record import (
    acquire_trial_lease,
    completed_score,
    read_trial_record,
    release_trial_lease,
    trial_directory,
)


class PolyxOptunaSweeper(_impl.OptunaSweeperImpl):  # type: ignore[misc]
    """The pinned Optuna sweeper plus trial-id injection and recovery."""

    def _configure_trials(
        self,
        trials: list[Trial],
        search_space_distributions: MutableMapping[str, Any],
        fixed_params: MutableMapping[str, Any],
    ) -> Sequence[Sequence[str]]:
        overrides = super()._configure_trials(trials, search_space_distributions, fixed_params)
        return [
            (*override, f"sweep_trial_id={trial.number}")
            for trial, override in zip(trials, overrides, strict=True)
        ]

    def sweep(self, arguments: list[str]) -> None:
        """Recover interrupted trials, then run the remaining configured trials.

        Optuna leaves a killed trial ``RUNNING``. Asking for a new trial would
        abandon its checkpoint and start a different parameter set. The pending
        records written before training name the old trial; this relaunches
        those exact ids first and only then asks the plugin for new trials.
        """
        recovered = self._recover_pending(arguments)
        self.n_trials = max(0, int(self.n_trials) - recovered)
        if self.n_trials == 0:
            self._write_results()
            return
        super().sweep(arguments)

    def _recover_pending(self, arguments: list[str]) -> int:
        """Relaunch or reconcile every interrupted trial of this study."""
        root = Path(str(self.config.output_dir))
        study = self._load_study()
        pending = self._pending_records(root)
        running = [
            int(trial.number)
            for trial in study.get_trials(deepcopy=False)
            if trial.state == optuna.trial.TrialState.RUNNING
        ]
        candidates = sorted({*pending, *running})
        if not candidates:
            return 0
        self._process_searchspace_config()
        params_conf = self._parse_sweeper_params_config()
        params_conf.extend(arguments)
        distributions, fixed = _impl.create_params_from_overrides(params_conf)
        if self.search_space_distributions:
            distributions = {**self.search_space_distributions, **distributions}
        for name in fixed:
            distributions.pop(name, None)
        recovered = 0
        for trial_id in candidates:
            trial = self._trial_by_number(study, trial_id)
            if trial is None or trial.state != optuna.trial.TrialState.RUNNING:
                continue
            directory = trial_directory(root, trial_id)
            lease = acquire_trial_lease(directory)
            if lease is None:
                # Another runner already recovered this trial.
                continue
            try:
                if _reconcile_finished(study, trial, directory):
                    recovered += 1
                    continue
                if not trial.params:
                    continue
                overrides = self._configure_trials([trial], distributions, fixed)
                returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
                self.job_idx += len(returns)
                recovered += 1
                for result in returns:
                    try:
                        value = float(result.return_value)
                    except Exception:  # noqa: BLE001 - a failed recovery stays a failure
                        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                    else:
                        study.tell(trial.number, value)
            finally:
                release_trial_lease(lease)
        return recovered

    def _load_study(self) -> Any:
        """The study this multirun owns, created on first use.

        The plugin's own ``setup`` already resolved the storage, name and
        directions from the composed config, so recovery reads them from the
        sweeper instance instead of asking Hydra for its global config again.
        """
        return optuna.create_study(
            study_name=str(self.study_name or "polyx-optuna"),
            storage=str(self.storage or ""),
            directions=self._get_directions(),
            load_if_exists=True,
        )

    @staticmethod
    def _trial_by_number(study: Any, trial_id: int) -> Any | None:
        """One stored trial by its number, or ``None`` when it does not exist."""
        trials = study.get_trials(deepcopy=False)
        return trials[trial_id] if 0 <= trial_id < len(trials) else None

    @staticmethod
    def _pending_records(root: Path) -> list[int]:
        """Trial ids whose record says the execution never completed."""
        trials = root / "trials"
        if not trials.is_dir():
            return []
        ids: list[int] = []
        for directory in sorted(trials.glob("trial_*")):
            record = read_trial_record(directory)
            if record is None or record.get("status") != "running":
                continue
            try:
                ids.append(int(record["trial_id"]))
            except (KeyError, TypeError, ValueError):
                continue
        return ids

    def _write_results(self) -> None:
        """Write the plugin's best-artifact when a run was only recovery."""
        study = self._load_study()
        completed = [
            trial
            for trial in study.get_trials(deepcopy=False)
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            # Nothing to report: a failed recovery leaves the study without a
            # complete trial, and inventing a best artifact would hide it.
            return
        best = study.best_trial
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.create(
                {
                    "name": "optuna",
                    "best_params": best.params,
                    "best_value": best.value,
                }
            ),
            sweep_dir / "optimization_results.yaml",
        )


def _reconcile_finished(study: Any, trial: Any, directory: Path) -> bool:
    """Tell a score the finished execution wrote but Optuna never received."""
    if not (directory / "completion.json").is_file():
        return False
    score = completed_score(directory)
    if score is None:
        return False
    study.tell(trial.number, score)
    return True


def install_trial_id_passthrough() -> None:
    """Replace the plugin class Hydra resolves with the trial-id subclass.

    Hydra only loads sweepers from the ``hydra_plugins`` package, so a project
    class cannot be named directly in ``hydra.sweeper._target_``. Replacing the
    loaded class keeps the stock config surface while making the code that
    acquires the trial inject its id into the job.
    """
    _impl.OptunaSweeperImpl = PolyxOptunaSweeper
