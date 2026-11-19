"""Hydra Optuna sweeper that names the real study trial in every job.

The stock plugin acquires a trial with ``study.ask()`` and launches its job,
but the job configuration never says which trial it is. Reconstructing that
identity from parameters or Hydra's job index is ambiguous after a preemption.
This thin subclass injects the trial's own number as a job override, so the
objective can persist and recover an execution keyed by the study's identity.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

import optuna
from hydra_plugins.hydra_optuna_sweeper import _impl
from omegaconf import OmegaConf
from optuna.trial import Trial


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
        """Relaunch pending trials that still have a checkpoint and a record."""
        root = Path(str(self.config.output_dir))
        pending = self._pending_records(root)
        if not pending:
            return 0
        self._process_searchspace_config()
        params_conf = self._parse_sweeper_params_config()
        params_conf.extend(arguments)
        distributions, fixed = _impl.create_params_from_overrides(params_conf)
        if self.search_space_distributions:
            distributions = {**self.search_space_distributions, **distributions}
        for name in fixed:
            distributions.pop(name, None)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            directions=self._get_directions(),
            load_if_exists=True,
        )
        recovered = 0
        for trial_id in pending:
            if trial_id >= len(study.trials):
                continue
            trial = study.trials[trial_id]
            if trial.state != optuna.trial.TrialState.RUNNING or not trial.params:
                continue
            overrides = self._configure_trials([trial], distributions, fixed)
            returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
            self.job_idx += len(returns)
            recovered += 1
            for result in returns:
                try:
                    value = float(result.return_value)
                except Exception:  # noqa: BLE001 - a failed recovery stays a failure
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                else:
                    study.tell(trial, value)
        return recovered

    @staticmethod
    def _pending_records(root: Path) -> list[int]:
        """Trial ids whose record is running and whose progress survived."""
        trials = root / "trials"
        if not trials.is_dir():
            return []
        ids: list[int] = []
        for directory in sorted(trials.glob("trial_*")):
            record = directory / "record.json"
            if not record.is_file() or not (directory / "progress.json").is_file():
                continue
            if (directory / "completion.json").is_file():
                continue
            try:
                loaded = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(loaded, dict) and loaded.get("status") == "running":
                ids.append(int(loaded["trial_id"]))
        return ids

    def _write_results(self) -> None:
        """Write the plugin's best-artifact when a run was only recovery."""
        study = optuna.load_study(study_name=self.study_name, storage=self.storage)
        best = study.best_trial
        OmegaConf.save(
            OmegaConf.create(
                {
                    "name": "optuna",
                    "best_params": best.params,
                    "best_value": best.value,
                }
            ),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )


def install_trial_id_passthrough() -> None:
    """Replace the plugin class Hydra resolves with the trial-id subclass.

    Hydra only loads sweepers from the ``hydra_plugins`` package, so a project
    class cannot be named directly in ``hydra.sweeper._target_``. Replacing the
    loaded class keeps the stock config surface while making the code that
    acquires the trial inject its id into the job.
    """
    _impl.OptunaSweeperImpl = PolyxOptunaSweeper
