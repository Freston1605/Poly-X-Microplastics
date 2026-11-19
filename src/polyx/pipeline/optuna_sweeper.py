"""Hydra Optuna sweeper that names the real study trial in every job.

The stock plugin acquires a trial with ``study.ask()`` and launches its job,
but the job configuration never says which trial it is. Reconstructing that
identity from parameters or Hydra's job index is ambiguous after a preemption.
This thin subclass injects the trial's own number as a job override, so the
objective can persist and recover an execution keyed by the study's identity.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

from hydra_plugins.hydra_optuna_sweeper import _impl
from optuna.trial import Trial


class PolyxOptunaSweeper(_impl.OptunaSweeperImpl):  # type: ignore[misc]
    """The pinned Optuna sweeper plus one ``sweep_trial_id`` override."""

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


def install_trial_id_passthrough() -> None:
    """Replace the plugin class Hydra resolves with the trial-id subclass.

    Hydra only loads sweepers from the ``hydra_plugins`` package, so a project
    class cannot be named directly in ``hydra.sweeper._target_``. Replacing the
    loaded class keeps the stock config surface while making the code that
    acquires the trial inject its id into the job.
    """
    _impl.OptunaSweeperImpl = PolyxOptunaSweeper
