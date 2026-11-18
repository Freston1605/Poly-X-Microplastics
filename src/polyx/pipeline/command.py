"""Headless training command with the local backend.

Usage:
    polyx-pipeline data_yaml=... output_dir=... protocol.epochs=2
    polyx-pipeline --sweep data_yaml=... output_dir=... hydra.sweeper.n_trials=3

Overrides are Hydra-style ``key=value`` strings. ``--sweep`` runs the small
Optuna driver over the same config, evaluator and artifact layout; the real
campaign uses the Hydra sweeper plugin through ``polyx-sweep --multirun``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from polyx.pipeline.config import load_config
from polyx.pipeline.execution import run_execution


def main(argv: Sequence[str] | None = None) -> int:
    """Compose one execution (or one sweep), run it and report where it landed."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    sweep = "--sweep" in arguments
    overrides = [argument for argument in arguments if argument != "--sweep"]
    try:
        if sweep:
            from polyx.pipeline.sweep import load_sweep_config, run_sweep

            summary = run_sweep(load_sweep_config(overrides))
        else:
            summary = run_execution(load_config(overrides))
    except Exception as exc:  # noqa: BLE001 - the command reports any failure as exit 1
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    selection = summary["selection"]
    print(f"OUTPUT: {summary['output_dir']}")
    if sweep:
        print(f"STUDY: {summary['study_name']}")
    else:
        print(f"CHECKPOINT: {summary['checkpoint']}")
    print(f"SELECTION: {selection['metric']}={selection['value']:.6f}")
    if selection.get("criterion"):
        print(f"CRITERION: {selection['criterion']}")
    if sweep:
        print(f"TRIALS_RUN: {summary['trials_run']}")
        print(f"TOTAL_TRIALS: {summary['total_trials']}")
        print(f"REUSED_STUDY: {str(summary['reused_study']).lower()}")
        print(f"BEST_TRIAL: {summary['best_trial']}")
        print(f"BEST_VALUE: {summary['best_value']:.6f}")
        print(f"BEST_PARAMS: {summary['best_params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
