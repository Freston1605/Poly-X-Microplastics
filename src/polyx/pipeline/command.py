"""Headless training command with the local backend.

Usage:
    polyx-pipeline data_yaml=... output_dir=... protocol.epochs=2

Overrides are Hydra-style ``key=value`` strings. Later tickets add the SLURM
backend and the Optuna sweeper to the same command without a second seam.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from polyx.pipeline.config import load_config
from polyx.pipeline.execution import run_execution


def main(argv: Sequence[str] | None = None) -> int:
    """Compose one execution, run it locally and report where it landed."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    try:
        config = load_config(overrides)
        summary = run_execution(config)
    except Exception as exc:  # noqa: BLE001 - the command reports any failure as exit 1
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    selection = summary["selection"]
    print(f"OUTPUT: {summary['output_dir']}")
    print(f"CHECKPOINT: {summary['checkpoint']}")
    print(f"SELECTION: {selection['metric']}={selection['value']:.6f}")
    print(f"CRITERION: {selection['criterion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
