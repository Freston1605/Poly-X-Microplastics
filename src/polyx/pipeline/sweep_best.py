"""Write the best-artifact of a real Optuna multirun.

The Hydra Optuna sweeper writes ``optimization_results.yaml`` inside its sweep
directory and leaves each trial's normal execution under
``<output_dir>/trials/trial_<number>``. This small post-multirun step folds the
two into the same ``sweep_best.json`` the one-process driver writes, so a
campaign or a reviewer finds the winning configuration and its criterion next to
the study and the per-trial artifacts.

Usage:
    python -m polyx.pipeline.sweep_best --output-dir <sweep root>
    python -m polyx.pipeline.sweep_best --output-dir <sweep root> \
        --storage sqlite:///.../polyx-optuna.db --study-name polyx-optuna
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from polyx.core.i18n import _
from polyx.pipeline.artifacts import write_json


def main(argv: list[str] | None = None) -> int:
    """Read the finished multirun, write the best artifact and report it."""
    parser = argparse.ArgumentParser(
        description=_("Write sweep_best.json from a finished Optuna multirun")
    )
    parser.add_argument("--output-dir", required=True, help=_("the sweep root"))
    parser.add_argument(
        "--storage",
        default="",
        help=_("study storage URL or SQLite path (default: the work layout path)"),
    )
    parser.add_argument("--study-name", default="polyx-optuna")
    args = parser.parse_args(argv)

    root = Path(args.output_dir).expanduser().resolve()
    results_path = root / "sweep" / "optimization_results.yaml"
    if not results_path.is_file():
        print(f"ERROR: no Optuna results at {results_path}", file=sys.stderr)
        return 1
    results = yaml.safe_load(results_path.read_text(encoding="utf-8")) or {}
    if "best_value" not in results:
        print(f"ERROR: {results_path} has no best_value", file=sys.stderr)
        return 1

    storage = _storage(args.storage, args.study_name, root)
    study = _load_study(args.study_name, storage)
    best_params = dict(results.get("best_params") or {})
    best_number = study.best_trial.number if study is not None else None
    best_directory = _find_best_directory(root / "trials", best_params, best_number)
    criterion = _criterion(best_directory)
    artifact: dict[str, Any] = {
        "study_name": args.study_name,
        "storage": storage,
        "direction": str(study.direction.name.lower()) if study is not None else "maximize",
        "metric": "with_class_score",
        "criterion": criterion,
        "best_trial": int(study.best_trial.number) if study is not None else -1,
        "best_value": float(results["best_value"]),
        "best_params": best_params,
        "total_trials": len(study.trials) if study is not None else 0,
        "pruner": "none",
    }
    artifact["best_output_dir"] = str(best_directory) if best_directory is not None else ""
    write_json(artifact, root / "sweep_best.json")

    print(f"STUDY: {args.study_name}")
    print(f"STORAGE: {storage}")
    print(f"BEST_VALUE: {artifact['best_value']:.6f}")
    print(f"BEST_PARAMS: {best_params}")
    print(f"ARTIFACT: {root / 'sweep_best.json'}")
    return 0


def _storage(configured: str, study_name: str, root: Path) -> str:
    """The study URL, defaulting to the same work-layout path the sweep uses."""
    if configured:
        return configured if configured.startswith("sqlite:///") else f"sqlite:///{configured}"
    work = os.environ.get("POLYX_HPC_WORK")
    base = Path(work).expanduser() if work else root
    return f"sqlite:///{base / 'studies' / f'{study_name}.db'}"


def _load_study(study_name: str, storage: str) -> Any:
    """The finished study, or ``None`` if Optuna cannot read it."""
    import optuna

    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except Exception:  # noqa: BLE001 - the artifact is still worth writing
        return None


def _find_best_directory(
    trials: Path, best_params: dict[str, Any], best_number: int | None = None
) -> Path | None:
    """The winning trial's execution directory, by trial id or by parameters."""
    if not trials.is_dir():
        return None
    index_path = trials / "index.json"
    if best_number is not None and index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            names = index.get(str(best_number)) or []
            if names:
                candidate = trials / str(names[-1])
                if candidate.is_dir():
                    return candidate
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    if not best_params:
        return None
    for directory in sorted(trials.glob("trial_*")):
        config_path = directory / "config.yaml"
        if not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        protocol = config.get("protocol") or {}
        if all(
            _matches(protocol.get(key.split(".", 1)[1]), value)
            for key, value in best_params.items()
        ):
            return directory
    return None


def _matches(observed: Any, expected: Any) -> bool:
    """Trial parameters round-trip through YAML; compare numbers by value."""
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    return observed == expected


def _criterion(directory: Path | None) -> str:
    """The selection criterion from the winning trial's metrics, when available."""
    if directory is None:
        return ""
    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        return ""
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(metrics.get("selection", {}).get("criterion", ""))


if __name__ == "__main__":
    raise SystemExit(main())
