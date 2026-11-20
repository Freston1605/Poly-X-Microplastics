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
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from polyx.core.i18n import _
from polyx.pipeline.artifacts import write_json
from polyx.pipeline.sweep_record import (
    belongs_to,
    completed_score,
    read_trial_record,
    trial_directory,
)


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
    parser.add_argument("--study-name", default=None)
    args = parser.parse_args(argv)

    root = Path(args.output_dir).expanduser().resolve()
    results_path = root / "sweep" / "optimization_results.yaml"
    if not results_path.is_file():
        print(
            "ERROR: " + _("no Optuna results at %(path)s") % {"path": results_path},
            file=sys.stderr,
        )
        return 1
    results = yaml.safe_load(results_path.read_text(encoding="utf-8")) or {}
    if "best_value" not in results:
        print(
            "ERROR: " + _("%(path)s has no best_value") % {"path": results_path},
            file=sys.stderr,
        )
        return 1

    record = _read_study_record(root)
    study_name = args.study_name or record.get("study_name") or "polyx-optuna"
    # One spelling of the study's location: a bare path and its ``sqlite:///``
    # URL name the same database, and the winner check matches on it.
    storage = _storage(args.storage or record.get("storage") or "", study_name, root)
    study = _load_study(study_name, storage)
    if study is None:
        print(
            "ERROR: "
            + _("cannot load study %(study)s at %(storage)s")
            % {"study": study_name, "storage": storage},
            file=sys.stderr,
        )
        return 1
    best_number = int(study.best_trial.number)
    best_directory = _find_best_directory(root / "trials", best_number, study_name, storage)
    if best_directory is None:
        print(
            "ERROR: "
            + _("the winner is study trial %(trial)d but its artifact is missing under %(root)s")
            % {"trial": best_number, "root": root / "trials"},
            file=sys.stderr,
        )
        return 1
    best_params = dict(study.best_trial.params)
    criterion = _criterion(best_directory)
    artifact: dict[str, Any] = {
        "study_name": study_name,
        "storage": storage,
        "direction": str(study.direction.name.lower()),
        "metric": "with_class_score",
        "criterion": criterion,
        "best_trial": best_number,
        "best_value": float(study.best_trial.value),
        "best_params": best_params,
        "total_trials": len(study.trials),
        "pruner": "none",
        "best_output_dir": str(best_directory),
    }
    write_json(artifact, root / "sweep_best.json")

    print(f"STUDY: {study_name}")
    print(f"STORAGE: {storage}")
    print(f"BEST_VALUE: {artifact['best_value']:.6f}")
    print(f"BEST_PARAMS: {best_params}")
    print(f"ARTIFACT: {root / 'sweep_best.json'}")
    return 0


def _read_study_record(root: Path) -> dict[str, Any]:
    """The study identity the sweep persisted before its first trial."""
    path = root / "sweep_config.yaml"
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
    trials: Path, best_number: int, study_name: str, storage: str
) -> Path | None:
    """The winner's complete execution directory, strictly by study trial number.

    A directory is the winner only when its record names the winning study
    trial *in the same database*, and it carries every artifact the sweep
    promises: the completion marker, readable metrics and the kept checkpoint.
    A half-written directory, or one left by a different study that happens to
    use the same trial number, is reported as missing instead of being
    exported as a best result.
    """
    if not trials.is_dir():
        return None
    candidate = trial_directory(trials.parent, best_number)
    if not belongs_to(
        read_trial_record(candidate),
        trial_id=best_number,
        study_name=study_name,
        storage=storage,
    ):
        return None
    if not (candidate / "completion.json").is_file():
        return None
    if completed_score(candidate) is None:
        return None
    if not (candidate / "checkpoints" / "best.pt").is_file():
        return None
    return candidate


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
