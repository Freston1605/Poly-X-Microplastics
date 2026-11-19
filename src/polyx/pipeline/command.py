"""Headless pipeline command with the local backend.

Usage:
    polyx-pipeline data_yaml=... output_dir=... protocol.epochs=2
    polyx-pipeline coco_dataset=... output_dir=... family=yolo_segmentation
    polyx-pipeline coco_import=... output_dir=... dataset_name=first_batch
    polyx-pipeline --sweep data_yaml=... output_dir=... hydra.sweeper.n_trials=3

Overrides are Hydra-style ``key=value`` strings. ``data_yaml`` trains a
detection family from YOLO box labels; ``coco_dataset`` trains a segmentation
family from the imported canonical COCO mask dataset, converted to the family's
format at job time into scratch. With ``coco_import`` the same command imports
and audits a CVAT COCO export as the canonical segmentation training dataset
instead of running an execution. ``--sweep`` runs the small Optuna driver over
the same config, evaluator and artifact layout; the real search uses the Hydra
sweeper plugin through ``python -m polyx.pipeline.sweep_app --multirun``.
A benchmark campaign across model families is a separate multirun,
``python -m polyx.pipeline.campaign_app --multirun``.

The printed lines are machine tokens consumed by job scripts and logs, not
interface copy, so they are deliberately not wrapped in ``_()``; the
human-facing GUI keeps using the translation catalog.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from polyx.core.provenance.observability import event_error, logger_for
from polyx.pipeline.artifacts import write_json, write_yaml
from polyx.pipeline.coco_import import audit_as_dict, import_coco_dataset, write_import_report
from polyx.pipeline.config import ExecutionConfig, config_to_dict, load_config
from polyx.pipeline.errors import PipelineError
from polyx.pipeline.execution import run_execution


def _import_dataset(config: ExecutionConfig) -> dict[str, Any]:
    """Import one COCO export and write the command-facing evidence."""
    assert config.coco_import is not None
    result = import_coco_dataset(config.coco_import, config.dataset_root, config.dataset_name)
    write_yaml(config_to_dict(config), config.output_dir / "config.yaml")
    write_json(result["manifest"], config.output_dir / "import_manifest.json")
    write_json(audit_as_dict(result["audit"]), config.output_dir / "dataset_audit.json")
    write_import_report(
        config.output_dir / "report.md", result["store"], result["manifest"], result["audit"]
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Compose one execution, import or sweep, run it and report where it landed."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    sweep = "--sweep" in arguments
    overrides = [argument for argument in arguments if argument != "--sweep"]
    try:
        if sweep:
            from polyx.pipeline.sweep import load_sweep_config, run_sweep

            summary = run_sweep(load_sweep_config(overrides))
        else:
            config = load_config(overrides)
            if config.coco_import is not None:
                result = _import_dataset(config)
                manifest = result["manifest"]
                audit = result["audit"]
                print(f"DATASET: {result['store']}")
                print(f"VERSION: {manifest['version']}")
                print(f"CLASSES: {', '.join(manifest['class_distribution'].get('train', {}))}")
                print(f"AUDIT: ok={str(audit.ok).lower()}")
                return 0
            summary = run_execution(config)
    except Exception as exc:  # noqa: BLE001 - the command reports any failure as exit 1
        if isinstance(exc, PipelineError):
            event_error(
                logger_for("pipeline"),
                "pipeline.error",
                exc,
                command="sweep" if sweep else "execution",
                error_code=exc.code,
                error_parameters=exc.parameters,
            )
            print(f"ERROR: {type(exc).__name__}: {exc.display()}", file=sys.stderr)
        else:
            event_error(
                logger_for("pipeline"),
                "pipeline.error",
                exc,
                command="sweep" if sweep else "execution",
            )
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
    localisation = summary.get("metrics", {}).get("figures", {}).get("localisation_score")
    if localisation:
        # Reported beside the selection score, never instead of it.
        print(f"LOCALISATION: {localisation['value']:.6f}")
        print(f"LOCALISATION_CRITERION: {localisation['criterion']}")
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
