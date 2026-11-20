"""Headless pipeline command with the local backend.

Usage:
    polyx-pipeline data_yaml=... output_dir=... protocol.epochs=2
    polyx-pipeline coco_dataset=... output_dir=... family=yolo_segmentation
    polyx-pipeline coco_import=... output_dir=... dataset_name=first_batch

Overrides are Hydra-style ``key=value`` strings. ``data_yaml`` trains a
detection family from YOLO box labels; ``coco_dataset`` trains a segmentation
family from the imported canonical COCO mask dataset, converted to the family's
format at job time into scratch. With ``coco_import`` the same command imports
and audits a CVAT COCO export as the canonical segmentation training dataset
instead of running an execution.

Hyperparameter search is the Hydra multirun and only the Hydra multirun:
``python -m polyx.pipeline.sweep_app --multirun`` runs the Optuna sweeper
plugin, and ``python -m polyx.pipeline.sweep_best --output-dir ...`` folds its
result into ``sweep_best.json``. A benchmark campaign across model families is
the other multirun, ``python -m polyx.pipeline.campaign_app --multirun``. This
command does not carry a second sweep loop.

The printed lines are machine tokens consumed by job scripts and logs, not
interface copy, so they are deliberately not wrapped in ``_()``; the
human-facing GUI keeps using the translation catalog.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from polyx.pipeline.artifacts import write_json, write_yaml
from polyx.pipeline.coco_import import audit_as_dict, import_coco_dataset, write_import_report
from polyx.pipeline.config import ExecutionConfig, config_to_dict, load_config
from polyx.pipeline.errors import report_failure
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
    """Compose one execution or import, run it and report where it landed."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    try:
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
        report_failure(exc, stage="execution")
        return 1

    selection = summary["selection"]
    print(f"OUTPUT: {summary['output_dir']}")
    print(f"CHECKPOINT: {summary['checkpoint']}")
    print(f"SELECTION: {selection['metric']}={selection['value']:.6f}")
    if selection.get("criterion"):
        print(f"CRITERION: {selection['criterion']}")
    localisation = summary.get("metrics", {}).get("figures", {}).get("localisation_score")
    if localisation:
        # Reported beside the selection score, never instead of it.
        print(f"LOCALISATION: {localisation['value']:.6f}")
        print(f"LOCALISATION_CRITERION: {localisation['criterion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
