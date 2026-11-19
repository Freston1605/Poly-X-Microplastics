"""The agreed artifact layout of one training execution.

Everything a later ticket or a reviewer needs is in one output directory:
resolved configuration, provenance, checkpoints, metrics, report and logs.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from polyx.core.i18n import _


@dataclass(frozen=True)
class ArtifactLayout:
    """Paths of one execution's artifacts, relative to ``root``."""

    root: Path

    def create(self) -> None:
        """Create every directory the execution writes into."""
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def resolved_protocol(self) -> Path:
        return self.root / "resolved_protocol.yaml"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance.json"

    @property
    def training_state(self) -> Path:
        return self.root / "training_state.json"

    @property
    def progress(self) -> Path:
        """Per-epoch snapshot used to resume an interrupted execution."""
        return self.root / "progress.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def training_log(self) -> Path:
        return self.logs_dir / "training.log"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints / "best.pt"

    @property
    def sweep_config(self) -> Path:
        """Resolved sweeper configuration, including the declared space."""
        return self.root / "sweep_config.yaml"

    @property
    def sweep_best(self) -> Path:
        """Best configuration and with-class value of the whole study."""
        return self.root / "sweep_best.json"


def write_yaml(data: dict, path: Path) -> None:
    """Write a YAML artifact with stable key order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(data: dict, path: Path) -> None:
    """Write a formatted JSON artifact."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_atomic(data: dict, path: Path) -> None:
    """Publish a JSON artifact in one rename, never as a half-written file.

    The progress snapshot is the resume contract. A killed process must leave
    either the previous complete snapshot or the new one, never truncated
    JSON that only looks like a valid checkpoint pointer.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def append_log(path: Path, message: str) -> None:
    """Append one plain-text log line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(message.rstrip("\n") + "\n")


def write_report(
    path: Path,
    metrics: dict,
    training_state: dict,
    config: dict,
) -> None:
    """Human-readable summary; every figure keeps the criterion beside it."""
    lines = [
        f"# {_('Training execution report')}",
        "",
        f"- {_('Family')}: {config['family']['name']}",
        f"- {_('Backend')}: {config['backend']['name']}",
        f"- {_('Selection metric')}: {metrics['selection']['metric']}",
        "",
        f"## {_('Figures')}",
        "",
    ]
    for name, figure in metrics["figures"].items():
        lines.append(f"- {name}: {figure['value']:.6f} — {figure['criterion']}")
    if metrics.get("per_class"):
        lines.extend(["", f"## {_('Per class')}", ""])
        for class_name, class_figures in metrics["per_class"].items():
            for figure_name, figure in class_figures.items():
                lines.append(
                    f"- {class_name} / {figure_name}: {figure['value']:.6f} — {figure['criterion']}"
                )
    lines.extend(["", f"## {_('Early stopping')}", ""])
    stopping = training_state["early_stopping"]
    lines.extend(
        f"- {key}: {stopping[key]}"
        for key in ("monitor", "mode", "patience", "min_delta", "best_epoch", "stopped_early")
    )
    lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def artifact_manifest(layout: ArtifactLayout) -> dict[str, Any]:
    """Names of the artifacts that must exist after a successful execution."""
    return {
        "config": str(layout.config),
        "resolved_protocol": str(layout.resolved_protocol),
        "provenance": str(layout.provenance),
        "training_state": str(layout.training_state),
        "progress": str(layout.progress),
        "metrics": str(layout.metrics),
        "report": str(layout.report),
        "training_log": str(layout.training_log),
        "best_checkpoint": str(layout.best_checkpoint),
    }
