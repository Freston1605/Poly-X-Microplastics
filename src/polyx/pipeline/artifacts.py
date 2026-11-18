"""The agreed artifact layout of one training execution.

Everything a later ticket or a reviewer needs is in one output directory:
resolved configuration, provenance, checkpoints, metrics, report and logs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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


def write_yaml(data: dict, path: Path) -> None:
    """Write a YAML artifact with stable key order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(data: dict, path: Path) -> None:
    """Write a formatted JSON artifact."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        "# Training execution report",
        "",
        f"- Family: {config['family']['name']}",
        f"- Backend: {config['backend']['name']}",
        f"- Selection metric: {metrics['selection']['metric']}",
        "",
        "## Figures",
        "",
    ]
    for name, figure in metrics["figures"].items():
        lines.append(f"- {name}: {figure['value']:.6f} — {figure['criterion']}")
    lines.extend(["", "## Early stopping", ""])
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
        "metrics": str(layout.metrics),
        "report": str(layout.report),
        "training_log": str(layout.training_log),
        "best_checkpoint": str(layout.best_checkpoint),
    }
