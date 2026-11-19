"""The one persisted record for a sweep trial.

Both the in-process driver and the Hydra multirun write the same artifact:
trial id, sampled protocol parameters, execution path and study identity. One
writer keeps the contract, atomicity and parameter meaning from drifting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polyx.pipeline.artifacts import write_json_atomic


def write_trial_record(
    root: Path,
    *,
    trial_id: int,
    parameters: dict[str, Any],
    execution: Path,
    study_name: str,
    storage: str,
    status: str = "running",
) -> Path:
    """Publish ``record.json`` for one trial in one atomic rename."""
    path = Path(execution) / "record.json"
    write_json_atomic(
        {
            "trial_id": int(trial_id),
            "parameters": dict(parameters),
            "execution": str(execution),
            "study_name": study_name,
            "storage": storage,
            "status": status,
        },
        path,
    )
    return path


def read_trial_record(execution: Path) -> dict[str, Any] | None:
    """Read a trial record, or ``None`` when it is absent or unreadable."""
    import json

    path = Path(execution) / "record.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None
