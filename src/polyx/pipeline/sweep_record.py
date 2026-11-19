"""The one persisted record for a sweep trial.

Both the in-process driver and the Hydra multirun write the same artifact:
trial id, sampled protocol parameters, execution path and study identity. One
writer keeps the contract, atomicity and parameter meaning from drifting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from polyx.pipeline.artifacts import write_json_atomic


def trial_directory(root: Path, trial_id: int) -> Path:
    """The one execution directory a study trial id owns."""
    return Path(root) / "trials" / f"trial_{int(trial_id):03d}"


def write_trial_record(
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


def completed_score(execution: Path) -> float | None:
    """The score a finished execution recorded, or ``None`` if unavailable."""
    path = Path(execution) / "metrics.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    selection = loaded.get("selection") if isinstance(loaded, dict) else None
    if not isinstance(selection, dict) or "value" not in selection:
        return None
    try:
        return float(selection["value"])
    except (TypeError, ValueError):
        return None


def acquire_trial_lease(execution: Path) -> Path | None:
    """Take an exclusive lease on one trial, or ``None`` when another runner has it.

    Two runners must never recover the same trial: the job would train twice
    and the second ``tell`` would fail. ``O_EXCL`` is the portable one-file
    mutex the work filesystem provides.
    """
    directory = Path(execution)
    locks = directory.parent / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    lease = locks / f"{directory.name}.lock"
    try:
        descriptor = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        # A job killed by the scheduler leaves its lease behind. The lease is
        # only a mutex between live runners, so a dead owner's file is stolen
        # instead of blocking recovery forever.
        if _lease_owner_alive(lease):
            return None
        lease.unlink(missing_ok=True)
        try:
            descriptor = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    return lease


def _lease_owner_alive(lease: Path) -> bool:
    """Whether the process that wrote a lease is still running."""
    try:
        owner = int(lease.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(owner, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # Permission or platform refusal: treat the owner as alive.
        return True
    return True


def release_trial_lease(lease: Path) -> None:
    """Release a trial's recovery lease."""
    Path(lease).unlink(missing_ok=True)
