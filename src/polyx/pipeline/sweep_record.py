"""The one persisted record for a sweep trial, and the lease that guards it.

Both the in-process driver and the Hydra multirun write the same artifact:
trial id, sampled protocol parameters, execution path and study identity. One
writer keeps the contract, atomicity and parameter meaning from drifting, and
one reader decides whether an execution belongs to the study asking about it.

A lease stops two runners from recovering the same trial. It is created with
its owner's identity already inside it, so a second runner can never observe a
half-written lock and mistake it for a dead owner's. When the owner cannot be
probed -- another node, Windows -- the lease falls back to an age, because a
job killed by the scheduler must not block recovery forever.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from polyx.core.platform import hostname, process_alive
from polyx.pipeline.artifacts import write_json_atomic

#: How long a lease whose owner cannot be probed may block recovery.
DEFAULT_LEASE_TTL_SECONDS = 3600.0


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


def normalize_storage(value: str) -> str:
    """A comparable form of a study's storage URL.

    ``sqlite:///`` and a bare path name the same database; comparing the two
    spellings literally would report one study's trial as another's. Any
    non-file storage (a server URL) is compared as written.
    """
    text = str(value or "").strip()
    prefix = "sqlite:///"
    if text.startswith(prefix):
        return f"{prefix}{Path(text[len(prefix) :]).expanduser().resolve()}"
    return text


def belongs_to(
    record: dict[str, Any] | None,
    *,
    trial_id: int,
    study_name: str,
    storage: str,
) -> bool:
    """Whether a record names this exact study trial.

    A trial number is only unique inside one study *and* one database. Two
    sweep roots can each hold a ``trial_007``; reading the wrong one would
    report one study's score for another study's trial.
    """
    if record is None:
        return False
    try:
        recorded_id = int(record.get("trial_id", -1))
    except (TypeError, ValueError):
        return False
    return (
        recorded_id == int(trial_id)
        and str(record.get("study_name", "")) == study_name
        and normalize_storage(str(record.get("storage", ""))) == normalize_storage(storage)
    )


def finished_score(
    execution: Path,
    *,
    trial_id: int,
    study_name: str,
    storage: str,
) -> float | None:
    """Score of a finished execution that provably belongs to this trial."""
    directory = Path(execution)
    if not belongs_to(
        read_trial_record(directory),
        trial_id=trial_id,
        study_name=study_name,
        storage=storage,
    ):
        return None
    if not (directory / "completion.json").is_file():
        return None
    return completed_score(directory)


def _lease_ttl() -> float:
    """Seconds after which a lease whose owner cannot be probed is taken over."""
    raw = os.environ.get("POLYX_LEASE_TTL", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_LEASE_TTL_SECONDS
    return value if value > 0 else DEFAULT_LEASE_TTL_SECONDS


def acquire_trial_lease(execution: Path, *, kind: str = "") -> Path | None:
    """Take an exclusive lease on one trial, or ``None`` when another runner has it.

    Two runners must never recover the same trial: the job would train twice
    and the second ``tell`` would fail. The lease is created with its owner
    already written -- a temporary file published with a hard link, which fails
    atomically when the lease exists -- so no runner can observe an empty lock
    and conclude its owner is dead. Taking over a stale lease is guarded by a
    second exclusive file, so only one runner can steal it.

    ``kind`` separates the two questions a trial answers: the default lease
    says "a process is training this trial", and ``".recovery"`` says "a runner
    is relaunching it". They must be distinct, because the recovery runner
    holds its lock while the launched job takes the training lease.
    """
    directory = Path(execution)
    locks = directory.parent / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    lease = locks / f"{directory.name}{kind}.lock"
    holder = {"host": hostname(), "pid": os.getpid(), "created": time.time()}
    temporary = locks / f".{directory.name}{kind}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(holder), encoding="utf-8")
    try:
        if _link_lease(temporary, lease):
            return lease
        takeover = locks / f"{directory.name}{kind}.takeover"
        if not _enter_takeover(takeover):
            # Someone else is taking the stale lease over; leave it to them.
            return None
        try:
            if not _stale(lease, directory):
                return None
            lease.unlink(missing_ok=True)
            return lease if _link_lease(temporary, lease) else None
        finally:
            takeover.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)


def _enter_takeover(takeover: Path) -> bool:
    """Enter the one-file critical section that steals a stale lease."""
    for _attempt in range(2):
        try:
            descriptor = os.open(takeover, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if _age(takeover) <= _lease_ttl():
                return False
            # The runner that held the takeover died; reclaim it once.
            takeover.unlink(missing_ok=True)
            continue
        except OSError:
            return False
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        return True
    return False


def _link_lease(temporary: Path, lease: Path) -> bool:
    """Publish the lease file atomically, or report that it already exists."""
    try:
        os.link(temporary, lease)
        return True
    except FileExistsError:
        return False
    except OSError:
        # A filesystem without hard links (or Windows without the privilege).
        # Fall back to exclusive creation and accept the tiny window in which
        # the file exists empty; ``_stale`` refuses to steal a young lock.
        try:
            descriptor = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(temporary.read_text(encoding="utf-8"))
        return True


def _read_owner(lease: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(Path(lease).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _stale(lease: Path, execution: Path) -> bool:
    """Whether the lease's owner is provably gone, or has been silent too long.

    A runner on another node cannot be probed, so the decision falls back to
    activity: the trial's own progress snapshot is written after every epoch,
    which makes it the heartbeat that a killed job stops. Silence for longer
    than ``POLYX_LEASE_TTL`` means no epoch has completed in that time.
    """
    owner = _read_owner(lease)
    this_host = hostname()
    if owner is not None and this_host and str(owner.get("host", "")) == this_host:
        try:
            pid = int(owner.get("pid", 0))
        except (TypeError, ValueError):
            pid = 0
        alive = process_alive(pid)
        if alive is not None:
            return not alive
    return _age(_newest_activity(lease, Path(execution))) > _lease_ttl()


def _newest_activity(lease: Path, execution: Path) -> Path:
    """The most recently written file that proves a runner is still working."""
    newest = lease
    for candidate in (execution / "progress.json", execution / "completion.json"):
        try:
            if candidate.stat().st_mtime > newest.stat().st_mtime:
                newest = candidate
        except OSError:
            continue
    return newest


def _age(lease: Path) -> float:
    try:
        return time.time() - Path(lease).stat().st_mtime
    except OSError:
        return 0.0


def release_trial_lease(lease: Path) -> None:
    """Release a trial's recovery lease."""
    Path(lease).unlink(missing_ok=True)
