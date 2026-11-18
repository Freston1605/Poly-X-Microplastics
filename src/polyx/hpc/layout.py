"""Work and scratch layout for a cluster execution.

Datasets, results, checkpoints and the user data location belong on the work
filesystem; temporary files belong on scratch. Every location comes from
environment variables, so no cluster path, username or account is committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClusterLayout:
    """The three roots a job needs, resolved from the submission environment."""

    work: Path
    scratch: Path
    user_data: Path

    @classmethod
    def from_env(cls) -> ClusterLayout:
        """Resolve the roots from ``POLYX_HPC_*`` and ``POLYX_HOME``."""
        work_raw = os.environ.get("POLYX_HPC_WORK")
        if not work_raw:
            raise ValueError("POLYX_HPC_WORK must point at the work filesystem")
        scratch_raw = os.environ.get("POLYX_HPC_SCRATCH")
        if not scratch_raw:
            raise ValueError("POLYX_HPC_SCRATCH must point at the scratch filesystem")
        work = Path(work_raw).expanduser().resolve()
        scratch = Path(scratch_raw).expanduser().resolve()
        home_raw = os.environ.get("POLYX_HOME")
        user_data = Path(home_raw).expanduser().resolve() if home_raw else work / "userdata"
        if not user_data.is_relative_to(work):
            raise ValueError("POLYX_HOME must point inside the work filesystem")
        return cls(work=work, scratch=scratch, user_data=user_data)

    @property
    def data(self) -> Path:
        """Datasets under the user data location on work."""
        return self.user_data / "data"

    @property
    def results(self) -> Path:
        """Final results on work, never on scratch."""
        return self.work / "results"

    @property
    def checkpoints(self) -> Path:
        """Checkpoints on work so a later job can resume them."""
        return self.work / "checkpoints"

    @property
    def tmp(self) -> Path:
        """Temporary files on the scratch filesystem."""
        return self.scratch / "polyx"
