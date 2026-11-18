"""Runnable pre-flight check for the V100 training environment.

Usage::

    python -m polyx.hpc.check_env

It prints what it actually observed and exits non-zero when anything is
missing. With no GPU it reports CUDA as unavailable and the AMP pass as not
run; it never substitutes a cached or synthetic pass. Run it on the GPU node
as the first smoke job (ticket #199 verifies that step on OCEANO).
"""

from __future__ import annotations

import sys
from typing import Any

from polyx.hpc.environment import EnvironmentReport, probe


def report_environment(report: EnvironmentReport) -> None:
    """Print one line per observation, then the result."""
    for line in report.lines():
        print(line)


def main(argv: list[str] | None = None, torch_module: Any | None = None) -> int:
    """Entry point; ``torch_module`` is the offline test seam."""
    if argv:
        # No options are needed for the check itself; fail loudly on a typo
        # rather than silently ignoring an argument someone meant to pass.
        print(f"unknown argument: {argv[0]}", file=sys.stderr)
        return 2
    report = probe(torch_module)
    report_environment(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
