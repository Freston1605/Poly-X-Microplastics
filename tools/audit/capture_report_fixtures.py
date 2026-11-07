#!/usr/bin/env python3
"""Capture the English baseline reports (benign and hostile).

Builds the two shared RunResult fixtures from tests/report_fixtures.py and
renders each one through polyx.gui.report_html.generate_report exactly as the
app does (all sections, real pipeline, stopwatch wrapper included). The
resulting HTML files are the English reference the per-section report tests
check against; there is no byte-identity gate and no digest file.

The language is forced to English (``POLYX_LOCALE=en``) before any polyx or
fixture module is imported, so the capture does not depend on the machine
locale or the saved preference. This is the same language the tests select.

Outputs (relative to the repo root):
- ``tests/fixtures/baseline_informe/benigna/informe.html``
- ``tests/fixtures/baseline_informe/hostil/informe.html``

Run from the repo root:  python tools/audit/capture_report_fixtures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The baseline is the English render: pin the language BEFORE importing
# tests.report_fixtures or any polyx module, because i18n resolves and caches
# the language at import time.
os.environ["POLYX_LOCALE"] = "en"

# Run as a raw file from tools/audit/, sys.path[0] points there: the repo
# root must go first so `import tests.report_fixtures` and `import polyx`
# resolve exactly as they do under pytest.
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from tests.report_fixtures import estado_benigno, estado_hostil  # noqa: E402

OUTPUTS = {
    "benigna": RAIZ / "tests" / "fixtures" / "baseline_informe" / "benigna",
    "hostil": RAIZ / "tests" / "fixtures" / "baseline_informe" / "hostil",
}


def capture() -> dict[str, Path]:
    """Render both fixture states and persist each report.

    Returns a dict variant -> written path, printed at the end so a capture
    is easy to verify by hand.
    """
    import polyx.gui.report_html as rh
    from polyx.gui.report import machine

    # The report embeds a machine-equipment table (_team) whose live values
    # (OS kernel, RAM, GPU, torch version) differ per runner. The baseline is
    # captured against the PINNED stack (CI: Python 3.11 + matplotlib 3.9.2),
    # so the equipment rows are pinned to canonical values here. The
    # per-section tests read section bodies, not the equipment table, but a
    # stable capture keeps the fixture readable and diffable.
    def patched_team():
        return [
            ("Operating system", "Linux"),
            ("Processor", "pinned"),
            ("RAM", "16.0 GB"),
            ("GPU", "not available"),
            ("PyTorch", "pinned"),
        ]

    machine._team = patched_team

    # Same environmental pins as before: the ultralytics version line in the
    # methods paragraph must not depend on which packages are importable on
    # the capturing machine.
    _ul_stub = type("ULStub", (), {"__version__": "8.3.40"})()
    sys.modules["ultralytics"] = _ul_stub

    states = {
        "benigna": estado_benigno(),
        "hostil": estado_hostil(),
    }
    written: dict[str, Path] = {}
    for variant, state in states.items():
        directory = OUTPUTS[variant]
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "informe.html"
        # Real pipeline: the same sections and signature the section and
        # injection tests use, never a trimmed render.
        rh.generate_report(state, target, sections=list(rh.IDS_SECTIONS))
        written[variant] = target
    return written


def main() -> int:
    """Entry point: capture both variants and report where they landed."""
    written = capture()
    for variant, target in written.items():
        print(f"{variant}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
