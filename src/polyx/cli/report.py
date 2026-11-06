"""Generate an HTML report from a detection run folder.

The default is the compact, Qt-free report. ``--full-report`` renders the
paper-quality report through the same ``RunResult`` seam as the GUI: a
canonical folder is reopened with ``load_run`` and a pre-canonical headless
folder through ``load_legacy_run``. ``--images`` is only needed by the legacy
reader when the original photographs no longer sit beside the run.

Usage:
    polyx-report --run ./run --output ./run/report.html
    polyx-report --run ./old-run --images ./photos --full-report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyx.core.report_simple import generate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an HTML report from a run folder")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--images",
        default=None,
        metavar="DIR",
        help="original photographs of a pre-canonical headless run",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="paper-quality report (requires PySide6, [gui] extra)",
    )
    return parser


def _write_full_report(run_dir: Path, output: Path | None, images: Path | None) -> Path:
    """Render the full report from a canonical or a legacy run folder."""
    from polyx.core.detection_run import load_legacy_run, load_run

    if (run_dir / "run.json").is_file():
        run = load_run(run_dir)
    else:
        run = load_legacy_run(run_dir, image_dir=images)

    from polyx.gui.qt import require_qt

    require_qt()
    from polyx.gui.report_html import generate_report

    return generate_report(run, output or (run_dir / "report.html"))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(args.run)
    output = Path(args.output) if args.output else None
    images = Path(args.images) if args.images else None

    if args.full_report:
        try:
            path = _write_full_report(run_dir, output, images)
        except Exception as exc:  # noqa: BLE001
            print(f"Full report unavailable ({exc}); writing compact report.")
            print(f"COMPACT REPORT: {generate(run_dir, output)}")
        else:
            print(f"FULL REPORT: {path}")
        return 0

    try:
        path = generate(run_dir, output)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"COMPACT REPORT: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
