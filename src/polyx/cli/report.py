"""Generate an HTML report from a canonical detection run folder.

``--full-report`` renders the paper-quality report through the same
``RunResult`` seam as the GUI; a canonical folder is reopened with ``load_run``.
Pre-canonical folders are not read: they need an older version of Poly-X and
are reported as such instead of falling back to a partial result.

Usage:
    polyx-report --run ./run --output ./run/report.html
    polyx-report --run ./run --full-report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyx.core.i18n import _
from polyx.core.report_simple import LegacyRunError, generate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_("Generate an HTML report from a run folder"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--full-report",
        action="store_true",
        help=_("paper-quality report (requires PySide6, [gui] extra)"),
    )
    return parser


def _write_full_report(run_dir: Path, output: Path | None) -> Path:
    """Render the full report from a canonical run folder only."""
    from polyx.core.detection_run import load_run

    if not (run_dir / "run.json").is_file():
        raise LegacyRunError(run_dir)
    run = load_run(run_dir)

    from polyx.gui.qt import require_qt

    require_qt()
    from polyx.gui.report_html import generate_report

    return generate_report(run, output or (run_dir / "report.html"))


def _reject_old_run(exc: LegacyRunError) -> int:
    """Tell the user plainly that this version cannot open a pre-canonical run."""
    print(_("Cannot open the run: %(error)s") % {"error": exc}, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(args.run)
    output = Path(args.output) if args.output else None

    if args.full_report:
        try:
            path = _write_full_report(run_dir, output)
        except LegacyRunError as exc:
            return _reject_old_run(exc)
        except Exception as exc:  # noqa: BLE001
            print(_("ERROR: %(error)s") % {"error": exc}, file=sys.stderr)
            return 1
        print(_("FULL REPORT: %(path)s") % {"path": path})
        return 0

    try:
        path = generate(run_dir, output)
    except LegacyRunError as exc:
        return _reject_old_run(exc)
    except Exception as exc:  # noqa: BLE001
        print(_("ERROR: %(error)s") % {"error": exc}, file=sys.stderr)
        return 1
    print(_("WROTE: %(path)s") % {"path": path})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
