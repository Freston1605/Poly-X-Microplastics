"""Minimal self-contained HTML report for canonical headless runs (no Qt).

Not the full paper-quality report (that lives in gui/report/ and needs Qt
state). This produces a compact, dependency-free summary from ``run.json`` and
``detections.csv``. A pre-canonical folder (``summary.json`` +
``centroids.csv``) is refused: old runs stay on disk and need an older version
of Poly-X.
"""

from __future__ import annotations

import html
from pathlib import Path

from polyx.core.detection_run.identity import display_name
from polyx.core.detection_run.schema import (
    is_canonical_run,
    read_csv_rows,
    validate_manifest_version,
)
from polyx.core.i18n import _, language, ngettext


class LegacyRunError(ValueError):
    """A pre-canonical run folder that this version cannot open.

    No reader is provided for old artifacts: the files are left untouched and
    the user is told to use an older Poly-X version instead.
    """

    def __init__(self, run_dir: Path) -> None:
        super().__init__(
            _(
                "Cannot open %(path)s: this is an old run (summary.json + "
                "centroids.csv). Old runs need an older version of Poly-X; "
                "see docs/specs/run-artifact-vocabulary.md."
            )
            % {"path": run_dir}
        )


def _canonical_totals(canonical: dict) -> tuple[int, int, dict[str, int]]:
    """Images, detections and summed TP/FP/FN/miscls across the run's models."""
    metrics = canonical.get("metrics") or {}
    totals = {
        key: sum(int(entry.get(key, 0) or 0) for entry in metrics.values())
        for key in ("tp", "fp", "fn", "miscls")
    }
    values = [entry for entry in metrics.values() if isinstance(entry, dict)]
    detections = sum(int(entry.get("detections", 0) or 0) for entry in values)
    return len(canonical.get("images") or []), detections, totals


def generate(run_dir: Path, output: Path | None = None) -> Path:
    """Render a canonical run folder, or refuse a pre-canonical one."""
    run_dir = Path(run_dir)
    if not is_canonical_run(run_dir):
        raise LegacyRunError(run_dir)

    canonical = validate_manifest_version(run_dir)
    images, detected, metrics = _canonical_totals(canonical)
    rows = read_csv_rows(run_dir / "detections.csv")
    detections = len(rows) if rows else detected

    out = output or (run_dir / "report.html")
    parts = [
        f"<!DOCTYPE html><html lang='{language()}'><head><meta charset='utf-8'>",
        f"<title>{_('MICRA run report')}</title>",
        "<style>body{font-family:sans-serif;margin:2rem;} table{border-collapse:collapse;} "  # noqa: ISC004
        "td,th{border:1px solid #ccc;padding:4px 8px;} img{max-width:320px;}</style>",
        "</head><body>",
    ]
    parts.append(f"<h1>{_('MICRA run report')}</h1>")
    parts.append(f"<p><b>{_('Run dir:')}</b> {html.escape(str(run_dir))}</p>")
    parts.append(
        f"<p><b>{_('Status:')}</b> {html.escape(str(canonical.get('status') or '?'))}"
        f" &nbsp; <b>{_('Schema:')}</b> "
        f"{html.escape(str(canonical.get('schema_version', '?')))}</p>"
    )
    # The canonical document is read raw, so a folder written before
    # ``alias_generated`` existed defaults to a chosen alias (verbatim).
    aliases = [
        display_name(str(model.get("alias", "")), bool(model.get("alias_generated", False)), index)
        for index, model in enumerate(canonical.get("models") or [])
    ]
    if aliases:
        parts.append(f"<p><b>{_('Models:')}</b> {html.escape(', '.join(aliases))}</p>")
    skipped = canonical.get("skipped_images") or []
    if skipped:
        skipped_count = len(skipped)
        label = ngettext("%(count)d skipped image", "%(count)d skipped images", skipped_count) % {
            "count": skipped_count
        }
        parts.append(f"<p><b>{label}</b></p>")
    parts.append(f"<p>{_('Images:')} {images} &nbsp; {_('Detections:')} {detections}</p>")
    if metrics:
        parts.append(
            f"<p>TP={metrics.get('tp')} FP={metrics.get('fp')} "
            f"FN={metrics.get('fn')} miscls={metrics.get('miscls')}</p>"
        )
    parts.append(
        f"<h2>{_('Detections')}</h2><table><tr>"
        + "".join(f"<th>{html.escape(k)}</th>" for k in (rows[0].keys() if rows else []))
        + "</tr>"
    )
    for row in rows[:500]:
        parts.append(  # noqa: PERF401
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row.values()) + "</tr>"
        )
    parts.append("</table>")
    if len(rows) > 500:
        omitted = len(rows) - 500
        label = ngettext(
            "... %(count)d more row omitted", "... %(count)d more rows omitted", omitted
        ) % {"count": omitted}
        parts.append(f"<p>{label}</p>")
    parts.append("</body></html>")
    out.write_text("".join(parts), encoding="utf-8")
    return out


__all__ = ["LegacyRunError", "generate"]
