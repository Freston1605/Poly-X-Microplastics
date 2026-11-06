"""Minimal self-contained HTML report for headless Poly-X runs (no Qt).

Not the full paper-quality report (that lives in gui/report/ and needs Qt
state). This produces a compact, dependency-free summary. It reads the
canonical ``run.json`` + ``detections.csv`` first, and falls back to the old
headless ``summary.json`` + ``centroids.csv`` so pre-canonical runs stay
readable without migration.
"""

from __future__ import annotations

import html
from pathlib import Path

from polyx.core.detection_run.schema import read_csv_rows, read_json_object


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
    run_dir = Path(run_dir)
    canonical = read_json_object(run_dir / "run.json")
    rows: list[dict] = []
    summary: dict = {}
    metrics: dict = {}
    images: object = "?"
    detections: object = "?"

    if canonical:
        images, detected, metrics = _canonical_totals(canonical)
        rows = read_csv_rows(run_dir / "detections.csv")
        detections = len(rows) if rows else detected
    else:
        summary = read_json_object(run_dir / "summary.json")
        rows = read_csv_rows(run_dir / "centroids.csv")
        images = summary.get("images", "?")
        detections = summary.get("detections", len(rows))
        metrics = summary.get("metrics") or {}

    out = output or (run_dir / "report.html")
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Poly-X run report</title>",
        "<style>body{font-family:sans-serif;margin:2rem;} table{border-collapse:collapse;} "  # noqa: ISC004
        "td,th{border:1px solid #ccc;padding:4px 8px;} img{max-width:320px;}</style>",
        "</head><body>",
    ]
    parts.append("<h1>Poly-X run report</h1>")
    parts.append(f"<p><b>Run dir:</b> {html.escape(str(run_dir))}</p>")
    if canonical:
        parts.append(
            f"<p><b>Status:</b> {html.escape(str(canonical.get('status') or '?'))}"
            f" &nbsp; <b>Schema:</b> {html.escape(str(canonical.get('schema_version', '?')))}</p>"
        )
        aliases = [str(model.get("alias", "")) for model in canonical.get("models") or []]
        if aliases:
            parts.append(f"<p><b>Models:</b> {html.escape(', '.join(aliases))}</p>")
        skipped = canonical.get("skipped_images") or []
        if skipped:
            parts.append(f"<p><b>Skipped images:</b> {len(skipped)}</p>")
    parts.append(f"<p>Images: {images} &nbsp; Detections: {detections}</p>")
    if metrics:
        parts.append(
            f"<p>TP={metrics.get('tp')} FP={metrics.get('fp')} "
            f"FN={metrics.get('fn')} miscls={metrics.get('miscls')}</p>"
        )
    parts.append(
        "<h2>Detections</h2><table><tr>"
        + "".join(f"<th>{html.escape(k)}</th>" for k in (rows[0].keys() if rows else []))
        + "</tr>"
    )
    for r in rows[:500]:
        parts.append(  # noqa: PERF401
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>"
        )
    parts.append("</table>")
    if len(rows) > 500:
        parts.append(f"<p>... {len(rows) - 500} more rows omitted</p>")
    parts.append("</body></html>")
    out.write_text("".join(parts), encoding="utf-8")
    return out


__all__ = ["generate"]
