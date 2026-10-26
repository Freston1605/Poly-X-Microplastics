"""Minimal self-contained HTML report for headless Poly-X runs (no Qt).

Not the full paper-quality report (that lives in gui/report_html.py and
needs Qt state). This produces a compact, dependency-free summary from
summary.json + centroids.csv so headless runs are self-documenting.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path


def generate(run_dir: Path, output: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    summary = {}
    sp = run_dir / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
    rows = []
    cp = run_dir / "centroids.csv"
    if cp.exists():
        with open(cp, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    out = output or (run_dir / "report.html")
    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
             "<title>Poly-X run report</title>",
             "<style>body{font-family:sans-serif;margin:2rem;} table{border-collapse:collapse;} "
             "td,th{border:1px solid #ccc;padding:4px 8px;} img{max-width:320px;}</style>",
             "</head><body>"]
    parts.append("<h1>Poly-X run report</h1>")
    parts.append(f"<p><b>Run dir:</b> {html.escape(str(run_dir))}</p>")
    parts.append(f"<p>Images: {summary.get('images', '?')} &nbsp; Detections: {summary.get('detections', len(rows))}</p>")
    if summary.get("metrics"):
        parts.append(f"<p>TP={summary['metrics'].get('tp')} FP={summary['metrics'].get('fp')} "
                     f"FN={summary['metrics'].get('fn')} miscls={summary['metrics'].get('miscls')}</p>")
    parts.append("<h2>Detections</h2><table><tr>" +
                 "".join(f"<th>{html.escape(k)}</th>" for k in (rows[0].keys() if rows else [])) +
                 "</tr>")
    for r in rows[:500]:
        parts.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>")
    parts.append("</table>")
    if len(rows) > 500:
        parts.append(f"<p>... {len(rows)-500} more rows omitted</p>")
    parts.append("</body></html>")
    out.write_text("".join(parts), encoding="utf-8")
    return out


__all__ = ["generate"]