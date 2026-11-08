"""``only_images`` narrows every report surface to the same subset (#84).

The filter is applied once, before the batch precompute, so the totals, the
aggregate figures and the per-image tables all describe the subset being
shown. The end-to-end proof is that narrowing a two-image run to one image
renders exactly the same document as a run that only ever had that image.
"""

from __future__ import annotations

import re
from pathlib import Path

from polyx.core.detect.protocol import Instance
from polyx.core.detection_run import ImageResult, ModelInfo, RunParams, RunResult, RunStatus

PNG_URI = re.compile(r"data:image/png;base64,[A-Za-z0-9+/=]+")

# The three sections that expose a batch total, an aggregate figure and a
# per-image table respectively.
REPORT_SECTIONS = ["resumen", "resultados", "conteo"]


def _particle(conf: float, diameter_um: float) -> Instance:
    """One measured PET particle with a distinguishable confidence and size."""
    return Instance(
        class_id=0,
        class_name="PET",
        conf=conf,
        x1=10.0,
        y1=10.0,
        x2=50.0,
        y2=50.0,
        diam_um=diameter_um,
        size_source="mask",
    )


def _run_with(root: Path, images: dict[str, list[Instance]]) -> RunResult:
    """Build a one-model run whose results are exactly ``images``."""
    photo_dir = root / "photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, detections in images.items():
        path = photo_dir / name
        path.write_bytes(b"not read by the summary, figure or count sections")
        rows.append(ImageResult(image_path=path, model_idx=0, predictions=detections))
    return RunResult(
        run_dir=root / "run",
        images=[row.image_path for row in rows],
        models=[ModelInfo(alias="best", path=root / "best.pt")],
        params=RunParams(),
        results={0: rows},
        status=RunStatus.COMPLETED,
    )


def _two_image_run(tmp_path: Path) -> tuple[RunResult, Path, Path]:
    """Alpha has one confident particle; beta has two weak ones."""
    alpha = tmp_path / "photos" / "alpha.png"
    beta = tmp_path / "photos" / "beta.png"
    run = _run_with(
        tmp_path,
        {
            "alpha.png": [_particle(0.9, 100.0)],
            "beta.png": [_particle(0.2, 200.0), _particle(0.3, 300.0)],
        },
    )
    return run, alpha, beta


def _kpi(html: str, label: str) -> str:
    match = re.search(rf"<div class='l'>{re.escape(label)}</div><div class='v'>([^<]+)</div>", html)
    assert match is not None, f"KPI {label!r} missing from the report"
    return match.group(1)


def test_only_images_narrows_the_totals_and_the_tables(tmp_path, selected_language):
    """The summary KPIs and the per-image table describe only the chosen subset."""
    selected_language("en")
    from polyx.gui.report_html import generate_report

    run, alpha, _beta = _two_image_run(tmp_path)

    full = generate_report(run, tmp_path / "full.html", sections=REPORT_SECTIONS).read_text("utf-8")
    assert _kpi(full, "Images") == "2"
    assert _kpi(full, "Detections") == "3"
    assert "alpha.png" in full
    assert "beta.png" in full

    narrowed = generate_report(
        run, tmp_path / "narrowed.html", sections=REPORT_SECTIONS, only_images=[alpha]
    ).read_text("utf-8")
    assert _kpi(narrowed, "Images") == "1"
    assert _kpi(narrowed, "Detections") == "1"
    assert _kpi(narrowed, "Mean conf.") == "0.900"
    assert "alpha.png" in narrowed
    assert "beta.png" not in narrowed


def test_only_images_gives_the_figures_the_same_subset(tmp_path, selected_language):
    """The narrowed render equals a run that only ever had the chosen image."""
    selected_language("en")
    from polyx.gui.report_html import generate_report

    run, alpha, _beta = _two_image_run(tmp_path)
    single = _run_with(tmp_path / "single", {"alpha.png": [_particle(0.9, 100.0)]})

    full = generate_report(run, tmp_path / "full.html", sections=REPORT_SECTIONS).read_text("utf-8")
    narrowed = generate_report(
        run, tmp_path / "narrowed.html", sections=REPORT_SECTIONS, only_images=[alpha]
    ).read_text("utf-8")
    single_html = generate_report(
        single, tmp_path / "single.html", sections=REPORT_SECTIONS
    ).read_text("utf-8")

    narrowed_figures = PNG_URI.findall(narrowed)
    assert narrowed_figures
    assert narrowed_figures == PNG_URI.findall(single_html)
    assert narrowed_figures != PNG_URI.findall(full)
    assert narrowed == single_html
