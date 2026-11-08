"""Sections."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from polyx.core.detection_run import RunStatus
from polyx.core.i18n import _, ngettext
from polyx.core.quality.assignment import resume
from polyx.core.quality.metrics import aggregate_per_class, confusion_matrix
from polyx.core.report_metrics import _ORDER_CLASSES, _count_per_class, _pr_f1
from polyx.core.stats import (
    _average_ic95,
    _kruskal_wallis,
    _morphotype,
    _sweep_confidence,
    _text_kruskal,
    _text_trend,
)
from polyx.gui import theme as T

from . import machine
from .batch import LotForms, ShapeBodies
from .figures import (
    _fig_agreement_by_image,
    _fig_boxplot_groups,
    _fig_class_distribution,
    _fig_comparison_metrics,
    _fig_confidence_hist,
    _fig_confusion_matrix,
    _fig_metrics_by_class,
    _fig_profile_depth,
    _fig_size_hist,
    _fig_sizes_by_section,
    _fig_sweep,
    _figure_calibration,
)
from .markup import _esc, _img_b64, _table_count, _table_width, _uri_gallery, method_label
from .particle import _record_particle
from .registry import NUM
from .samples import _MAX_FOLDER_PER_PHOTO, _ORDER_STATION, _clue_order, _sample_of, choose_records

_common = machine

if TYPE_CHECKING:
    from polyx.core.detection_run import RunResult

    from .context import ReportContext


def _clue_label(station: str, stretch: str, plate: str | None = None) -> str:
    """``station · stretch N · plate X`` as the grouped tables render it.

    The three tables that group by station/stretch build this label; keeping
    it in one place keeps the separators and escaping identical.
    """
    label = _esc(station) + " &middot; " + _("stretch %(stretch)s") % {"stretch": _esc(stretch)}
    if plate is not None:
        label += " &middot; " + _("plate %(plate)s") % {"plate": _esc(plate)}
    return label


def _section_team(active) -> str:
    """Table of components and of how each loaded model was trained."""
    rows_eq = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in _common._team())
    html = (
        f"<h3>@@N@@.1 {_('Computing hardware')}</h3>"
        f"<table class='data'><tr>{_('<th>Component</th><th>Detail</th>')}</tr>"
        f"{rows_eq}</table>"
    )

    training_runs = []
    for s in active:
        if s.path is None:
            continue
        info = _common._training_of(Path(s.path))
        if info:
            training_runs.append((s.alias, info))
    if not training_runs:
        return html

    def _n(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    rows_2 = ""
    for alias, i in training_runs:
        rows_2 += (
            f"<tr><td>{_esc(alias)}</td><td>{_esc(i['base'])}</td>"
            f"<td class='r'>{_esc(i['imgsz'])}</td><td class='r'>{_esc(i['batch'])}</td>"
            f"<td class='r'>{_esc(i['epocas'])}</td><td>{_esc(i['optimizador'])}</td>"
            f"<td class='r'>{_n(i['precision'])}</td><td class='r'>{_n(i['recall'])}</td>"
            f"<td class='r'>{_n(i['map50'])}</td><td class='r'>{_n(i['map'])}</td></tr>"
        )

    html += (
        f"<h3>@@N@@.2 {_('Training of each model')}</h3>"
        + _(
            "<p>Validation settings and metrics each weight was trained with, read from the <code>.pt</code> file itself.</p>"
        )
        + "<table class='data'>"
        + _(
            "<tr><th>Model</th><th>Base architecture</th><th>imgsz</th><th>batch</th><th>epochs</th><th>optimiser</th><th>Precision</th><th>Recall</th><th>mAP@50</th><th>mAP@50-95</th></tr>"
        )
        + f"{rows_2}</table>"
    )
    return html


def _run_status_note(run) -> str:
    """A visible banner for a run that did not finish, or empty when it did.

    A failed or aborted batch is still reportable -- its partial results are
    real data -- but the document must never present it as a complete run.
    """
    status = RunStatus.parse(run.status)
    if status is RunStatus.FAILED:
        title = _("Incomplete batch (failed run).")
        detail = _("The report describes only the results written before the error: %(error)s") % {
            "error": _esc(run.error or "—")
        }
        colour = T.DOC.ERR
    elif status is RunStatus.ABORTED:
        title = _("Partial batch (aborted run).")
        detail = _("The report describes only the images processed before the interruption.")
        colour = T.DOC.WARN
    else:
        return ""
    return (
        f"<p class='run-status' style='border-left:4px solid {colour};"
        "padding:6px 12px;background:var(--bg_soft)'>"
        f"<strong>{title}</strong> {detail} <code>{status.value}</code></p>"
    )


def _section_count(results_2: dict[int, list], run, classes: list[str]) -> str:
    """Section 8: how many particles of each polymer there are per sample.

    It is the table requested for the manuscript: count by plastic type
    and sample, with the manual count and the model count one next to the
    other. **Not** an evaluation of the detector: the model column is all
    of its detections, without discounting false positives. That reading
    lives in the error section.

    With several models loaded the first one is used. Putting two models in
    the same cell would make the table illegible, and the report already
    compares models in its own section.
    """
    if not run.models:
        return ""
    rs = results_2.get(0, [])
    if not rs:
        return ""

    rs = sorted(rs, key=lambda r: _clue_order(r.image_path))
    there_is_a_manual = any(r.has_gt for r in rs)

    # -- Per image --
    rows_img = []
    for r in rs:
        sample = _sample_of(r.image_path)
        label_2 = _esc(r.image_path.name)
        if sample:
            station, range_value, plate = sample
            label_2 = (
                f"{_esc(r.image_path.name)}<br>"
                f"<span style='font-size:8.5pt;color:var(--ink3)'>"
                + _clue_label(station, range_value, plate)
                + "</span>"
            )
        rows_img.append(
            (
                label_2,
                _count_per_class(r.gt) if r.has_gt else Counter(),
                _count_per_class(r.predictions),
            )
        )
    table_img = _table_width(rows_img, classes, "Sample (image)", there_is_a_manual)

    # -- Per stretch and per station, if the names follow the pattern --
    grouped = [(r, _sample_of(r.image_path)) for r in rs]
    parseable = [(r, m) for r, m in grouped if m is not None]
    table_range = table_station = ""
    note_grouping = ""
    if parseable:
        per_section: dict[tuple, list] = {}
        per_station: dict[str, list] = {}
        for r, (station, range_value, _plate) in parseable:
            per_section.setdefault((station, range_value), []).append(r)
            per_station.setdefault(station, []).append(r)

        def _add(items_2):
            man, det = Counter(), Counter()
            for r in items_2:
                if r.has_gt:
                    man.update(_count_per_class(r.gt))
                det.update(_count_per_class(r.predictions))
            return man, det

        def _order_est(e):
            return _ORDER_STATION.index(e) if e in _ORDER_STATION else 99

        rows_t = []
        for clue in sorted(per_section, key=lambda k: (_order_est(k[0]), k[1])):
            station, range_value = clue
            items_2 = per_section[clue]
            man, det = _add(items_2)
            palabra = ngettext("%(count)s image", "%(count)s images", len(items_2)) % {
                "count": len(items_2)
            }
            rows_t.append(
                (
                    _clue_label(station, range_value) + " "
                    f"<span style='font-size:8.5pt;color:var(--ink3)'>"
                    f"({palabra})</span>",
                    man,
                    det,
                )
            )
        table_range = (
            "<h3>@@N@@.2 "
            + _("By depth interval")
            + "</h3>"
            + _(
                "<p>Dishes from the same depth interval are <strong>added together</strong>: they are subsamples of the same mass of sediment, not repeated photographs. The interval is the unit of analysis.</p>"
            )
            + _table_width(rows_t, classes, _("Interval"), there_is_a_manual)
        )

        rows_e = []
        for station in sorted(per_station, key=_order_est):
            man, det = _add(per_station[station])
            rows_e.append((_esc(station), man, det))
        table_station = (
            "<h3>@@N@@.3 "
            + _("By station")
            + "</h3>"
            + _table_width(rows_e, classes, _("Station"), there_is_a_manual, totals=False)
        )

        outside = len(grouped) - len(parseable)
        if outside:
            note_grouping = (
                "<p class='caption'>"
                + ngettext(
                    "%(n)s image does not follow the <code>interval.core</code> naming and is left out of the grouped tables; it is still in the per-image table.",
                    "%(n)s images do not follow the <code>interval.core</code> naming and are left out of the grouped tables; they are still in the per-image table.",
                    outside,
                )
                % {"n": outside}
                + "</p>"
            )

    tagline = _(", the first active model.") if len(run.models) > 1 else "."
    intro = (
        _(
            "<p>Particles counted in each sample, broken down by polymer. The <em>manual</em> column is the human annotation (ground truth) and the <em>model</em> column is every detection made by <strong>%(alias)s</strong>%(suffix)s</p>"
        )
        % {"alias": _esc(run.models[0].alias), "suffix": tagline}
        + _(
            "<p><strong>Read it as a count, not as an evaluation.</strong> The model's column does not subtract false positives nor match box to box: it is how many particles of each polymer it reported. Agreeing on the total does not imply getting each particle right; that is what the error analysis is for.</p>"
        )
        + "<h3>@@N@@.1 "
        + _("Per image")
        + "</h3>"
    )
    return intro + table_img + note_grouping + table_range + table_station


# ────────────────────────────────────────────────────────────────────
# ── Report sections ────────────────────────────────────────────────────────
# The order of this list is the order of the report, and the numbering comes from here:
# if a section is unchecked the following ones renumber themselves, instead of leaving
# a gap (`4, 6, 7`) that in a deliverable document reads as an error.
#
# Adding a new section is adding a tuple: the interface draws its checkboxes
# by reading this list, so there is no second list to keep in
# sync.


def _section_summary(
    total_imgs: int, total_dets: int, avg_conf: float, avg_size: float, active, any_gt: bool
) -> str:
    """Body of the 'resumen' section: KPI cards plus the abstract paragraph.

    Returns the HTML body WITHOUT its ``<h2>`` title: the assembler owns the
    heading and the section number.
    """
    # Outside the paragraph f-string: it carries a plural and a numeric format,
    # and nesting them inside `_()` would require quotes of the same type.
    _summary_paragraph = _(
        "<strong>%(images)s images</strong> were analysed with <strong>%(n)s YOLO %(models)s</strong> trained to detect PET, PP and LDPE microplastics under Nile Red fluorescence (254 nm). Detections totalled <strong>%(detections)s</strong> with a mean confidence of <strong>%(conf)s</strong>."
    ) % {
        "images": total_imgs,
        "n": len(active),
        "models": ngettext("model", "models", len(active)),
        "detections": total_dets,
        "conf": f"{avg_conf:.3f}",
    }

    return f"""
<div class='kpis'>
  <div class='kpi'><div class='l'>{_("Images")}</div><div class='v'>{total_imgs}</div><div class='b' style='background:var(--accent)'></div></div>
  <div class='kpi'><div class='l'>{_("Detections")}</div><div class='v'>{total_dets}</div><div class='b' style='background:var(--accent)'></div></div>
  <div class='kpi'><div class='l'>{_("Mean conf.")}</div><div class='v'>{avg_conf:.3f}</div><div class='b' style='background:var(--vio)'></div></div>
  <div class='kpi'><div class='l'>{_("Mean size (μm)")}</div><div class='v'>{(f"{avg_size:.1f}" if avg_size else "—")}</div><div class='b' style='background:var(--warn)'></div></div>
</div>
<p>{_summary_paragraph}
{_("Error analysis against ground truth was included (true positives, false positives, false negatives and misclassified).") if any_gt else _("No ground truth was supplied, so no error metrics are reported.")}
</p>"""


def _section_errors(run, results_2, any_gt: bool) -> str:
    """Body of the 'errores' section: confusion matrix plus per-class metrics.

    Returns the HTML body WITHOUT its ``<h2>`` title. Empty when there is no
    ground truth or no model: the assembler omits a section with no content
    rather than inventing data.
    """
    if not (any_gt and run.models):
        return ""
    active = run.models
    rs = results_2.get(0, [])
    if run.legacy:
        # Old headless folders kept per-image TP/FP/FN counts but not the GT
        # boxes. Rebuilding the confusion matrix from the predictions alone
        # would turn every legacy detection into a false positive, so report
        # the counts the old run actually recorded instead.
        rows_legacy = ""
        for index, model in enumerate(active):
            counts = run.metrics.get(index, {})
            if not counts:
                model_rows = rs if index == 0 else []
                counts = {
                    "tp": sum(row.tp for row in model_rows),
                    "fp": sum(row.fp for row in model_rows),
                    "fn": sum(row.fn for row in model_rows),
                    "miscls": sum(row.miscls for row in model_rows),
                }
            rows_legacy += (
                f"<tr><td>{_esc(model.alias)}</td>"
                f"<td class='r'>{int(counts.get('tp', 0))}</td>"
                f"<td class='r'>{int(counts.get('fp', 0))}</td>"
                f"<td class='r'>{int(counts.get('fn', 0))}</td>"
                f"<td class='r'>{int(counts.get('miscls', 0))}</td></tr>"
            )
        return f"""
        <p>{_("This report comes from a folder written before the canonical format. That folder stored per-image counts but not the ground-truth boxes, so the confusion matrix and per-class metrics cannot be rebuilt. These are the counts the run did record:")}</p>
        <table class='data'><tr>{_("<th>Model</th>")}<th>TP</th><th>FP</th><th>FN</th><th>MISCLS</th></tr>{rows_legacy}</table>
        """

    gts_only = [r.gt for r in rs if r.has_gt]
    preds_only = [r.predictions for r in rs if r.has_gt]
    cls_ids = sorted({d.class_id for lst in gts_only + preds_only for d in lst})
    cls_names = []
    for cid in cls_ids:
        name = next(
            (d.class_name for lst in gts_only + preds_only for d in lst if d.class_id == cid),
            str(cid),
        )
        cls_names.append(name)
    cm = confusion_matrix(preds_only, gts_only, cls_ids, iou_thr=run.params.iou_tp)
    per_class_metrics = aggregate_per_class(
        preds_only, gts_only, cls_ids, iou_thr=run.params.iou_tp
    )

    cm_img = _fig_confusion_matrix(cm, cls_names)
    rows = ""
    for cid, name in zip(cls_ids, cls_names):  # noqa: B905
        cm_ = per_class_metrics[cid]
        rows += (
            f"<tr><td>{name}</td>"
            f"<td class='r'>{cm_.tp}</td><td class='r'>{cm_.fp}</td><td class='r'>{cm_.fn}</td>"
            f"<td class='r'>{cm_.precision:.3f}</td><td class='r'>{cm_.recall:.3f}</td>"
            f"<td class='r'>{cm_.f1:.3f}</td></tr>"
        )
    # The classes that the model knows but that appear neither in the GT nor in
    # the predictions are left out of the table, because they have no metric
    # to compute. They are declared anyway: a report that announces PET/PP/LDPE and
    # then shows two rows looks broken, and the reader cannot distinguish
    # `absent from the batch` from `lost along the way`.
    known = dict(active[0].names or {})
    missing = [n for cid, n in sorted(known.items()) if cid not in set(cls_ids)]
    note_missing = ""
    if missing:
        note_missing = (
            "<p class='caption' style='text-align:left'>"
            + _(
                "No row for %(classes)s: it appears neither in the manual annotation nor among this batch's predictions, so there is no metric to report. The model is trained for that class all the same."
            )
            % {"classes": _esc(", ".join(missing))}
            + "</p>"
        )

    return f"""
        
        <h3>@@N@@.1 {_("Confusion matrix")}</h3>
        <div class='fig'><img src='data:image/png;base64,{cm_img}' />
            <div class='caption'>{_("Figure. Confusion matrix (main model: %(model)s, IoU = %(iou)s).") % {"model": _esc(active[0].alias), "iou": run.params.iou_tp}}</div></div>
        <h3>@@N@@.2 {_("Precision / recall / F1 per class")}</h3>
        <table class='data'><tr>{_("<th>Class</th><th>True Positives</th><th>False Positives</th><th>False Negatives</th>")}
        {_("<th>Precision</th><th>Recall</th><th>F1</th>")}</tr>{rows}</table>
        {note_missing}
        """


def _section_models(run, results_2, all_results, any_gt: bool) -> str:
    """Body of the 'modelos' section: per-model comparison table plus F1 note.

    Returns the HTML body WITHOUT its ``<h2>`` title (header table included:
    the builder owns the complete section body the assembler used to compose).
    """
    # TWO F1 values are reported, and the difference between them is not a detail:
    #
    #         localization  tp/(tp+fp) and tp/(tp+fn). Boxes that are well placed but with
    #                                                                 the wrong class fall outside both denominators,
    #                                                                 so it measures only whether the detector finds the particle.
    #         with class    those same boxes count as a false positive of the predicted
    #                                                                 class and a false negative of the real one, which is the strict criterion.
    #
    # Publishing only the first overestimates performance, and also contradicts the
    # per-class table, which does penalize confusion. Both are declared.
    rows_models = ""
    total_mc = 0
    for mi, slot in enumerate(run.models):
        rs = results_2.get(mi, [])
        n_img = len({r.image_path for r in rs})
        n_det = sum(len(r.predictions) for r in rs)
        cf = [p.conf for r in rs for p in r.predictions]
        tp = sum(r.tp for r in rs)
        fp = sum(r.fp for r in rs)
        fn = sum(r.fn for r in rs)
        mc = sum(r.miscls for r in rs)
        total_mc += mc
        _unused, _unused, f1_loc = _pr_f1(tp, fp, fn)
        _unused, _unused, f1_cls = _pr_f1(tp, fp + mc, fn + mc)
        avg_cf = (sum(cf) / len(cf)) if cf else 0
        there_is_gt = any(r.has_gt for r in rs)
        c_loc = f"{f1_loc:.3f}" if there_is_gt else "—"
        c_cls = f"{f1_cls:.3f}" if there_is_gt else "—"
        rows_models += (
            f"<tr><td>{_esc(slot.alias)}</td><td class='r'>{n_img}</td><td class='r'>{n_det}</td>"
            f"<td class='r'>{avg_cf:.3f}</td>"
            f"<td class='r'>{tp}</td><td class='r'>{fp}</td><td class='r'>{fn}</td>"
            f"<td class='r'>{mc}</td>"
            f"<td class='r'>{c_loc}</td><td class='r'>{c_cls}</td></tr>"
        )

    # Note that accompanies the table, with the figures from the batch itself.
    note_miscls = ""
    if any_gt:
        tp_g = sum(r.tp for r in all_results)
        fp_g = sum(r.fp for r in all_results)
        fn_g = sum(r.fn for r in all_results)
        p_loc, r_loc, f_loc = _pr_f1(tp_g, fp_g, fn_g)
        p_cls, r_cls, f_cls = _pr_f1(tp_g, fp_g + total_mc, fn_g + total_mc)
        note_miscls = ngettext(
            "<p><strong>The two F1 figures measure different things.</strong> <em>Localisation</em> answers whether the detector finds the particle (P %(pl)s · R %(rl)s · <strong>F1 %(fl)s</strong>). <em>With class</em> also requires getting the polymer right, counting each misclassified box as a false positive of the predicted class and a false negative of the real one (P %(pc)s · R %(rc)s · <strong>F1 %(fc)s</strong>).</p><p>The difference corresponds to <strong>%(mc)s</strong> particle located correctly but assigned to the wrong class. It is the figure that reconciles this table with the per-class precision in the error section.</p>",
            "<p><strong>The two F1 figures measure different things.</strong> <em>Localisation</em> answers whether the detector finds the particle (P %(pl)s · R %(rl)s · <strong>F1 %(fl)s</strong>). <em>With class</em> also requires getting the polymer right, counting each misclassified box as a false positive of the predicted class and a false negative of the real one (P %(pc)s · R %(rc)s · <strong>F1 %(fc)s</strong>).</p><p>The difference corresponds to <strong>%(mc)s</strong> particles located correctly but assigned to the wrong class. It is the figure that reconciles this table with the per-class precision in the error section.</p>",
            total_mc,
        ) % {
            "pl": f"{p_loc:.3f}",
            "rl": f"{r_loc:.3f}",
            "fl": f"{f_loc:.3f}",
            "pc": f"{p_cls:.3f}",
            "rc": f"{r_cls:.3f}",
            "fc": f"{f_cls:.3f}",
            "mc": total_mc,
        }

    return f"""
<table class='data'><tr>{_("<th>Model</th><th>Images</th><th>Detections</th><th>Mean conf.</th>")}
{_("<th>True Positives</th><th>False Positives</th><th>False Negatives</th>")}
{_("<th>Misclassified</th>")}<th>F1<br><span style='font-weight:400;font-size:8.5pt'>{_("localisation")}</span></th>
<th>F1<br><span style='font-weight:400;font-size:8.5pt'>{_("with class")}</span></th></tr>{rows_models}</table>
{note_miscls}"""


def _section_verdict(results_2, run) -> str:
    """Body of the 'veredicto' section: best model by F1-with-class, from the
    confidence sweep. Arguments are the precomputed locals the inline block
    consumed before extraction."""
    text_2 = ""
    if any(r.has_gt for rs in results_2.values() for r in rs):
        sweep = _sweep_confidence(results_2, run.models, run.params.iou_tp, run.params.conf)
        if sweep:
            summary = []
            for alias, rows_2 in sweep.items():
                best = max(rows_2, key=lambda f: f[3])
                current = rows_2[0]
                summary.append((alias, current, best))

            # Verdict: the winner is whoever has the best F1 with class at their best threshold.
            summary.sort(key=lambda x: x[2][3], reverse=True)
            g_alias, g_act, g_best = summary[0]  # noqa: RUF059
            text_2 = _(
                "<p><strong>Best performance: %(alias)s</strong>, with F1 %(f1)s at threshold %(threshold)s (P %(p)s · R %(r)s).</p>"
            ) % {
                "alias": _esc(g_alias),
                "f1": f"{g_best[3]:.3f}",
                "threshold": f"{g_best[0]:.2f}",
                "p": f"{g_best[1]:.3f}",
                "r": f"{g_best[2]:.3f}",
            }
            if len(summary) > 1:
                p_alias, _unused, p_best = summary[-1]
                d = g_best[3] - p_best[3]
                text_2 += _(
                    "<p>The gap against %(other)s is <strong>%(d)s</strong> of F1. With a single training run per architecture, a small difference does not tell the network design apart from initialisation luck: it would take repeating with different seeds to claim one is better.</p>"
                ) % {"other": _esc(p_alias), "d": f"{d:.3f}"}

            return text_2
    return text_2


def _section_comparison(run, results_2) -> str:
    """Body of the 'comparacion' section: per-photo model table, the F1 of
    localization note and the comparison figures. Arguments are the
    precomputed locals the inline block consumed before extraction."""
    # ── Models side by side, photo by photo ──
    # The global table above says which model wins overall; this one says in
    # which photos it wins. With uneven batches like those from the Loa, where one stretch
    # concentrates most of the particles, the total is decided by that stretch and
    # can hide that in the rest the other model does better.
    compare_html = ""
    active_2 = list(range(len(run.models)))
    if len(active_2) > 1:
        by_photo = {}
        for mi in active_2:
            for r in results_2.get(mi, []):
                by_photo.setdefault(r.image_path, {})[mi] = r

        there_is_gt = any(r.has_gt for rs in results_2.values() for r in rs)
        header_line = _("<tr><th>Image</th>")
        for mi in active_2:
            alias = run.models[mi].alias
            header_line += (
                f"<th>{_esc(alias)}<br>"
                f"<span style='font-weight:400;font-size:8.5pt'>{_('dets')}</span></th>"
            )
            if there_is_gt:
                header_line += (
                    "<th><span style='font-weight:400;font-size:8.5pt'>"
                    + _("True Positives/False Positives/False Negatives")
                    + "</span></th>"
                )
        header_line += "</tr>"

        rows_2 = ""
        for path_2 in sorted(by_photo):
            rows_2 += f"<tr><td>{_esc(Path(path_2).name)}</td>"
            for mi in active_2:
                r = by_photo[path_2].get(mi)
                if r is None:
                    rows_2 += "<td class='r'>—</td>"
                    if there_is_gt:
                        rows_2 += "<td class='r'>—</td>"
                    continue
                rows_2 += f"<td class='r'>{len(r.predictions)}</td>"
                if there_is_gt:
                    cell = f"{r.tp}/{r.fp}/{r.fn}" if r.has_gt else "—"
                    rows_2 += f"<td class='r'>{cell}</td>"
            rows_2 += "</tr>"

        # Verdict: only declared if there is ground truth to judge with. Without it,
        # more detections is not better, and saying otherwise would be misleading.
        verdict = ""
        if there_is_gt:
            marker_2 = []
            for mi in active_2:
                rs = results_2.get(mi, [])
                tp = sum(r.tp for r in rs)
                fp = sum(r.fp for r in rs)
                fn = sum(r.fn for r in rs)
                pr = tp / (tp + fp) if (tp + fp) else 0
                rc = tp / (tp + fn) if (tp + fn) else 0
                marker_2.append((2 * pr * rc / (pr + rc) if (pr + rc) else 0, run.models[mi].alias))
            marker_2.sort(reverse=True)
            best_f1, best = marker_2[0]
            # This F1 is the permissive one (localization). It used to say only `F1 global` and
            # coexisted with the strict verdict above: two different numbers
            # called the same in the same section.
            verdict = _(
                "<p><b>Best localisation F1:</b> %(alias)s (%(f1)s) — finding the particle, without requiring the polymer to be right. The with-class verdict is further up.</p>"
            ) % {"alias": _esc(best), "f1": f"{best_f1:.3f}"}
        else:
            verdict = (
                "<p class='caption' style='text-align:left'>"
                + _(
                    "Without ground truth no winner can be declared: a model with more detections may be getting them right or making them up. Load annotations to obtain F1 per model."
                )
                + "</p>"
            )

        # ── Side-by-side figures ──
        figures_cmp = ""
        if there_is_gt:
            data_bars = []
            by_model_class: dict[str, dict[str, tuple]] = {}
            for mi in active_2:
                alias = run.models[mi].alias
                rs = [r for r in results_2.get(mi, []) if r.has_gt]
                tp = sum(r.tp for r in rs)
                fp = sum(r.fp for r in rs)
                fn = sum(r.fn for r in rs)
                mc = sum(r.miscls for r in rs)
                p_l, r_l, f_l = _pr_f1(tp, fp, fn)
                p_c, r_c, f_c = _pr_f1(tp, fp + mc, fn + mc)
                data_bars.append((alias, p_l, r_l, f_l, p_c, r_c, f_c))

                gts_m = [r.gt for r in rs]
                preds_m = [r.predictions for r in rs]
                ids = sorted({d.class_id for lst in gts_m + preds_m for d in lst})
                if ids:
                    metrics = aggregate_per_class(preds_m, gts_m, ids, iou_thr=run.params.iou_tp)
                    names_cls = {
                        cid: next(
                            (
                                d.class_name
                                for lst in gts_m + preds_m
                                for d in lst
                                if d.class_id == cid
                            ),
                            str(cid),
                        )
                        for cid in ids
                    }
                    by_model_class[alias] = {
                        names_cls[cid]: (
                            metrics[cid].precision,
                            metrics[cid].recall,
                            metrics[cid].f1,
                        )
                        for cid in ids
                    }

            f_bars = _fig_comparison_metrics(data_bars)
            if f_bars:
                figures_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_bars}' />"
                    "<div class='caption'>"
                    + _(
                        "Figure. Precision, recall and F1 of each model at threshold %(threshold)s, under both criteria. The gap between panels is the confusion between polymers."
                    )
                    % {"threshold": f"{run.params.conf:g}"}
                    + "</div></div>"
                )

            f_class_id = _fig_metrics_by_class(by_model_class)
            if f_class_id:
                figures_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_class_id}' />"
                    "<div class='caption'>"
                    + _(
                        "Figure. F1 per class. A model can win overall and lose on the polymer that matters."
                    )
                    + "</div></div>"
                )

            # Photo-by-photo agreement between the first two models.
            if len(active_2) >= 2:
                a, b = active_2[0], active_2[1]
                pares = []
                for path_2 in sorted(by_photo):
                    ra, rb = by_photo[path_2].get(a), by_photo[path_2].get(b)
                    if ra is not None and rb is not None:
                        pares.append((len(ra.predictions), len(rb.predictions), Path(path_2).name))
                f_agreement = _fig_agreement_by_image(
                    run.models[a].alias, run.models[b].alias, pares
                )
                if f_agreement:
                    figures_cmp += (
                        f"<div class='fig'><img src='data:image/png;base64,{f_agreement}' />"
                        "<div class='caption'>"
                        + _(
                            "Figure. Detections per image of one model against the other; the diagonal is exact agreement. Symmetric-log scale, because almost every dish has few particles and a handful have hundreds."
                        )
                        + "</div></div>"
                    )

            # F1 curve against the threshold: it supports the choice of the operating
            # point with evidence, instead of asserting it.
            sweep = _sweep_confidence(results_2, run.models, run.params.iou_tp, run.params.conf)
            f_sweep = _fig_sweep(sweep)
            if f_sweep:
                figures_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_sweep}' />"
                    "<div class='caption'>"
                    + _(
                        "Figure. F1 (with class) against the confidence threshold; the star marks each model's maximum. The curve starts at %(threshold)s, the threshold it was run with: below that the detections were never computed."
                    )
                    % {"threshold": f"{run.params.conf:g}"}
                    + "</div></div>"
                )

        compare_html = (
            _(
                "<p>Each model's detections image by image. The total can be dominated by a few dense photographs, so it is worth looking at the detail before choosing a model.</p>"
            )
            + verdict
            + figures_cmp
            + f"<table class='data'>{header_line}{rows_2}</table>"
        )
    elif len(active_2) == 1:
        compare_html = (
            "<p class='caption' style='text-align:left'>"
            + _(
                "Only one model was run, so there is no comparison. Load a second model in the Models tab to compare them over the same images."
            )
            + "</p>"
        )
    return compare_html


def _section_gallery(
    results_2: dict[int, list],
    run,
    classes_batch,
    max_gallery: int = 60,
    include_gallery: bool = True,
) -> str:
    """Body of the 'galeria' section: side-by-side prediction vs Ground Truth.

    Returns the HTML body WITHOUT its ``<h2>`` title: the assembler owns the
    heading and the section number. Empty when the gallery is disabled or no
    candidate image has prediction bytes.
    """
    # ── Side-by-side gallery: Prediction vs Ground Truth ──
    gallery_html = ""
    omitted = 0
    if include_gallery:
        blocks = []
        # Gallery image cap: each one is embedded as base64, so
        # without a limit a batch of several hundred produces a gigabyte HTML.
        # It is grouped by image and not by model. Flattening model by model, the
        # cap was exhausted inside the first one and the second never appeared in
        # the gallery: with two models loaded only the photos from one were seen,
        # which is precisely the visual side-by-side that is wanted. Now the cap counts
        # images, and from each one all models enter, one below the other.
        per_image: dict = {}
        for mi in sorted(results_2):
            for r in results_2[mi]:
                per_image.setdefault(r.image_path, []).append((mi, r))
        # Order by station/stretch/plate when the plate name
        # allows it: the report is read following the core downwards, not in the
        # arbitrary order in which the files were loaded.
        order_imgs = sorted(per_image, key=_clue_order)
        candidates = []
        for path_2 in order_imgs[:max_gallery]:
            candidates.extend(per_image[path_2])
        omitted = max(0, len(per_image) - max_gallery)
        for mi, r in candidates:
            slot = run.models[mi]
            if True:
                # Prediction image (we prefer pred_png; otherwise the combined one)
                pred_bytes = getattr(r, "pred_png", None) or r.annotated_png
                if not pred_bytes:
                    continue
                left = (
                    f"<figure><img src='{_uri_gallery(pred_bytes)}' />"
                    f"<figcaption>{_('Model prediction')} · {_esc(slot.alias)}"
                    f"<span class='sub'>"
                    + ngettext(
                        "%(n)s detection by YOLO",
                        "%(n)s detections by YOLO",
                        len(r.predictions),
                    )
                    % {"n": len(r.predictions)}
                    + "</span>"
                    "</figcaption></figure>"
                )

                # Ground Truth image (control)
                gt_bytes = getattr(r, "gt_png", None)
                if r.has_gt and gt_bytes:
                    right = (
                        f"<figure><img src='{_uri_gallery(gt_bytes)}' />"
                        f"<figcaption>{_('Ground Truth (control)')}"
                        f"<span class='sub'>"
                        + ngettext(
                            "%(count)s real label",
                            "%(count)s real labels",
                            len(r.gt),
                        )
                        % {"count": len(r.gt)}
                        + "</span></figcaption></figure>"
                    )
                else:
                    right = (
                        f"<figure><div class='nogt'>"
                        f"{_('No ground truth for this image')}</div>"
                        f"<figcaption>{_('Ground Truth (control)')}"
                        f"<span class='sub'>{_('not available')}</span>"
                        f"</figcaption></figure>"
                    )

                # Photo header: name and, if the name allows it,
                # which stretch and plate it comes from.
                sample = _sample_of(r.image_path)
                where = ""
                if sample:
                    station, range_value, plate = sample
                    where = (
                        " &nbsp;<span class='tag'>"
                        + _clue_label(station, range_value, plate)
                        + "</span>"
                    )
                meta = (
                    f"<strong>{_esc(r.image_path.name)}</strong>{where}"
                    "&nbsp;<span class='tag'>"
                    + _("model: %(model)s") % {"model": _esc(slot.alias)}
                    + "</span>"
                )

                # Photo table: particles per polymer. Matches and
                # failures paired box by box are another question and live in
                # the error analysis section.
                table = _table_count(
                    _count_per_class(r.gt) if r.has_gt else None,
                    _count_per_class(r.predictions),
                    classes_batch,
                )

                blocks.append(
                    f"<div class='compare'><div class='compare-pair'>{left}{right}</div>"
                    f"<div class='compare-meta'>{meta}</div>{table}</div>"
                )

        if blocks:
            note = ""
            if omitted:
                n_shown = len({r.image_path for _unused, r in candidates})
                note = (
                    "<p class='caption'>"
                    + _(
                        "The first %(n)s images are shown, each with every model; %(omitted)s were left out of the gallery to keep the file manageable. The metrics in the earlier sections do cover all of them."
                    )
                    % {"n": n_shown, "omitted": omitted}
                    + "</p>"
                )
            gallery_html = (
                _(
                    "<p>Each block shows, on the left, the model's detections (<em>bounding boxes</em> drawn by YOLO with their class and confidence) and, on the right, the real control labels (<em>ground truth</em>). This side-by-side view lets you judge visually where the model got it right or wrong.</p>"
                )
                + note
                + "".join(blocks)
            )
    return gallery_html


def _section_results(per_class: dict[str, int], confs: list[float], sizes: list[float]) -> str:
    """Body of the ``'resultados'`` section: aggregate figures plus captions.

    Returns the HTML body WITHOUT its ``<h2>`` title: the assembler owns
    the heading and the section number. Empty when no figure was
    produced.
    """
    # ── Aggregate figures ──
    fig_classes = _fig_class_distribution(dict(per_class))
    fig_conf = _fig_confidence_hist(confs)
    fig_size = _fig_size_hist(sizes) if sizes else ""

    figures_html = ""
    if fig_classes:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_classes}' /><div class='caption'>{_('Figure. Distribution of detections per class.')}</div></div>"
    if fig_conf:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_conf}' /><div class='caption'>{_('Figure. Confidence histogram.')}</div></div>"
    if fig_size:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_size}' /><div class='caption'>{_('Figure. Size distribution (equivalent diameter in μm).')}</div></div>"
    return figures_html


def _section_methods(
    run, active, all_results, total_imgs: int, total_dets: int, any_gt: bool
) -> tuple:
    """Body of the 'metodos' section: parameters table, team and redacted paragraph.

    Returns ``(methods_html, equipo_html, methods_para)`` — the three pieces
    the assembler's ``metodos`` entry composes — WITHOUT any ``<h2>`` title.
    The ``import ultralytics`` stays lazy inside this body: it must not load
    at module import time (S01 laziness gate).
    """
    # ── Methods ──
    p = run.params
    # Calibration is per photo: there is no single run parameter to quote.
    # The effective scales live in ``run.calibrations``, so the batch mean is
    # given with its 95% confidence interval -- a number that can be cited in
    # a method even when every photo has its own factor (the calibration
    # section expands the minimum, median and maximum).
    _ums_methods = [c.um_por_px for c in run.calibrations.values() if getattr(c, "valid", False)]
    _mic = _average_ic95(_ums_methods)
    if _mic:
        _average_m, _margin_m, _n_m = _mic
        um_px_txt = _("%(mean)s ± %(margin)s (95%% CI, n=%(n)s photographs)") % {
            "mean": f"{_average_m:.4f}",
            "margin": f"{_margin_m:.4f}",
            "n": _n_m,
        }
    elif _ums_methods:
        um_px_txt = f"{_ums_methods[0]:.4f} " + _("(a single calibrated photograph, no CI)")
    else:
        um_px_txt = "—"
    methods_rows = [
        (_("Models loaded"), _esc(", ".join(s.alias for s in active)) or "—"),
        (_("Minimum confidence"), f"{p.conf:g}"),
        (_("IoU NMS"), f"{p.iou_nms:g}"),
        (_("IoU for matching true positives"), f"{p.iou_tp:g}"),
        (_("Image size (imgsz)"), f"{p.imgsz}"),
        (_("Device"), p.device),
        (_("μm per pixel"), um_px_txt),
        (
            _("Size filter (μm)"),
            f"{p.size_min_um} – {p.size_max_um}"
            if (p.size_min_um or p.size_max_um)
            else _("no filter"),
        ),
        (_("Images processed"), str(total_imgs)),
        (_("Total detections"), str(total_dets)),
    ]
    # Abstention: how many particles were counted without attributing a polymer to them. It goes in
    # Methods and not buried in a results table because it is an analysis
    # decision, and whoever reads the report needs to know it was taken.
    _res_assignment = resume(
        [d for r in all_results for d in r.predictions], getattr(p, "conf_assignment", 0.0)
    )
    if _res_assignment.active:
        methods_rows.append((_("Minimum confidence to assign a polymer"), f"{p.conf_assignment:g}"))
        methods_rows.append(
            (
                _("Particles with no assignable polymer"),
                f"{_res_assignment.no_assignables} / {_res_assignment.total} "  # noqa: ISC004
                f"({_res_assignment.percentage_no_assignable:.1f} %)",
            )
        )
    methods_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in methods_rows)
    team_html = _section_team(active)

    # Redacted methods paragraph, ready to copy into the manuscript
    # (reproducibility: model, version and exact parameters).
    try:
        import ultralytics

        _ul_see = ultralytics.__version__
    except Exception:  # noqa: BLE001
        _ul_see = "—"
    _model_names = _esc(", ".join(s.alias for s in active)) if active else "—"
    methods_for = (
        "<p><strong>" + _("Summary of the settings used:") + "</strong></p>"
        "<blockquote style='border-left:3px solid var(--accent); margin:8px 0; "
        "padding:6px 14px; color:#424a53; background:#f6f8fa; border-radius:0 6px 6px 0;'>"
        + _(
            "Automated detection was carried out with the YOLO model «%(models)s» (Ultralytics %(ul)s) at an input resolution of %(imgsz)s px, confidence threshold %(conf)s and non-maximum suppression at IoU %(iou)s. "
        )
        % {
            "models": _model_names,
            "ul": _ul_see,
            "imgsz": p.imgsz,
            "conf": f"{p.conf:g}",
            "iou": f"{p.iou_nms:g}",
        }
        + (
            _(
                "Error metrics were computed against independent manual annotation, matching predictions and labels at IoU ≥ %(iou)s. "
            )
            % {"iou": f"{p.iou_tp:g}"}
            if any_gt
            else ""
        )
        + (
            _("Optical calibration was %(um)s μm/pixel. ") % {"um": um_px_txt}
            if _ums_methods
            else ""
        )
        + _("%(images)s images were processed with a total of %(detections)s detections.")
        % {"images": total_imgs, "detections": total_dets}
        + "</blockquote>"
    )
    return methods_html, team_html, methods_for


def _section_references(include_refs: bool = True) -> str:
    """Body of the 'referencias' section: static list of cited papers.

    Returns the HTML body WITHOUT its ``<h2>`` title. Empty when the section
    is not included, so the assembler drops it like it does today.
    """
    refs_html = ""
    if include_refs:
        refs_html = """
        
        <ol>
          <li>Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).
              Development of a new methodology for the determination of PET microplastics in sediment,
              based on microwave-assisted acid digestion.
              <em>PLoS ONE</em> 19(12): e0314520.
              <a href='https://doi.org/10.1371/journal.pone.0314520'>https://doi.org/10.1371/journal.pone.0314520</a></li>
          <li>Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).
              Evaluation of microwave-assisted acid/oxidant digestion method for the detection of
              polyethylene microplastics in <em>Merluccius gayi</em> fish by Nile Red fluorescent
              staining and image analysis. <em>J. Chil. Chem. Soc.</em> 69(1): 6082-6085.
              <a href='https://doi.org/10.4067/s0717-97072024000106082'>https://doi.org/10.4067/s0717-97072024000106082</a></li>
        </ol>
        """
    return refs_html


def _section_calibration(run) -> str:
    """Body of the 'calibracion' section: scale provenance and per-photo calibration.

    Returns the HTML body WITHOUT its ``<h2>`` title. Empty when the run has
    no effective calibrations, so the assembler drops the section like it does
    today. The ``polyx.core.calibration`` import stays lazy inside this body:
    it must not load at module import time (S01 laziness gate).
    """
    # ── Scale calibration ──
    # The provenance is reported, not only the number: a size in µm without saying
    # against which standard it was measured is not verifiable. If there is no calibration the
    # section is not emitted, instead of showing an `—` that looks like data.
    calib_html = ""
    cals = run.calibrations
    if cals:
        from polyx.core.measure.calibration import (
            BIAS_MAXIMUM_WALL_PCT,
            THICKNESS_WALL_MM,
            summary_batch,
        )

        res = summary_batch(list(cals.values()))
        if res["n"]:
            label_2 = {
                "placa": _("measured on this photograph's Petri dish"),
                "indice": _("inherited from the crop (calibration index)"),
                "manual": _("entered by hand in Parameters"),
            }
            rows_2 = "".join(
                f"<tr><td>{label_2.get(o, o)}</td>"
                f"<td>{ngettext('%(count)s image', '%(count)s images', k) % {'count': k}}</td></tr>"
                for o, k in sorted(res["origenes"].items())
            )
            warning = ""
            if res["variacion"] > 1.05:
                warning = _(
                    "<p><strong>The scale is not shared by the whole batch:</strong> it varies %(var)s× between the nearest and the farthest photograph. That is why each image is converted with its own factor; using a single one for all of them would give sizes off by as much as %(pct)s%%.</p>"
                ) % {"var": f"{res['variacion']:.2f}", "pct": f"{100 * (res['variacion'] - 1):.0f}"}
            _mic_calib = _average_ic95(
                [c.um_por_px for c in cals.values() if getattr(c, "valid", False)]
            )
            _row_average = ""
            if _mic_calib:
                _average_c, _margin_c, _n_c = _mic_calib
                _row_average = (
                    f"<tr>{_('<td>mean ± 95% CI</td>')}"
                    f"<td>{_average_c:.4f} ± {_margin_c:.4f} (n={_n_c})</td></tr>"
                )
            # Each provenance gets only the caveat that applies to it: a run
            # calibrated by hand must not claim the scale was fitted to the
            # dish ring, and an index run must not claim it was measured here.
            scale_notes = ""
            if "placa" in res["origenes"]:
                scale_notes += _(
                    "<p>The length reference is the nominal outer diameter of the Petri dish. The radius in pixels comes from a least-squares circle fitted to the sampled ring edge in 720 directions, with outlier rejection; the Hough transform only supplies the approximate centre, because its radius can be off by 12&nbsp;% and that error would enter every reported size in full.</p>"
                )
                scale_notes += (
                    f"<p style='border-left:3px solid {T.DOC.INK3};padding:6px 12px;background:var(--bg_soft)'>"
                    + _(
                        "<strong>Uncertainty of the scale, and in which direction.</strong> The rim of the dish has a wall about <strong>%(pared)g&nbsp;mm</strong> thick (measured on the photographs of this study: the inner edge falls at 0.960 of the fitted radius and the outer edge at 1.000). The nominal diameter of a Petri dish is ambiguous at that level: it may refer to the <em>outer</em> diameter or to the <em>usable inner</em> one. Here the <strong>outer</strong> is taken, since that is the edge the circle is fitted to. If the nominal figure referred to the inner edge, the correct scale would be <strong>%(sesgo)g&nbsp;%% larger</strong> and every size in this report would be <strong>underestimated</strong> by that amount. The bias can only run in that direction, because the outer edge is the larger of the two possible ones."
                    )
                    % {"pared": THICKNESS_WALL_MM, "sesgo": BIAS_MAXIMUM_WALL_PCT}
                    + "</p>"
                )
            if "manual" in res["origenes"]:
                scale_notes += _(
                    "<p>The scale was entered by hand in Parameters. It is not traced to a length standard, so any variation in shooting distance between photographs goes uncorrected.</p>"
                )
            if "indice" in res["origenes"]:
                scale_notes += _(
                    "<p>The scale is inherited from the calibration index, which associates it with each crop or plate. It is a declared value, not one measured on the photograph.</p>"
                )
            calib_html = f"""
<table class='data'><tr>{_("<th>Source of the scale</th><th>Images</th>")}</tr>{rows_2}</table>
<table class='data'>
<tr>{_("<th>µm per pixel</th><th>Value</th>")}</tr>
<tr>{_("<td>minimum</td>")}<td>{res["min"]:.4f}</td></tr>
<tr>{_("<td>median</td>")}<td>{res["mediana"]:.4f}</td></tr>
<tr>{_("<td>maximum</td>")}<td>{res["max"]:.4f}</td></tr>
{_row_average}
</table>
{warning}
{scale_notes}
{_figure_calibration(run)}"""
            if res["avisos"]:
                items = "".join(f"<li>{a}</li>" for a in res["avisos"][:12] if a)
                if items:
                    calib_html += f"<p>{_('Issues during calibration:')}</p><ul>{items}</ul>"
    return calib_html


def _section_shape(run, shapes, n_en_el_aro, n_formless):
    """Body of the 'forma' section, plus the companion 'talla_carpetas' body.

    Returns ``(forma_html, por_carpeta_html)``, both WITHOUT their ``<h2>``
    titles (the assembler owns them; the ``{NUM}.N`` sub-header placeholders
    stay intact). ``por_carpeta_html`` is built in the same pass because it
    shares the per-class size grouping computed here, but it is its own
    optional section in the report (see SECCIONES) and never lands inside
    ``forma_html``. ``shapes`` arrives precomputed from
    :func:`polyx.gui.report.batch.compute_lot_forms` because the depth profile
    and the particle fichas read the same list.
    """
    record_elderly = None
    record_minor = None
    table_morphotype = tables = count_2 = method_2 = example = ""
    # A batch without measured particles has no forma body: with this
    # initialization the section stays empty and the assembler drops it,
    # instead of crashing with UnboundLocalError in the return below. Same
    # meaning as the por_carpeta_html one: it is its own optional section.
    shape_html = ""
    # Independent of `forma`: it is its own optional section in the report
    # (see SECCIONES below), so it does not enter forma_html.
    by_html_folder = ""

    if shapes:
        with_size = [(d, path_2) for d, path_2 in shapes if d.length_um]
        fibers = sum(1 for d, _unused in shapes if d.morphotype == "fibra")
        curves = sum(1 for d, _unused in shapes if d.curvature and d.curvature >= 1.15)
        pc_f = 100.0 * fibers / len(shapes)

        # ── Particles whose size warrants a look ──
        # It is declared in the section, not only in each card: whoever reads the size
        # figures has to know how many of them relied on a mask
        # that had to be cropped. It affects above all the large tail of the
        # distribution, which is precisely the one reported as `the largest
        # particle`.
        #
        # WATCH THE DIRECTION OF THE BIAS: while the mask was not cropped, the
        # leak inflated the size and the bias went UP. Now the crop
        # bounds it, so what is reported is the piece inside the box and the
        # bias goes DOWN. The text below says it in that direction.
        n_review = sum(1 for d, _unused in shapes if getattr(d, "review", False))
        warning_review = ""
        if n_review:
            # One decimal below 1%: with `:.0f` a proportion of 4 out of
            # 909 came out as `(0 %)` next to `4 particulas`, which reads as if
            # the figure were broken.
            _pc = 100.0 * n_review / len(shapes)
            _pc_txt = f"{_pc:.1f}" if _pc < 1 else f"{_pc:.0f}"
            warning_review = (
                f"<p style='border-left:3px solid {T.DOC.WARN};padding:6px 12px;"
                f"background:#fff8e6'>"
                + _(
                    "<strong>%(n)s of %(total)s particles (%(pct)s&nbsp;%%) have a size that needs checking</strong>: when segmenting them, the mask ran well beyond the detector's box, almost always because the threshold caught a neighbouring particle or a bright patch of background. In those cases the mask is <em>clipped to the box</em>, so the reported size is not inflated; what it may be is <em>short</em>, if the particle really did continue beyond the box. They are marked with ⚠ on their cards."
                )
                % {"n": n_review, "total": len(shapes), "pct": _pc_txt}
                + "</p>"
            )

        # ── What was set apart for falling on the ring ──
        if n_en_el_aro:
            from polyx.core.measure.calibration import FRACTION_AREA_USEFUL

            warning_review += (
                f"<p style='border-left:3px solid {T.DOC.INK3};padding:6px 12px;"
                f"background:var(--bg_soft)'>"
                + ngettext(
                    "<strong>%(n)s detection falls on the dish's rim</strong> —beyond %(pct)s %% of the fitted radius— and is left out of this section: the rim is a bright band the detector mistakes for material, and being huge it would displace the real particles at the large end. It is still counted in the counting section.",
                    "<strong>%(n)s detections fall on the dish's rim</strong> —beyond %(pct)s %% of the fitted radius— and are left out of this section: the rim is a bright band the detector mistakes for material, and being huge it would displace the real particles at the large end. They are still counted in the counting section.",
                    n_en_el_aro,
                )
                % {"n": n_en_el_aro, "pct": f"{100 * FRACTION_AREA_USEFUL:.0f}"}
                + "</p>"
            )

        # ── Distribution by morphotype ──
        table_morphotype = (
            warning_review
            + f"""
<table class='data'><tr>{_("<th>Morphotype</th><th>How many</th><th>%</th>")}</tr>
<tr>{_("<td>Fibre (aspect ratio ≥ 3)</td>")}<td>{fibers}</td><td>{pc_f:.1f} %</td></tr>
<tr>{_("<td>Particle (non-fibrous)</td>")}<td>{len(shapes) - fibers}</td><td>{100 - pc_f:.1f} %</td></tr>
</table>"""
        )

        if with_size:
            lengths = np.array([d.length_um for d, _unused in with_size], dtype=float)
            elderly, path_elderly = max(with_size, key=lambda t: t[0].length_um)
            minor_2, path_minor = min(with_size, key=lambda t: t[0].length_um)

            def _num(v, dec=0):
                """Cell with the value, or a dash when that magnitude was not measured."""
                return f"{v:.{dec}f}" if v else "—"

            def _row(label_3, d, path_2):
                return (
                    f"<tr><td><strong>{label_3}</strong></td><td>{_esc(d.class_name)}</td>"
                    f"<td>{_num(d.length_um)}</td><td>{_num(d.width_um)}</td>"
                    f"<td>{_num(d.area_um2)}</td><td>{_num(d.aspect, 1)}</td>"
                    f"<td>{_num(d.curvature, 2)}</td><td>{_morphotype(d.morphotype)}</td>"
                    f"<td style='font-size:9pt'>{_esc(Path(path_2).name)}</td></tr>"
                )

            tables = f"""
<h3>{NUM}.4 {_("Largest and smallest particle")}</h3>
<table class='data'>
<tr>{_("<th></th><th>Class</th><th>Length</th><th>Width</th><th>Area</th><th>Aspect</th><th>Curvature</th><th>Morphotype</th><th>Image</th>")}</tr>
{_row(_("Largest"), elderly, path_elderly)}
{_row(_("Smallest"), minor_2, path_minor)}
</table>

<h3>{NUM}.5 {_("Size distribution")}</h3>
<table class='data'>
<tr>{_("<th>Statistic</th><th>Length (µm)</th>")}</tr>
<tr>{_("<td>minimum</td>")}<td>{lengths.min():.0f}</td></tr>
<tr>{_("<td>10th percentile</td>")}<td>{np.percentile(lengths, 10):.0f}</td></tr>
<tr>{_("<td>median</td>")}<td>{np.median(lengths):.0f}</td></tr>
<tr>{_("<td>90th percentile</td>")}<td>{np.percentile(lengths, 90):.0f}</td></tr>
<tr>{_("<td>maximum</td>")}<td>{lengths.max():.0f}</td></tr>
</table>"""

            # ── Sizes by stretch, stacked by class ──
            per_class_2: dict[str, list[float]] = {}
            for d, _path in with_size:
                per_class_2.setdefault(d.class_name, []).append(d.length_um)
            fig = _fig_sizes_by_section(per_class_2)
            if fig:
                tables += f"<img src='data:image/png;base64,{fig}' style='max-width:100%'>"

            # ── Table by class, with its largest ──
            rows_cls = ""
            for cls in _ORDER_CLASSES + [c for c in per_class_2 if c not in _ORDER_CLASSES]:
                v = sorted(per_class_2.get(cls, []))
                if not v:
                    continue
                nf = sum(
                    1 for d, _r in with_size if d.class_name == cls and d.morphotype == "fibra"
                )
                rows_cls += (
                    f"<tr><td>{cls}</td><td>{len(v)}</td>"
                    f"<td>{v[0]:.0f}</td><td>{v[len(v) // 2]:.0f}</td>"
                    f"<td>{v[-1]:.0f}</td><td>{nf}</td></tr>"
                )
            if rows_cls:
                tables += (
                    f"<h3>{NUM}.6 {_('Size by polymer type')}</h3>"
                    + _(
                        "<table class='data'><tr><th>Class</th><th>n</th>"
                        "<th>smallest<br>(µm)</th><th>median<br>(µm)</th>"
                        "<th>largest<br>(µm)</th><th>Fibers</th></tr>"
                    )
                    + f"{rows_cls}</table>"
                )

            # The example card is drawn further below, AFTER explaining the
            # method: showing the crop with a yellow line on top before
            # saying that this line is the Feret diameter explains nothing.
            cal_elderly = run.calibration_for(Path(path_elderly))
            um_elderly = cal_elderly.um_por_px if cal_elderly and cal_elderly.valid else None
            record_elderly = (elderly, path_elderly, um_elderly)

            # The smallest one is shown too, not only tabulated: a minimum size
            # without a photo cannot be distinguished from noise, or the shape card
            # and a reader has no way to check that there really is a
            # particle there and not a background speck that was badly segmented.
            cal_minor = run.calibration_for(Path(path_minor))
            um_minor = cal_minor.um_por_px if cal_minor and cal_minor.valid else None
            record_minor = (minor_2, path_minor, um_minor)
            by_html_folder = _size_per_folder(with_size)

        # ── Fibre/fragment count per photo ──
        # It goes inside the forma section: it answers `how many fibres and how many
        # fragments are there on each plate`, which is the question asked of the
        # table when reviewing photo by photo.
        por_img: dict[str, list] = {}
        for d, path_2 in shapes:
            por_img.setdefault(Path(path_2).name, []).append(d)
        rows_img = ""
        for name_2 in sorted(por_img):
            ds = por_img[name_2]
            nf = sum(1 for d in ds if d.morphotype == "fibra")
            lengths_i = [d.length_um for d in ds if d.length_um]
            rows_img += (
                f"<tr><td style='font-size:9pt'>{_esc(name_2)}</td><td>{len(ds)}</td>"
                f"<td>{nf}</td><td>{len(ds) - nf}</td>"
                f"<td>{(f'{np.median(lengths_i):.0f}' if lengths_i else '—')}</td>"
                f"<td>{(f'{max(lengths_i):.0f}' if lengths_i else '—')}</td></tr>"
            )
        # The first column is renamed `Total` and not `Partículas`: with the
        # nomenclature change there would be two columns with the same name, one
        # as a sum and another as a morphotype. `Total = Fibras + Partículas`.
        count_2 = (
            f"<h3>{NUM}.7 {_('Per-image counts')}</h3>"
            f"<table class='data'><tr>"
            f"{_('<th>Image</th><th>Total</th><th>Fibres</th><th>Particles</th>')}"
            f"{_('<th>Median length<br>(µm)</th><th>Largest<br>(µm)</th>')}"
            f"</tr>{rows_img}</table>"
        )

        # An `—` in the size columns is not an analysis failure: the
        # particle WAS measured, in pixels, but without µm/px there is no way to
        # turn it into micrometres. Before, the dash appeared without explanation and
        # it looked as if the measurement had failed. It says what is missing and how
        # to fix it, instead of inventing a number or saying nothing.
        no_scale = [n for n, ds in por_img.items() if not any(d.length_um for d in ds)]
        if no_scale:
            all_value = len(no_scale) == len(por_img)
            which_2 = (
                _("<strong>No image has a scale</strong>")
                if all_value
                else _("<strong>%(n)s of %(total)s images have no scale</strong> (%(which)s)")
                % {
                    "n": len(no_scale),
                    "total": len(por_img),
                    "which": ", ".join(sorted(no_scale)[:6]) + (", …" if len(no_scale) > 6 else ""),
                }
            )
            count_2 += (
                "<p>"
                + which_2
                + _(
                    ", which is why their length shows as «—». The particles <em>were</em> measured —the counts and the shape columns are correct— but a measurement in pixels cannot be turned into micrometres without knowing how many µm each pixel is.</p>"
                )
                + _(
                    "<p>To get one, in <em>Parameters</em>: tick <strong>«Measure the Petri dish»</strong> —each photograph then gets its own scale from the dish's rim, which is the most reliable route— or type a <strong>µm/pixel</strong> value"
                )
                + (
                    "."
                    if all_value
                    else _(
                        ". If it is already on, then in those particular photographs the dish's rim was not found: check that it is fully inside the frame."
                    )
                )
                + "</p>"
            )

        # Outside the f-string: inside it we would have to nest quotes of the same type.
        pct_curves = f"{100.0 * curves / len(shapes):.1f}"
        method_2 = f"""
<h3>{NUM}.1 {_("How length is measured")}</h3>
{_("<p>Every magnitude is measured on <strong>each particle's mask</strong>, not on the detector's box. The box of an elongated particle is almost empty and depends on how it happened to land: a fibre lying diagonally has a square box, so measuring on the box would report it as if it were not a fibre, and with the wrong size. The mask is obtained by thresholding each crop halfway between the background level —the median of the ring surrounding the box— and the particle's own —the 90th percentile inside it—.</p>")}

{_("<p><strong>The general criterion is the longest straight line that fits inside the particle</strong>, that is the greatest distance between two points of its outline: the <em>maximum Feret diameter</em>. For a particle of irregular but unbent shape, that line is its size, and it has two properties that make it reliable: it does not depend on the orientation the particle happened to land in, and a jagged edge does not disturb it.</p>")}

{_("<p><strong>That straight line stops working when the particle is contorted.</strong> In a bent fibre the distance between its ends is the chord, not its length: on a half-circle arc it falls 35&nbsp;% short. For those cases the <em>geodesic diameter</em> is measured, which is the longest path that fits <strong>inside</strong> the particle. Since it cannot leave the mask, that path follows the curve and returns the distance travelled.</p>")}

<table class='data'>
<tr>{_("<th>Particle shape</th><th>What is reported as length</th>")}</tr>
<tr>{_("<td>Compact or irregular, but not bent</td><td>Maximum Feret (the longest straight line)</td>")}</tr>
<tr>{_("<td>Elongated and contorted (fibre)</td><td>Geodesic diameter (follows the curve)</td>")}</tr>
</table>

{_("<p>The distinction is made by <strong>thickness</strong>: the geodesic is used only if the length exceeds at least four times the particle's maximum inscribed thickness. The reason is that in a thick particle any concavity forces the geodesic path to skirt around it instead of crossing it, and the number is then inflated; in a thin fibre that cannot happen.</p>")}

{_("<p>Checked against shapes of known size —straight bars, bars rotated 45&nbsp;°, arcs of 60, 120 and 180&nbsp;°, a circle and a bar with a jagged edge— the length measured this way gives a <strong>median error of 0.8&nbsp;% and 4.7&nbsp;% in the worst case</strong>.</p>")}

{_("<p><strong>%(n)s particles (%(pct)s %%) are contorted</strong>, meaning that their length exceeds their straight-line extent by more than 15&nbsp;%%.</p>") % {"n": curves, "pct": pct_curves}}

{_("<p><em>Touching particles.</em> Two particles in contact form a single blob, and measuring them together would add their sizes up. They are separated by <em>watershed</em> on the distance transform: the centre of each one sits far from the background while the neck joining them sits close, so the cut falls on the neck. Against circles of known size it separates them up to 27&nbsp;% overlap of the diameter, without splitting any particle that is a single piece.</p>")}

{_("<p><em>Known limitations.</em> Two particles overlapping by more than 40&nbsp;% of their diameter are still measured as one: past that point there is no neck left to cut along. And in a tightly coiled fibre the geodesic path cuts the corner at each bend, underestimating the length by up to 19&nbsp;% in the tightest case tested.</p>")}"""
        if n_formless:
            method_2 += _(
                "<p>In %(n)s particles the particle could not be separated from the background; their size comes from the box and is not comparable with the rest.</p>"
            ) % {"n": n_formless}

        # ── The examples, once the method is explained ──
        # Both, not only the largest: a minimum size without a photo cannot be
        # distinguished by eye from badly segmented noise, and it is the one most easily
        # questioned in a review.
        if record_elderly is not None:
            example = _record_particle(
                *record_elderly, f"2 {_('An example: the largest particle, measured')}"
            )
        if record_minor is not None:
            example += _record_particle(
                *record_minor, f"3 {_('An example: the smallest particle, measured')}"
            )

        # Section order: first how it is measured, then a measured case, and
        # finally the figures. The other way round, the tables arrive before anything has
        # explained what `largo` means here.
        shape_html = method_2 + example + table_morphotype + tables + count_2
    return shape_html, by_html_folder


def _section_records(run, shapes: list[tuple], max_records: int = 12) -> str:
    """Body of the 'fichas' section: one annotated card per sampled particle.

    Returns the HTML body WITHOUT its ``<h2>`` title: the assembler owns the
    heading and the section number. Empty when there are no measured particles
    or the cap is zero. The ``cv2`` / ``morfologia`` / ``calibracion`` imports
    stay lazy inside this body, exactly as lazy as when the block was inline
    (S01 fresh-interpreter laziness gate).
    """
    # One entry per particle, with its number, its crop and the measurement drawn
    # on top. That is what makes it possible to check particle by particle that the size
    # reported corresponds to what is in the photo, instead of having to
    # trust the table.
    records_html = ""
    if shapes and max_records:
        import cv2

        from polyx.core.measure.drawing import draw_measurement

        chosen = choose_records(shapes, max_records)
        n_fibers_sample = sum(1 for d, _unused in chosen if d.morphotype == "fibra")
        por_img_f: dict[str, list] = {}
        for d, path_2 in chosen:
            por_img_f.setdefault(str(path_2), []).append(d)
        parts, made = [], 0
        for path_2 in sorted(por_img_f):
            bgr = None
            try:
                from polyx.core.measure.calibration import read_image

                bgr = read_image(path_2)
            except Exception:  # noqa: BLE001
                bgr = None
            if bgr is None:
                continue
            cal = run.calibration_for(Path(path_2))
            um = cal.um_por_px if cal and cal.valid else None
            cards = ""
            for d in sorted(por_img_f[path_2], key=lambda z: z.number or 0):
                if made >= max_records:
                    break
                img, m = draw_measurement(bgr, d)
                if img is None or m is None or not m.ok:
                    continue
                ok, buf = cv2.imencode(".png", img)
                if not ok:
                    continue
                made += 1
                account = ""
                if d.length_um and um:
                    account = (
                        f"{m.length_px:.1f} px &times; {um:.4f} = "
                        f"<strong>{d.length_um:.0f} µm</strong>"
                    )
                # d.aspecto/morfotipo are filled by segmentation alone (they do not need
                # calibration), but d.ancho_um only if the photo had a valid um_por_px
                # when it was measured -- a photo with no detected plate leaves
                # aspecto with a value and ancho_um at None. Without this guard, that
                # combination crashes the `:.0f` format on None.
                detail = (
                    [_("width %(width)s µm") % {"width": f"{d.width_um:.0f}"}] if d.width_um else []
                )
                if d.aspect:
                    detail.append(_("aspect %(aspect)s") % {"aspect": f"{d.aspect:.1f}"})
                detail.append(_("measured by %(method)s") % {"method": method_label(m.method)})
                # If an appreciable part of the line falls outside, it is stated with a
                # number. Below 5% it is contour discretization and
                # saying it would only add noise to the card.
                if getattr(m, "feret_outside", 0) > 0.05:
                    detail.append(
                        _("%(pct)s %% of the line outside the outline")
                        % {"pct": f"{100 * m.feret_outside:.0f}"}
                    )
                color = "#1f6b5e" if d.morphotype == "fibra" else "#656d76"
                label_morph = _morphotype(d.morphotype, lowercase=True)
                # The warning is SHOWN. Before, it was computed and thrown away: a size
                # inflated by several particles in contact reached the report
                # as a clean number, and whoever read it had no way to know.
                # It is read from the Detection, NOT from the Morfologia returned by
                # dibujar_medicion: that one comes from medir(), and the warning is set by
                # medir_deteccion() when measuring against the box diagonal. In
                # the Morfologia here the field is always False, so
                # looking at it meant the `⚠` never appeared.
                alerta = ""
                if getattr(d, "review", False):
                    alerta = (
                        f"<div style='font-size:8.5pt;color:{T.DOC.WARN};margin-top:3px'>"
                        f"⚠ {d.warning_shape or _('the mask ran outside the box and was clipped to it')}</div>"
                    )
                cards += (
                    f"<div style='display:inline-block;vertical-align:top;margin:0 14px 18px 0;"
                    f"max-width:290px'>"
                    f"<div style='font-weight:700'>#{d.number} · {_esc(d.class_name)} · "
                    f"<span style='color:{color}'>{label_morph}</span></div>"
                    f"<img src='data:image/png;base64,{_img_b64(buf.tobytes())}' style='max-width:100%'>"
                    f"<div style='font-size:9pt;color:#656d76'>{account}<br>"
                    f"{' · '.join(detail)}</div>{alerta}</div>"
                )
            if cards:
                parts.append(f"<h3>{_esc(Path(path_2).name)}</h3>{cards}")
        if parts:
            # It is declared HOW the sample was chosen. Without this, a reader could
            # take it as representative of the batch and it is not: it is deliberately biased
            # towards the large ones and towards fibres.
            n_others_sample = made - n_fibers_sample
            if n_fibers_sample:
                criterion = ngettext(
                    "The <strong>%(fibres)s fibre and the %(others)s largest particles</strong> are shown",
                    "The <strong>%(fibres)s fibres and the %(others)s largest particles</strong> are shown",
                    n_fibers_sample,
                ) % {"fibres": n_fibers_sample, "others": n_others_sample}
            else:
                criterion = _(
                    "The <strong>%(others)s largest particles</strong> are shown. No fibre was detected in this batch, so the sample includes none"
                ) % {"others": n_others_sample}
            records_html = (
                _(
                    "<p>Each particle with the number it carries in the annotated image, its crop and the measurement drawn <strong>on it</strong>: in yellow the Feret line, in magenta the geodesic path, whichever of the two decided its size. The green outline is the mask that was measured. On the left the particle without marks, so you can judge whether the outline follows it.</p>"
                )
                + _(
                    "<p>On a <strong>concave</strong> particle the yellow line crosses outside the outline, and that is correct: Feret is the separation of two caliper jaws, not a path through the particle. So that this is visible, the stretch that falls outside is <strong>dashed</strong>. The path that does stay inside is the geodesic, in magenta, used when the particle is thin and bent.</p>"
                )
                + "<p>"
                + criterion
                + _(
                    ". The split is deliberate: fibres are the case where the geodesic method comes into play —and they are a minority, so a sample taken at random might show none— and the largest of each type is what any claim about maximum size rests on. <strong>It is not a representative sample of the batch</strong>, but the selection that lets the method be checked where it can fail most.</p>"
                )
                + "".join(parts)
            )
    return records_html


def section_depth(shapes, results_2) -> str:
    """How many particles and of which size sit at each depth of the core.

    The interval is not just any folder: it is an ORDERED variable. That
    makes it possible to ask about TRENDS — whether there are more
    particles at the bottom than at the top, or whether the bottom ones
    are bigger — and not only whether the groups differ, which is all the
    generic per-folder comparison can answer.

    ``formas`` is ``[(Detection, ruta)]`` with a measured size;
    ``resultados`` are the ImageResult records, needed to count ALL the
    particles and not only the ones that could be measured in
    micrometers.

    Returns "" when the names do not follow the ``tramo.testigo``
    pattern: without an interval there is no depth, and the report cannot
    demand that naming.
    """
    # -- Count by stretch, summing the plates of the same stretch --
    # They are summed because they are subsamples of the same sediment mass, not
    # photographic replicates. It is the same rule as the count section.
    count: dict[tuple, int] = {}
    plates: dict[tuple, set] = {}
    for r in results_2:
        m = _sample_of(r.image_path)
        if m is None:
            continue
        station, range_value, plate = m
        count[(station, range_value)] = count.get((station, range_value), 0) + len(r.predictions)
        plates.setdefault((station, range_value), set()).add(plate)

    # -- Sizes by stretch --
    sizes_2: dict[tuple, list[float]] = {}
    for d, path_2 in shapes:
        m = _sample_of(path_2)
        if m is None or not d.length_um:
            continue
        sizes_2.setdefault((m[0], m[1]), []).append(float(d.length_um))

    if len(count) < 3:
        # With one or two stretches there is no profile to draw or trend to
        # measure; saying so is more useful than a two-point figure.
        return ""

    profiles: dict[str, list] = {}
    for (station, range_value), n in sorted(count.items()):
        vals = np.asarray(sizes_2.get((station, range_value), []), dtype=float)
        if len(vals):
            median_2 = float(np.median(vals))
            q1, q3 = (float(np.percentile(vals, 25)), float(np.percentile(vals, 75)))
        else:
            median_2 = q1 = q3 = float("nan")
        profiles.setdefault(station, []).append((range_value, n, median_2, q1, q3))

    fig = _fig_profile_depth(profiles)

    # -- Trend, by station --
    # n is the number of STRETCHES, not of particles: two particles from the same
    # plate are not independent observations of depth, and counting them
    # as such would inflate any significance. With few stretches the
    # correlation is weak by design, and that is why n is declared.
    paragraphs = []
    for station, rows_2 in sorted(profiles.items()):
        rows_2 = sorted(rows_2)  # noqa: PLW2901
        x = [f[0] for f in rows_2]
        n_part = [f[1] for f in rows_2]
        med = [f[2] for f in rows_2]
        with_size = [(a, b) for a, b in zip(x, med) if b == b]  # discards NaN  # noqa: B905
        parts = [_text_trend(x, n_part, _("the number of particles"))]
        if len(with_size) >= 3:
            parts.append(
                _text_trend(
                    [a for a, _unused in with_size],
                    [b for _unused, b in with_size],
                    _("the median size"),
                )
            )
        parts = [p for p in parts if p]
        if parts:
            paragraphs.append(f"<p><strong>{_esc(station)}.</strong> " + " ".join(parts) + "</p>")

    # -- Table --
    rows_table = ""
    for station, rows_2 in sorted(profiles.items()):
        for range_value, n, median_2, _q1, _q3 in sorted(rows_2):
            vals = sizes_2.get((station, range_value), [])
            # A stretch can have counted particles and none with a size, if
            # that photo was never calibrated. The count is shown and a dash in
            # the size columns, which is what actually happened.
            cells = [
                _esc(station),
                str(range_value),
                str(len(plates.get((station, range_value), ()))),
                str(n),
                str(len(vals)),
                f"{median_2:.0f}" if vals else "—",
                f"{max(vals):.0f}" if vals else "—",
            ]
            rows_table += (
                "<tr><td>"
                + cells[0]
                + "</td>"
                + "".join(f"<td class='r'>{c}</td>" for c in cells[1:])
                + "</tr>"
            )

    return (
        _(
            "<p>The interval is the <strong>depth</strong> the sediment was taken from and, unlike an arbitrary folder, it is <em>ordered</em>. That makes it possible to ask about a trend —whether there are more particles deeper down, or whether the deeper ones are larger— and not merely whether the groups differ.</p>"
        )
        + (f"<img src='data:image/png;base64,{fig}' style='max-width:100%'>" if fig else "")
        + "".join(paragraphs)
        + _(
            "<p class='caption' style='text-align:left'>The correlation is computed over the <strong>intervals</strong>, not over the particles: two particles from the same dish are not independent observations of depth, and counting them as such would inflate any significance. The p value comes from shuffling the series 20,000 times, not from an approximation that would not hold with so few intervals.</p>"
        )
        + f"<table class='data'><tr>{_('<th>Station</th><th>Interval</th><th>Dishes</th><th>Particles</th><th>With size</th><th>Median (µm)</th><th>Largest (µm)</th>')}</tr>{rows_table}</table>"
    )


def _size_per_folder(with_size):
    """Body of the 'talla_carpetas' section: size by folder, then by photo.

    Returns the HTML body WITHOUT its ``<h2>`` title, with the ``{NUM}.N``
    sub-header placeholders intact. Empty when fewer than two folders hold
    measured particles, so the assembler drops the section like it does
    today.
    """
    # ── Size by folder (`{_('Size by folder')}`), and by photo within each folder ──
    # Generalizes the `by station/stretch` contrast from the paper
    # pipeline (which depended on a filename pattern specific
    # to one study) to the only thing ANY batch has: which
    # folder each photo came from. It is useful for contrasting sampling sites,
    # treatments, or any other grouping that the user has
    # expressed as folders when choosing the images.
    by_html_folder = ""
    groups_folder: dict[str, list[float]] = {}
    for d, path_2 in with_size:
        folder = Path(path_2).parent.name or str(Path(path_2).parent)
        groups_folder.setdefault(folder, []).append(d.length_um)

    if len(groups_folder) >= 2:
        fig_folder = _fig_boxplot_groups(groups_folder, _("Size by folder"), _("source folder"))
        verdict_folder = _text_kruskal(
            _kruskal_wallis(list(groups_folder.values())), len(groups_folder)
        )
        by_html_folder = (
            f"<h3>{NUM}.1 {_('Size by folder')}</h3>"
            + _(
                "<p>Compares size across the folders of the analysed batch, each folder as a separate group. Useful when each folder is a sampling site, a station or a condition.</p>"
            )
            + f"<img src='data:image/png;base64,{fig_folder}' style='max-width:100%'>"
            f"<p>{verdict_folder}</p>"
        )

        # -- Per photo, within folders with more than one image --
        parts_photo = []
        for folder in sorted(groups_folder, key=lambda c: -len(groups_folder[c])):
            if len(parts_photo) >= _MAX_FOLDER_PER_PHOTO:
                break
            folder_photos: dict[str, list[float]] = {}
            for d, path_2 in with_size:
                if Path(path_2).parent.name == folder:
                    folder_photos.setdefault(Path(path_2).name, []).append(d.length_um)
            if len(folder_photos) < 2:
                continue
            fig_photo = _fig_boxplot_groups(
                folder_photos,
                _("Size per photograph — %(folder)s") % {"folder": folder},
                _("photograph"),
                max_groups=20,
            )
            verdict_photo = _text_kruskal(
                _kruskal_wallis(list(folder_photos.values())), len(folder_photos)
            )
            parts_photo.append(
                f"<p><strong>{_esc(folder)}</strong> "
                + _("(%(n)s photographs)") % {"n": len(folder_photos)}
                + "</p>"
                f"<img src='data:image/png;base64,{fig_photo}' style='max-width:100%'>"
                f"<p>{verdict_photo}</p>"
            )
        if parts_photo:
            n_folders_with_photos = sum(
                1
                for c in groups_folder
                if len({Path(r).name for d, r in with_size if Path(r).parent.name == c}) >= 2
            )
            note_tope = (
                _(
                    " The %(n)s folders with the most particles are shown, out of the %(total)s that have more than one photograph."
                )
                % {"n": _MAX_FOLDER_PER_PHOTO, "total": n_folders_with_photos}
                if n_folders_with_photos > _MAX_FOLDER_PER_PHOTO
                else ""
            )
            by_html_folder += (
                f"<h3>{NUM}.2 {_('Size by photo, within each folder')}</h3>"
                + _(
                    "<p>Compares size across the individual photographs of one folder: for instance, whether size changes with depth or with the sampling moment within the same site.%(note)s</p>"
                )
                % {"note": note_tope}
                + "".join(parts_photo)
            )
    return by_html_folder


# ────────────────────────────────────────────────────────────────────
# ── Section builders ─────────────────────────────────────────────────────
# Every section id in the catalogue has one builder fed by a ReportContext.
# The assembler owns the id -> builder mapping and the section order; the
# builders only turn their slice of the context into the section body, without
# the <h2> title (the assembler owns the heading and the number).


def compute_shape_bodies(run: RunResult, lot: LotForms) -> ShapeBodies:
    """Build the shape section's two bodies once, from the named precomputes.

    ``forma`` and ``talla_carpetas`` are separate sections, but the shape pass
    computes the per-class size grouping they both need. Running it once here
    keeps the two builders from repeating that work and from disagreeing about
    the same batch.
    """
    forma, by_folder = _section_shape(run, lot.shapes, lot.ring_count, lot.formless_count)
    return ShapeBodies(forma=forma, by_folder=by_folder)


def build_summary(ctx: ReportContext) -> str:
    """Body of the ``resumen`` section."""
    return _section_summary(
        ctx.batch.image_count,
        ctx.batch.detection_count,
        ctx.batch.mean_confidence,
        ctx.batch.mean_size_um,
        list(ctx.run.models),
        ctx.batch.has_gt,
    )


def build_methods(ctx: ReportContext) -> str:
    """Body of the ``metodos`` section: table, team and redacted paragraph."""
    methods_html, team_html, methods_for = _section_methods(
        ctx.run,
        list(ctx.run.models),
        ctx.batch.all_results,
        ctx.batch.image_count,
        ctx.batch.detection_count,
        ctx.batch.has_gt,
    )
    return f"""
<table class='data'><tr>{_("<th>Parameter</th><th>Value</th>")}</tr>{methods_html}</table>
{team_html}
{methods_for}"""


def build_calibration(ctx: ReportContext) -> str:
    """Body of the ``calibracion`` section."""
    return _section_calibration(ctx.run)


def build_shape(ctx: ReportContext) -> str:
    """Body of the ``forma`` section, read from the shared shape precompute."""
    return ctx.shape_bodies.forma


def build_size_by_folder(ctx: ReportContext) -> str:
    """Body of the ``talla_carpetas`` section, from the same precompute."""
    return ctx.shape_bodies.by_folder


def build_depth(ctx: ReportContext) -> str:
    """Body of the ``profundidad`` section; empty when the names carry no depth."""
    if not ctx.lot.shapes:
        return ""
    return section_depth(ctx.lot.shapes, ctx.batch.all_results)


def build_records(ctx: ReportContext) -> str:
    """Body of the ``fichas`` section."""
    return _section_records(ctx.run, ctx.lot.shapes, ctx.max_records)


def build_results(ctx: ReportContext) -> str:
    """Body of the ``resultados`` section."""
    return _section_results(ctx.batch.per_class, ctx.batch.confidences, ctx.batch.sizes_um)


def build_models(ctx: ReportContext) -> str:
    """Body of the ``modelos`` section."""
    return _section_models(
        ctx.run, ctx.batch.results_by_model, ctx.batch.all_results, ctx.batch.has_gt
    )


def build_errors(ctx: ReportContext) -> str:
    """Body of the ``errores`` section."""
    return _section_errors(ctx.run, ctx.batch.results_by_model, ctx.batch.has_gt)


def build_comparison(ctx: ReportContext) -> str:
    """Body of the ``comparacion`` section: verdict plus per-photo comparison."""
    verdict_html = _section_verdict(ctx.batch.results_by_model, ctx.run)
    compare_html = _section_comparison(ctx.run, ctx.batch.results_by_model)
    return f"{verdict_html}\n{compare_html}"


def build_gallery(ctx: ReportContext) -> str:
    """Body of the ``galeria`` section."""
    return _section_gallery(
        ctx.batch.results_by_model,
        ctx.run,
        ctx.batch.classes,
        max_gallery=ctx.max_gallery,
    )


def build_count(ctx: ReportContext) -> str:
    """Body of the ``conteo`` section."""
    return _section_count(ctx.batch.results_by_model, ctx.run, ctx.batch.classes)


def build_references(ctx: ReportContext) -> str:
    """Body of the ``referencias`` section.

    ``include_refs`` is resolved by the assembler, which drops the section id
    from the requested set; a requested reference builder always renders the
    bibliography.
    """
    return _section_references(include_refs=True)
