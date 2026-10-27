"""Sections."""

import sys

from ._common import *
from .stats import *
import polyx.gui.report._common as _common

globals().update({k: v for k, v in vars(_common).items() if not k.startswith("__")})
from .figures import *  # noqa: E402
import polyx.gui.report.figures as _figures  # noqa: E402

globals().update({k: v for k, v in vars(_figures).items() if not k.startswith("__")})

import polyx.gui.report.stats as _stats  # noqa: E402

globals().update({k: v for k, v in vars(_stats).items() if not k.startswith("__")})


def _section_team(active) -> str:
    """Table of components and of how each loaded model was trained."""
    rows_eq = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
        for k, v in sys.modules.get("polyx.gui.report_html", sys.modules[__name__])._team()
    )
    html = (
        f"<h3>@@N@@.1 {tr('Equipo de cómputo')}</h3>"
        f"<table class='data'><tr>{tr('<th>Componente</th><th>Detalle</th>')}</tr>"
        f"{rows_eq}</table>"
    )

    training_runs = []
    for s in active:
        if s.path is None:
            continue
        info = sys.modules.get("polyx.gui.report_html", sys.modules[__name__])._training_of(
            Path(s.path)
        )
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
        f"<h3>@@N@@.2 {tr('Entrenamiento de cada modelo')}</h3>"
        + tr(
            "<p>Configuración y métricas de validación con que se entrenó cada peso, "
            "leídas del propio archivo <code>.pt</code>.</p>"
        )
        + "<table class='data'>"
        + tr(
            "<tr><th>Modelo</th><th>Arquitectura base</th><th>imgsz</th><th>batch</th>"
            "<th>épocas</th><th>optimizador</th><th>Precisión</th><th>Recall</th>"
            "<th>mAP@50</th><th>mAP@50-95</th></tr>"
        )
        + f"{rows_2}</table>"
    )
    return html


def _section_count(results_2: dict[int, list], state, active, classes: list[str]) -> str:
    """Section 8: how many particles of each polymer there are per sample.

    It is the table requested for the manuscript: count by plastic type
    and sample, with the manual count and the model count one next to the
    other. **Not** an evaluation of the detector: the model column is all
    of its detections, without discounting false positives. That reading
    lives in the error section.

    With several models loaded the first active one is used. Putting two
    models in the same cell would make the table illegible, and the
    report already compares models in its own section.
    """
    if not active:
        return ""
    main_mi = state.model_slots.index(active[0])
    rs = results_2.get(main_mi, [])
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
                f"{_esc(station)} &middot; tramo {_esc(range_value)} &middot; placa {_esc(plate)}</span>"
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
            palabra = tr("imágenes") if len(items_2) != 1 else tr("imagen")
            rows_t.append(
                (
                    f"{_esc(station)} &middot; tramo {_esc(range_value)} "  # noqa: ISC004
                    f"<span style='font-size:8.5pt;color:var(--ink3)'>"
                    f"({len(items_2)} {palabra})</span>",
                    man,
                    det,
                )
            )
        table_range = (
            "<h3>@@N@@.2 "
            + tr("Por tramo de profundidad")
            + "</h3>"
            + tr(
                "<p>Las placas del mismo tramo se <strong>suman</strong>: son "
                "submuestras de la misma masa de sedimento, no repeticiones "
                "fotográficas. El tramo es la unidad de análisis.</p>"
            )
            + _table_width(rows_t, classes, tr("Tramo"), there_is_a_manual)
        )

        rows_e = []
        for station in sorted(per_station, key=_order_est):
            man, det = _add(per_station[station])
            rows_e.append((_esc(station), man, det))
        table_station = (
            "<h3>@@N@@.3 "
            + tr("Por estación")
            + "</h3>"
            + _table_width(rows_e, classes, tr("Estación"), there_is_a_manual, totals=False)
        )

        outside = len(grouped) - len(parseable)
        if outside:
            note_grouping = (
                "<p class='caption'>"
                + tr(
                    "{n} imagen(es) no siguen la nomenclatura <code>tramo.testigo</code> y "
                    "quedan fuera de las tablas agrupadas; sí están en la tabla por "
                    "imagen."
                ).format(n=outside)
                + "</p>"
            )

    tagline = tr(", el primer modelo activo.") if len(active) > 1 else "."
    intro = (
        tr(
            "<p>Partículas contadas en cada muestra, desglosadas por polímero. "
            "La columna <em>manual</em> es la anotación humana (Ground Truth) y la "
            "columna <em>modelo</em> son todas las detecciones de "
            "<strong>{alias}</strong>{coletilla}</p>"
        ).format(alias=_esc(active[0].alias), coletilla=tagline)
        + tr(
            "<p><strong>Léase como conteo, no como evaluación.</strong> La "
            "columna del modelo no descuenta falsos positivos ni empareja caja a caja: "
            "es cuántas partículas de cada polímero reportó. Coincidir en el total no "
            "implica haber acertado partícula por partícula; para eso está el análisis "
            "de errores.</p>"
        )
        + "<h3>@@N@@.1 "
        + tr("Por imagen")
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
    # and nesting them inside {tr(...)} would require quotes of the same type.
    _summary_paragraph = tr(
        "Se analizaron <strong>{imgs} imágenes</strong> con <strong>{n} {modelos}</strong> "
        "YOLO entrenado para detectar microplásticos de PET, PP y LDPE bajo "
        "fluorescencia Nile Red (254 nm). El total de detecciones fue "
        "<strong>{dets}</strong> con una confianza media de <strong>{conf}</strong>."
    ).format(
        imgs=total_imgs,
        n=len(active),
        modelos=tr("modelos") if len(active) != 1 else tr("modelo"),
        dets=total_dets,
        conf=f"{avg_conf:.3f}",
    )

    return f"""
<div class='kpis'>
  <div class='kpi'><div class='l'>{tr("Imágenes")}</div><div class='v'>{total_imgs}</div><div class='b' style='background:var(--accent)'></div></div>
  <div class='kpi'><div class='l'>{tr("Detecciones")}</div><div class='v'>{total_dets}</div><div class='b' style='background:var(--accent)'></div></div>
  <div class='kpi'><div class='l'>{tr("Conf. media")}</div><div class='v'>{avg_conf:.3f}</div><div class='b' style='background:var(--vio)'></div></div>
  <div class='kpi'><div class='l'>{tr("Tamaño medio (μm)")}</div><div class='v'>{(f"{avg_size:.1f}" if avg_size else "—")}</div><div class='b' style='background:var(--warn)'></div></div>
</div>
<p>{_summary_paragraph}
{tr("Se incluyó análisis de errores con Ground Truth (Verdaderos Positivos, Falsos Positivos, Falsos Negativos y Mal Clasificados).") if any_gt else tr("No se aportó Ground Truth, por lo que no se reportan métricas de error.")}
</p>"""


def _section_errors(state, results_2, active, any_gt: bool) -> str:
    """Body of the 'errores' section: confusion matrix plus per-class metrics.

    Returns the HTML body WITHOUT its ``<h2>`` title. Empty when there is no
    ground truth or no active model: the assembler omits a section with no
    content rather than inventing data.
    """
    if not (any_gt and active):
        return ""
    main_mi = state.model_slots.index(active[0])
    rs = results_2.get(main_mi, [])
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
    cm = confusion_matrix(preds_only, gts_only, cls_ids, iou_thr=state.params.iou_tp)
    per_class_metrics = aggregate_per_class(
        preds_only, gts_only, cls_ids, iou_thr=state.params.iou_tp
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
    known = dict(getattr(getattr(active[0], "loaded", None), "names", {}) or {})
    missing = [n for cid, n in sorted(known.items()) if cid not in set(cls_ids)]
    note_missing = ""
    if missing:
        note_missing = (
            "<p class='caption' style='text-align:left'>"
            + tr(
                "Sin fila para {clases}: no aparece ni en la anotación manual ni "
                "entre las predicciones de este lote, de modo que no hay métrica "
                "que informar. El modelo sí está entrenado para esa clase."
            ).format(clases=_esc(", ".join(missing)))
            + "</p>"
        )

    return f"""
        
        <h3>@@N@@.1 {tr("Matriz de confusión")}</h3>
        <div class='fig'><img src='data:image/png;base64,{cm_img}' />
            <div class='caption'>{tr("Figura. Matriz de confusión (modelo principal: {modelo}, IoU = {iou}).").format(modelo=_esc(active[0].alias), iou=state.params.iou_tp)}</div></div>
        <h3>@@N@@.2 {tr("Precisión / Recall / F1 por clase")}</h3>
        <table class='data'><tr>{tr("<th>Clase</th>")}<th>True Positives</th><th>False Positives</th><th>False Negatives</th>
        {tr("<th>Precisión</th><th>Recall</th><th>F1</th>")}</tr>{rows}</table>
        {note_missing}
        """


def _section_models(state, results_2, all_results, any_gt: bool) -> str:
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
    for mi, slot in enumerate(state.model_slots):
        if slot.path is None:
            continue
        rs = results_2.get(mi, [])
        n_img = len({r.image_path for r in rs})
        n_det = sum(len(r.predictions) for r in rs)
        cf = [p.conf for r in rs for p in r.predictions]
        tp = sum(r.tp for r in rs)
        fp = sum(r.fp for r in rs)
        fn = sum(r.fn for r in rs)
        mc = sum(r.miscls for r in rs)
        total_mc += mc
        _, _, f1_loc = _pr_f1(tp, fp, fn)
        _, _, f1_cls = _pr_f1(tp, fp + mc, fn + mc)
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
        note_miscls = tr(
            "<p><strong>Los dos F1 miden cosas distintas.</strong> "
            "<em>Localización</em> responde si el detector encuentra la partícula "
            "(P {pl} · R {rl} · <strong>F1 {fl}</strong>). "
            "<em>Con clase</em> exige además acertar el polímero, contando cada "
            "caja mal clasificada como falso positivo de la clase predicha y "
            "falso negativo de la real "
            "(P {pc} · R {rc} · <strong>F1 {fc}</strong>).</p>"
            "<p>La diferencia corresponde a <strong>{mc}</strong> "
            "partícula(s) bien localizada(s) pero asignada(s) a la clase "
            "incorrecta. Es la cifra que concilia esta tabla con la de "
            "precisión por clase de la sección de errores.</p>"
        ).format(
            pl=f"{p_loc:.3f}",
            rl=f"{r_loc:.3f}",
            fl=f"{f_loc:.3f}",
            pc=f"{p_cls:.3f}",
            rc=f"{r_cls:.3f}",
            fc=f"{f_cls:.3f}",
            mc=total_mc,
        )

    return f"""
<table class='data'><tr>{tr("<th>Modelo</th><th>Imágenes</th><th>Detecciones</th><th>Conf. media</th>")}
<th>True Positives</th><th>False Positives</th><th>False Negatives</th>
<th>Misclassified</th><th>F1<br><span style='font-weight:400;font-size:8.5pt'>{tr("localización")}</span></th>
<th>F1<br><span style='font-weight:400;font-size:8.5pt'>{tr("con clase")}</span></th></tr>{rows_models}</table>
{note_miscls}"""


def _section_verdict(results_2, state) -> str:
    """Body of the 'veredicto' section: best model by F1-with-class, from the
    confidence sweep. Arguments are the precomputed locals the inline block
    consumed before extraction."""
    text_2 = ""
    if any(r.has_gt for rs in results_2.values() for r in rs):
        sweep = _sweep_confidence(results_2, state, state.params.iou_tp, state.params.conf)
        if sweep:
            summary = []
            for alias, rows_2 in sweep.items():
                best = max(rows_2, key=lambda f: f[3])
                current = rows_2[0]
                summary.append((alias, current, best))

            # Verdict: the winner is whoever has the best F1 with class at their best threshold.
            summary.sort(key=lambda x: x[2][3], reverse=True)
            g_alias, g_act, g_best = summary[0]  # noqa: RUF059
            text_2 = tr(
                "<p><strong>Mejor desempeño: {alias}</strong>, con F1 {f1} "
                "al umbral {umbral} (P {p} · R {r}).</p>"
            ).format(
                alias=_esc(g_alias),
                f1=f"{g_best[3]:.3f}",
                umbral=f"{g_best[0]:.2f}",
                p=f"{g_best[1]:.3f}",
                r=f"{g_best[2]:.3f}",
            )
            if len(summary) > 1:
                p_alias, _, p_best = summary[-1]
                d = g_best[3] - p_best[3]
                text_2 += tr(
                    "<p>La diferencia con {otro} es de <strong>{d}</strong> de F1. Con "
                    "un solo entrenamiento por arquitectura, una diferencia pequeña no "
                    "distingue el diseño de la red del azar de inicialización: haría "
                    "falta repetir con distintas semillas para afirmar que una es "
                    "superior.</p>"
                ).format(otro=_esc(p_alias), d=f"{d:.3f}")

            return text_2
    return text_2


def _section_comparison(state, results_2) -> str:
    """Body of the 'comparacion' section: per-photo model table, the F1 of
    localization note and the comparison figures. Arguments are the
    precomputed locals the inline block consumed before extraction."""
    # ── Models side by side, photo by photo ──
    # The global table above says which model wins overall; this one says in
    # which photos it wins. With uneven batches like those from the Loa, where one stretch
    # concentrates most of the particles, the total is decided by that stretch and
    # can hide that in the rest the other model does better.
    compare_html = ""
    active_2 = [mi for mi, s in enumerate(state.model_slots) if s.path is not None]
    if len(active_2) > 1:
        by_photo = {}
        for mi in active_2:
            for r in results_2.get(mi, []):
                by_photo.setdefault(r.image_path, {})[mi] = r

        there_is_gt = any(r.has_gt for rs in results_2.values() for r in rs)
        header_line = "<tr><th>Image</th>"
        for mi in active_2:
            alias = state.model_slots[mi].alias
            header_line += f"<th>{_esc(alias)}<br><span style='font-weight:400;font-size:8.5pt'>dets</span></th>"
            if there_is_gt:
                header_line += (
                    "<th><span style='font-weight:400;font-size:8.5pt'>"
                    "True Positives/False Positives/False Negatives</span></th>"
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
                marker_2.append(
                    (2 * pr * rc / (pr + rc) if (pr + rc) else 0, state.model_slots[mi].alias)
                )
            marker_2.sort(reverse=True)
            best_f1, best = marker_2[0]
            # This F1 is the permissive one (localization). It used to say only `F1 global` and
            # coexisted with the strict verdict above: two different numbers
            # called the same in the same section.
            verdict = tr(
                "<p><b>Mejor F1 de localización:</b> {alias} ({f1}) — encontrar la "
                "partícula, sin exigir que acierte el polímero. El veredicto con clase "
                "está más arriba.</p>"
            ).format(alias=_esc(best), f1=f"{best_f1:.3f}")
        else:
            verdict = (
                "<p class='caption' style='text-align:left'>"
                + tr(
                    "Sin ground truth no se puede declarar un ganador: un "
                    "modelo con más detecciones puede estar acertando o "
                    "inventando. Carga anotaciones para obtener F1 por "
                    "modelo."
                )
                + "</p>"
            )

        # ── Side-by-side figures ──
        figures_cmp = ""
        if there_is_gt:
            data_bars = []
            by_model_class: dict[str, dict[str, tuple]] = {}
            for mi in active_2:
                alias = state.model_slots[mi].alias
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
                    metrics = aggregate_per_class(preds_m, gts_m, ids, iou_thr=state.params.iou_tp)
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
                    + tr(
                        "Figura. Precisión, Recall y F1 de cada modelo al umbral "
                        "{umbral}, en los dos criterios. La distancia entre paneles es "
                        "la confusión entre polímeros."
                    ).format(umbral=f"{state.params.conf:g}")
                    + "</div></div>"
                )

            f_class_id = _fig_metrics_by_class(by_model_class)
            if f_class_id:
                figures_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_class_id}' />"
                    "<div class='caption'>"
                    + tr(
                        "Figura. F1 por clase. Un modelo puede ganar en el agregado y "
                        "perder en el polímero que interesa."
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
                    state.model_slots[a].alias, state.model_slots[b].alias, pares
                )
                if f_agreement:
                    figures_cmp += (
                        f"<div class='fig'><img src='data:image/png;base64,{f_agreement}' />"
                        "<div class='caption'>"
                        + tr(
                            "Figura. Detecciones por imagen de un modelo frente al "
                            "otro; la diagonal es el acuerdo exacto. Escala "
                            "simétrica-logarítmica, porque casi todas las placas "
                            "tienen pocas partículas y unas pocas cientos."
                        )
                        + "</div></div>"
                    )

            # F1 curve against the threshold: it supports the choice of the operating
            # point with evidence, instead of asserting it.
            sweep = _sweep_confidence(results_2, state, state.params.iou_tp, state.params.conf)
            f_sweep = _fig_sweep(sweep)
            if f_sweep:
                figures_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_sweep}' />"
                    "<div class='caption'>"
                    + tr(
                        "Figura. F1 (con clase) frente al umbral de confianza; la "
                        "estrella marca el máximo de cada modelo. La curva arranca en "
                        "{umbral}, el umbral con que se ejecutó: por debajo las "
                        "detecciones no se calcularon."
                    ).format(umbral=f"{state.params.conf:g}")
                    + "</div></div>"
                )

        compare_html = (
            tr(
                "<p>Detecciones de cada modelo imagen por imagen. El total puede "
                "estar dominado por unas pocas fotos densas, así que conviene mirar "
                "el detalle antes de elegir modelo.</p>"
            )
            + verdict
            + figures_cmp
            + f"<table class='data'>{header_line}{rows_2}</table>"
        )
    elif len(active_2) == 1:
        compare_html = (
            "<p class='caption' style='text-align:left'>"
            + tr(
                "Solo se ejecutó un modelo, así que no hay comparación. "
                "Carga un segundo modelo en la pestaña Modelos para "
                "compararlos sobre las mismas imágenes."
            )
            + "</p>"
        )
    return compare_html


def _section_gallery(
    results_2: dict[int, list],
    state,
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
            slot = state.model_slots[mi]
            if True:
                # Prediction image (we prefer pred_png; otherwise the combined one)
                pred_bytes = getattr(r, "pred_png", None) or r.annotated_png
                if not pred_bytes:
                    continue
                left = (
                    f"<figure><img src='{_uri_gallery(pred_bytes)}' />"
                    f"<figcaption>{tr('Predicción del modelo')} · {_esc(slot.alias)}"
                    f"<span class='sub'>"
                    + tr("{n} detección(es) por YOLO").format(n=len(r.predictions))
                    + "</span>"
                    "</figcaption></figure>"
                )

                # Ground Truth image (control)
                gt_bytes = getattr(r, "gt_png", None)
                if r.has_gt and gt_bytes:
                    right = (
                        f"<figure><img src='{_uri_gallery(gt_bytes)}' />"
                        f"<figcaption>Ground Truth (control)"
                        f"<span class='sub'>{len(r.gt)} etiqueta(s) reales</span>"
                        f"</figcaption></figure>"
                    )
                else:
                    right = (
                        f"<figure><div class='nogt'>"
                        f"{tr('Sin Ground Truth para esta imagen')}</div>"
                        f"<figcaption>Ground Truth (control)"
                        f"<span class='sub'>{tr('no disponible')}</span>"
                        f"</figcaption></figure>"
                    )

                # Photo header: name and, if the name allows it,
                # which stretch and plate it comes from.
                sample = _sample_of(r.image_path)
                where = ""
                if sample:
                    station, range_value, plate = sample
                    where = (
                        f" &nbsp;<span class='tag'>{_esc(station)} · tramo {_esc(range_value)} "
                        f"· placa {_esc(plate)}</span>"
                    )
                meta = (
                    f"<strong>{_esc(r.image_path.name)}</strong>{where}"
                    f"&nbsp;<span class='tag'>modelo: {_esc(slot.alias)}</span>"
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
                n_shown = len({r.image_path for _, r in candidates})
                note = (
                    "<p class='caption'>"
                    + tr(
                        "Se muestran las primeras {n} imágenes, cada una con todos los "
                        "modelos; {fuera} quedaron fuera de la galería para que el archivo "
                        "siga siendo manejable. Las métricas de las secciones anteriores "
                        "sí incluyen todas."
                    ).format(n=n_shown, fuera=omitted)
                    + "</p>"
                )
            gallery_html = (
                tr(
                    "<p>Cada bloque muestra, a la izquierda, las detecciones del "
                    "modelo (<em>bounding boxes</em> dibujadas por YOLO con su clase y "
                    "confianza) y, a la derecha, las etiquetas reales de control "
                    "(<em>Ground Truth</em>). Esta vista lado a lado permite evaluar "
                    "visualmente dónde acertó o falló el modelo.</p>"
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
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_classes}' /><div class='caption'>{tr('Figura. Distribución de detecciones por clase.')}</div></div>"
    if fig_conf:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_conf}' /><div class='caption'>{tr('Figura. Histograma de confianza.')}</div></div>"
    if fig_size:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_size}' /><div class='caption'>{tr('Figura. Distribución de tamaños (diámetro equivalente en μm).')}</div></div>"
    return figures_html


def _section_methods(
    state, active, all_results, total_imgs: int, total_dets: int, any_gt: bool
) -> tuple:
    """Body of the 'metodos' section: parameters table, team and redacted paragraph.

    Returns ``(methods_html, equipo_html, methods_para)`` — the three pieces
    the assembler's ``metodos`` entry composes — WITHOUT any ``<h2>`` title.
    The ``import ultralytics`` stays lazy inside this body: it must not load
    at module import time (S01 laziness gate).
    """
    # ── Methods ──
    p = state.params
    # When calibration is per photo (medir_placa) there is no SINGLE um_per_px
    # value to report here -- p.um_per_px stays at 0 on purpose.
    # Instead of an `—` that says nothing, the batch mean is given with its
    # 95% confidence interval: a number that can be cited in a
    # method even if the real calibration is photo by photo (that is explained in
    # detail in the `Calibracion de escala` section, with the minimum and
    # maximum that show how much it varies).
    if p.um_per_px > 0:
        um_px_txt = f"{p.um_per_px:g}"
    else:
        _ums_methods = [
            c.um_por_px
            for c in (getattr(state, "calibrations", None) or {}).values()
            if getattr(c, "valid", False)
        ]
        _mic = _average_ic95(_ums_methods)
        if _mic:
            _average_m, _margin_m, _n_m = _mic
            um_px_txt = f"{_average_m:.4f} ± {_margin_m:.4f} (IC 95%, n={_n_m} fotos)"
        elif _ums_methods:
            um_px_txt = f"{_ums_methods[0]:.4f} " + tr("(una sola foto calibrada, sin IC)")
        else:
            um_px_txt = "—"
    methods_rows = [
        (tr("Modelos cargados"), _esc(", ".join(s.alias for s in active)) or "—"),
        (tr("Confianza mínima"), f"{p.conf:g}"),
        ("IoU NMS", f"{p.iou_nms:g}"),
        (tr("IoU para emparejar Verdaderos Positivos"), f"{p.iou_tp:g}"),
        (tr("Tamaño de imagen (imgsz)"), f"{p.imgsz}"),
        (tr("Dispositivo"), p.device),
        (tr("μm por píxel"), um_px_txt),
        (
            tr("Filtro tamaño (μm)"),
            f"{p.size_min_um} – {p.size_max_um}"
            if (p.size_min_um or p.size_max_um)
            else tr("sin filtro"),
        ),
        (tr("Imágenes procesadas"), str(total_imgs)),
        (tr("Total de detecciones"), str(total_dets)),
    ]
    # Abstention: how many particles were counted without attributing a polymer to them. It goes in
    # Methods and not buried in a results table because it is an analysis
    # decision, and whoever reads the report needs to know it was taken.
    _res_assignment = resume(
        [d for r in all_results for d in r.predictions], getattr(p, "conf_assignment", 0.0)
    )
    if _res_assignment.active:
        methods_rows.append(
            (tr("Confianza mínima para asignar polímero"), f"{p.conf_assignment:g}")
        )
        methods_rows.append(
            (
                tr("Partículas sin polímero asignable"),
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
        "<p><strong>" + tr("Resumen de la configuración empleada:") + "</strong></p>"
        "<blockquote style='border-left:3px solid var(--accent); margin:8px 0; "
        "padding:6px 14px; color:#424a53; background:#f6f8fa; border-radius:0 6px 6px 0;'>"
        + tr(
            "La detección automatizada se realizó con el modelo YOLO «{modelos}» "
            "(Ultralytics {ul}) a una resolución de entrada de {imgsz} px, umbral de "
            "confianza {conf} y supresión de no-máximos con IoU {iou}. "
        ).format(
            modelos=_model_names,
            ul=_ul_see,
            imgsz=p.imgsz,
            conf=f"{p.conf:g}",
            iou=f"{p.iou_nms:g}",
        )
        + (
            tr(
                "Las métricas de error se calcularon contra anotación manual "
                "independiente, emparejando predicciones y etiquetas con IoU ≥ {iou}. "
            ).format(iou=f"{p.iou_tp:g}")
            if any_gt
            else ""
        )
        + (
            tr("La calibración óptica fue de {um} μm/píxel. ").format(um=f"{p.um_per_px:g}")
            if p.um_per_px > 0
            else ""
        )
        + tr("Se procesaron {imgs} imágenes con un total de {dets} detecciones.").format(
            imgs=total_imgs, dets=total_dets
        )
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


def _section_calibration(state) -> str:
    """Body of the 'calibracion' section: scale provenance and per-photo calibration.

    Returns the HTML body WITHOUT its ``<h2>`` title. Empty when the state has
    no calibraciones and no manual scale, so the assembler drops the section
    like it does today. The ``polyx.core.calibration`` import stays lazy
    inside this body: it must not load at module import time (S01 laziness
    gate).
    """
    # ── Scale calibration ──
    # The provenance is reported, not only the number: a size in µm without saying
    # against which standard it was measured is not verifiable. If there is no calibration the
    # section is not emitted, instead of showing an `—` that looks like data.
    p = state.params
    calib_html = ""
    cals = getattr(state, "calibrations", None) or {}
    if cals:
        from polyx.core.measure.calibration import (
            BIAS_MAXIMUM_WALL_PCT,
            THICKNESS_WALL_MM,
            summary_batch,
        )

        res = summary_batch(list(cals.values()))
        if res["n"]:
            label_2 = {
                "placa": tr("medida sobre la placa Petri de esta foto"),
                "indice": tr("heredada del recorte (índice de calibración)"),
                "manual": tr("introducida a mano en Parámetros"),
            }
            rows_2 = "".join(
                f"<tr><td>{label_2.get(o, o)}</td>"
                f"<td>{k} {tr('imágenes') if k != 1 else tr('imagen')}</td></tr>"
                for o, k in sorted(res["origenes"].items())
            )
            warning = ""
            if res["variacion"] > 1.05:
                warning = tr(
                    "<p><strong>La escala no es común a todo el lote:</strong> varía "
                    "{var}× entre la foto más cercana y la más lejana. Por eso cada "
                    "imagen se convierte con su propio factor; usar uno solo para todas "
                    "daría tamaños con hasta un {pct}% de error.</p>"
                ).format(var=f"{res['variacion']:.2f}", pct=f"{100 * (res['variacion'] - 1):.0f}")
            _mic_calib = _average_ic95(
                [c.um_por_px for c in cals.values() if getattr(c, "valid", False)]
            )
            _row_average = ""
            if _mic_calib:
                _average_c, _margin_c, _n_c = _mic_calib
                _row_average = (
                    f"<tr>{tr('<td>media ± IC 95%</td>')}"
                    f"<td>{_average_c:.4f} ± {_margin_c:.4f} (n={_n_c})</td></tr>"
                )
            calib_html = f"""
<table class='data'><tr>{tr("<th>Procedencia de la escala</th><th>Imágenes</th>")}</tr>{rows_2}</table>
<table class='data'>
<tr>{tr("<th>µm por píxel</th><th>Valor</th>")}</tr>
<tr>{tr("<td>mínimo</td>")}<td>{res["min"]:.4f}</td></tr>
<tr>{tr("<td>mediana</td>")}<td>{res["mediana"]:.4f}</td></tr>
<tr>{tr("<td>máximo</td>")}<td>{res["max"]:.4f}</td></tr>
{_row_average}
</table>
{warning}
<p>{tr("El patrón de longitud es el diámetro externo nominal de la placa Petri. El radio en píxeles se obtiene ajustando un círculo por mínimos cuadrados al borde del anillo muestreado en 720 direcciones, con rechazo de atípicos; la transformada de Hough solo aporta el centro aproximado, porque su radio llega a errar un 12&nbsp;% y ese error entraría entero en todos los tamaños reportados.")}</p>

<p style='border-left:3px solid {T.DOC.INK3};padding:6px 12px;background:var(--bg_soft)'>
{tr("<strong>Incertidumbre de la escala, y en qué dirección.</strong> El anillo de la placa tiene una pared de unos <strong>{pared:g}&nbsp;mm</strong> (medido sobre las fotos de este estudio: el borde interno cae en 0,960 del radio ajustado y el externo en 1,000). El diámetro nominal de una placa Petri es ambiguo a ese nivel: puede referirse al diámetro <em>externo</em> o al <em>útil interior</em>. Aquí se toma el <strong>externo</strong>, que es el borde al que ajusta el círculo. Si el nominal se refiriera al interior, la escala correcta sería un <strong>{sesgo:g}&nbsp;% mayor</strong> y todas las tallas de este informe estarían <strong>subestimadas</strong> en esa cifra. El sesgo solo puede ir en ese sentido, porque el externo es el mayor de los dos bordes posibles.").format(pared=THICKNESS_WALL_MM, sesgo=BIAS_MAXIMUM_WALL_PCT)}</p>
{_figure_calibration(state)}"""
            if res["avisos"]:
                items = "".join(f"<li>{a}</li>" for a in res["avisos"][:12] if a)
                if items:
                    calib_html += (
                        f"<p>{tr('Incidencias durante la calibración:')}</p><ul>{items}</ul>"
                    )
    elif p.um_per_px > 0:
        calib_html = tr(
            "<p>Escala única para todo el lote, introducida a mano: "
            "<strong>{um} µm/píxel</strong>. No se midió ninguna placa, "
            "de modo que este factor no está trazado a un patrón de longitud y "
            "cualquier variación en la distancia de disparo entre fotos queda sin "
            "corregir.</p>"
        ).format(um=f"{p.um_per_px:g}")
    return calib_html


def _morphotype_table(shapes, warning_review: str) -> str:
    """Morphotype count table, with the review/ring warnings above it."""
    fibers = sum(1 for d, _ in shapes if d.morphotype == "fibra")
    pc_f = 100.0 * fibers / len(shapes)
    return (
        warning_review
        + f"""
<table class='data'><tr>{tr("<th>Morfotipo</th><th>Cuántas</th><th>%</th>")}</tr>
<tr>{tr("<td>Fibra (relación de aspecto ≥ 3)</td>")}<td>{fibers}</td><td>{pc_f:.1f} %</td></tr>
<tr>{tr("<td>Partícula (no fibrosa)</td>")}<td>{len(shapes) - fibers}</td><td>{100 - pc_f:.1f} %</td></tr>
</table>"""
    )


def _ring_warning(n_en_el_aro: int) -> str:
    """Paragraph for particles that fell on the plate ring, or '' when none."""
    if not n_en_el_aro:
        return ""
    from polyx.core.measure.calibration import FRACTION_AREA_USEFUL

    return (
        f"<p style='border-left:3px solid {T.DOC.INK3};padding:6px 12px;"
        f"background:var(--bg_soft)'>"
        + tr(
            "<strong>{n} detección(es) caen sobre el anillo de la placa</strong> "
            "—más allá del {pct} % del radio ajustado— y quedan fuera de esta "
            "sección: el aro es una banda brillante que el detector confunde "
            "con material, y como sale enorme desplazaría a las partículas de "
            "verdad del extremo grande. Siguen contadas en la sección de "
            "conteo."
        ).format(n=n_en_el_aro, pct=f"{100 * FRACTION_AREA_USEFUL:.0f}")
        + "</p>"
    )


def _review_warning(shapes, n_review: int) -> str:
    """Paragraph for particles whose size warrants manual review, or ''."""
    if not n_review:
        return ""
    _pc = 100.0 * n_review / len(shapes)
    _pc_txt = f"{_pc:.1f}" if _pc < 1 else f"{_pc:.0f}"
    return (
        f"<p style='border-left:3px solid {T.DOC.WARN};padding:6px 12px;"
        f"background:#fff8e6'>"
        + tr(
            "<strong>{n} de {total} partículas ({pct}&nbsp;%) tienen una talla "
            "que pide comprobación</strong>: al segmentarlas, la máscara se "
            "salía bastante más allá de la caja del detector, casi siempre "
            "porque el umbral enganchó una partícula vecina o una zona "
            "brillante del fondo. En esos casos la máscara se <em>recorta a la "
            "caja</em>, de modo que la talla reportada no está inflada; lo que "
            "puede es quedarse <em>corta</em>, si la partícula seguía de verdad "
            "más allá de la caja. Van marcadas con ⚠ en sus fichas."
        ).format(n=n_review, total=len(shapes), pct=_pc_txt)
        + "</p>"
    )


def _count_per_image(por_img: dict[str, list]) -> str:
    """Per-image fibre/fragment count table."""
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
    return (
        f"<h3>{NUM}.7 {tr('Recuento por imagen')}</h3>"
        f"<table class='data'><tr>"
        f"{tr('<th>Imagen</th><th>Total</th><th>Fibras</th><th>Partículas</th>')}"
        f"{tr('<th>Largo mediano<br>(µm)</th><th>Mayor<br>(µm)</th>')}"
        f"</tr>{rows_img}</table>"
    )


def _example_cards(record_elderly, record_minor) -> str:
    """Annotated cards for the largest and smallest measured particle."""
    example = ""
    if record_elderly is not None:
        example = _record_particle(
            *record_elderly, f"2 {tr('Un ejemplo: la partícula mayor, medida')}"
        )
    if record_minor is not None:
        example += _record_particle(
            *record_minor, f"3 {tr('Un ejemplo: la partícula menor, medida')}"
        )
    return example


def _section_shape(state, shapes, n_en_el_aro, n_formless):
    """Body of the 'forma' section, plus the companion 'talla_carpetas' body.

    Returns ``(forma_html, por_carpeta_html)``, both WITHOUT their ``<h2>``
    titles (the assembler owns them; the ``{NUM}.N`` sub-header placeholders
    stay intact). ``por_carpeta_html`` is built in the same pass because it
    shares the per-class size grouping computed here, but it is its own
    optional section in the report (see SECCIONES) and never lands inside
    ``forma_html``. ``formas`` arrives precomputed from
    :func:`_formas_de_lote` because the depth profile and the particle
    fichas read the same list.
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
        curves = sum(1 for d, _ in shapes if d.curvature and d.curvature >= 1.15)

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
        n_review = sum(1 for d, _ in shapes if getattr(d, "review", False))
        warning_review = _review_warning(shapes, n_review)

        # ── What was set apart for falling on the ring ──
        warning_review += _ring_warning(n_en_el_aro)

        # ── Distribution by morphotype ──
        table_morphotype = _morphotype_table(shapes, warning_review)

        if with_size:
            lengths = np.array([d.length_um for d, _ in with_size], dtype=float)
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
<h3>{NUM}.4 {tr("Partícula mayor y menor")}</h3>
<table class='data'>
<tr>{tr("<th></th><th>Clase</th><th>Largo</th><th>Ancho</th><th>Área</th><th>Aspecto</th><th>Curvatura</th><th>Morfotipo</th><th>Imagen</th>")}</tr>
{_row(tr("Mayor"), elderly, path_elderly)}
{_row(tr("Menor"), minor_2, path_minor)}
</table>

<h3>{NUM}.5 {tr("Distribución de tallas")}</h3>
<table class='data'>
<tr>{tr("<th>Estadístico</th><th>Largo (µm)</th>")}</tr>
<tr>{tr("<td>mínimo</td>")}<td>{lengths.min():.0f}</td></tr>
<tr>{tr("<td>percentil 10</td>")}<td>{np.percentile(lengths, 10):.0f}</td></tr>
<tr>{tr("<td>mediana</td>")}<td>{np.median(lengths):.0f}</td></tr>
<tr>{tr("<td>percentil 90</td>")}<td>{np.percentile(lengths, 90):.0f}</td></tr>
<tr>{tr("<td>máximo</td>")}<td>{lengths.max():.0f}</td></tr>
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
                    f"<h3>{NUM}.6 {tr('Talla por tipo de plástico')}</h3>"
                    f"<table class='data'><tr><th>Class</th><th>n</th>"
                    f"<th>smallest<br>(µm)</th><th>median<br>(µm)</th>"
                    f"<th>largest<br>(µm)</th><th>Fibers</th></tr>{rows_cls}</table>"
                )

            # The example card is drawn further below, AFTER explaining the
            # method: showing the crop with a yellow line on top before
            # saying that this line is the Feret diameter explains nothing.
            cal_elderly = (getattr(state, "calibrations", {}) or {}).get(Path(path_elderly).name)
            um_elderly = cal_elderly.um_por_px if cal_elderly and cal_elderly.valid else None
            record_elderly = (elderly, path_elderly, um_elderly)

            # The smallest one is shown too, not only tabulated: a minimum size
            # without a photo cannot be distinguished from noise, or the shape card
            # and a reader has no way to check that there really is a
            # particle there and not a background speck that was badly segmented.
            cal_minor = (getattr(state, "calibrations", {}) or {}).get(Path(path_minor).name)
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
        count_2 = _count_per_image(por_img)

        # An `—` in the size columns is not an analysis failure: the
        # particle WAS measured, in pixels, but without µm/px there is no way to
        # turn it into micrometres. Before, the dash appeared without explanation and
        # it looked as if the measurement had failed. It says what is missing and how
        # to fix it, instead of inventing a number or saying nothing.
        no_scale = [n for n, ds in por_img.items() if not any(d.length_um for d in ds)]
        if no_scale:
            all_value = len(no_scale) == len(por_img)
            which_2 = (
                tr("<strong>Ninguna imagen tiene escala</strong>")
                if all_value
                else tr(
                    "<strong>{n} de {total} imágenes no tienen escala</strong> ({cuales})"
                ).format(
                    n=len(no_scale),
                    total=len(por_img),
                    cuales=", ".join(sorted(no_scale)[:6]) + (", …" if len(no_scale) > 6 else ""),
                )
            )
            count_2 += (
                "<p>"
                + which_2
                + tr(
                    ", y por eso su largo aparece como «—». Las partículas <em>sí</em> "
                    "se midieron —el conteo y la forma de la tabla son correctos—, pero "
                    "una medida en píxeles no se puede pasar a micrómetros sin saber "
                    "cuántos µm mide cada píxel.</p>"
                )
                + tr(
                    "<p>Para obtenerlo, en <em>Parámetros</em>: activa <strong>«Medir "
                    "la placa Petri»</strong> —cada foto obtiene su propia escala del "
                    "anillo de la placa, que es lo más fiable— o escribe un valor de "
                    "<strong>µm/píxel</strong>"
                )
                + (
                    "."
                    if all_value
                    else tr(
                        ". Si ya está activo, en esas fotos concretas no se encontró el "
                        "anillo de la placa: revisa que se vea entero en el encuadre."
                    )
                )
                + "</p>"
            )

        # Outside the f-string: inside it we would have to nest quotes of the same type.
        pct_curves = f"{100.0 * curves / len(shapes):.1f}"
        method_2 = f"""
<h3>{NUM}.1 {tr("Cómo se mide el largo")}</h3>
{tr("<p>Las magnitudes se miden sobre la <strong>máscara de cada partícula</strong>, no sobre la caja del detector. La caja de una partícula alargada está casi vacía y depende de cómo haya caído: una fibra tumbada en diagonal tiene caja cuadrada, de modo que medir sobre la caja la reportaría como si no fuera fibra, y con una talla equivocada. La máscara se obtiene umbralizando cada recorte a medio camino entre el nivel del fondo —la mediana del anillo que rodea la caja— y el de la partícula —el percentil 90 dentro de ella—.</p>")}

{tr("<p><strong>El criterio general es la línea recta más larga que cabe en la partícula</strong>, es decir la mayor distancia entre dos puntos de su contorno: el <em>diámetro de Feret máximo</em>. Para una partícula de forma irregular pero no doblada, esa recta es su talla, y tiene dos propiedades que la hacen fiable: no depende de la orientación con que la partícula haya caído, y un borde dentado no la altera.</p>")}

{tr("<p><strong>Esa recta deja de servir cuando la partícula está contorsionada.</strong> En una fibra doblada la distancia entre sus extremos es la cuerda, no su longitud: en un arco de media circunferencia se queda un 35&nbsp;% corta. Para esos casos se mide el <em>diámetro geodésico</em>, que es el camino más largo que cabe <strong>dentro</strong> de la partícula. Al no poder salirse de la máscara, ese camino rodea la curva y devuelve la longitud recorrida.</p>")}

<table class='data'>
<tr>{tr("<th>Forma de la partícula</th><th>Qué se reporta como largo</th>")}</tr>
<tr>{tr("<td>Compacta o irregular, pero no doblada</td><td>Feret máximo (la recta más larga)</td>")}</tr>
<tr>{tr("<td>Alargada y contorsionada (fibra)</td><td>Diámetro geodésico (sigue la curva)</td>")}</tr>
</table>

{tr("<p>La distinción se hace por el <strong>grosor</strong>: solo se usa el geodésico si el largo supera al menos cuatro veces el grosor máximo inscrito de la partícula. La razón es que en una partícula gruesa cualquier concavidad obliga al camino geodésico a bordearla en vez de atravesarla, y entonces el número se infla; en una fibra delgada eso no puede ocurrir.</p>")}

{tr("<p>Contrastado con formas de talla conocida —rectas, rectas giradas 45&nbsp;°, arcos de 60, 120 y 180&nbsp;°, un círculo y una recta de borde dentado— el largo así medido da un <strong>error mediano del 0,8&nbsp;% y del 4,7&nbsp;% en el peor caso</strong>.</p>")}

{tr("<p><strong>{n} partículas ({pct} %) están contorsionadas</strong>, entendiendo por tal que su largo supera en más de un 15&nbsp;% su extensión en línea recta.</p>").format(n=curves, pct=pct_curves)}

{tr("<p><em>Partículas en contacto.</em> Dos partículas que se tocan forman una sola mancha, y medirlas juntas sumaría sus tallas. Se separan por <em>watershed</em> sobre la transformada de distancia: el centro de cada una queda lejos del fondo y el cuello que las une queda cerca, de modo que el corte cae por el cuello. Sobre círculos de talla conocida las separa hasta un 27&nbsp;% de solapamiento del diámetro, sin partir ninguna partícula de una sola pieza.</p>")}

{tr("<p><em>Limitaciones conocidas.</em> Dos partículas solapadas más allá de un 40&nbsp;% de su diámetro se siguen midiendo como una sola: a esa altura ya no hay un cuello por el que cortar. Y en una fibra muy enroscada el camino geodésico ataja por el interior de cada codo, subestimando la longitud hasta un 19&nbsp;% en el caso más cerrado que se ensayó.</p>")}"""
        if n_formless:
            method_2 += tr(
                "<p>En {n} partículas no se pudo separar la partícula del fondo; su "
                "talla proviene de la caja y no es comparable con el resto.</p>"
            ).format(n=n_formless)

        # ── The examples, once the method is explained ──
        # Both, not only the largest: a minimum size without a photo cannot be
        # distinguished by eye from badly segmented noise, and it is the one most easily
        # questioned in a review.
        example = _example_cards(record_elderly, record_minor)

        # Section order: first how it is measured, then a measured case, and
        # finally the figures. The other way round, the tables arrive before anything has
        # explained what `largo` means here.
        shape_html = method_2 + example + table_morphotype + tables + count_2
    return shape_html, by_html_folder


def _section_records(state, shapes: list[tuple], max_records: int = 12) -> str:
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

        from polyx.core.measure.morphology import draw_measurement

        chosen = choose_records(shapes, max_records)
        n_fibers_sample = sum(1 for d, _ in chosen if d.morphotype == "fibra")
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
            cal = (getattr(state, "calibrations", {}) or {}).get(Path(path_2).name)
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
                detail = [f"ancho {d.width_um:.0f} µm"] if d.width_um else []
                if d.aspect:
                    detail.append(f"aspecto {d.aspect:.1f}")
                detail.append(tr("medido por {metodo}").format(metodo=tr(m.method)))
                # If an appreciable part of the line falls outside, it is stated with a
                # number. Below 5% it is contour discretization and
                # saying it would only add noise to the card.
                if getattr(m, "feret_outside", 0) > 0.05:
                    detail.append(
                        tr("{pct} % de la recta fuera del contorno").format(
                            pct=f"{100 * m.feret_outside:.0f}"
                        )
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
                        f"⚠ {d.warning_shape or tr('la máscara se salía de la caja y se recortó a ella')}</div>"
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
                criterion = tr(
                    "Se muestran las <strong>{fibras} fibra(s) y las {otras} "
                    "partículas más grandes</strong>"
                ).format(fibras=n_fibers_sample, otras=n_others_sample)
            else:
                criterion = tr(
                    "Se muestran las <strong>{otras} partículas más "
                    "grandes</strong>. En este lote no se detectó ninguna "
                    "fibra, así que la muestra no incluye ninguna"
                ).format(otras=n_others_sample)
            records_html = (
                tr(
                    "<p>Cada partícula con el número que lleva en la imagen anotada, su "
                    "recorte y la medida dibujada <strong>sobre ella</strong>: en "
                    "amarillo la recta de Feret, en magenta el camino geodésico, según "
                    "cuál de las dos haya decidido su talla. El contorno verde es la "
                    "máscara que se midió. A la izquierda va la partícula sin marcas, "
                    "para poder juzgar si el contorno la sigue.</p>"
                )
                + tr(
                    "<p>En una partícula <strong>cóncava</strong> la recta amarilla "
                    "cruza por fuera del contorno, y eso es correcto: Feret es la "
                    "separación de dos mordazas de calibre, no un camino por dentro "
                    "de la partícula. Para que se vea, el tramo que cae fuera va "
                    "<strong>a trazos</strong>. El camino que sí va por dentro es el "
                    "geodésico, en magenta, y se usa cuando la partícula es delgada "
                    "y está doblada.</p>"
                )
                + "<p>"
                + criterion
                + tr(
                    ". El reparto es deliberado: las fibras son el caso donde actúa el "
                    "método geodésico —y son minoría, de modo que una muestra tomada al "
                    "azar podría no enseñar ninguna—, y la mayor de cada tipo es la que "
                    "sostiene cualquier afirmación sobre talla máxima. <strong>No es "
                    "una muestra representativa del lote</strong>, sino la selección "
                    "que permite comprobar el método donde más puede fallar.</p>"
                )
                + "".join(parts)
            )
    return records_html
