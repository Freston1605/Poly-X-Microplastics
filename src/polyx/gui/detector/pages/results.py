"""Page 6 — `Resultados`. Large KPIs + size histogram + per-image table."""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from polyx.core.i18n import _, lazy, ngettext
from polyx.core.platform import open_in_system
from polyx.core.quality.metrics import LABEL_FN, LABEL_FP, LABEL_MISCLS, LABEL_TP, match_image
from polyx.gui import iconos
from polyx.gui import theme as T
from polyx.gui.widgets import KPICard

from ._base import DetectorPage


def histogram_backgrounds() -> tuple[str, str]:
    """Axes (sunken) and figure (raised) backgrounds for the size histogram.

    Read from the active theme at call time so the chart follows it instead of
    staying a white card on a dark window.
    """
    return T.BG_SOFT, T.BG


class ResultsPage(DetectorPage):
    STEP_N = 6
    STEP_TITLE = lazy("Results")
    STEP_DESCRIPTION = lazy(
        "Quantitative summary of the analysis. For full charts and a per-image gallery, generate the HTML report in the Report tab."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── KPIs ──
        c1, l1 = self.card(_("General metrics"), "resultados")
        grid = QGridLayout()
        grid.setSpacing(12)
        self.kpi_imgs = KPICard(_("Images"), T.ACCENT)
        self.kpi_dets = KPICard(_("Detections"), T.ACCENT)
        self.kpi_conf = KPICard(_("Mean confidence"), T.VIO)
        self.kpi_size = KPICard(_("Mean size (μm)"), T.WARN)
        self.kpi_tp = KPICard(str(LABEL_TP), T.OK)
        self.kpi_fp = KPICard(str(LABEL_FP), T.WARN)
        self.kpi_fn = KPICard(str(LABEL_FN), T.ERR)
        self.kpi_f1 = KPICard("F1", T.ACCENT_D)
        grid.addWidget(self.kpi_imgs, 0, 0)
        grid.addWidget(self.kpi_dets, 0, 1)
        grid.addWidget(self.kpi_conf, 0, 2)
        grid.addWidget(self.kpi_size, 0, 3)
        grid.addWidget(self.kpi_tp, 1, 0)
        grid.addWidget(self.kpi_fp, 1, 1)
        grid.addWidget(self.kpi_fn, 1, 2)
        grid.addWidget(self.kpi_f1, 1, 3)
        l1.addLayout(grid)
        self.body.addWidget(c1)

        # ── Detailed metrics ──
        c2, l2 = self.card(_("Detailed metrics (summed over all models)"), "lista")
        row = QHBoxLayout()
        row.setSpacing(20)
        self.lbl_prec = QLabel(_("Precision:  —"))
        self.lbl_rec = QLabel(_("Recall:  —"))
        self.lbl_mis = QLabel(f"{LABEL_MISCLS}:  —")
        for l in (self.lbl_prec, self.lbl_rec, self.lbl_mis):  # noqa: E741
            l.setStyleSheet(f"color: {T.INK2}; font-size: 10.5pt; font-weight: 500; border: none;")
            row.addWidget(l)
        row.addStretch(1)
        l2.addLayout(row)
        # Automatic suggestion of the optimal confidence threshold (maximum F1)
        self.lbl_sugg = QLabel("")
        self.lbl_sugg.setWordWrap(True)
        self.lbl_sugg.setStyleSheet(
            f"color: {T.OK_TX}; font-size: 10pt; font-weight: 600; border: none;"
        )
        l2.addWidget(self.lbl_sugg)
        self.body.addWidget(c2)

        # ── Size histogram (calibration only) ──
        self.c_hist, l_hist = self.card(
            _("Size distribution (μm) — only with calibration active"), "visor"
        )
        self._hist_placeholder = QLabel(_("Set µm/px in Parameters to see the size distribution."))
        self._hist_placeholder.setAlignment(Qt.AlignCenter)
        self._hist_placeholder.setStyleSheet(
            f"color: {T.INK3}; font-size: 10pt; border: none; padding: 24px;"
        )
        l_hist.addWidget(self._hist_placeholder)
        self._hist_canvas = None
        self.body.addWidget(self.c_hist)

        # ── Reload GT from disk ──
        c_gt, l_gt = self.card(_("Ground truth"), "recargar")
        note_gt = QLabel(
            _(
                "It asks for no folder: it re-reads the .txt files sitting next to each photo, the same ones you see under Manual GT. This is needed because a run freezes the ground truth it read, so fixing an annotation afterwards does not by itself change these metrics or the report. It recalculates without running the model again, and only redraws the images whose .txt actually changed."
            )
        )
        note_gt.setWordWrap(True)
        note_gt.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l_gt.addWidget(note_gt)
        self.lbl_recharge = QLabel("")
        self.lbl_recharge.setWordWrap(True)
        self.lbl_recharge.setStyleSheet(
            f"color: {T.OK_TX}; font-size: 10pt; font-weight: 600; border: none;"
        )
        row_gt = QHBoxLayout()
        self.btn_reload = QPushButton(_("Re-read the .txt files from disk and recalculate"))
        self.btn_reload.setIcon(iconos.icon("recargar", 15, T.INK2))
        self.btn_reload.clicked.connect(self._reload_gt)
        row_gt.addWidget(self.btn_reload)
        row_gt.addStretch(1)
        l_gt.addLayout(row_gt)
        l_gt.addWidget(self.lbl_recharge)
        self.body.addWidget(c_gt)

        # ── Export CSV ──
        c_csv, l_csv = self.card(_("Export data"), "guardar")
        row_csv = QHBoxLayout()
        btn_csv = QPushButton(_("Export detections CSV"))
        btn_csv.setIcon(iconos.icon("texto", 15, T.INK2))
        btn_csv.clicked.connect(self._export_csv)
        row_csv.addWidget(btn_csv)
        row_csv.addStretch(1)
        l_csv.addLayout(row_csv)
        self.body.addWidget(c_csv)

        # ── Per-image table ──
        c3, l3 = self.card(_("Per image"), "imagenes")
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                _("Model"),
                _("Image"),
                _("Pred"),
                "GT",
                str(LABEL_TP),
                str(LABEL_FP),
                str(LABEL_FN),
                _("Mean conf."),
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._open_row)
        self.table.setMinimumHeight(280)
        l3.addWidget(self.table)
        row_rev = QHBoxLayout()
        btn_rev = QPushButton(_("Review particle by particle in the Viewer"))
        btn_rev.setIcon(iconos.icon("buscar", 15, T.INK2))
        btn_rev.clicked.connect(self._review_in_viewer)
        row_rev.addWidget(btn_rev)
        row_rev.addStretch(1)
        l3.addLayout(row_rev)
        note_rev = QLabel(
            _(
                "Opens the selected image in the Viewer with these very detections, numbered. There each particle shows what it was measured on and whether it is a fibre or a fragment — without running the model again."
            )
        )
        note_rev.setWordWrap(True)
        note_rev.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
        l3.addWidget(note_rev)
        self.body.addWidget(c3)

        # Subscribe
        self.state.run_finished.connect(self.refresh)
        self.state.run_aborted.connect(self.refresh)

    # ──────────────────────────────────────────
    def refresh(self):
        state = self.state
        # KPIs
        total_imgs = len({r.image_path for rs in state.results.values() for r in rs})
        all_results = [r for rs in state.results.values() for r in rs]
        total_dets = sum(len(r.predictions) for r in all_results)
        confs = [p.conf for r in all_results for p in r.predictions]
        sizes = [p.diam_um for r in all_results for p in r.predictions if p.diam_um]
        tp = sum(r.tp for r in all_results)
        fp = sum(r.fp for r in all_results)
        fn = sum(r.fn for r in all_results)

        # A retranslation rebuilds the page and refreshes it: an empty batch
        # must keep the "no data yet" dash, not report a measured zero.
        has_results = bool(all_results)
        self.kpi_imgs.set_value(total_imgs if has_results else None)
        self.kpi_dets.set_value(total_dets if has_results else None)
        self.kpi_conf.set_value(f"{(sum(confs) / len(confs)):.3f}" if confs else None)
        self.kpi_size.set_value(f"{(sum(sizes) / len(sizes)):.1f}" if sizes else None)
        self.kpi_tp.set_value(tp if has_results else None)
        self.kpi_fp.set_value(fp if has_results else None)
        self.kpi_fn.set_value(fn if has_results else None)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        any_gt = any(r.has_gt for r in all_results)
        self.kpi_f1.set_value(f"{f1:.3f}" if any_gt else None)
        self.lbl_prec.setText(
            _("Precision:  %(value).3f") % {"value": prec} if any_gt else _("Precision:  —")
        )
        self.lbl_rec.setText(
            _("Recall:  %(value).3f") % {"value": rec} if any_gt else _("Recall:  —")
        )
        mc = sum(r.miscls for r in all_results)
        self.lbl_mis.setText(f"{LABEL_MISCLS}:  {mc}" if any_gt else f"{LABEL_MISCLS}:  —")

        # Optimal threshold suggestion, model by model
        self.lbl_sugg.setText(self._text_suggestion() if any_gt else "")

        # Size histogram (only with calibration)
        self._update_histogram()

        # Table
        rows = []
        for mi, rs in state.results.items():
            slot = state.model_slots[mi]
            for r in rs:
                avg_c = (
                    sum(p.conf for p in r.predictions) / len(r.predictions) if r.predictions else 0
                )
                rows.append(
                    (
                        slot.alias,
                        r.image_path,
                        len(r.predictions),
                        len(r.gt),
                        r.tp,
                        r.fp,
                        r.fn,
                        avg_c,
                    )
                )
        self.table.setRowCount(len(rows))
        for i, (alias, p, pred, gt, tp, fp, fn, ac) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(alias))
            self.table.setItem(i, 1, QTableWidgetItem(p.name))
            for col, v in enumerate([pred, gt, tp, fp, fn], start=2):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, it)
            self.table.setItem(i, 7, QTableWidgetItem(f"{ac:.3f}"))
            # we store the path for double-click
            self.table.item(i, 1).setData(Qt.UserRole, str(p))
            self.table.item(i, 0).setData(Qt.UserRole, alias)

    def _reload_gt(self):
        """Re-reads the .txt files from disk and recomputes, without inferring again."""
        from ..recargar_gt import reload_gt

        if not self.state.results:
            self.lbl_recharge.setText(
                _("There are no results to recompute: run the detection first.")
            )
            return

        dlg = QProgressDialog(_("Re-reading ground truth…"), _("Cancel"), 0, 0, self)
        dlg.setWindowTitle(_("Reload GT"))
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        cancelled = {"si": False}

        def progress(made, total, name_2):
            if dlg.wasCanceled():
                cancelled["si"] = True
                return False
            dlg.setMaximum(total)
            dlg.setValue(made)
            dlg.setLabelText(f"{name_2}  ({made}/{total})")
            QApplication.processEvents()

        self.btn_reload.setEnabled(False)
        try:
            summary = reload_gt(self.state, progress=progress)
        except Exception as e:  # noqa: BLE001
            dlg.close()
            self.btn_reload.setEnabled(True)
            QMessageBox.warning(self, _("Reload GT"), f"{type(e).__name__}: {e}")
            return
        finally:
            dlg.close()
            self.btn_reload.setEnabled(True)

        changed = summary["cambiadas"]
        if changed:
            items_2 = ", ".join(changed[:8])
            if len(changed) > 8:
                items_2 += _(" and %(count)s more") % {"count": len(changed) - 8}
            text = ngettext(
                "✓ %(count)s image with different ground truth: %(items)s. Metrics and control images updated; you can generate the report now.",
                "✓ %(count)s images with different ground truth: %(items)s. Metrics and control images updated; you can generate the report now.",
                len(changed),
            ) % {"count": len(changed), "items": items_2}
        else:
            text = _(
                "✓ Reviewed %(count)s images: no .txt changed since the run. There was nothing to recompute."
            ) % {"count": summary["revisadas"]}
        if summary["sin_gt"]:
            text += "  " + _("(%(count)s with no control .txt.)") % {
                "count": len(summary["sin_gt"])
            }
        if cancelled["si"]:
            text += "  " + _("Cancelled halfway: press again to finish.")
        self.lbl_recharge.setText(text)
        self.refresh()

    def _results_by_model(self):
        """[(alias, [ImageResult with GT])], in the order of the slots."""
        output = []
        for mi, rs in sorted(self.state.results.items()):
            if mi >= len(self.state.model_slots):
                continue
            con_gt = [r for r in rs if r.has_gt]
            if con_gt:
                output.append((self.state.model_slots[mi].alias, con_gt))
        return output

    def _threshold_common(self, by_model):
        """Single threshold that maximizes the mean F1 of all the models.

        Comparing two architectures is only clean if both are measured at the
        same operating point. If each uses its own optimum, the threshold
        tuning gets compared too, and the report's verdict stops being about
        the architecture. Returns (threshold, mean_f1) or None with a single
        model.
        """
        if len(by_model) < 2:
            return None
        candidates = sorted(
            {round(p.conf, 2) for _unused, rs in by_model for r in rs for p in r.predictions}
        )
        if not candidates:
            return None
        paso = max(1, len(candidates) // 25)
        best = None
        for t in candidates[::paso]:
            f1s = [self._metrics_to(rs, t)[0] for _unused, rs in by_model]
            half = sum(f1s) / len(f1s)
            if best is None or half > best[1]:
                best = (t, half)
        return best

    def _text_suggestion(self) -> str:
        """Confidence advice, one line per model plus the recommendation.

        The sweep goes model by model on purpose: adding the two up, the
        optimum comes out of a mixture that belongs to neither, and it is
        exactly that number what decides which one wins.
        """
        by_model = self._results_by_model()
        if not by_model:
            return ""
        used_2 = self.state.params.conf
        lines: list[str] = []
        optimal: list[float] = []
        for alias, rs in by_model:
            sugg = self._suggest_operating_point(rs)
            if not sugg:
                continue
            (tb, f1b, pb, rb), (ta, f1a, pa, ra) = sugg
            head = f"{alias} — conf {ta:g}: F1 {f1a:.3f} (P {pa:.3f} / R {ra:.3f})."
            # A real improvement is required, not just a different threshold: with a flat F1, the
            # maximum falls anywhere, and advising a move there would be noise.
            if tb > ta + 1e-9 and (f1b - f1a) > 0.002:
                sign = "⚠" if (f1b - f1a) >= 0.01 else "·"
                lines.append(
                    _(
                        "%(sign)s %(head)s  Its best threshold is %(threshold).2f: F1 %(f1b).3f (%(delta)+.3f), P %(pb).3f / R %(rb).3f."
                    )
                    % {
                        "sign": sign,
                        "head": head,
                        "threshold": tb,
                        "f1b": f1b,
                        "delta": f1b - f1a,
                        "pb": pb,
                        "rb": rb,
                    }
                )
                optimal.append(tb)
            else:
                lines.append(
                    _(
                        "✓ %(head)s  It is the best in the explored range; below %(threshold)g there are no computed detections and they cannot be recovered by filtering."
                    )
                    % {"head": head, "threshold": ta}
                )
                optimal.append(ta)
        if not lines:
            return ""

        common = self._threshold_common(by_model)
        distinct = len({round(t, 2) for t in optimal}) > 1
        if common and distinct:
            lines.append(
                _(
                    "➜ Each model prefers a different threshold. Use %(threshold).2f for both (mean F1 %(mean).3f): it is the same operating point and the comparison stays clean. Change it in Parameters and run again."
                )
                % {"threshold": common[0], "mean": common[1]}
            )
        elif max(optimal) > used_2 + 1e-9:
            lines.append(
                _(
                    "➜ Change the confidence to %(threshold).2f in Parameters and run again before generating the report."
                )
                % {"threshold": max(optimal)}
            )
        else:
            lines.append(
                _(
                    "➜ Confidence %(threshold)g is already a good operating point. You can generate the report."
                )
                % {"threshold": used_2}
            )
        return "\n".join(lines)

    def _metrics_to(self, gt_results, threshold: float):
        """Strict P/R/F1 filtering the predictions at a given threshold.

        Strict: a well placed box with the wrong class counts as a false
        positive of the predicted class and a false negative of the true one.
        The permissive criterion, which leaves it out of both denominators,
        overestimates the performance and moreover contradicts the per-class
        precision table.
        """
        iou_thr = self.state.params.iou_tp
        tp = fp = fn = mc = 0
        for r in gt_results:
            preds = [p for p in r.predictions if p.conf >= threshold]
            m = match_image(preds, r.gt, iou_thr)
            tp += m.tp
            fp += m.fp
            fn += m.fn
            mc += m.miscls
        prec = tp / (tp + fp + mc) if (tp + fp + mc) else 0.0
        rec = tp / (tp + fn + mc) if (tp + fn + mc) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return f1, prec, rec

    def _suggest_operating_point(self, all_results):
        """Searches for the confidence threshold with the best F1 over what was already detected.

        Returns (best, current) as (threshold, f1, precision, recall) each.
        It can only explore UPWARD from the threshold the run used: the
        detections below it were never computed.
        """
        gt_results = [r for r in all_results if r.has_gt]
        if not gt_results:
            return None
        confs = sorted({round(p.conf, 2) for r in gt_results for p in r.predictions})
        if not confs:
            return None
        step = max(1, len(confs) // 25)  # limit to ~25 thresholds
        best = None
        for t in confs[::step]:
            f1, prec, rec = self._metrics_to(gt_results, t)
            if best is None or f1 > best[1]:
                best = (t, f1, prec, rec)
        used_2 = self.state.params.conf
        f1_a, p_a, r_a = self._metrics_to(gt_results, used_2)
        return best, (used_2, f1_a, p_a, r_a)

    def _update_histogram(self):
        """Draws a histogram of diam_um per class; only when there is calibration."""
        um_per_px = self.state.calibration.um_per_px
        all_results = [r for rs in self.state.results.values() for r in rs]
        sizes_by_class: dict[str, list[float]] = {}
        for r in all_results:
            for p in r.predictions:
                if p.diam_um and p.diam_um > 0:
                    sizes_by_class.setdefault(p.class_name, []).append(p.diam_um)

        has_data = um_per_px > 0 and any(sizes_by_class.values())

        if not has_data:
            if self._hist_canvas:
                self._hist_canvas.setVisible(False)
            self._hist_placeholder.setVisible(True)
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            if self._hist_canvas is None:
                fig = Figure(figsize=(7, 3), tight_layout=True)
                self._hist_ax = fig.add_subplot(111)
                self._hist_canvas = FigureCanvasQTAgg(fig)
                self._hist_canvas.setMinimumHeight(220)
                # Insert after the placeholder
                l_hist = self.c_hist.layout()
                l_hist.addWidget(self._hist_canvas)

            ax = self._hist_ax
            ax.clear()
            axes_bg, figure_bg = histogram_backgrounds()
            ax.set_facecolor(axes_bg)
            ax.figure.set_facecolor(figure_bg)

            for cls_name, vals in sizes_by_class.items():
                color = T.CLASS_COLOR_HEX.get(cls_name, T.FALLBACK_COLOR_HEX)
                ax.hist(
                    vals,
                    bins=20,
                    alpha=0.72,
                    label=f"{cls_name} (n={len(vals)})",
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                )

            ax.set_xlabel(_("Equivalent diameter (μm)"), fontsize=9)
            ax.set_ylabel(_("Frequency"), fontsize=9)
            ax.set_title(_("Size distribution by class"), fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(labelsize=8)
            self._hist_canvas.draw()
            self._hist_placeholder.setVisible(False)
            self._hist_canvas.setVisible(True)
        except Exception as e:  # noqa: BLE001
            self._hist_placeholder.setText(
                _("matplotlib is not available: %(error)s") % {"error": e}
            )
            self._hist_placeholder.setVisible(True)

    def _export_csv(self):
        """Exports all the detections to a CSV file."""
        path, _unused = QFileDialog.getSaveFileName(
            self, _("Export CSV"), "detecciones.csv", _("CSV (*.csv)")
        )
        if not path:
            return
        all_results = [r for rs in self.state.results.values() for r in rs]  # noqa: F841
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                # The real shape is exported in addition to the equivalent diameter: for
                # a fibre, the equivalent diameter is not the quantity
                # reported in the literature, which is its length.
                w.writerow(
                    [
                        "modelo",
                        "imagen",
                        "clase",
                        "conf",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "um_por_px",
                        "origen_escala",
                        "largo_um",
                        "metodo_largo",
                        "feret_um",
                        "geodesico_um",
                        "largo_rect_eq_um",
                        "ancho_um",
                        "area_um2",
                        "diam_um",
                        "aspecto",
                        "curvatura",
                        "morfotipo",
                    ]
                )
                cals = getattr(self.state, "calibrations", None) or {}
                for mi, rs in self.state.results.items():
                    alias = self.state.model_slots[mi].alias
                    for r in rs:
                        cal = cals.get(r.image_path.name)
                        um = f"{cal.um_por_px:.4f}" if cal and cal.valid else ""
                        origin_2 = cal.origin if cal and cal.valid else ""
                        for p in r.predictions:

                            def _n(v, dec=2):
                                return f"{v:.{dec}f}" if v else ""

                            w.writerow(
                                [
                                    alias,
                                    r.image_path.name,
                                    p.class_name,
                                    f"{p.conf:.4f}",
                                    f"{p.x1:.1f}",
                                    f"{p.y1:.1f}",
                                    f"{p.x2:.1f}",
                                    f"{p.y2:.1f}",
                                    um,
                                    origin_2,
                                    _n(p.length_um, 1),
                                    p.method_length or "",
                                    _n(p.feret_um, 1),
                                    _n(p.geodesic_um, 1),
                                    _n(p.equivalent_rect_length_um, 1),
                                    _n(p.width_um, 1),
                                    _n(p.area_um2, 1),
                                    _n(p.diam_um),
                                    _n(p.aspect),
                                    _n(p.curvature),
                                    p.morphotype or "",
                                ]
                            )
        except Exception as e:  # noqa: BLE001
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, _("Error"), _("Could not save: %(error)s") % {"error": e})

    def _open_row(self, row: int, col: int):
        if row < 0 or self.state.run_dir is None:
            return
        alias = self.table.item(row, 0).data(Qt.UserRole)
        img = Path(self.table.item(row, 1).data(Qt.UserRole))
        annot = self.state.run_dir / alias / f"{img.stem}_annot.png"
        if annot.exists():
            open_in_system(str(annot))

    def _review_in_viewer(self):
        """Opens the selected image in the Viewer, with its detections loaded.

        Reviewing particle by particle is an interactive task and does not fit
        in a report: twelve fact sheets already weigh more than a megabyte. But
        it should not force leaving the Detector, searching for the file and
        detecting again, which would also run the model a second time over the
        same photo. Here the Viewer is handed the detections that were ALREADY
        computed, so what gets reviewed is exactly what the run produced.
        """
        rows_2 = self.table.selectionModel().selectedRows()
        if not rows_2:
            QMessageBox.information(
                self, _("Review"), _("Select a row from the table below first.")
            )
            return
        row = rows_2[0].row()
        alias = self.table.item(row, 0).data(Qt.UserRole)
        img = Path(self.table.item(row, 1).data(Qt.UserRole))

        # The detections of THAT image with THAT model.
        dets = []
        for mi, rs in self.state.results.items():
            if self.state.model_slots[mi].alias != alias:
                continue
            for r in rs:
                if Path(r.image_path) == img:
                    dets = list(r.predictions)
                    break
        if not dets:
            QMessageBox.information(self, _("Review"), _("That image has no detections to review."))
            return

        try:
            from polyx.gui.viewer.window import ViewerWindow
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(
                self, _("Review"), _("Could not open the Viewer: %(error)s") % {"error": e}
            )
            return

        # It is stored in self so the garbage collector does not take it when
        # leaving the function: a Qt window with no live reference closes itself.
        self._viewer = ViewerWindow()
        self._viewer.state.load_single(img)
        cal = (getattr(self.state, "calibrations", None) or {}).get(img.name)
        if cal is not None and getattr(cal, "valid", False):
            self._viewer.state.um_per_px = cal.um_por_px
        elif self.state.calibration.um_per_px > 0:
            self._viewer.state.um_per_px = self.state.calibration.um_per_px
        self._viewer.state.detections = dets
        self._viewer.state.detections_changed.emit(dets)
        self._viewer.show()
        self._viewer.raise_()
