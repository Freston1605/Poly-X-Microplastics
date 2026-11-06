"""Page 9 — Detection report in HTML."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
)

from polyx.core.detection_run import RunStatus
from polyx.core.i18n import _, lazy, ngettext
from polyx.core.platform import open_in_system
from polyx.gui import iconos, pdf_export
from polyx.gui import theme as T
from polyx.gui.report_html import PRESETS, SECTIONS, generate_report

from ._base import DetectorPage


class ReportPage(DetectorPage):
    STEP_N = 9
    STEP_TITLE = lazy("Detection report")
    STEP_DESCRIPTION = lazy(
        "Generates a self-contained HTML report (every image embedded as base64, so nothing breaks when you send it to someone else) with methods, metrics, charts, a comparison gallery, per-sample counts and error analysis. You can export it straight to PDF to send it, or save each photo separately with its labels drawn on."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Card: Generate report ──
        c1, l1 = self.card(_("Generate report"), "texto")
        info = QLabel(
            _(
                "After running at least one detection, press <b>Generate HTML report</b> (<code>informe_deteccion.html</code> is written inside the run folder and opened in your browser) or <b>Export to PDF</b> for a file ready to send. Every image is embedded as base64, so the file is self-contained."
            )
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l1.addWidget(info)

        # ── Which sections are included ──
        self.body.addWidget(self._card_sections())

        # ── Report scope ──
        self.body.addWidget(self._card_scope())

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_gen = QPushButton(_("Generate HTML report"))
        self.btn_gen.setIcon(iconos.icon("texto", 15, T.ON_ACCENT))
        self.btn_gen.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_gen.setCursor(Qt.PointingHandCursor)
        self.btn_gen.clicked.connect(self._generate)
        row.addWidget(self.btn_gen)

        self.btn_pdf = QPushButton(_("Export to PDF"))
        self.btn_pdf.setIcon(iconos.icon("reporte", 15, T.ON_ACCENT))
        self.btn_pdf.setStyleSheet(
            f"background: {T.OK}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_pdf.setCursor(Qt.PointingHandCursor)
        self.btn_pdf.setToolTip(_("Generates the report and saves it as a PDF ready to send."))
        self.btn_pdf.clicked.connect(self._export_pdf)
        row.addWidget(self.btn_pdf)

        row.addStretch(1)
        l1.addLayout(row)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l1.addWidget(self.lbl_status)
        self.body.addWidget(c1)

        # ── Save photos with labels drawn ──
        self.body.addWidget(self._card_photos())

        # ── Report content ──
        c2, l2 = self.card(_("Report contents"), "lista")
        contents = QLabel(
            _(
                "<ol><li><b>Summary</b> with detections, mean confidence and mean size.</li><li><b>Methods</b> — model, parameters, calibration and device.</li><li><b>Scale calibration</b> — where the µm/px of each photo came from.</li><li><b>Shape and size</b> — fibre/particle split, size distribution, the largest and the smallest, per-image counts and how length is measured.</li><li><b>Size by folder and by photo</b> (optional, off by default) — compares size across the folders of the batch and, within each one, across its photos, with a significance test. Useful when each folder is a different sampling site.</li><li><b>Cards</b> — a sample of particles with their measurement drawn on them.</li><li><b>General results</b> — classes, confidence and sizes.</li><li><b>Per-model summary</b> (comparison table)</li><li><b>Error analysis</b> (only with ground truth) — confusion matrix and per-class P/R/F1.</li><li><b>Model comparison</b></li><li><b>Per-image gallery</b> — prediction and ground truth side by side.</li><li><b>Counts by sample and polymer type</b> — per image, per depth section and per station.</li><li><b>References</b></li></ol><p>Every section is optional: you pick them above.</p>"
            )
        )
        contents.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; border: none;")
        contents.setTextFormat(Qt.RichText)
        l2.addWidget(contents)
        self.body.addWidget(c2)

    # ── Labelled photos ───────────────────────────────────────
    def _card_photos(self):
        """Card to save each photo with its labels as a standalone file.

        The report already carries them embedded, but there they go
        recompressed and 1100 px wide so the file stays manageable. Whoever
        wants the image for a paper figure or to review it with a magnifier
        needs the original, and that is what this card copies.
        """
        c, l = self.card(_("Save the photos with their labels"), "imagenes")  # noqa: E741

        aid = QLabel(
            _(
                "Saves every analysed photo with its boxes drawn on, at its original resolution. You pick a folder and a subfolder is created inside it for each option you tick (<code>conteo_manual</code>, <code>deteccion_modelo</code>, <code>ambas_superpuestas</code>), so they do not get mixed up. It honours the <b>scope</b> chosen above: if you ticked specific photos, only those are saved."
            )
        )
        aid.setWordWrap(True)
        aid.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l.addWidget(aid)

        self.chk_f_manual = QCheckBox(_("Manual labels (ground truth)"))
        self.chk_f_model = QCheckBox(_("Model detections"))
        self.chk_f_both = QCheckBox(_("Both overlaid on the same photo"))
        self.chk_f_manual.setChecked(True)
        self.chk_f_model.setChecked(True)
        for chk in (self.chk_f_manual, self.chk_f_model, self.chk_f_both):
            chk.setStyleSheet(f"color: {T.INK2}; border: none;")
            l.addWidget(chk)

        row_2 = QHBoxLayout()
        self.btn_photos = QPushButton(_("Save labelled photos"))
        self.btn_photos.setIcon(iconos.icon("imagenes", 15, T.ON_ACCENT))
        self.btn_photos.setStyleSheet(
            f"background: {T.VIO}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_photos.setCursor(Qt.PointingHandCursor)
        self.btn_photos.clicked.connect(self._save_photos)
        row_2.addWidget(self.btn_photos)
        row_2.addStretch(1)
        l.addLayout(row_2)

        self.lbl_photos = QLabel("")
        self.lbl_photos.setWordWrap(True)
        self.lbl_photos.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l.addWidget(self.lbl_photos)
        return c

    # One subfolder per image type: mixing them forces one to read the name of
    # each file to know which is which, and there are hundreds.
    FOLDER_MANUAL = "conteo_manual"
    FOLDER_MODEL = "deteccion_modelo"
    FOLDER_BOTH = "ambas_superpuestas"

    def _photos_to_export(self):
        """[(source_path, subfolder, destination_name)] according to checkboxes and scope."""
        marked = self._photos_marked()
        # With `"solo las marcadas"` it filters; with `"completo"` or `"ambos"` everything is included,
        # because `"ambos"` includes the complete work.
        solo_marked = self.rb_chosen.isChecked()
        # The alias only gets in the way when there is a single model: without it, the file
        # is named the same as the original photo, which is convenient for a figure.
        several = len(self.state.results) > 1

        tasks, seen = [], set()
        for mi, results_2 in self.state.results.items():
            alias = self.state.model_slots[mi].alias
            brand = f"__{alias}" if several else ""
            for r in results_2:
                if solo_marked and Path(r.image_path) not in marked:
                    continue
                stem = Path(r.image_path).stem
                candidates = []
                if self.chk_f_manual.isChecked() and r.gt_path:
                    # GT does not depend on the model: no alias in the name, and
                    # deduplicated so the same file is not written N times.
                    candidates.append((r.gt_path, self.FOLDER_MANUAL, stem))
                if self.chk_f_model.isChecked() and r.pred_path:
                    candidates.append((r.pred_path, self.FOLDER_MODEL, stem + brand))
                if self.chk_f_both.isChecked() and r.annotated_path:
                    candidates.append((r.annotated_path, self.FOLDER_BOTH, stem + brand))
                for origin_2, folder, name_2 in candidates:
                    if origin_2 is None or (folder, name_2) in seen:
                        continue
                    origin_2 = Path(origin_2)  # noqa: PLW2901
                    if not origin_2.exists():
                        continue
                    seen.add((folder, name_2))
                    tasks.append((origin_2, folder, name_2 + origin_2.suffix))
        return tasks

    def _save_photos(self):
        if not self._check_results():
            return
        if not any(c.isChecked() for c in (self.chk_f_manual, self.chk_f_model, self.chk_f_both)):
            QMessageBox.warning(
                self,
                _("Nothing to save"),
                _(
                    "Tick at least one of the three options: manual labels, model detections, or both overlaid."
                ),
            )
            return
        if self.rb_chosen.isChecked() and not self._photos_marked():
            QMessageBox.warning(
                self,
                _("No photos ticked"),
                _("The scope is set to 'only the photos I tick' and none are ticked."),
            )
            return

        tasks = self._photos_to_export()
        if not tasks:
            QMessageBox.warning(
                self,
                _("Nothing to save"),
                _(
                    "No annotated image was found on disk for what you asked for.\n\nIf you ticked 'manual labels', remember that these only exist for photos that have ground truth."
                ),
            )
            return

        base = (
            self.state.run_dir
            if (self.state.run_dir and self.state.run_dir.exists())
            else Path.home()
        )
        destination = QFileDialog.getExistingDirectory(
            self, _("Folder to save the labelled photos in"), str(base)
        )
        if not destination:
            return
        destination = Path(destination)

        self.btn_photos.setEnabled(False)
        self.lbl_photos.setText(
            ngettext("Saving %(count)d image…", "Saving %(count)d images…", len(tasks))
            % {"count": len(tasks)}
        )
        QApplication.processEvents()
        copied, failures = 0, []
        used = set()
        try:
            for origin_2, folder, name_2 in tasks:
                try:
                    sub_dir = destination / folder
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origin_2, sub_dir / name_2)
                    used.add(folder)
                    copied += 1
                except OSError as e:
                    failures.append(f"{folder}/{name_2}: {e}")
            msg = "✓ " + ngettext(
                "%(count)d image in %(folders)d folder inside %(destination)s",
                "%(count)d images in %(folders)d folders inside %(destination)s",
                copied,
            ) % {"count": copied, "folders": len(used), "destination": destination}
            if failures:
                msg += " · " + ngettext("%(count)d failed", "%(count)d failed", len(failures)) % {
                    "count": len(failures)
                }
            self.lbl_photos.setText(msg)
            if copied and not failures:
                try:  # noqa: SIM105
                    open_in_system(str(destination))
                except Exception:  # noqa: BLE001, S110
                    pass
            if failures:
                QMessageBox.warning(self, _("Some could not be saved"), "\n".join(failures[:12]))
        finally:
            self.btn_photos.setEnabled(True)

    # ── Scope ─────────────────────────────────────────────────
    def _card_scope(self):
        """Card to choose whether the report covers the whole job or a few photos."""
        c, l = self.card(_("Report scope"), "diana")  # noqa: E741

        aid = QLabel(
            _(
                "You can generate the report for the <b>whole job</b>, for <b>only the photos you choose</b>, or both at once. Figures, charts and the confusion matrix are recomputed over your choice, so the report always describes exactly the photos it shows."
            )
        )
        aid.setWordWrap(True)
        aid.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l.addWidget(aid)

        self.rb_complete = QRadioButton(_("Whole job (every analysed photo)"))
        self.rb_chosen = QRadioButton(_("Only the photos I tick below"))
        self.rb_both = QRadioButton(_("Both: one complete report and one with the ticked photos"))
        self.rb_complete.setChecked(True)
        self._cluster_scope = QButtonGroup(self)
        for rb in (self.rb_complete, self.rb_chosen, self.rb_both):
            rb.setStyleSheet(f"color: {T.INK2}; border: none;")
            self._cluster_scope.addButton(rb)
            rb.toggled.connect(self._sync_scope)
            l.addWidget(rb)

        row_2 = QHBoxLayout()
        self.btn_all = QPushButton(_("Tick all"))
        self.btn_none = QPushButton(_("Untick all"))
        self.btn_all.clicked.connect(lambda: self._mark_all(True))
        self.btn_none.clicked.connect(lambda: self._mark_all(False))
        for b in (self.btn_all, self.btn_none):
            b.setCursor(Qt.PointingHandCursor)
            row_2.addWidget(b)
        row_2.addStretch(1)
        self.lbl_sel = QLabel("")
        self.lbl_sel.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        row_2.addWidget(self.lbl_sel)
        l.addLayout(row_2)

        self.lst_photos = QListWidget()
        self.lst_photos.setMaximumHeight(220)
        self.lst_photos.itemChanged.connect(self._sync_scope)
        l.addWidget(self.lst_photos)

        self._sync_scope()
        return c

    def refresh(self):
        """Fills the list with the photos that were actually analyzed."""
        super_refresh = getattr(super(), "refresh", None)
        if callable(super_refresh):
            super_refresh()
        paths = sorted(
            {r.image_path for rs in self.state.results.values() for r in rs}, key=lambda p: p.name
        )
        previous = self._photos_marked()
        self.lst_photos.blockSignals(True)
        self.lst_photos.clear()
        for p in paths:
            it = QListWidgetItem(p.name)
            it.setData(Qt.UserRole, str(p))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if p in previous else Qt.Unchecked)
            self.lst_photos.addItem(it)
        self.lst_photos.blockSignals(False)
        self._sync_scope()

    def _mark_all(self, mark: bool):
        self.lst_photos.blockSignals(True)
        for i in range(self.lst_photos.count()):
            self.lst_photos.item(i).setCheckState(Qt.Checked if mark else Qt.Unchecked)
        self.lst_photos.blockSignals(False)
        self._sync_scope()

    def _photos_marked(self) -> set:
        marked = set()
        for i in range(self.lst_photos.count()):
            it = self.lst_photos.item(i)
            if it.checkState() == Qt.Checked:
                marked.add(Path(it.data(Qt.UserRole)))
        return marked

    def _sync_scope(self):
        """Enables the list only when the scope uses it, and warns when it is empty."""
        use_list = self.rb_chosen.isChecked() or self.rb_both.isChecked()
        for w in (self.lst_photos, self.btn_all, self.btn_none):
            w.setEnabled(use_list)
        n = len(self._photos_marked())
        total = self.lst_photos.count()
        if not use_list:
            self.lbl_sel.setText(
                ngettext("%(count)d photo in the job", "%(count)d photos in the job", total)
                % {"count": total}
            )
        elif n == 0:
            self.lbl_sel.setText(_("⚠ none ticked"))
        else:
            self.lbl_sel.setText(_("%(count)d of %(total)d ticked") % {"count": n, "total": total})

    def _check_results(self) -> bool:
        """Warns unless the current run finished (completed, aborted or failed).

        A failed batch is reportable: the document marks it as partial, so the
        results written before the error are not thrown away.
        """
        run = self.state.run_result
        if run is not None and RunStatus.parse(run.status) is not RunStatus.RUNNING:
            return True
        message = (
            _("Detection is still running; wait for it to finish before generating the report.")
            if run is not None
            else _(
                "You have not run any detection yet.\nGo to the Run tab and start the analysis first."
            )
        )
        QMessageBox.warning(self, _("No results"), message)
        return False

    def _card_sections(self):
        """Checkboxes to choose which sections go into the report.

        They are drawn by reading ``SECCIONES`` from the generator, not a list
        repeated here: when a section is added to the report tomorrow its
        checkbox appears on its own, and no section can end up impossible to
        uncheck.
        """
        c, l = self.card(_("What to include in the report"), "reporte")  # noqa: E741
        aid = QLabel(
            _(
                "The full report is long. Untick whatever you do not need: sections renumber themselves and the table of contents adjusts. A ticked section with no data — error analysis without ground truth, for instance — is left out anyway."
            )
        )
        aid.setWordWrap(True)
        aid.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l.addWidget(aid)

        row_2 = QHBoxLayout()
        row_2.setSpacing(8)
        for clue, label in (
            ("completo", _("Full")),
            ("resumen", _("Short summary")),
            ("metodologico", _("Methodological")),
        ):
            b = QPushButton(label)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                f"background: white; color: {T.INK2}; border: 1px solid #d0d7de; "
                f"border-radius: 6px; padding: 5px 12px;"
            )
            b.clicked.connect(lambda _unused=False, k=clue: self._apply_preset(k))
            row_2.addWidget(b)
        row_2.addStretch(1)
        l.addLayout(row_2)

        self.chk_sections = {}
        for sid, title in SECTIONS:
            # The title is translated here and not in SECCIONES because that same list
            # feeds the HTML document, which is generated in Spanish.
            chk = QCheckBox(_(title))
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {T.INK2}; border: none;")
            l.addWidget(chk)
            self.chk_sections[sid] = chk

        # The gallery is what makes the file large: each photo is embedded in
        # base64 inside the HTML itself.
        self.chk_sections["galeria"].setText(
            _("Per-image gallery  (this is what makes the file big)")
        )

        # Unlike the rest, this one is not checked by default: it only makes
        # sense when the analyzed images are spread across several
        # folders (sites, treatments...) and the user wants that
        # particular side-by-side view, not in every report.
        self.chk_sections["talla_carpetas"].setChecked(False)
        return c

    def _apply_preset(self, clue: str):
        chosen = set(PRESETS.get(clue, []))
        for sid, chk in self.chk_sections.items():
            chk.setChecked(sid in chosen)

    def _sections_marked(self) -> list:
        return [sid for sid, chk in self.chk_sections.items() if chk.isChecked()]

    def _write_html(self, out_path: Path, solo_images_2=None) -> Path:
        """Generates the self-contained HTML (images in base64) at `out_path`.

        The report consumes the canonical ``RunResult`` of the current run;
        the GUI's ``results`` view is only for the pages, not for rendering.
        """
        run = self.state.run_result
        if run is None:
            raise RuntimeError("no run result to report")
        generate_report(
            run,
            out_path,
            sections=self._sections_marked(),
            solo_images=solo_images_2,
        )
        return out_path

    def _jobs(self) -> list:
        """Reports to generate according to the scope: (suffix, photos, label).

        ``photos`` None means the whole job. Returns an empty list when the
        scope asks for checked photos and there are none, so an empty report
        is not generated in silence that would look like nothing was detected.
        """
        marked = self._photos_marked()
        ask_marked = self.rb_chosen.isChecked() or self.rb_both.isChecked()
        if ask_marked and not marked:
            QMessageBox.warning(
                self,
                _("No photos ticked"),
                _(
                    "You chose a report of specific photos, but none are ticked.\n\nTick at least one in the list, or change the scope to 'Whole job'."
                ),
            )
            return []

        jobs = []
        if self.rb_complete.isChecked() or self.rb_both.isChecked():
            jobs.append(("", None, _("whole job")))
        if ask_marked:
            jobs.append(
                (
                    "_seleccion",
                    marked,
                    ngettext("%(count)d ticked photo", "%(count)d ticked photos", len(marked))
                    % {"count": len(marked)},
                )
            )
        return jobs

    def _generate(self):
        if not self._check_results():
            return

        out_dir = self.state.run_dir
        if out_dir is None or not out_dir.exists():
            # Ask for the destination folder
            d = QFileDialog.getExistingDirectory(self, _("Report destination folder"))
            if not d:
                return
            out_dir = Path(d)

        jobs = self._jobs()
        if not jobs:
            return

        self.btn_gen.setEnabled(False)
        self.lbl_status.setText(_("Generating report… (this may take a few seconds)"))
        generated = []
        try:
            for suffix_2, photos, label in jobs:  # noqa: B007
                out_path = out_dir / f"informe_deteccion{suffix_2}.html"
                self._write_html(out_path, solo_images_2=photos)
                generated.append(out_path)
            self.lbl_status.setText(
                "✓ "
                + ngettext(
                    "%(count)d report generated in %(folder)s",
                    "%(count)d reports generated in %(folder)s",
                    len(generated),
                )
                % {"count": len(generated), "folder": out_dir}
            )
            for p in generated:
                try:  # noqa: SIM105
                    open_in_system(str(p))
                except Exception:  # noqa: BLE001, S110
                    pass
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, _("Error generating report"), f"{type(e).__name__}: {e}")
            self.lbl_status.setText(_("✗ Generation failed."))
        finally:
            self.btn_gen.setEnabled(True)

    def _export_pdf(self):
        if not self._check_results():
            return

        if not pdf_export.is_available():
            QMessageBox.warning(
                self,
                _("PDF not available"),
                _(
                    "Exporting to PDF needs QtWebEngine (bundled with PySide6-Addons).\n\nAlternative: press 'Generate HTML report' and, in the browser, use\nCtrl+P → 'Save as PDF'."
                ),
            )
            return

        jobs = self._jobs()
        if not jobs:
            return

        # Run folder as base; if it does not exist, next to the chosen PDF
        base_dir = (
            self.state.run_dir
            if (self.state.run_dir and self.state.run_dir.exists())
            else Path.home()
        )
        suggested = str(base_dir / "informe_deteccion.pdf")
        path, _unused = QFileDialog.getSaveFileName(
            self, _("Save report as PDF"), suggested, _("PDF (*.pdf)")
        )
        if not path:
            return
        pdf_base = Path(path)
        html_dir = (
            self.state.run_dir
            if (self.state.run_dir and self.state.run_dir.exists())
            else pdf_base.parent
        )

        self.btn_pdf.setEnabled(False)
        self.btn_gen.setEnabled(False)
        self.lbl_status.setText(
            _("Generating PDF… (rendering with the browser engine, a few seconds)")
        )
        QApplication.processEvents()
        try:
            facts, failed = [], []
            for suffix_2, photos, label in jobs:
                html_path = html_dir / f"informe_deteccion{suffix_2}.html"
                # With `"ambos"` two PDFs are written: the name chosen by the
                # user for the complete one, and that same name with a suffix for the
                # selection. Reusing the name would overwrite the first one.
                pdf_path = (
                    pdf_base
                    if not suffix_2
                    else pdf_base.with_name(pdf_base.stem + suffix_2 + ".pdf")
                )
                self._write_html(html_path, solo_images_2=photos)
                if pdf_export.html_to_pdf(html_path, pdf_path):
                    facts.append(pdf_path)
                else:
                    failed.append(label)

            if facts:
                self.lbl_status.setText(
                    "✓ "
                    + ngettext(
                        "%(count)d PDF generated in %(folder)s",
                        "%(count)d PDFs generated in %(folder)s",
                        len(facts),
                    )
                    % {"count": len(facts), "folder": pdf_base.parent}
                )
                for p in facts:
                    try:  # noqa: SIM105
                        open_in_system(str(p))
                    except Exception:  # noqa: BLE001, S110
                        pass
            if failed:
                self.lbl_status.setText(_("✗ Could not generate the whole PDF."))
                QMessageBox.warning(
                    self,
                    _("The PDF could not be generated"),
                    _(
                        "Failed to render: %(files)s.\n\nAs an alternative, generate the "
                        "HTML and use\nCtrl+P → 'Save as PDF' in your browser."
                    )
                    % {"files": ", ".join(failed)},
                )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, _("Error generating PDF"), f"{type(e).__name__}: {e}")
            self.lbl_status.setText(_("✗ PDF generation failed."))
        finally:
            self.btn_pdf.setEnabled(True)
            self.btn_gen.setEnabled(True)
