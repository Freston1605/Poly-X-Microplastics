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

from polyx.core.i18n import tr
from polyx.core.platform import open_in_system
from polyx.gui import iconos, pdf_export
from polyx.gui import theme as T
from polyx.gui.report_html import PRESETS, SECTIONS, generate_report

from ._base import DetectorPage


class ReportPage(DetectorPage):
    STEP_N = 9
    STEP_TITLE = tr("Informe de detección")
    STEP_DESCRIPTION = tr(
        "Genera un informe HTML autocontenido (todas las imágenes embebidas en base64, "
        "así no se rompen al enviarlo a otra persona) con métodos, métricas, gráficos, "
        "galería comparativa, conteo por muestra y análisis de errores. Puedes exportarlo "
        "directamente a PDF para enviarlo, o guardar aparte cada foto con sus etiquetas "
        "dibujadas."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Card: Generate report ──
        c1, l1 = self.card(tr("Generar informe"), "texto")
        info = QLabel(
            tr(
                "Después de ejecutar al menos una detección, presiona <b>Generar reporte HTML</b> "
                "(se crea <code>informe_deteccion.html</code> dentro de la carpeta del run y se abre "
                "en tu navegador) o <b>Exportar a PDF</b> para obtener un archivo listo para enviar. "
                "Todas las imágenes van embebidas en base64, por lo que el archivo es autocontenido."
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
        self.btn_gen = QPushButton(tr("Generar informe HTML"))
        self.btn_gen.setIcon(iconos.icon("texto", 15, T.ON_ACCENT))
        self.btn_gen.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_gen.setCursor(Qt.PointingHandCursor)
        self.btn_gen.clicked.connect(self._generate)
        row.addWidget(self.btn_gen)

        self.btn_pdf = QPushButton(tr("Exportar a PDF"))
        self.btn_pdf.setIcon(iconos.icon("reporte", 15, T.ON_ACCENT))
        self.btn_pdf.setStyleSheet(
            f"background: {T.OK}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_pdf.setCursor(Qt.PointingHandCursor)
        self.btn_pdf.setToolTip(tr("Genera el reporte y lo guarda como PDF listo para enviar."))
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
        c2, l2 = self.card(tr("Contenido del reporte"), "lista")
        contents = QLabel(
            tr(
                "<ol>"
                "<li><b>Resumen</b> con detecciones, confianza media y tamaño medio.</li>"
                "<li><b>Métodos</b> — modelo, parámetros, calibración y dispositivo.</li>"
                "<li><b>Calibración de escala</b> — de dónde salió el µm/px de cada foto.</li>"
                "<li><b>Forma y talla</b> — reparto fibra/fragmento, distribución de tallas, "
                "la mayor y la menor, recuento por imagen y cómo se mide el largo.</li>"
                "<li><b>Talla por carpeta y por foto</b> (opcional, desmarcada de fábrica) — "
                "compara el tamaño entre las carpetas del lote y, dentro de cada una, entre "
                "sus fotos, con una prueba de significancia. Sirve cuando cada carpeta es un "
                "sitio de muestreo distinto.</li>"
                "<li><b>Fichas</b> — una muestra de partículas con su medición dibujada.</li>"
                "<li><b>Resultados generales</b> — clases, confianza y tamaños.</li>"
                "<li><b>Resumen por modelo</b> (tabla comparativa)</li>"
                "<li><b>Análisis de errores</b> (solo si hay ground truth) — matriz de "
                "confusión y P/R/F1 por clase.</li>"
                "<li><b>Comparación entre modelos</b></li>"
                "<li><b>Galería por imagen</b> — predicción y ground truth lado a lado.</li>"
                "<li><b>Conteo por muestra y tipo de plástico</b> — por imagen, por tramo "
                "y por estación.</li>"
                "<li><b>Referencias bibliográficas</b></li>"
                "</ol>"
                "<p>Todas las secciones son opcionales: se eligen arriba.</p>"
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
        c, l = self.card(tr("Guardar las fotos con las etiquetas"), "imagenes")

        aid = QLabel(
            tr(
                "Guarda cada foto analizada con sus cajas dibujadas, en su resolución "
                "original. Eliges una carpeta y dentro se crea una subcarpeta por cada "
                "opción que marques (<code>conteo_manual</code>, "
                "<code>deteccion_modelo</code>, <code>ambas_superpuestas</code>), así "
                "no se mezclan. Respeta el <b>alcance</b> elegido arriba: si marcaste "
                "fotos concretas, solo se guardan esas."
            )
        )
        aid.setWordWrap(True)
        aid.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l.addWidget(aid)

        self.chk_f_manual = QCheckBox(tr("Etiquetas manuales (Ground Truth)"))
        self.chk_f_model = QCheckBox(tr("Detecciones del modelo"))
        self.chk_f_both = QCheckBox(tr("Las dos superpuestas en la misma foto"))
        self.chk_f_manual.setChecked(True)
        self.chk_f_model.setChecked(True)
        for chk in (self.chk_f_manual, self.chk_f_model, self.chk_f_both):
            chk.setStyleSheet(f"color: {T.INK2}; border: none;")
            l.addWidget(chk)

        row_2 = QHBoxLayout()
        self.btn_photos = QPushButton(tr("Guardar fotos etiquetadas"))
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
                    origin_2 = Path(origin_2)
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
                tr("Nada que guardar"),
                tr(
                    "Marca al menos una de las tres opciones: etiquetas manuales, "
                    "detecciones del modelo, o las dos superpuestas."
                ),
            )
            return
        if self.rb_chosen.isChecked() and not self._photos_marked():
            QMessageBox.warning(
                self,
                tr("Sin fotos marcadas"),
                tr("El alcance está en 'solo las fotos que marque' y no hay ninguna marcada."),
            )
            return

        tasks = self._photos_to_export()
        if not tasks:
            QMessageBox.warning(
                self,
                tr("Nada que guardar"),
                tr(
                    "No se encontró ninguna imagen anotada en disco para lo que "
                    "pediste.\n\nSi marcaste 'etiquetas manuales', recuerda que "
                    "solo existen para las fotos que tienen Ground Truth."
                ),
            )
            return

        base = (
            self.state.run_dir
            if (self.state.run_dir and self.state.run_dir.exists())
            else Path.home()
        )
        destination = QFileDialog.getExistingDirectory(
            self, tr("Carpeta donde guardar las fotos etiquetadas"), str(base)
        )
        if not destination:
            return
        destination = Path(destination)

        self.btn_photos.setEnabled(False)
        self.lbl_photos.setText(tr("Guardando {} imagen(es)…").format(len(tasks)))
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
            msg = "✓ " + tr("{} imagen(es) en {} carpeta(s) dentro de {}").format(
                copied, len(used), destination
            )
            if failures:
                msg += " · " + tr("{} fallaron").format(len(failures))
            self.lbl_photos.setText(msg)
            if copied and not failures:
                try:
                    open_in_system(str(destination))
                except Exception:
                    pass
            if failures:
                QMessageBox.warning(
                    self, tr("Algunas no se pudieron guardar"), "\n".join(failures[:12])
                )
        finally:
            self.btn_photos.setEnabled(True)

    # ── Scope ─────────────────────────────────────────────────
    def _card_scope(self):
        """Card to choose whether the report covers the whole job or a few photos."""
        c, l = self.card(tr("Alcance del informe"), "diana")

        aid = QLabel(
            tr(
                "Puedes generar el informe del <b>trabajo completo</b>, solo de las "
                "<b>fotos que elijas</b>, o ambos de una vez. Las cifras, los gráficos "
                "y la matriz de confusión se recalculan sobre lo que elijas, así que "
                "el informe siempre describe exactamente las fotos que muestra."
            )
        )
        aid.setWordWrap(True)
        aid.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l.addWidget(aid)

        self.rb_complete = QRadioButton(tr("Trabajo completo (todas las fotos analizadas)"))
        self.rb_chosen = QRadioButton(tr("Solo las fotos que marque abajo"))
        self.rb_both = QRadioButton(tr("Ambos: un informe completo y otro con las marcadas"))
        self.rb_complete.setChecked(True)
        self._cluster_scope = QButtonGroup(self)
        for rb in (self.rb_complete, self.rb_chosen, self.rb_both):
            rb.setStyleSheet(f"color: {T.INK2}; border: none;")
            self._cluster_scope.addButton(rb)
            rb.toggled.connect(self._sync_scope)
            l.addWidget(rb)

        row_2 = QHBoxLayout()
        self.btn_all = QPushButton(tr("Marcar todas"))
        self.btn_none = QPushButton(tr("Desmarcar todas"))
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
            self.lbl_sel.setText(tr("{} foto(s) en el trabajo").format(total))
        elif n == 0:
            self.lbl_sel.setText(tr("⚠ ninguna marcada"))
        else:
            self.lbl_sel.setText(tr("{} de {} marcadas").format(n, total))

    def _check_results(self) -> bool:
        """Warns when no detections have been run yet."""
        if not self.state.results:
            QMessageBox.warning(
                self,
                tr("Sin resultados"),
                tr(
                    "Aún no has ejecutado ninguna detección.\n"
                    "Ve a la pestaña Ejecutar e inicia el análisis primero."
                ),
            )
            return False
        return True

    def _card_sections(self):
        """Checkboxes to choose which sections go into the report.

        They are drawn by reading ``SECCIONES`` from the generator, not a list
        repeated here: when a section is added to the report tomorrow its
        checkbox appears on its own, and no section can end up impossible to
        uncheck.
        """
        c, l = self.card(tr("Qué incluir en el informe"), "reporte")
        aid = QLabel(
            tr(
                "El informe completo es largo. Desmarca lo que no necesites: las secciones "
                "se renumeran solas y el índice se ajusta. Una sección marcada que no tenga "
                "datos —errores sin ground truth, por ejemplo— se omite igualmente."
            )
        )
        aid.setWordWrap(True)
        aid.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l.addWidget(aid)

        row_2 = QHBoxLayout()
        row_2.setSpacing(8)
        for clue, label in (
            ("completo", tr("Completo")),
            ("resumen", tr("Resumen breve")),
            ("metodologico", tr("Metodológico")),
        ):
            b = QPushButton(label)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                f"background: white; color: {T.INK2}; border: 1px solid #d0d7de; "
                f"border-radius: 6px; padding: 5px 12px;"
            )
            b.clicked.connect(lambda _=False, k=clue: self._apply_preset(k))
            row_2.addWidget(b)
        row_2.addStretch(1)
        l.addLayout(row_2)

        self.chk_sections = {}
        for sid, title in SECTIONS:
            # The title is translated here and not in SECCIONES because that same list
            # feeds the HTML document, which is generated in Spanish.
            chk = QCheckBox(tr(title))
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {T.INK2}; border: none;")
            l.addWidget(chk)
            self.chk_sections[sid] = chk

        # The gallery is what makes the file large: each photo is embedded in
        # base64 inside the HTML itself.
        self.chk_sections["galeria"].setText(tr("Galería por imagen  (es lo que más pesa)"))

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
        """Generates the self-contained HTML (images in base64) at `out_path`."""
        generate_report(
            self.state,
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
                tr("Sin fotos marcadas"),
                tr(
                    "Elegiste generar el informe de fotos concretas, pero no hay "
                    "ninguna marcada.\n\nMarca al menos una en la lista, o cambia "
                    "el alcance a 'Trabajo completo'."
                ),
            )
            return []

        jobs = []
        if self.rb_complete.isChecked() or self.rb_both.isChecked():
            jobs.append(("", None, tr("trabajo completo")))
        if ask_marked:
            jobs.append(("_seleccion", marked, tr("{} foto(s) marcada(s)").format(len(marked))))
        return jobs

    def _generate(self):
        if not self._check_results():
            return

        out_dir = self.state.run_dir
        if out_dir is None or not out_dir.exists():
            # Ask for the destination folder
            d = QFileDialog.getExistingDirectory(self, "Carpeta destino del reporte")
            if not d:
                return
            out_dir = Path(d)

        jobs = self._jobs()
        if not jobs:
            return

        self.btn_gen.setEnabled(False)
        self.lbl_status.setText(tr("Generando reporte… (puede tardar unos segundos)"))
        generated = []
        try:
            for suffix_2, photos, label in jobs:
                out_path = out_dir / f"informe_deteccion{suffix_2}.html"
                self._write_html(out_path, solo_images_2=photos)
                generated.append(out_path)
            self.lbl_status.setText(
                "✓ " + tr("{} informe(s) generado(s) en {}").format(len(generated), out_dir)
            )
            for p in generated:
                try:
                    open_in_system(str(p))
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, tr("Error generando reporte"), f"{type(e).__name__}: {e}")
            self.lbl_status.setText(tr("✗ Falló la generación."))
        finally:
            self.btn_gen.setEnabled(True)

    def _export_pdf(self):
        if not self._check_results():
            return

        if not pdf_export.is_available():
            QMessageBox.warning(
                self,
                tr("PDF no disponible"),
                tr(
                    "Para exportar a PDF se necesita QtWebEngine (incluido en PySide6-Addons).\n\n"
                    "Alternativa: pulsa 'Generar reporte HTML' y, en el navegador, usa\n"
                    "Ctrl+P → 'Guardar como PDF'."
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
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar informe como PDF", suggested, "PDF (*.pdf)"
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
            tr("Generando PDF… (renderizando con el motor del navegador, unos segundos)")
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
                    "✓ " + tr("{} PDF generado(s) en {}").format(len(facts), pdf_base.parent)
                )
                for p in facts:
                    try:
                        open_in_system(str(p))
                    except Exception:
                        pass
            if failed:
                self.lbl_status.setText(tr("✗ No se pudo generar todo el PDF."))
                QMessageBox.warning(
                    self,
                    tr("No se pudo generar el PDF"),
                    tr(
                        "Falló el renderizado de: "
                        + ", ".join(failed)
                        + ".\n\nComo alternativa, genera el HTML y usa\n"
                        "Ctrl+P → 'Guardar como PDF' en tu navegador."
                    ),
                )
        except Exception as e:
            QMessageBox.critical(self, tr("Error generando PDF"), f"{type(e).__name__}: {e}")
            self.lbl_status.setText(tr("✗ Falló la generación de PDF."))
        finally:
            self.btn_pdf.setEnabled(True)
            self.btn_gen.setEnabled(True)
