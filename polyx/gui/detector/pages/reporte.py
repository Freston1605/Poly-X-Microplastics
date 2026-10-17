"""Página 9 — Informe de detección en HTML."""
from __future__ import annotations
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QCheckBox, QMessageBox,
    QFileDialog, QApplication, QRadioButton, QButtonGroup, QListWidget,
    QListWidgetItem,
)

from ._base import DetectorPage
from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.gui import pdf_export
from polyx.gui.report_html import generate_report, SECCIONES, PRESETS
from polyx.model.i18n import tr
from polyx.model.plataforma import abrir_en_el_sistema


class ReportePage(DetectorPage):
    STEP_N = 9
    STEP_TITLE = tr("Informe de detección")
    STEP_DESCRIPTION = (
        tr("Genera un informe HTML autocontenido (todas las imágenes embebidas en base64, "
        "así no se rompen al enviarlo a otra persona) con métodos, métricas, gráficos, "
        "galería comparativa, conteo por muestra y análisis de errores. Puedes exportarlo "
        "directamente a PDF para enviarlo, o guardar aparte cada foto con sus etiquetas "
        "dibujadas.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Tarjeta: Generar informe ──
        c1, l1 = self.card(tr("Generar informe"), "texto")
        info = QLabel(
            tr("Después de ejecutar al menos una detección, presiona <b>Generar reporte HTML</b> "
            "(se crea <code>informe_deteccion.html</code> dentro de la carpeta del run y se abre "
            "en tu navegador) o <b>Exportar a PDF</b> para obtener un archivo listo para enviar. "
            "Todas las imágenes van embebidas en base64, por lo que el archivo es autocontenido.")
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l1.addWidget(info)

        # ── Qué secciones entran ──
        self.body.addWidget(self._card_secciones())

        # ── Alcance del informe ──
        self.body.addWidget(self._card_alcance())

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_gen = QPushButton(tr("Generar informe HTML"))
        self.btn_gen.setIcon(iconos.icono("texto", 15, T.ON_ACCENT))
        self.btn_gen.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_gen.setCursor(Qt.PointingHandCursor)
        self.btn_gen.clicked.connect(self._generate)
        row.addWidget(self.btn_gen)

        self.btn_pdf = QPushButton(tr("Exportar a PDF"))
        self.btn_pdf.setIcon(iconos.icono("reporte", 15, T.ON_ACCENT))
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

        # ── Guardar las fotos con las etiquetas dibujadas ──
        self.body.addWidget(self._card_fotos())

        # ── Contenido del reporte ──
        c2, l2 = self.card(tr("Contenido del reporte"), "lista")
        contents = QLabel(tr(
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
            "<p>Todas las secciones son opcionales: se eligen arriba.</p>")
        )
        contents.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; border: none;")
        contents.setTextFormat(Qt.RichText)
        l2.addWidget(contents)
        self.body.addWidget(c2)

    # ── Fotos etiquetadas ───────────────────────────────────────
    def _card_fotos(self):
        """Tarjeta para guardar cada foto con sus etiquetas como archivo suelto.

        El informe ya las lleva embebidas, pero ahí van recomprimidas y a 1100 px
        de ancho para que el archivo sea manejable. Quien quiera la imagen para
        una figura del paper o para revisarla con lupa necesita el original, y
        eso es lo que copia esta tarjeta.
        """
        c, l = self.card(tr("Guardar las fotos con las etiquetas"), "imagenes")

        ayuda = QLabel(tr(
            "Guarda cada foto analizada con sus cajas dibujadas, en su resolución "
            "original. Eliges una carpeta y dentro se crea una subcarpeta por cada "
            "opción que marques (<code>conteo_manual</code>, "
            "<code>deteccion_modelo</code>, <code>ambas_superpuestas</code>), así "
            "no se mezclan. Respeta el <b>alcance</b> elegido arriba: si marcaste "
            "fotos concretas, solo se guardan esas."))
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l.addWidget(ayuda)

        self.chk_f_manual = QCheckBox(tr("Etiquetas manuales (Ground Truth)"))
        self.chk_f_modelo = QCheckBox(tr("Detecciones del modelo"))
        self.chk_f_ambas = QCheckBox(tr("Las dos superpuestas en la misma foto"))
        self.chk_f_manual.setChecked(True)
        self.chk_f_modelo.setChecked(True)
        for chk in (self.chk_f_manual, self.chk_f_modelo, self.chk_f_ambas):
            chk.setStyleSheet(f"color: {T.INK2}; border: none;")
            l.addWidget(chk)

        fila = QHBoxLayout()
        self.btn_fotos = QPushButton(tr("Guardar fotos etiquetadas"))
        self.btn_fotos.setIcon(iconos.icono("imagenes", 15, T.ON_ACCENT))
        self.btn_fotos.setStyleSheet(
            f"background: {T.VIO}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_fotos.setCursor(Qt.PointingHandCursor)
        self.btn_fotos.clicked.connect(self._guardar_fotos)
        fila.addWidget(self.btn_fotos)
        fila.addStretch(1)
        l.addLayout(fila)

        self.lbl_fotos = QLabel("")
        self.lbl_fotos.setWordWrap(True)
        self.lbl_fotos.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l.addWidget(self.lbl_fotos)
        return c

    # Una subcarpeta por tipo de imagen: mezclarlas obliga a leer el nombre de
    # cada archivo para saber cual es cual, y son cientos.
    CARPETA_MANUAL = "conteo_manual"
    CARPETA_MODELO = "deteccion_modelo"
    CARPETA_AMBAS = "ambas_superpuestas"

    def _fotos_a_exportar(self):
        """[(ruta_origen, subcarpeta, nombre_destino)] segun casillas y alcance."""
        marcadas = self._fotos_marcadas()
        # Con "solo las marcadas" se filtra; con "completo" o "ambos" entra todo,
        # porque "ambos" incluye el trabajo completo.
        solo_marcadas = self.rb_elegidas.isChecked()
        # El alias solo estorba cuando hay un unico modelo: sin el, el archivo
        # se llama igual que la foto original, que es lo comodo para una figura.
        varios = len(self.state.results) > 1

        tareas, vistos = [], set()
        for mi, resultados in self.state.results.items():
            alias = self.state.model_slots[mi].alias
            marca = f"__{alias}" if varios else ""
            for r in resultados:
                if solo_marcadas and Path(r.image_path) not in marcadas:
                    continue
                stem = Path(r.image_path).stem
                candidatos = []
                if self.chk_f_manual.isChecked() and r.gt_path:
                    # El GT no depende del modelo: sin alias en el nombre, y
                    # deduplicado para no escribir el mismo archivo N veces.
                    candidatos.append((r.gt_path, self.CARPETA_MANUAL, stem))
                if self.chk_f_modelo.isChecked() and r.pred_path:
                    candidatos.append((r.pred_path, self.CARPETA_MODELO, stem + marca))
                if self.chk_f_ambas.isChecked() and r.annotated_path:
                    candidatos.append((r.annotated_path, self.CARPETA_AMBAS, stem + marca))
                for origen, carpeta, nombre in candidatos:
                    if origen is None or (carpeta, nombre) in vistos:
                        continue
                    origen = Path(origen)
                    if not origen.exists():
                        continue
                    vistos.add((carpeta, nombre))
                    tareas.append((origen, carpeta, nombre + origen.suffix))
        return tareas

    def _guardar_fotos(self):
        if not self._check_results():
            return
        if not any(c.isChecked() for c in (self.chk_f_manual, self.chk_f_modelo,
                                           self.chk_f_ambas)):
            QMessageBox.warning(
                self, tr("Nada que guardar"),
                tr("Marca al menos una de las tres opciones: etiquetas manuales, "
                   "detecciones del modelo, o las dos superpuestas."))
            return
        if self.rb_elegidas.isChecked() and not self._fotos_marcadas():
            QMessageBox.warning(
                self, tr("Sin fotos marcadas"),
                tr("El alcance está en 'solo las fotos que marque' y no hay "
                   "ninguna marcada."))
            return

        tareas = self._fotos_a_exportar()
        if not tareas:
            QMessageBox.warning(
                self, tr("Nada que guardar"),
                tr("No se encontró ninguna imagen anotada en disco para lo que "
                   "pediste.\n\nSi marcaste 'etiquetas manuales', recuerda que "
                   "solo existen para las fotos que tienen Ground Truth."))
            return

        base = self.state.run_dir if (self.state.run_dir and self.state.run_dir.exists()) else Path.home()
        destino = QFileDialog.getExistingDirectory(
            self, tr("Carpeta donde guardar las fotos etiquetadas"), str(base))
        if not destino:
            return
        destino = Path(destino)

        self.btn_fotos.setEnabled(False)
        self.lbl_fotos.setText(tr("Guardando {} imagen(es)…").format(len(tareas)))
        QApplication.processEvents()
        copiadas, fallos = 0, []
        usadas = set()
        try:
            for origen, carpeta, nombre in tareas:
                try:
                    sub_dir = destino / carpeta
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origen, sub_dir / nombre)
                    usadas.add(carpeta)
                    copiadas += 1
                except OSError as e:
                    fallos.append(f"{carpeta}/{nombre}: {e}")
            msg = "✓ " + tr("{} imagen(es) en {} carpeta(s) dentro de {}").format(
                copiadas, len(usadas), destino)
            if fallos:
                msg += " · " + tr("{} fallaron").format(len(fallos))
            self.lbl_fotos.setText(msg)
            if copiadas and not fallos:
                try:
                    abrir_en_el_sistema(str(destino))
                except Exception:
                    pass
            if fallos:
                QMessageBox.warning(self, tr("Algunas no se pudieron guardar"),
                                    "\n".join(fallos[:12]))
        finally:
            self.btn_fotos.setEnabled(True)

    # ── Alcance ─────────────────────────────────────────────────
    def _card_alcance(self):
        """Tarjeta para elegir si el informe cubre todo el trabajo o unas fotos."""
        c, l = self.card(tr("Alcance del informe"), "diana")

        ayuda = QLabel(tr(
            "Puedes generar el informe del <b>trabajo completo</b>, solo de las "
            "<b>fotos que elijas</b>, o ambos de una vez. Las cifras, los gráficos "
            "y la matriz de confusión se recalculan sobre lo que elijas, así que "
            "el informe siempre describe exactamente las fotos que muestra."))
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l.addWidget(ayuda)

        self.rb_completo = QRadioButton(tr("Trabajo completo (todas las fotos analizadas)"))
        self.rb_elegidas = QRadioButton(tr("Solo las fotos que marque abajo"))
        self.rb_ambos = QRadioButton(tr("Ambos: un informe completo y otro con las marcadas"))
        self.rb_completo.setChecked(True)
        self._grupo_alcance = QButtonGroup(self)
        for rb in (self.rb_completo, self.rb_elegidas, self.rb_ambos):
            rb.setStyleSheet(f"color: {T.INK2}; border: none;")
            self._grupo_alcance.addButton(rb)
            rb.toggled.connect(self._sync_alcance)
            l.addWidget(rb)

        fila = QHBoxLayout()
        self.btn_todas = QPushButton(tr("Marcar todas"))
        self.btn_ninguna = QPushButton(tr("Desmarcar todas"))
        self.btn_todas.clicked.connect(lambda: self._marcar_todas(True))
        self.btn_ninguna.clicked.connect(lambda: self._marcar_todas(False))
        for b in (self.btn_todas, self.btn_ninguna):
            b.setCursor(Qt.PointingHandCursor)
            fila.addWidget(b)
        fila.addStretch(1)
        self.lbl_sel = QLabel("")
        self.lbl_sel.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        fila.addWidget(self.lbl_sel)
        l.addLayout(fila)

        self.lst_fotos = QListWidget()
        self.lst_fotos.setMaximumHeight(220)
        self.lst_fotos.itemChanged.connect(self._sync_alcance)
        l.addWidget(self.lst_fotos)

        self._sync_alcance()
        return c

    def refresh(self):
        """Rellena la lista con las fotos que realmente se analizaron."""
        super_refresh = getattr(super(), "refresh", None)
        if callable(super_refresh):
            super_refresh()
        rutas = sorted({r.image_path for rs in self.state.results.values() for r in rs},
                       key=lambda p: p.name)
        previas = self._fotos_marcadas()
        self.lst_fotos.blockSignals(True)
        self.lst_fotos.clear()
        for p in rutas:
            it = QListWidgetItem(p.name)
            it.setData(Qt.UserRole, str(p))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if p in previas else Qt.Unchecked)
            self.lst_fotos.addItem(it)
        self.lst_fotos.blockSignals(False)
        self._sync_alcance()

    def _marcar_todas(self, marcar: bool):
        self.lst_fotos.blockSignals(True)
        for i in range(self.lst_fotos.count()):
            self.lst_fotos.item(i).setCheckState(Qt.Checked if marcar else Qt.Unchecked)
        self.lst_fotos.blockSignals(False)
        self._sync_alcance()

    def _fotos_marcadas(self) -> set:
        marcadas = set()
        for i in range(self.lst_fotos.count()):
            it = self.lst_fotos.item(i)
            if it.checkState() == Qt.Checked:
                marcadas.add(Path(it.data(Qt.UserRole)))
        return marcadas

    def _sync_alcance(self):
        """Habilita la lista solo cuando el alcance la usa, y avisa si esta vacia."""
        usa_lista = self.rb_elegidas.isChecked() or self.rb_ambos.isChecked()
        for w in (self.lst_fotos, self.btn_todas, self.btn_ninguna):
            w.setEnabled(usa_lista)
        n = len(self._fotos_marcadas())
        total = self.lst_fotos.count()
        if not usa_lista:
            self.lbl_sel.setText(tr("{} foto(s) en el trabajo").format(total))
        elif n == 0:
            self.lbl_sel.setText(tr("⚠ ninguna marcada"))
        else:
            self.lbl_sel.setText(tr("{} de {} marcadas").format(n, total))

    def _check_results(self) -> bool:
        """Avisa si todavía no hay detecciones ejecutadas."""
        if not self.state.results:
            QMessageBox.warning(
                self, tr("Sin resultados"),
                tr("Aún no has ejecutado ninguna detección.\n"
                "Ve a la pestaña Ejecutar e inicia el análisis primero.")
            )
            return False
        return True

    def _card_secciones(self):
        """Casillas para elegir qué secciones entran en el informe.

        Se dibujan leyendo ``SECCIONES`` del generador, no una lista repetida
        aquí: si mañana se añade una sección al informe aparece sola su casilla,
        y no puede quedar una sección imposible de desmarcar.
        """
        c, l = self.card(tr("Qué incluir en el informe"), "reporte")
        ayuda = QLabel(tr(
            "El informe completo es largo. Desmarca lo que no necesites: las secciones "
            "se renumeran solas y el índice se ajusta. Una sección marcada que no tenga "
            "datos —errores sin ground truth, por ejemplo— se omite igualmente."))
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l.addWidget(ayuda)

        fila = QHBoxLayout()
        fila.setSpacing(8)
        for clave, etiqueta in (("completo", tr("Completo")),
                                ("resumen", tr("Resumen breve")),
                                ("metodologico", tr("Metodológico"))):
            b = QPushButton(etiqueta)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                f"background: white; color: {T.INK2}; border: 1px solid #d0d7de; "
                f"border-radius: 6px; padding: 5px 12px;")
            b.clicked.connect(lambda _=False, k=clave: self._aplicar_preset(k))
            fila.addWidget(b)
        fila.addStretch(1)
        l.addLayout(fila)

        self.chk_secciones = {}
        for sid, titulo in SECCIONES:
            # El titulo se traduce aqui y no en SECCIONES porque esa misma lista
            # alimenta el documento HTML, que se genera en espanol.
            chk = QCheckBox(tr(titulo))
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {T.INK2}; border: none;")
            l.addWidget(chk)
            self.chk_secciones[sid] = chk

        # La galería es lo que hace que el archivo pese: cada foto va embebida en
        # base64 dentro del propio HTML.
        self.chk_secciones["galeria"].setText(
            tr("Galería por imagen  (es lo que más pesa)"))

        # A diferencia del resto, esta no viene marcada de fábrica: solo tiene
        # sentido cuando las imágenes analizadas están repartidas en varias
        # carpetas (sitios, tratamientos...) y el usuario quiere esa
        # comparación en particular, no en cualquier informe.
        self.chk_secciones["talla_carpetas"].setChecked(False)
        return c

    def _aplicar_preset(self, clave: str):
        elegidas = set(PRESETS.get(clave, []))
        for sid, chk in self.chk_secciones.items():
            chk.setChecked(sid in elegidas)

    def _secciones_marcadas(self) -> list:
        return [sid for sid, chk in self.chk_secciones.items() if chk.isChecked()]

    def _write_html(self, out_path: Path, solo_imagenes=None) -> Path:
        """Genera el HTML autocontenido (imágenes en base64) en `out_path`."""
        generate_report(
            self.state, out_path,
            secciones=self._secciones_marcadas(),
            solo_imagenes=solo_imagenes,
        )
        return out_path

    def _trabajos(self) -> list:
        """Informes a generar según el alcance: (sufijo, fotos, etiqueta).

        ``fotos`` None significa el trabajo completo. Devuelve lista vacía si el
        alcance pide fotos marcadas y no hay ninguna, para no generar en silencio
        un informe vacío que parecería decir que no se detectó nada.
        """
        marcadas = self._fotos_marcadas()
        pide_marcadas = self.rb_elegidas.isChecked() or self.rb_ambos.isChecked()
        if pide_marcadas and not marcadas:
            QMessageBox.warning(
                self, tr("Sin fotos marcadas"),
                tr("Elegiste generar el informe de fotos concretas, pero no hay "
                   "ninguna marcada.\n\nMarca al menos una en la lista, o cambia "
                   "el alcance a 'Trabajo completo'."))
            return []

        trabajos = []
        if self.rb_completo.isChecked() or self.rb_ambos.isChecked():
            trabajos.append(("", None, tr("trabajo completo")))
        if pide_marcadas:
            trabajos.append(("_seleccion", marcadas,
                             tr("{} foto(s) marcada(s)").format(len(marcadas))))
        return trabajos

    def _generate(self):
        if not self._check_results():
            return

        out_dir = self.state.run_dir
        if out_dir is None or not out_dir.exists():
            # Pedir carpeta destino
            d = QFileDialog.getExistingDirectory(self, "Carpeta destino del reporte")
            if not d:
                return
            out_dir = Path(d)

        trabajos = self._trabajos()
        if not trabajos:
            return

        self.btn_gen.setEnabled(False)
        self.lbl_status.setText(tr("Generando reporte… (puede tardar unos segundos)"))
        generados = []
        try:
            for sufijo, fotos, etiqueta in trabajos:
                out_path = out_dir / f"informe_deteccion{sufijo}.html"
                self._write_html(out_path, solo_imagenes=fotos)
                generados.append(out_path)
            self.lbl_status.setText(
                "✓ " + tr("{} informe(s) generado(s) en {}").format(len(generados), out_dir))
            for p in generados:
                try:
                    abrir_en_el_sistema(str(p))
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
                self, tr("PDF no disponible"),
                tr("Para exportar a PDF se necesita QtWebEngine (incluido en PySide6-Addons).\n\n"
                "Alternativa: pulsa 'Generar reporte HTML' y, en el navegador, usa\n"
                "Ctrl+P → 'Guardar como PDF'.")
            )
            return

        trabajos = self._trabajos()
        if not trabajos:
            return

        # Carpeta del run como base; si no existe, junto al PDF elegido
        base_dir = self.state.run_dir if (self.state.run_dir and self.state.run_dir.exists()) else Path.home()
        suggested = str(base_dir / "informe_deteccion.pdf")
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar informe como PDF", suggested, "PDF (*.pdf)"
        )
        if not path:
            return
        pdf_base = Path(path)
        html_dir = self.state.run_dir if (self.state.run_dir and self.state.run_dir.exists()) else pdf_base.parent

        self.btn_pdf.setEnabled(False)
        self.btn_gen.setEnabled(False)
        self.lbl_status.setText(tr("Generando PDF… (renderizando con el motor del navegador, unos segundos)"))
        QApplication.processEvents()
        try:
            hechos, fallidos = [], []
            for sufijo, fotos, etiqueta in trabajos:
                html_path = html_dir / f"informe_deteccion{sufijo}.html"
                # Con "ambos" se escriben dos PDF: el nombre elegido por el
                # usuario para el completo, y ese mismo con sufijo para la
                # seleccion. Reutilizar el nombre pisaria el primero.
                pdf_path = (pdf_base if not sufijo
                            else pdf_base.with_name(pdf_base.stem + sufijo + ".pdf"))
                self._write_html(html_path, solo_imagenes=fotos)
                if pdf_export.html_to_pdf(html_path, pdf_path):
                    hechos.append(pdf_path)
                else:
                    fallidos.append(etiqueta)

            if hechos:
                self.lbl_status.setText(
                    "✓ " + tr("{} PDF generado(s) en {}").format(len(hechos), pdf_base.parent))
                for p in hechos:
                    try:
                        abrir_en_el_sistema(str(p))
                    except Exception:
                        pass
            if fallidos:
                self.lbl_status.setText(tr("✗ No se pudo generar todo el PDF."))
                QMessageBox.warning(
                    self, tr("No se pudo generar el PDF"),
                    tr("Falló el renderizado de: " + ", ".join(fallidos) +
                       ".\n\nComo alternativa, genera el HTML y usa\n"
                       "Ctrl+P → 'Guardar como PDF' en tu navegador.")
                )
        except Exception as e:
            QMessageBox.critical(self, tr("Error generando PDF"), f"{type(e).__name__}: {e}")
            self.lbl_status.setText(tr("✗ Falló la generación de PDF."))
        finally:
            self.btn_pdf.setEnabled(True)
            self.btn_gen.setEnabled(True)
