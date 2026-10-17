"""Página 6 — Resultados. KPIs grandes + histograma de tamaños + tabla por imagen."""
from __future__ import annotations
import csv
import io
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QFileDialog, QFrame, QProgressDialog, QMessageBox,
    QApplication,
)

from ._base import DetectorPage
from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.gui.widgets import KPICard
from polyx.model.metrics import LABEL_TP, LABEL_FP, LABEL_FN, LABEL_MISCLS, match_image
from polyx.model.i18n import tr
from polyx.model.plataforma import abrir_en_el_sistema


class ResultadosPage(DetectorPage):
    STEP_N = 6
    STEP_TITLE = tr("Resultados")
    STEP_DESCRIPTION = (
        tr("Resumen cuantitativo del análisis. Para gráficos completos y galería por imagen, "
        "genera el reporte HTML en la pestaña Reporte.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── KPIs ──
        c1, l1 = self.card(tr("Métricas generales"), "resultados")
        grid = QGridLayout()
        grid.setSpacing(12)
        self.kpi_imgs   = KPICard(tr("Imágenes"), T.ACCENT)
        self.kpi_dets   = KPICard(tr("Detecciones"), T.ACCENT)
        self.kpi_conf   = KPICard(tr("Confianza media"), T.VIO)
        self.kpi_size   = KPICard(tr("Tamaño medio (μm)"), T.WARN)
        self.kpi_tp     = KPICard(LABEL_TP, T.OK)
        self.kpi_fp     = KPICard(LABEL_FP, T.WARN)
        self.kpi_fn     = KPICard(LABEL_FN, T.ERR)
        self.kpi_f1     = KPICard("F1", T.ACCENT_D)
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

        # ── Métricas detalladas ──
        c2, l2 = self.card(tr("Métricas detalladas (suma de todos los modelos)"), "lista")
        row = QHBoxLayout()
        row.setSpacing(20)
        self.lbl_prec = QLabel(tr("Precisión:  —"))
        self.lbl_rec  = QLabel(tr("Recall:  —"))
        self.lbl_mis  = QLabel(f"{LABEL_MISCLS}:  —")
        for l in (self.lbl_prec, self.lbl_rec, self.lbl_mis):
            l.setStyleSheet(f"color: {T.INK2}; font-size: 10.5pt; font-weight: 500; border: none;")
            row.addWidget(l)
        row.addStretch(1)
        l2.addLayout(row)
        # Sugerencia automática del umbral de confianza óptimo (F1 máximo)
        self.lbl_sugg = QLabel("")
        self.lbl_sugg.setWordWrap(True)
        self.lbl_sugg.setStyleSheet(
            f"color: {T.OK_TX}; font-size: 10pt; font-weight: 600; border: none;")
        l2.addWidget(self.lbl_sugg)
        self.body.addWidget(c2)

        # ── Histograma de tamaños (solo con calibración) ──
        self.c_hist, l_hist = self.card(
            tr("Distribución de tamaños (μm) — solo con calibración activa"), "visor"
        )
        self._hist_placeholder = QLabel(
            tr("Configura μm/px en Parámetros para ver la distribución de tamaños.")
        )
        self._hist_placeholder.setAlignment(Qt.AlignCenter)
        self._hist_placeholder.setStyleSheet(
            f"color: {T.INK3}; font-size: 10pt; border: none; padding: 24px;"
        )
        l_hist.addWidget(self._hist_placeholder)
        self._hist_canvas = None
        self.body.addWidget(self.c_hist)

        # ── Recargar GT del disco ──
        c_gt, l_gt = self.card(tr("Ground truth"), "recargar")
        nota_gt = QLabel(tr(
            "No pide ninguna carpeta: relee los .txt que están junto a cada "
            "foto, los mismos que ves en GT manual. Hace falta porque la corrida "
            "guarda el ground truth que leyó, así que corregir una anotación "
            "después no cambia por sí solo estas métricas ni el informe. "
            "Recalcula sin pasar el modelo otra vez y solo redibuja las imágenes "
            "cuyo .txt cambió."))
        nota_gt.setWordWrap(True)
        nota_gt.setStyleSheet(
            f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l_gt.addWidget(nota_gt)
        self.lbl_recarga = QLabel("")
        self.lbl_recarga.setWordWrap(True)
        self.lbl_recarga.setStyleSheet(
            f"color: {T.OK_TX}; font-size: 10pt; font-weight: 600; border: none;")
        row_gt = QHBoxLayout()
        self.btn_recargar = QPushButton(tr("Releer los .txt del disco y recalcular"))
        self.btn_recargar.setIcon(iconos.icono("recargar", 15, T.INK2))
        self.btn_recargar.clicked.connect(self._recargar_gt)
        row_gt.addWidget(self.btn_recargar)
        row_gt.addStretch(1)
        l_gt.addLayout(row_gt)
        l_gt.addWidget(self.lbl_recarga)
        self.body.addWidget(c_gt)

        # ── Exportar CSV ──
        c_csv, l_csv = self.card(tr("Exportar datos"), "guardar")
        row_csv = QHBoxLayout()
        btn_csv = QPushButton(tr("Exportar CSV de detecciones"))
        btn_csv.setIcon(iconos.icono("texto", 15, T.INK2))
        btn_csv.clicked.connect(self._export_csv)
        row_csv.addWidget(btn_csv)
        row_csv.addStretch(1)
        l_csv.addLayout(row_csv)
        self.body.addWidget(c_csv)

        # ── Tabla por imagen ──
        c3, l3 = self.card(tr("Por imagen"), "imagenes")
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Modelo", "Imagen", "Pred", "GT", LABEL_TP, LABEL_FP, LABEL_FN, "Conf media"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._open_row)
        self.table.setMinimumHeight(280)
        l3.addWidget(self.table)
        fila_rev = QHBoxLayout()
        btn_rev = QPushButton(tr("Revisar partícula a partícula en el Visor"))
        btn_rev.setIcon(iconos.icono("buscar", 15, T.INK2))
        btn_rev.clicked.connect(self._revisar_en_visor)
        fila_rev.addWidget(btn_rev)
        fila_rev.addStretch(1)
        l3.addLayout(fila_rev)
        nota_rev = QLabel(tr(
            "Abre la imagen seleccionada en el Visor con estas mismas detecciones, "
            "numeradas. Allí cada partícula muestra sobre qué se midió y si es "
            "fibra o fragmento — sin volver a pasar el modelo."))
        nota_rev.setWordWrap(True)
        nota_rev.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
        l3.addWidget(nota_rev)
        self.body.addWidget(c3)

        # Suscribir
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

        self.kpi_imgs.set_value(total_imgs)
        self.kpi_dets.set_value(total_dets)
        self.kpi_conf.set_value(f"{(sum(confs)/len(confs)):.3f}" if confs else None)
        self.kpi_size.set_value(f"{(sum(sizes)/len(sizes)):.1f}" if sizes else None)
        self.kpi_tp.set_value(tp)
        self.kpi_fp.set_value(fp)
        self.kpi_fn.set_value(fn)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        any_gt = any(r.has_gt for r in all_results)
        self.kpi_f1.set_value(f"{f1:.3f}" if any_gt else None)
        self.lbl_prec.setText(f"Precisión:  {prec:.3f}" if any_gt else "Precisión:  —")
        self.lbl_rec.setText(f"Recall:  {rec:.3f}" if any_gt else "Recall:  —")
        mc = sum(r.miscls for r in all_results)
        self.lbl_mis.setText(f"{LABEL_MISCLS}:  {mc}" if any_gt else f"{LABEL_MISCLS}:  —")

        # Sugerencia de umbral óptimo, modelo por modelo
        self.lbl_sugg.setText(self._texto_sugerencia() if any_gt else "")

        # Histograma de tamaños (solo si hay calibración)
        self._update_histogram()

        # Tabla
        rows = []
        for mi, rs in state.results.items():
            slot = state.model_slots[mi]
            for r in rs:
                avg_c = sum(p.conf for p in r.predictions) / len(r.predictions) if r.predictions else 0
                rows.append((slot.alias, r.image_path, len(r.predictions),
                             len(r.gt), r.tp, r.fp, r.fn, avg_c))
        self.table.setRowCount(len(rows))
        for i, (alias, p, pred, gt, tp, fp, fn, ac) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(alias))
            self.table.setItem(i, 1, QTableWidgetItem(p.name))
            for col, v in enumerate([pred, gt, tp, fp, fn], start=2):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, it)
            self.table.setItem(i, 7, QTableWidgetItem(f"{ac:.3f}"))
            # guardamos la ruta para doble-clic
            self.table.item(i, 1).setData(Qt.UserRole, str(p))
            self.table.item(i, 0).setData(Qt.UserRole, alias)

    def _recargar_gt(self):
        """Relee los .txt del disco y recalcula, sin volver a inferir."""
        from ..recargar_gt import recargar_gt

        if not self.state.results:
            self.lbl_recarga.setText(
                "No hay resultados que recalcular: ejecuta primero la detección.")
            return

        dlg = QProgressDialog(tr("Releyendo ground truth…"), tr("Cancelar"),
                              0, 0, self)
        dlg.setWindowTitle(tr("Recargar GT"))
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        cancelado = {"si": False}

        def progreso(hechas, total, nombre):
            if dlg.wasCanceled():
                cancelado["si"] = True
                return False
            dlg.setMaximum(total)
            dlg.setValue(hechas)
            dlg.setLabelText(f"{nombre}  ({hechas}/{total})")
            QApplication.processEvents()

        self.btn_recargar.setEnabled(False)
        try:
            resumen = recargar_gt(self.state, progreso=progreso)
        except Exception as e:
            dlg.close()
            self.btn_recargar.setEnabled(True)
            QMessageBox.warning(self, tr("Recargar GT"), f"{type(e).__name__}: {e}")
            return
        finally:
            dlg.close()
            self.btn_recargar.setEnabled(True)

        cambiadas = resumen["cambiadas"]
        if cambiadas:
            lista = ", ".join(cambiadas[:8])
            if len(cambiadas) > 8:
                lista += f" y {len(cambiadas) - 8} más"
            texto = (f"✓ {len(cambiadas)} imagen(es) con ground truth distinto: "
                     f"{lista}. Métricas e imágenes de control actualizadas; "
                     f"ya puedes generar el informe.")
        else:
            texto = (f"✓ Revisadas {resumen['revisadas']} imágenes: ningún .txt "
                     f"cambió respecto a la corrida. No había nada que recalcular.")
        if resumen["sin_gt"]:
            texto += f"  ({len(resumen['sin_gt'])} sin .txt de control.)"
        if cancelado["si"]:
            texto += "  Se canceló a mitad: vuelve a pulsar para terminar."
        self.lbl_recarga.setText(texto)
        self.refresh()

    def _resultados_por_modelo(self):
        """[(alias, [ImageResult con GT])], en el orden de los slots."""
        salida = []
        for mi, rs in sorted(self.state.results.items()):
            if mi >= len(self.state.model_slots):
                continue
            con_gt = [r for r in rs if r.has_gt]
            if con_gt:
                salida.append((self.state.model_slots[mi].alias, con_gt))
        return salida

    def _umbral_comun(self, por_modelo):
        """Umbral unico que maximiza el F1 medio de todos los modelos.

        Comparar dos arquitecturas solo es limpio si ambas se miden en el mismo
        punto de operacion. Si cada una usa su propio optimo se compara tambien
        el ajuste del umbral, y el veredicto del informe deja de hablar de la
        arquitectura. Devuelve (umbral, f1_medio) o None con un solo modelo.
        """
        if len(por_modelo) < 2:
            return None
        candidatos = sorted({round(p.conf, 2) for _, rs in por_modelo
                             for r in rs for p in r.predictions})
        if not candidatos:
            return None
        paso = max(1, len(candidatos) // 25)
        mejor = None
        for t in candidatos[::paso]:
            f1s = [self._metricas_a(rs, t)[0] for _, rs in por_modelo]
            medio = sum(f1s) / len(f1s)
            if mejor is None or medio > mejor[1]:
                mejor = (t, medio)
        return mejor

    def _texto_sugerencia(self) -> str:
        """Consejo de confianza, una linea por modelo mas la recomendacion.

        El barrido va por modelo a proposito: sumando los dos, el optimo sale
        de una mezcla que no corresponde a ninguno, y es justo el numero con el
        que se decide cual gana.
        """
        por_modelo = self._resultados_por_modelo()
        if not por_modelo:
            return ""
        usado = self.state.params.conf
        lineas: list[str] = []
        optimos: list[float] = []
        for alias, rs in por_modelo:
            sugg = self._suggest_operating_point(rs)
            if not sugg:
                continue
            (tb, f1b, pb, rb), (ta, f1a, pa, ra) = sugg
            cabeza = f"{alias} — conf {ta:g}: F1 {f1a:.3f} (P {pa:.3f} / R {ra:.3f})."
            # Se exige mejora real, no solo un umbral distinto: con F1 plano el
            # maximo cae en cualquier punto y aconsejar moverse ahi seria ruido.
            if tb > ta + 1e-9 and (f1b - f1a) > 0.002:
                señal = "⚠" if (f1b - f1a) >= 0.01 else "·"
                lineas.append(
                    f"{señal} {cabeza}  Su mejor umbral es {tb:.2f}: F1 {f1b:.3f} "
                    f"({f1b - f1a:+.3f}), P {pb:.3f} / R {rb:.3f}.")
                optimos.append(tb)
            else:
                lineas.append(
                    f"✓ {cabeza}  Es el mejor del rango explorado; por debajo de "
                    f"{ta:g} no hay detecciones calculadas y no se recuperan "
                    f"filtrando.")
                optimos.append(ta)
        if not lineas:
            return ""

        comun = self._umbral_comun(por_modelo)
        distintos = len({round(t, 2) for t in optimos}) > 1
        if comun and distintos:
            lineas.append(
                f"➜ Cada modelo prefiere un umbral distinto. Usa {comun[0]:.2f} "
                f"para los dos (F1 medio {comun[1]:.3f}): es el mismo punto de "
                f"operación y la comparación queda limpia. Cámbialo en "
                f"Parámetros y vuelve a Ejecutar.")
        elif max(optimos) > usado + 1e-9:
            lineas.append(
                f"➜ Cambia la confianza a {max(optimos):.2f} en Parámetros y "
                f"vuelve a Ejecutar antes de generar el informe.")
        else:
            lineas.append(
                f"➜ La confianza {usado:g} ya es el punto de operación bueno. "
                f"Puedes generar el informe.")
        return "\n".join(lineas)

    def _metricas_a(self, gt_results, umbral: float):
        """P/R/F1 estrictos filtrando las predicciones a un umbral dado.

        Estricto: una caja bien situada con la clase equivocada cuenta como
        falso positivo de la clase predicha y falso negativo de la real. El
        criterio permisivo, que la deja fuera de ambos denominadores, sobreestima
        el desempeno y ademas contradice la tabla de precision por clase.
        """
        iou_thr = self.state.params.iou_tp
        tp = fp = fn = mc = 0
        for r in gt_results:
            preds = [p for p in r.predictions if p.conf >= umbral]
            m = match_image(preds, r.gt, iou_thr)
            tp += m.tp; fp += m.fp; fn += m.fn; mc += m.miscls
        prec = tp / (tp + fp + mc) if (tp + fp + mc) else 0.0
        rec = tp / (tp + fn + mc) if (tp + fn + mc) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return f1, prec, rec

    def _suggest_operating_point(self, all_results):
        """Busca el umbral de confianza con mejor F1 sobre lo ya detectado.

        Devuelve (mejor, actual) con (umbral, f1, precision, recall) cada uno.
        Solo puede explorar hacia ARRIBA del umbral con que se ejecuto: las
        detecciones por debajo nunca se calcularon.
        """
        gt_results = [r for r in all_results if r.has_gt]
        if not gt_results:
            return None
        confs = sorted({round(p.conf, 2) for r in gt_results for p in r.predictions})
        if not confs:
            return None
        step = max(1, len(confs) // 25)   # acotar a ~25 umbrales
        mejor = None
        for t in confs[::step]:
            f1, prec, rec = self._metricas_a(gt_results, t)
            if mejor is None or f1 > mejor[1]:
                mejor = (t, f1, prec, rec)
        usado = self.state.params.conf
        f1_a, p_a, r_a = self._metricas_a(gt_results, usado)
        return mejor, (usado, f1_a, p_a, r_a)

    def _update_histogram(self):
        """Dibuja histograma de diam_um por clase; solo si hay calibración."""
        um_per_px = getattr(self.state.params, "um_per_px", 0.0)
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
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            if self._hist_canvas is None:
                fig = Figure(figsize=(7, 3), tight_layout=True)
                self._hist_ax = fig.add_subplot(111)
                self._hist_canvas = FigureCanvasQTAgg(fig)
                self._hist_canvas.setMinimumHeight(220)
                # Insertar después del placeholder
                l_hist = self.c_hist.layout()
                l_hist.addWidget(self._hist_canvas)

            ax = self._hist_ax
            ax.clear()
            ax.set_facecolor("#f6f8fa")
            ax.figure.set_facecolor("#ffffff")

            colors = {"PET": "#e3342f", "PP": "#ff8c00", "LDPE": "#ffd700"}
            for cls_name, vals in sizes_by_class.items():
                color = colors.get(cls_name, "#0969da")
                ax.hist(vals, bins=20, alpha=0.72, label=f"{cls_name} (n={len(vals)})",
                        color=color, edgecolor="white", linewidth=0.5)

            ax.set_xlabel("Diámetro equivalente (μm)", fontsize=9)
            ax.set_ylabel("Frecuencia", fontsize=9)
            ax.set_title("Distribución de tamaños por clase", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(labelsize=8)
            self._hist_canvas.draw()
            self._hist_placeholder.setVisible(False)
            self._hist_canvas.setVisible(True)
        except Exception as e:
            self._hist_placeholder.setText(f"matplotlib no disponible: {e}")
            self._hist_placeholder.setVisible(True)

    def _export_csv(self):
        """Exporta todas las detecciones a un archivo CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar CSV", "detecciones.csv", "CSV (*.csv)"
        )
        if not path:
            return
        all_results = [r for rs in self.state.results.values() for r in rs]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                # Se exporta la forma real ademas del diametro equivalente: para
                # una fibra el diametro equivalente no es la magnitud que se
                # reporta en la literatura, que es su largo.
                w.writerow(["modelo", "imagen", "clase", "conf",
                            "x1", "y1", "x2", "y2",
                            "um_por_px", "origen_escala",
                            "largo_um", "metodo_largo", "feret_um",
                            "geodesico_um", "largo_rect_eq_um",
                            "ancho_um", "area_um2", "diam_um",
                            "aspecto", "curvatura", "morfotipo"])
                cals = getattr(self.state, "calibraciones", None) or {}
                for mi, rs in self.state.results.items():
                    alias = self.state.model_slots[mi].alias
                    for r in rs:
                        cal = cals.get(r.image_path.name)
                        um = f"{cal.um_por_px:.4f}" if cal and cal.valida else ""
                        origen = cal.origen if cal and cal.valida else ""
                        for p in r.predictions:
                            def _n(v, dec=2):
                                return f"{v:.{dec}f}" if v else ""
                            w.writerow([
                                alias, r.image_path.name,
                                p.class_name, f"{p.conf:.4f}",
                                f"{p.x1:.1f}", f"{p.y1:.1f}",
                                f"{p.x2:.1f}", f"{p.y2:.1f}",
                                um, origen,
                                _n(p.largo_um, 1), p.metodo_largo or "",
                                _n(p.feret_um, 1), _n(p.geodesico_um, 1),
                                _n(p.largo_rect_eq_um, 1), _n(p.ancho_um, 1),
                                _n(p.area_um2, 1), _n(p.diam_um),
                                _n(p.aspecto), _n(p.curvatura),
                                p.morfotipo or "",
                            ])
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, tr("Error"), f"No se pudo guardar: {e}")

    def _open_row(self, row: int, col: int):
        if row < 0 or self.state.run_dir is None: return
        alias = self.table.item(row, 0).data(Qt.UserRole)
        img = Path(self.table.item(row, 1).data(Qt.UserRole))
        annot = self.state.run_dir / alias / f"{img.stem}_annot.png"
        if annot.exists():
            abrir_en_el_sistema(str(annot))

    def _revisar_en_visor(self):
        """Abre la imagen seleccionada en el Visor, con sus detecciones puestas.

        Revisar partícula a partícula es una tarea interactiva y no cabe en un
        informe: doce fichas ya pesan más de un megabyte. Pero tampoco debería
        obligar a salir del Detector, buscar el archivo y volver a detectar, que
        además correría el modelo por segunda vez sobre la misma foto. Aquí se le
        pasan al Visor las detecciones que YA se calcularon, de modo que lo que
        se revisa es exactamente lo que produjo la corrida.
        """
        filas = self.table.selectionModel().selectedRows()
        if not filas:
            QMessageBox.information(
                self, tr("Revisar"),
                tr("Selecciona primero una fila de la tabla de abajo."))
            return
        row = filas[0].row()
        alias = self.table.item(row, 0).data(Qt.UserRole)
        img = Path(self.table.item(row, 1).data(Qt.UserRole))

        # Las detecciones de ESA imagen con ESE modelo.
        dets = []
        for mi, rs in self.state.results.items():
            if self.state.model_slots[mi].alias != alias:
                continue
            for r in rs:
                if Path(r.image_path) == img:
                    dets = list(r.predictions)
                    break
        if not dets:
            QMessageBox.information(
                self, tr("Revisar"),
                tr("Esa imagen no tiene detecciones que revisar."))
            return

        try:
            from polyx.gui.visor.window import VisorWindow
        except Exception as e:
            QMessageBox.warning(self, tr("Revisar"),
                                f"No se pudo abrir el Visor: {e}")
            return

        # Se guarda en self para que no lo recoja el recolector de basura al
        # salir de la función: una ventana Qt sin referencia viva se cierra sola.
        self._visor = VisorWindow()
        self._visor.state.load_single(img)
        cal = (getattr(self.state, "calibraciones", None) or {}).get(img.name)
        if cal is not None and getattr(cal, "valida", False):
            self._visor.state.um_per_px = cal.um_por_px
        elif self.state.params.um_per_px > 0:
            self._visor.state.um_per_px = self.state.params.um_per_px
        self._visor.state.detections = dets
        self._visor.state.detections_changed.emit(dets)
        self._visor.show()
        self._visor.raise_()
