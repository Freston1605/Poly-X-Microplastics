"""Page 3 — `Parámetros`: training parameters.

Includes the key utility `Maximizar imgsz para mi GPU`, which detects the
available VRAM and proposes the highest imgsz that fits without OOM (the
user's goal: always train as large as possible).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from polyx.core.i18n import tr
from polyx.gui import iconos
from polyx.gui import theme as T

from ..hw import detect_gpu, estimate_vram_gb, humanize_gb, recommend_batch, recommend_max_imgsz
from ._base import TrainerPage


# GPU detection in the background so the GUI does not freeze at startup
# (torch.cuda.is_available() can take 10–30 s the first time).
class _GpuDetectWorker(QThread):
    detected = Signal(object)  # GPUInfo

    def run(self):
        info = detect_gpu()
        self.detected.emit(info)


IMGSZ_CHOICES = [320, 416, 512, 640, 800, 960, 1024, 1280, 1600, 1920, 2560, 3840]


def _hint(text: str) -> QLabel:
    l = QLabel(text)  # noqa: E741
    l.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
    l.setWordWrap(True)
    return l


class ParametersPage(TrainerPage):
    PAGE_ICON = "ajustes"
    PAGE_TITLE = tr("Parámetros de entrenamiento")
    PAGE_DESCRIPTION = tr(
        "Cada parámetro tiene un hint con su explicación. Los valores por defecto son "
        "sensatos para microplásticos. Si tocas algo, el preset cambia a 'Personalizado'."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Basic configuration (includes Maximize imgsz) ──
        c1, l1 = self.card(tr("Configuración básica"), "piezas")
        g = QGridLayout()
        g.setHorizontalSpacing(20)
        g.setVerticalSpacing(10)

        # Imgsz: combobox with high sizes as the main values
        g.addWidget(QLabel(tr("Tamaño imagen (imgsz):")), 0, 0)
        self.combo_imgsz = QComboBox()
        self.combo_imgsz.setEditable(True)
        for v in IMGSZ_CHOICES:
            self.combo_imgsz.addItem(str(v))
        self.combo_imgsz.setCurrentText(str(state.params.imgsz))
        self.combo_imgsz.currentTextChanged.connect(self._on_change)
        g.addWidget(self.combo_imgsz, 0, 1)
        g.addWidget(
            _hint(tr("MÁS ALTO = mejor para partículas pequeñas. Recomendado: 1280+.")), 0, 2
        )

        # Epochs
        g.addWidget(QLabel(tr("Épocas:")), 1, 0)
        self.sb_epochs = QSpinBox()
        self.sb_epochs.setRange(1, 2000)
        self.sb_epochs.setValue(state.params.epochs)
        self.sb_epochs.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_epochs, 1, 1)
        g.addWidget(
            _hint(tr("150–300 típico. Early stopping detiene si no mejora (ver Patience).")), 1, 2
        )

        # Batch
        g.addWidget(QLabel(tr("Batch size:")), 2, 0)
        self.sb_batch = QSpinBox()
        self.sb_batch.setRange(1, 256)
        self.sb_batch.setValue(state.params.batch)
        self.sb_batch.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_batch, 2, 1)
        g.addWidget(_hint(tr("Bajar si hay OOM. Subir si la GPU tiene RAM de sobra.")), 2, 2)

        # Cache
        g.addWidget(QLabel(tr("Cache:")), 3, 0)
        self.combo_cache = QComboBox()
        self.combo_cache.addItems(["disk", "ram", "False"])
        self.combo_cache.setCurrentText(state.params.cache)
        self.combo_cache.currentTextChanged.connect(self._on_change)
        g.addWidget(self.combo_cache, 3, 1)
        g.addWidget(
            _hint(
                tr(
                    "'ram' es lo más rápido si cabe; 'disk' es seguro; 'False' para datasets gigantes."
                )
            ),
            3,
            2,
        )

        l1.addLayout(g)

        # ── Sub-panel: GPU detection + maximize imgsz ──
        gpu_frame = QFrame()
        gpu_frame.setStyleSheet(
            f"QFrame {{ background: {T.BG_SOFT}; border: 1px solid {T.RULE}; border-radius: 6px; }}"
        )
        gf = QVBoxLayout(gpu_frame)
        gf.setContentsMargins(14, 10, 14, 12)
        gf.setSpacing(8)

        title = QLabel(tr("Maximizar imgsz para mi GPU"))
        title.setStyleSheet(f"color: {T.INK}; font-size: 11pt; font-weight: 600; border: none;")
        gf.addWidget(title)
        self.lbl_gpu = QLabel(tr("Detectando GPU…"))
        self.lbl_gpu.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; border: none;")
        gf.addWidget(self.lbl_gpu)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_detect = QPushButton(tr("Detectar GPU / VRAM"))
        self.btn_detect.setIcon(iconos.icon("recargar", 15, T.ON_ACCENT))
        self.btn_detect.clicked.connect(self._detect_gpu)
        btn_row.addWidget(self.btn_detect)

        self.btn_maximize = QPushButton(tr("Sugerir imgsz MÁXIMO"))
        self.btn_maximize.setIcon(iconos.icon("rayo", 15, T.ON_ACCENT))
        self.btn_maximize.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 7px 14px; font-weight: 600;"
        )
        self.btn_maximize.setCursor(Qt.PointingHandCursor)
        self.btn_maximize.clicked.connect(self._maximize_imgsz)
        btn_row.addWidget(self.btn_maximize)

        self.btn_suggest_batch = QPushButton(tr("Sugerir batch para mi GPU"))
        self.btn_suggest_batch.clicked.connect(self._suggest_batch)
        btn_row.addWidget(self.btn_suggest_batch)
        btn_row.addStretch(1)
        gf.addLayout(btn_row)

        # A single button that applies the full priority order, instead of
        # letting the user guess whether to raise imgsz or batch first.
        self.btn_optimize = QPushButton(tr("Optimizar todo (imgsz → batch → velocidad)"))
        self.btn_optimize.setIcon(iconos.icon("diana", 15, T.ON_ACCENT))
        self.btn_optimize.setStyleSheet(
            f"background: {T.OK}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 14px; font-weight: 600;"
        )
        self.btn_optimize.setCursor(Qt.PointingHandCursor)
        self.btn_optimize.setToolTip(
            tr(
                "Fija imgsz al máximo que aguanta la tarjeta (sin pasar de la resolución "
                "nativa del dataset), después sube el batch con lo que sobre, y al final "
                "ajusta AMP, workers y cache. En ese orden."
            )
        )
        self.btn_optimize.clicked.connect(self._optimize_all)
        gf.addWidget(self.btn_optimize)

        self.lbl_est = QLabel(tr("Estimación de VRAM: —"))
        self.lbl_est.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        gf.addWidget(self.lbl_est)

        l1.addWidget(gpu_frame)
        self.body.addWidget(c1)

        # ── Optimizer and LR ──
        c2, l2 = self.card(tr("Optimizador y learning rate"), "curva")
        grid2 = QGridLayout()
        grid2.setHorizontalSpacing(20)
        grid2.setVerticalSpacing(10)
        grid2.addWidget(QLabel(tr("Learning rate (lr0):")), 0, 0)
        self.sb_lr0 = QDoubleSpinBox()
        self.sb_lr0.setRange(0.0001, 0.1)
        self.sb_lr0.setDecimals(4)
        self.sb_lr0.setSingleStep(0.001)
        self.sb_lr0.setValue(state.params.lr0)
        self.sb_lr0.valueChanged.connect(self._on_change)
        grid2.addWidget(self.sb_lr0, 0, 1)
        grid2.addWidget(QLabel(tr("Weight decay:")), 0, 2)
        self.sb_wd = QDoubleSpinBox()
        self.sb_wd.setRange(0.0, 0.01)
        self.sb_wd.setDecimals(5)
        self.sb_wd.setSingleStep(0.0001)
        self.sb_wd.setValue(state.params.weight_decay)
        self.sb_wd.valueChanged.connect(self._on_change)
        grid2.addWidget(self.sb_wd, 0, 3)

        grid2.addWidget(QLabel(tr("Momentum:")), 1, 0)
        self.sb_mom = QDoubleSpinBox()
        self.sb_mom.setRange(0.5, 0.999)
        self.sb_mom.setDecimals(3)
        self.sb_mom.setSingleStep(0.01)
        self.sb_mom.setValue(state.params.momentum)
        self.sb_mom.valueChanged.connect(self._on_change)
        grid2.addWidget(self.sb_mom, 1, 1)
        grid2.addWidget(QLabel(tr("Warmup épocas:")), 1, 2)
        self.sb_warm = QSpinBox()
        self.sb_warm.setRange(0, 50)
        self.sb_warm.setValue(state.params.warmup_epochs)
        self.sb_warm.valueChanged.connect(self._on_change)
        grid2.addWidget(self.sb_warm, 1, 3)

        grid2.addWidget(QLabel(tr("Close mosaic:")), 2, 0)
        self.sb_close = QSpinBox()
        self.sb_close.setRange(0, 100)
        self.sb_close.setValue(state.params.close_mosaic)
        self.sb_close.valueChanged.connect(self._on_change)
        grid2.addWidget(self.sb_close, 2, 1)
        grid2.addWidget(QLabel(tr("Label smoothing:")), 2, 2)
        self.sb_ls = QDoubleSpinBox()
        self.sb_ls.setRange(0.0, 0.3)
        self.sb_ls.setDecimals(2)
        self.sb_ls.setSingleStep(0.01)
        self.sb_ls.setValue(state.params.label_smoothing)
        self.sb_ls.valueChanged.connect(self._on_change)
        grid2.addWidget(self.sb_ls, 2, 3)

        self.cb_cos = QCheckBox(tr("Cosine LR (curva coseno)"))
        self.cb_cos.setChecked(state.params.cos_lr)
        self.cb_cos.stateChanged.connect(self._on_change)
        grid2.addWidget(self.cb_cos, 3, 0, 1, 2)
        self.cb_amp = QCheckBox(tr("AMP (Mixed Precision FP16)"))
        self.cb_amp.setChecked(state.params.amp)
        self.cb_amp.stateChanged.connect(self._on_change)
        grid2.addWidget(self.cb_amp, 3, 2, 1, 2)
        l2.addLayout(grid2)
        self.body.addWidget(c2)

        # ── Early stopping ──
        c3, l3 = self.card(tr("Early stopping y checkpoints"), "guardar")
        g3 = QGridLayout()
        g3.setHorizontalSpacing(20)
        g3.setVerticalSpacing(10)
        g3.addWidget(QLabel(tr("Patience:")), 0, 0)
        self.sb_pat = QSpinBox()
        self.sb_pat.setRange(0, 500)
        self.sb_pat.setValue(state.params.patience)
        self.sb_pat.valueChanged.connect(self._on_change)
        g3.addWidget(self.sb_pat, 0, 1)
        g3.addWidget(_hint(tr("Épocas sin mejora antes de detener. 50 = sensato.")), 0, 2)
        g3.addWidget(QLabel(tr("Save period:")), 1, 0)
        self.sb_save = QSpinBox()
        self.sb_save.setRange(1, 100)
        self.sb_save.setValue(state.params.save_period)
        self.sb_save.valueChanged.connect(self._on_change)
        g3.addWidget(self.sb_save, 1, 1)
        g3.addWidget(_hint(tr("Guardar checkpoint cada N épocas.")), 1, 2)
        l3.addLayout(g3)
        self.body.addWidget(c3)

        # ── Hardware / IO ──
        c4, l4 = self.card(tr("Hardware e I/O"), "equipo")
        g4 = QGridLayout()
        g4.setHorizontalSpacing(20)
        g4.setVerticalSpacing(10)
        g4.addWidget(QLabel(tr("Device:")), 0, 0)
        self.ed_device = QLineEdit(state.params.device)
        self.ed_device.editingFinished.connect(self._on_change)
        g4.addWidget(self.ed_device, 0, 1)
        g4.addWidget(_hint(tr("'0' = primera GPU. 'cpu' = CPU (muy lento).")), 0, 2)
        g4.addWidget(QLabel(tr("Workers:")), 1, 0)
        self.sb_workers = QSpinBox()
        self.sb_workers.setRange(0, 32)
        self.sb_workers.setValue(state.params.workers)
        self.sb_workers.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_workers, 1, 1)
        g4.addWidget(QLabel(tr("Nombre del run:")), 2, 0)
        self.ed_run = QLineEdit(state.params.run_name)
        self.ed_run.setPlaceholderText(tr("vacío = auto-fecha"))
        self.ed_run.editingFinished.connect(self._on_change)
        g4.addWidget(self.ed_run, 2, 1, 1, 2)
        l4.addLayout(g4)
        self.body.addWidget(c4)

        # Subscribe to preset changes
        self.state.params_changed.connect(self._reload_from_state)
        self.state.model_changed.connect(self._update_estimate)

        # Deferred GPU detection (in the background) so startup is not blocked
        self._gpu_worker: _GpuDetectWorker | None = None
        self.lbl_gpu.setText(
            tr("⌛  Detectando GPU en background (puede tardar 10–30 s la primera vez)…")
        )
        self.lbl_gpu.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        QTimer.singleShot(100, self._detect_gpu)
        self._update_estimate()

    # ──────────────────────────────────────────────────────────────
    def _reload_from_state(self):
        p = self.state.params
        self.combo_imgsz.setCurrentText(str(p.imgsz))
        self.sb_epochs.setValue(p.epochs)
        self.sb_batch.setValue(p.batch)
        self.combo_cache.setCurrentText(p.cache)
        self.sb_lr0.setValue(p.lr0)
        self.sb_wd.setValue(p.weight_decay)
        self.sb_mom.setValue(p.momentum)
        self.sb_warm.setValue(p.warmup_epochs)
        self.sb_close.setValue(p.close_mosaic)
        self.sb_ls.setValue(p.label_smoothing)
        self.cb_cos.setChecked(p.cos_lr)
        self.cb_amp.setChecked(p.amp)
        self.sb_pat.setValue(p.patience)
        self.sb_save.setValue(p.save_period)
        self.ed_device.setText(p.device)
        self.sb_workers.setValue(p.workers)
        self.ed_run.setText(p.run_name)
        self._update_estimate()

    def _on_change(self, *_):
        p = self.state.params
        try:  # noqa: SIM105
            p.imgsz = int(self.combo_imgsz.currentText())
        except ValueError:
            pass
        p.epochs = self.sb_epochs.value()
        p.batch = self.sb_batch.value()
        p.cache = self.combo_cache.currentText()
        p.lr0 = self.sb_lr0.value()
        p.weight_decay = self.sb_wd.value()
        p.momentum = self.sb_mom.value()
        p.warmup_epochs = self.sb_warm.value()
        p.close_mosaic = self.sb_close.value()
        p.label_smoothing = self.sb_ls.value()
        p.cos_lr = self.cb_cos.isChecked()
        p.amp = self.cb_amp.isChecked()
        p.patience = self.sb_pat.value()
        p.save_period = self.sb_save.value()
        p.device = self.ed_device.text().strip() or "0"
        p.workers = self.sb_workers.value()
        p.run_name = self.ed_run.text().strip()
        # Manual change → `Personalizado` preset
        if self.state.model.preset_name != "Personalizado":
            self.state.model.preset_name = "Personalizado"
            self.state.model_changed.emit()
        self._update_estimate()

    # ── GPU detection (in the background) ──────────────────────
    def _detect_gpu(self):
        # If a worker is already running, do not spawn another
        if self._gpu_worker is not None and self._gpu_worker.isRunning():
            return
        self.btn_detect.setEnabled(False)
        self.btn_maximize.setEnabled(False)
        self.btn_suggest_batch.setEnabled(False)
        self.btn_optimize.setEnabled(False)
        self._gpu_worker = _GpuDetectWorker()
        self._gpu_worker.detected.connect(self._on_gpu_detected)
        self._gpu_worker.start()

    def _on_gpu_detected(self, info):
        self._last_gpu_info = info  # cache for the other buttons
        self.btn_detect.setEnabled(True)
        self.btn_maximize.setEnabled(True)
        self.btn_suggest_batch.setEnabled(True)
        self.btn_optimize.setEnabled(True)
        self.lbl_gpu.setTextFormat(Qt.RichText)
        if info.available:
            extra = []
            if info.n_gpus > 1:
                extra.append(f"usando la GPU {info.index} de {info.n_gpus} (la de más VRAM)")
            if info.capacity:
                extra.append(f"capacidad {info.capacity}")
            if info.driver:
                extra.append(f"driver {info.driver}")
            if info.is_rocm and (info.torch_hip or info.torch_rocm):
                extra.append(
                    f"torch {info.torch_version} / ROCm {info.torch_hip or info.torch_rocm}"
                )
            elif info.torch_cuda:
                extra.append(f"torch {info.torch_version} / CUDA {info.torch_cuda}")
            self.lbl_gpu.setText(
                f"<b>GPU detectada:</b> {info.name} · "
                f"VRAM total: <b>{humanize_gb(info.vram_total_gb)}</b> · "
                f"libre: <b>{humanize_gb(info.vram_free_gb)}</b>"
                + (f"<br><span style='font-size:9pt'>{' · '.join(extra)}</span>" if extra else "")
            )
            self.lbl_gpu.setStyleSheet(f"color: {T.OK_TX}; font-size: 10pt; border: none;")
        elif info.gpu_sin_torch:
            # The distinction matters: the card works, what is wrong is the
            # installation. Saying `no GPU` here would send someone to buy hardware.
            self.lbl_gpu.setText(
                f"⚠ <b>{info.name}</b> está presente "
                f"({humanize_gb(info.vram_total_gb)}, driver {info.driver}) "
                f"<b>pero PyTorch no la puede usar.</b><br>"
                f"<span style='font-size:9pt'>{info.detail} "
                f"Reinstala PyTorch con CUDA (SETUP.bat) — hasta entonces se "
                f"entrena por CPU.</span>"
            )
            self.lbl_gpu.setStyleSheet(f"color: {T.ERR_TX}; font-size: 10pt; border: none;")
        else:
            self.lbl_gpu.setText(
                "✗ No se detectó GPU NVIDIA. Entrenarás en CPU (será MUY lento)."
                + (f"<br><span style='font-size:9pt'>{info.detail}</span>" if info.detail else "")
            )
            self.lbl_gpu.setStyleSheet(f"color: {T.WARN_TX}; font-size: 10pt; border: none;")
        self._update_estimate()

    def _maximize_imgsz(self):
        info = getattr(self, "_last_gpu_info", None) or detect_gpu()
        if not info.available:
            QMessageBox.information(
                self,
                tr("Sin GPU"),
                tr("No hay GPU NVIDIA disponible. En CPU recomendamos imgsz ≤ 640."),
            )
            self.combo_imgsz.setCurrentText("640")
            return
        size = self.state.model.size
        batch = self.state.params.batch
        amp = self.state.params.amp
        sz = recommend_max_imgsz(size, batch, info.vram_free_gb, amp=amp)
        self.combo_imgsz.setCurrentText(str(sz))
        self.lbl_est.setText(
            f"✓ Recomendación: imgsz = {sz} (modelo {size}, batch {batch}, "
            f"VRAM libre {humanize_gb(info.vram_free_gb)})"
        )

    def _suggest_batch(self):
        info = getattr(self, "_last_gpu_info", None) or detect_gpu()
        if not info.available:
            QMessageBox.information(self, tr("Sin GPU"), tr("Sin GPU no aplica esta sugerencia."))
            return
        try:
            imgsz = int(self.combo_imgsz.currentText())
        except ValueError:
            imgsz = 1280
        b = recommend_batch(
            self.state.model.size, imgsz, info.vram_free_gb, amp=self.cb_amp.isChecked()
        )
        self.sb_batch.setValue(b)
        self.lbl_est.setText(
            f"✓ Recomendación: batch = {b} para imgsz {imgsz} ({humanize_gb(info.vram_free_gb)} libres)"
        )

    def _measure_dataset(self) -> tuple[int, float]:
        """(larger native side, GB on disk) of the training dataset.

        The native side is the useful imgsz ceiling: training at 4096 over
        1630 px crops adds no signal, it only interpolates and takes the VRAM
        the batch needs. A few images are sampled instead of walking all of
        them, which with thousands of files would cost seconds every time.
        """
        from polyx.core.detect.yolo import size_image

        yaml_path = self.state.dataset.yaml_path
        if not yaml_path or not Path(yaml_path).exists():
            return 0, 0.0
        root = Path(yaml_path).parent
        imgs: list[Path] = []
        for pattern in ("**/images/**/*.jpg", "**/images/**/*.png", "**/*.jpg"):
            imgs = [p for p in root.glob(pattern) if p.is_file()]
            if imgs:
                break
        if not imgs:
            return 0, 0.0
        gb = sum(p.stat().st_size for p in imgs) / (1024**3)
        side = 0
        for p in imgs[:25]:
            wh = size_image(p)
            if wh:
                side = max(side, max(wh))  # noqa: PLW3301
        return side, gb

    def _optimize_all(self):
        """Applies the priority order: imgsz, then batch, then speed."""
        from ..hw import recommend_config

        info = getattr(self, "_last_gpu_info", None) or detect_gpu()
        if not info.available:
            msg = (
                "No hay GPU utilizable, así que no hay nada que optimizar: en CPU "
                "el techo práctico es imgsz 640."
            )
            if info.gpu_sin_torch:
                msg = (
                    f"La tarjeta está ahí ({info.name}, "
                    f"{humanize_gb(info.vram_total_gb)}) pero PyTorch no la puede "
                    f"usar.\n\n{info.detail}\n\nHay que reinstalar PyTorch con "
                    f"CUDA; hasta entonces el entrenamiento va por CPU."
                )
            QMessageBox.information(self, tr("Sin GPU utilizable"), msg)
            return

        side_native, dataset_gb = self._measure_dataset()
        ram_gb = 0.0
        try:
            import psutil

            ram_gb = psutil.virtual_memory().available / (1024**3)
        except Exception:  # noqa: BLE001, S110
            pass

        r = recommend_config(
            self.state.model.size,
            info.vram_free_gb,
            imgsz_native=side_native,
            dataset_gb=dataset_gb,
            ram_free_gb=ram_gb,
        )

        # 1. imgsz — the combo is editable, so any value is accepted
        self.combo_imgsz.setCurrentText(str(r.imgsz))
        # 2. batch
        self.sb_batch.setValue(r.batch)
        # 3. speed, without touching the previous steps
        self.cb_amp.setChecked(r.amp)
        self.sb_workers.setValue(r.workers)
        if r.cache:
            self.combo_cache.setCurrentText(r.cache)

        limit = {
            "vram": f"la VRAM ({humanize_gb(info.vram_free_gb)} libres)",
            "resolucion_nativa": f"la resolución nativa del dataset ({side_native} px)",
            "tope_batch": "el tope de batch útil",
        }.get(r.limited_by, r.limited_by)
        text_2 = (
            f"imgsz <b>{r.imgsz}</b> → batch <b>{r.batch}</b> → AMP on, "
            f"{r.workers} workers, cache {r.cache or 'sin cambio'}. "
            f"~{humanize_gb(r.vram_est_gb)} de {humanize_gb(r.vram_cap_gb)} "
            f"utilizables. Limitado por {limit}."
        )
        if not side_native:
            text_2 += (
                " <i>Sin dataset cargado no se pudo medir la resolución nativa: "
                "carga el dataset y vuelve a optimizar.</i>"
            )
        self.lbl_est.setText(text_2)
        self.lbl_est.setTextFormat(Qt.RichText)
        if r.notes:
            QMessageBox.information(
                self, tr("Configuración optimizada"), "\n\n".join(f"• {n}" for n in r.notes)
            )

    def _update_estimate(self):
        try:
            imgsz = int(self.combo_imgsz.currentText())
        except ValueError:
            imgsz = 640
        try:
            est = estimate_vram_gb(
                self.state.model.size, imgsz, self.state.params.batch, amp=self.cb_amp.isChecked()
            )
            # The key that gets translated is the TEMPLATE, not the assembled text:
            # with the data inside it would change on every call and would never match
            # an entry in the dictionary.
            self.lbl_est.setText(
                tr(
                    "Estimación de VRAM: ~{gb} (modelo {size}, imgsz {imgsz}, "
                    "batch {batch}, AMP={amp})"
                ).format(
                    gb=humanize_gb(est),
                    size=self.state.model.size,
                    imgsz=imgsz,
                    batch=self.state.params.batch,
                    amp="on" if self.cb_amp.isChecked() else "off",
                )
            )
        except Exception:  # noqa: BLE001
            self.lbl_est.setText(tr("Estimación de VRAM: —"))
