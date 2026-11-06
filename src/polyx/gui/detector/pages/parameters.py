"""Page 4 — Inference parameters + optical calibration + filters."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
)

from polyx.core.i18n import _, lazy
from polyx.core.platform import IS_MAC, architecture_mac, label_devices
from polyx.gui import iconos
from polyx.gui import theme as T

from ._base import DetectorPage


def _hint(text: str) -> QLabel:
    l = QLabel(text)  # noqa: E741
    l.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
    l.setWordWrap(True)
    return l


# Quality profiles: they hide YOLO jargon behind a simple choice.
# `Maximum detection` targets tiny particles in high-resolution photos
# (if the GPU cannot handle the imgsz, the auto-fallback lowers the size on its own).
PRESETS = {
    "Fast": {"conf": 0.25, "imgsz": 1280},
    "Balanced": {"conf": 0.15, "imgsz": 2560},
    "Maximum detection": {"conf": 0.10, "imgsz": 4096},
}


class _ProbeThread(QThread):
    """Probes in the background the maximum imgsz the GPU can take for the loaded model."""

    progress = Signal(int)
    done = Signal(int, dict)
    failed = Signal(str)

    def __init__(self, model, image_path, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.image_path = image_path
        self.device = device

    def run(self):
        try:
            mx, detail = self.model.probe_max_imgsz(
                self.image_path,
                device=self.device,
                progress=lambda sz: self.progress.emit(sz),  # noqa: PLW0108
            )
            self.done.emit(mx, detail)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


class ParametersPage(DetectorPage):
    STEP_N = 4
    STEP_TITLE = lazy("Parameters")
    STEP_DESCRIPTION = lazy(
        "Set confidence, IoU, optical calibration and filters. The μm/px calibration is optional but recommended for scientific reports."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Inference ──
        c1, l1 = self.card(_("Inference"), "ajustes")

        # Quality profile (preset): applies conf + imgsz at once
        self._applying_preset = False
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        preset_row.addWidget(QLabel(_("Profile:")))
        self.combo_preset = QComboBox()
        self.combo_preset.addItem(_("Custom"), None)
        for _name in PRESETS:
            self.combo_preset.addItem(_(_name), _name)
        self.combo_preset.currentIndexChanged.connect(self._apply_preset)
        self.combo_preset.setMinimumWidth(190)
        preset_row.addWidget(self.combo_preset)
        preset_row.addWidget(
            _hint(
                _(
                    "Pick a profile and you are done. «Maximum detection» is recommended for small microplastics in high-resolution microscopy photos."
                )
            ),
            1,
        )
        l1.addLayout(preset_row)

        g = QGridLayout()
        g.setHorizontalSpacing(24)
        g.setVerticalSpacing(10)

        # Confidence
        g.addWidget(QLabel(_("Min. confidence (conf):")), 0, 0)
        self.sb_conf = QDoubleSpinBox()
        self.sb_conf.setRange(0.01, 1.0)
        self.sb_conf.setSingleStep(0.05)
        self.sb_conf.setDecimals(2)
        self.sb_conf.setValue(state.params.conf)
        self.sb_conf.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_conf, 0, 1)
        g.addWidget(
            _hint(_("0.05–0.95. Higher = fewer detections (and fewer false positives).")), 0, 2
        )

        # IoU NMS
        g.addWidget(QLabel(_("IoU NMS:")), 1, 0)
        self.sb_iou_nms = QDoubleSpinBox()
        self.sb_iou_nms.setRange(0.1, 0.9)
        self.sb_iou_nms.setSingleStep(0.05)
        self.sb_iou_nms.setDecimals(2)
        self.sb_iou_nms.setValue(state.params.iou_nms)
        self.sb_iou_nms.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_iou_nms, 1, 1)
        g.addWidget(_hint(_("0.30–0.70. Suppresses overlapping boxes.")), 1, 2)

        # imgsz + `detectar máximo` button
        g.addWidget(QLabel(_("Image size (imgsz):")), 2, 0)
        imgsz_row = QHBoxLayout()
        imgsz_row.setSpacing(8)
        self.sb_imgsz = QSpinBox()
        self.sb_imgsz.setRange(320, 8192)
        self.sb_imgsz.setSingleStep(64)
        self.sb_imgsz.setValue(state.params.imgsz)
        self.sb_imgsz.valueChanged.connect(self._on_change)
        imgsz_row.addWidget(self.sb_imgsz)
        self.btn_probe = QPushButton(_("Detect maximum (GPU)"))
        self.btn_probe.setIcon(iconos.icon("buscar", 15, T.INK2))
        self.btn_probe.setCursor(Qt.PointingHandCursor)
        self.btn_probe.clicked.connect(self._probe_max_imgsz)
        imgsz_row.addWidget(self.btn_probe)
        imgsz_row.addStretch(1)
        g.addLayout(imgsz_row, 2, 1)
        g.addWidget(
            _hint(
                _(
                    "Larger = detects smaller particles (key for microplastics), but slower and heavier on GPU memory. For high-resolution photos go up to 4096+. The button probes the maximum your GPU can take with the loaded model."
                )
            ),
            2,
            2,
        )

        # device
        # The label and hint depend on the system: on a Mac there is no
        # "0" GPU to choose, and offering it only leads to the analysis failing.
        device_label = label_devices()
        g.addWidget(QLabel(_(device_label)), 3, 0)
        self.ed_device = QLineEdit(state.params.device)
        self.ed_device.editingFinished.connect(self._on_change)
        g.addWidget(self.ed_device, 3, 1)
        if IS_MAC:
            clue_2 = (
                _("'mps' = the Mac's built-in GPU (Apple Silicon). 'cpu' = processor.")
                if architecture_mac() == "Apple Silicon"
                else _("This is an Intel Mac: 'cpu' only. MPS requires Apple Silicon.")
            )
        else:
            clue_2 = _("'0' = first GPU. 'cpu' = CPU. '0,1' = GPU 0 and 1.")
        g.addWidget(_hint(clue_2), 3, 2)

        l1.addLayout(g)
        self.body.addWidget(c1)

        # ── Automatic tiling ──
        c5, l5 = self.card(_("Automatic tiling (large photos)"), "piezas")
        l5.addWidget(
            _hint(
                _(
                    "A full-plate photo enters the network rescaled to imgsz: at 4096 px reduced to 2080, every particle shrinks by half and disappears below the network stride. With tiling, each tile enters at native resolution and the boxes return to full-photo coordinates, merged with a global NMS — nothing has to be cut by hand nor the counts added up afterwards."
                )
            )
        )
        g5 = QGridLayout()
        g5.setHorizontalSpacing(24)
        g5.setVerticalSpacing(10)

        g5.addWidget(QLabel(_("When to tile:")), 0, 0)
        self.combo_tiling = QComboBox()
        # The value goes in userData, not in the displayed text: if the text were the
        # value, translating the interface would break the check against `"siempre"` and
        # `"nunca"` in predict_auto(). The text is translated, the data does not change.
        for _value, _label in (
            ("auto", _("automatic")),
            ("siempre", _("always")),
            ("nunca", _("never")),
        ):
            self.combo_tiling.addItem(_label, _value)
        self.combo_tiling.setCurrentIndex(max(0, self.combo_tiling.findData(state.params.tiling)))
        self.combo_tiling.currentTextChanged.connect(self._on_change)
        g5.addWidget(self.combo_tiling, 0, 1)
        g5.addWidget(
            _hint(_("«auto» decides on its own from the longest side of each photo.")), 0, 2
        )

        g5.addWidget(QLabel(_("Threshold (longest side, px):")), 1, 0)
        self.sb_threshold = QSpinBox()
        self.sb_threshold.setRange(512, 20000)
        self.sb_threshold.setSingleStep(128)
        self.sb_threshold.setValue(state.params.tiling_threshold_px)
        self.sb_threshold.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_threshold, 1, 1)
        g5.addWidget(
            _hint(
                _(
                    "Above this, tiling kicks in. 2000 lets the study crops (1630 px) through whole and tiles the full plates (~4096 px)."
                )
            ),
            1,
            2,
        )

        g5.addWidget(QLabel(_("Tile side (px):")), 2, 0)
        self.sb_tile = QSpinBox()
        self.sb_tile.setRange(0, 8192)
        self.sb_tile.setSingleStep(64)
        self.sb_tile.setValue(state.params.tiling_tile)
        self.sb_tile.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_tile, 2, 1)
        g5.addWidget(
            _hint(
                _(
                    "0 = automatic, min(threshold, imgsz). It should match the size of the crops the model was trained on: if the model saw particles at one scale and the tile presents them at another, recall drops."
                )
            ),
            2,
            2,
        )

        g5.addWidget(QLabel(_("Overlap between tiles:")), 3, 0)
        self.sb_overlap = QDoubleSpinBox()
        self.sb_overlap.setRange(0.0, 0.9)
        self.sb_overlap.setSingleStep(0.05)
        self.sb_overlap.setDecimals(2)
        self.sb_overlap.setValue(state.params.tiling_overlap)
        self.sb_overlap.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_overlap, 3, 1)
        g5.addWidget(
            _hint(
                _(
                    "0.25 = each tile overlaps a quarter of its neighbour. The overlap is what avoids losing or duplicating a particle that falls right on the seam."
                )
            ),
            3,
            2,
        )

        l5.addLayout(g5)
        self.lbl_plan = QLabel("")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt; border: none;")
        l5.addWidget(self.lbl_plan)
        self.body.addWidget(c5)
        self._refresh_plan()

        # ── Error analysis ──
        c2, l2 = self.card(_("Error analysis (when GT is available)"), "diana")
        grid2 = QGridLayout()
        grid2.setHorizontalSpacing(24)
        grid2.setVerticalSpacing(10)
        grid2.addWidget(QLabel(_("IoU for matching True Positives:")), 0, 0)
        self.sb_iou_tp = QDoubleSpinBox()
        self.sb_iou_tp.setRange(0.1, 0.95)
        self.sb_iou_tp.setSingleStep(0.05)
        self.sb_iou_tp.setDecimals(2)
        self.sb_iou_tp.setValue(state.params.iou_tp)
        self.sb_iou_tp.valueChanged.connect(self._on_change)
        grid2.addWidget(self.sb_iou_tp, 0, 1)
        grid2.addWidget(_hint(_("COCO standard: 0.50. Stricter: 0.75.")), 0, 2)
        l2.addLayout(grid2)
        l2.addWidget(
            _hint(
                _(
                    "True Positives: correct class + IoU ≥ threshold.  False Positives: prediction with no nearby GT.  False Negatives: GT not detected.  Misclassified: high IoU but wrong class."
                )
            )
        )
        self.body.addWidget(c2)

        # ── Optical calibration ──
        c3, l3 = self.card(_("Optical calibration (particle size)"), "visor")
        g3 = QGridLayout()
        g3.setHorizontalSpacing(24)
        g3.setVerticalSpacing(10)
        g3.addWidget(QLabel(_("µm per pixel:")), 0, 0)
        self.sb_umpx = QDoubleSpinBox()
        # Up to 1000 and not 10: the plates in this study range from 31 to 49 µm/px, so
        # the previous cap silently clipped the real calibration and
        # left the value at 10 without warning about anything.
        self.sb_umpx.setRange(0.0, 1000.0)
        self.sb_umpx.setSingleStep(0.01)
        self.sb_umpx.setDecimals(4)
        self.sb_umpx.setValue(state.calibration.um_per_px)
        self.sb_umpx.valueChanged.connect(self._on_change)
        g3.addWidget(self.sb_umpx, 0, 1)
        g3.addWidget(
            _hint(_("Fallback: used only when the dish cannot be measured. 0 = no calibration.")),
            0,
            2,
        )
        l3.addLayout(g3)

        # ── Automatic calibration against the dish ──
        self.chk_plate = QCheckBox(
            _("Measure the dish in every photo and use its scale (recommended)")
        )
        self.chk_plate.setChecked(state.calibration.measure_plate)
        self.chk_plate.setStyleSheet(f"color: {T.INK2}; border: none;")
        self.chk_plate.toggled.connect(self._on_change)
        l3.addWidget(self.chk_plate)

        g4 = QGridLayout()
        g4.setHorizontalSpacing(24)
        g4.setVerticalSpacing(10)
        g4.addWidget(QLabel(_("Actual dish diameter (mm):")), 0, 0)
        self.sb_plate_mm = QDoubleSpinBox()
        self.sb_plate_mm.setRange(1.0, 1000.0)
        self.sb_plate_mm.setDecimals(2)
        self.sb_plate_mm.setValue(state.calibration.diameter_plate_mm)
        self.sb_plate_mm.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_plate_mm, 0, 1)
        g4.addWidget(
            _hint(_("Nominal outer diameter. The Petri dish used in this study is 100 mm.")), 0, 2
        )

        g4.addWidget(QLabel(_("Dish height (mm):")), 1, 0)
        self.sb_height = QDoubleSpinBox()
        self.sb_height.setRange(0.0, 100.0)
        self.sb_height.setDecimals(1)
        self.sb_height.setValue(state.calibration.height_plate_mm)
        self.sb_height.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_height, 1, 1)
        g4.addWidget(
            _hint(_("0 = no correction. A 100 mm Petri dish is usually 15 mm tall.")), 1, 2
        )

        g4.addWidget(QLabel(_("Camera-to-base distance (mm):")), 2, 0)
        self.sb_distance = QDoubleSpinBox()
        self.sb_distance.setRange(0.0, 5000.0)
        self.sb_distance.setDecimals(0)
        self.sb_distance.setValue(state.calibration.distance_camera_mm)
        self.sb_distance.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_distance, 2, 1)
        g4.addWidget(
            _hint(_("0 = no correction. Measure to the BOTTOM of the dish, not to its rim.")), 2, 2
        )
        l3.addLayout(g4)

        l3.addWidget(
            _hint(
                _(
                    "Nothing has to be marked: the dish rim is found automatically. The radius comes from fitting a circle to the rim sampled along 720 directions, with outlier rejection; the Hough transform only supplies the approximate centre, because its radius can be off by as much as 12 %. In crops the dish is not visible, so there the scale is inherited from the calibration index left behind when the crop was made."
                )
            )
        )
        l3.addWidget(
            _hint(
                _(
                    "The dish rim sits closer to the camera than the base where the particles lie, so it projects larger and the resulting scale is too small: left uncorrected, ALL sizes are underestimated. With a 15 mm dish and the camera at 100 mm that is almost 16 percentage points; at 500 mm, about 3 %. Fill in both fields above to correct it."
                )
            )
        )
        l3.addWidget(
            _hint(
                _(
                    "The detector computes the area and equivalent diameter (of a circle with the same area) for every detected particle. With calibration, also in μm and μm²."
                )
            )
        )
        self.body.addWidget(c3)

        # ── Size filter ──
        c4, l4 = self.card(_("Size filter (optional)"), "visor")
        g4 = QGridLayout()
        g4.setHorizontalSpacing(24)
        g4.setVerticalSpacing(10)
        g4.addWidget(QLabel(_("Min size (µm):")), 0, 0)
        self.sb_min = QDoubleSpinBox()
        self.sb_min.setRange(0.0, 10000.0)
        self.sb_min.setDecimals(2)
        self.sb_min.setValue(state.params.size_min_um)
        self.sb_min.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_min, 0, 1)

        g4.addWidget(QLabel(_("Max size (µm):")), 0, 2)
        self.sb_max = QDoubleSpinBox()
        self.sb_max.setRange(0.0, 10000.0)
        self.sb_max.setDecimals(2)
        self.sb_max.setValue(state.params.size_max_um)
        self.sb_max.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_max, 0, 3)
        g4.addWidget(_hint(_("0 = no filter. Applies to the equivalent diameter.")), 1, 0, 1, 4)
        l4.addLayout(g4)
        self.body.addWidget(c4)

        # ── How far the polymer holds ──
        c5, l5 = self.card(_("Polymer assignment (optional)"), "ensayo")
        g5 = QGridLayout()
        g5.setHorizontalSpacing(24)
        g5.setVerticalSpacing(10)
        g5.addWidget(QLabel(_("Minimum confidence to assign:")), 0, 0)
        self.sb_assignment = QDoubleSpinBox()
        self.sb_assignment.setRange(0.0, 1.0)
        self.sb_assignment.setDecimals(2)
        self.sb_assignment.setSingleStep(0.05)
        self.sb_assignment.setValue(state.params.conf_assignment)
        self.sb_assignment.valueChanged.connect(self._on_change)
        g5.addWidget(self.sb_assignment, 0, 1)
        g5.addWidget(
            _hint(
                _(
                    '0 = off. Below this value the particle is still counted, but reported as "unassignable" instead of being given a polymer. Nile Red responds to the polarity of its environment, not to chemical identity: it separates PET —a polyester— cleanly from the rest, while PP and LDPE are both polyolefins and differ only in brightness, which depends on exposure and focus. Abstaining is more honest than inventing the class, and the report states what percentage was left unassigned.'
                )
            ),
            1,
            0,
            1,
            4,
        )
        l5.addLayout(g5)
        self.body.addWidget(c5)

    def _apply_preset(self, _index: int):
        """Applies the chosen profile (conf + imgsz) to the controls."""
        cfg = PRESETS.get(self.combo_preset.currentData())
        if not cfg:
            return
        self._applying_preset = True
        try:
            self.sb_conf.setValue(cfg["conf"])
            self.sb_imgsz.setValue(cfg["imgsz"])
        finally:
            self._applying_preset = False

    def _probe_max_imgsz(self):
        """Launches the maximum-imgsz probe in the background over the active model/image."""
        models = self.state.active_models()
        if not models:
            QMessageBox.warning(
                self, _("No model"), _("Load at least one model in the Models tab.")
            )
            return
        if not self.state.images:
            QMessageBox.warning(
                self,
                _("No images"),
                _("Select images in the Images tab (one of them is used to measure)."),
            )
            return
        slot = models[0]
        if slot.loaded is None:
            from polyx.core.detect.yolo import YoloModel

            slot.loaded = YoloModel(str(slot.path), alias=slot.alias)
        device = self.state.params.device
        self.btn_probe.setEnabled(False)
        self.btn_probe.setText(_("Testing…"))
        self._probe = _ProbeThread(slot.loaded, str(self.state.images[0]), device, self)
        self._probe.progress.connect(
            lambda size: self.btn_probe.setText(_("⏳ Testing %(size)d px…") % {"size": size})
        )
        self._probe.done.connect(self._on_probe_done)
        self._probe.failed.connect(self._on_probe_failed)
        self._probe.start()

    def _on_probe_done(self, max_ok: int, detail: dict):
        self.btn_probe.setEnabled(True)
        self.btn_probe.setText(_("Detect maximum (GPU)"))
        self.sb_imgsz.setValue(max_ok)
        oom = [str(k) for k, v in detail.items() if v == "oom"]
        msg = (
            f"Máximo seguro para «{self.state.active_models()[0].alias}»: "
            f"{max_ok} px.\n\nSe fijó imgsz = {max_ok}."
        )
        if oom:
            msg += f"\n(A {oom[0]} px se quedó sin memoria.)"
        if any(v == "cpu" for v in detail.values()):
            msg = (
                f"En CPU no hay tope de memoria GPU. Se fijó imgsz = {max_ok} "
                "pero será lento. Con GPU el detector es mucho más rápido."
            )
        QMessageBox.information(self, _("maximum imgsz"), msg)

    def _on_probe_failed(self, err: str):
        self.btn_probe.setEnabled(True)
        self.btn_probe.setText(_("Detect maximum (GPU)"))
        QMessageBox.warning(self, _("Could not measure"), err)

    def _on_change(self, *_unused):
        # Manual edit → the profile stops applying (returns to `"Personalizado"`)
        if not self._applying_preset and hasattr(self, "combo_preset"):
            self.combo_preset.blockSignals(True)
            self.combo_preset.setCurrentIndex(0)
            self.combo_preset.blockSignals(False)
        p = self.state.params
        p.conf = float(self.sb_conf.value())
        p.iou_nms = float(self.sb_iou_nms.value())
        p.iou_tp = float(self.sb_iou_tp.value())
        p.imgsz = int(self.sb_imgsz.value())
        p.device = self.ed_device.text().strip() or "0"
        c = self.state.calibration
        c.um_per_px = float(self.sb_umpx.value())
        c.measure_plate = self.chk_plate.isChecked()
        c.diameter_plate_mm = float(self.sb_plate_mm.value())
        c.height_plate_mm = float(self.sb_height.value())
        c.distance_camera_mm = float(self.sb_distance.value())
        p.size_min_um = float(self.sb_min.value())
        p.size_max_um = float(self.sb_max.value())
        p.conf_assignment = float(self.sb_assignment.value())
        p.tiling = self.combo_tiling.currentData() or "auto"
        p.tiling_threshold_px = int(self.sb_threshold.value())
        p.tiling_tile = int(self.sb_tile.value())
        p.tiling_overlap = float(self.sb_overlap.value())
        self._refresh_plan()
        self.state.params_changed.emit()

    def _refresh_plan(self):
        """Shows what the tiling will do with the already selected images.

        Seeing the decision before launching the batch avoids the surprise of
        discovering two hours later that everything was inferred in one piece.
        """
        from polyx.core.detect.yolo import policy_tiling, size_image

        p = self.state.params
        if not self.state.images:
            self.lbl_plan.setText(_("Select images to see what would be tiled."))
            return
        wh = size_image(self.state.images[0])
        if wh is None:
            self.lbl_plan.setText(_("Could not read the size of the first image."))
            return
        if p.tiling == "nunca":
            self.lbl_plan.setText(
                _(
                    "First image %(width)d×%(height)d px → single pass at imgsz %(imgsz)d "
                    "(tiling off)."
                )
                % {"width": wh[0], "height": wh[1], "imgsz": p.imgsz}
            )
            return
        plan = policy_tiling(
            wh[0],
            wh[1],
            p.imgsz,
            threshold_px=0 if p.tiling == "siempre" else p.tiling_threshold_px,
            tile=p.tiling_tile,
            overlap=p.tiling_overlap,
        )
        if plan is None:
            self.lbl_plan.setText(
                _(
                    "First image %(width)d×%(height)d px → single pass: below the "
                    "%(threshold)d px threshold."
                )
                % {"width": wh[0], "height": wh[1], "threshold": p.tiling_threshold_px}
            )
        else:
            scale = p.imgsz / plan["tile"]
            self.lbl_plan.setText(
                _(
                    "First image %(width)d×%(height)d px → %(tiles)d tiles of %(tile)d px "
                    "with %(overlap)d%% overlap, each at imgsz %(imgsz)d (%(scale).2f× the "
                    "tile's native resolution). Takes ~%(tiles)d× longer than a single pass."
                )
                % {
                    "width": wh[0],
                    "height": wh[1],
                    "tiles": plan["n_tiles"],
                    "tile": plan["tile"],
                    "overlap": int(plan["overlap"] * 100),
                    "imgsz": p.imgsz,
                    "scale": scale,
                }
            )
