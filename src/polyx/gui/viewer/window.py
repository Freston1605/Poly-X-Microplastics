"""Main window of the Viewer — canvas (left) | control panel (right)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from polyx.core.annotate import find_run_labels
from polyx.core.i18n import _, ngettext
from polyx.core.paths import RUNS_DIR
from polyx.gui import i18n_qt, iconos, marca
from polyx.gui import theme as T
from polyx.gui.widgets import HLine, LogoBadge, reset_central_widget

from .canvas import ViewerCanvas
from .export import csv_rows, render_annotated_frame
from .state import ViewerState


def _there_are_gpu() -> bool:
    """CUDA available. Queried lazily: importing torch is slow."""

    try:
        import torch

        return bool(torch.cuda.is_available())

    except Exception:  # noqa: BLE001
        return False


class ViewerWindow(i18n_qt.RetranslateOnLanguageChange, QMainWindow):
    def __init__(self):

        super().__init__()

        i18n_qt.activate()

        marca.titular(self, "visor")

        self.resize(1420, 900)

        self.state = ViewerState()
        self._model_name = ""

        self._build_ui()

        self._connect_signals()

        self._install_shortcuts()

        self.setAcceptDrops(True)

    # ── Language ───────────────────────────────────────────────

    def retranslate(self) -> None:
        """Rebuild the window and re-read the state that must survive."""
        if not i18n_qt.is_alive(self):
            return

        conf = self.spin_conf.value()

        imgsz = self.spin_imgsz.value()

        gpu = self.chk_gpu.isChecked()

        self._build_ui()

        marca.titular(self, "visor")

        self.spin_conf.setValue(conf)

        self.spin_imgsz.setValue(imgsz)

        self.chk_gpu.setChecked(gpu)

        if self._model_name:
            self.lbl_model.setText(self._model_name)

        self.nav_frame.setVisible(len(self.state.images) > 1)

        if self.state.current_image is not None:
            self._on_image_loaded(self.state.current_image)

        else:
            self._update_status_bar()

        if self.state.detections:
            self._on_detections_changed(self.state.detections)

        if self.state.um_per_px > 0:
            self.lbl_calib_result.setText(_("%(value).4f μm/px") % {"value": self.state.um_per_px})

            self.lbl_calib_result.setStyleSheet(
                f"color: {T.OK_TX}; font-size: 9.5pt; font-weight: 600;"
            )

            self._update_status_bar()

    def _build_ui(self) -> None:

        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG_SOFT}; }}")

        central = reset_central_widget(self)

        root = QHBoxLayout(central)

        root.setContentsMargins(0, 0, 0, 0)

        root.setSpacing(0)

        # Canvas (center/left)

        canvas_wrap = QWidget()

        canvas_wrap.setStyleSheet(f"background: {T.BG_SOFT};")

        cw_lay = QVBoxLayout(canvas_wrap)

        cw_lay.setContentsMargins(0, 0, 0, 0)

        cw_lay.setSpacing(0)

        self.canvas = ViewerCanvas(self.state)

        cw_lay.addWidget(self.canvas, 1)

        # Bottom status/calibration bar

        self.status_bar = QLabel(_("  No image  —  Not calibrated"))

        self.status_bar.setFixedHeight(28)

        self.status_bar.setStyleSheet(
            f"background: {T.BG}; border-top: 1px solid {T.RULE}; "
            f"color: {T.INK2}; font-size: 9.5pt; padding: 0 12px;"
        )

        cw_lay.addWidget(self.status_bar)

        root.addWidget(canvas_wrap, 1)

        # Right panel

        right_scroll = QScrollArea()

        right_scroll.setFixedWidth(280)

        right_scroll.setWidgetResizable(True)

        right_scroll.setStyleSheet(
            f"QScrollArea {{ border: none; border-left: 1px solid {T.RULE}; background: {T.BG}; }}"
        )

        right_content = QWidget()

        right_content.setStyleSheet(f"background: {T.BG};")

        self._rv = QVBoxLayout(right_content)

        self._rv.setContentsMargins(14, 14, 14, 14)

        self._rv.setSpacing(10)

        right_scroll.setWidget(right_content)

        root.addWidget(right_scroll)

        self._build_right_panel()

    # ── Right panel ─────────────────────────────────────────

    def _build_right_panel(self):

        rv = self._rv

        rv.addWidget(LogoBadge("POLY-X", _("Viewer")))

        rv.addSpacing(4)

        # ── Open image / folder ──

        lbl_open = QLabel(_("Image"))

        lbl_open.setStyleSheet("font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_open)

        row_open = QHBoxLayout()

        btn_img = QPushButton(_("Image"))
        btn_img.setIcon(iconos.icon("imagenes", 15, T.INK2))

        btn_fol = QPushButton(_("Folder"))
        btn_fol.setIcon(iconos.icon("dataset", 15, T.INK2))

        btn_img.clicked.connect(self._open_image)

        btn_fol.clicked.connect(self._open_folder)

        row_open.addWidget(btn_img)

        row_open.addWidget(btn_fol)

        rv.addLayout(row_open)

        self.lbl_img_name = QLabel(_("No image"))

        self.lbl_img_name.setWordWrap(True)

        self.lbl_img_name.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")

        rv.addWidget(self.lbl_img_name)

        # Navigation (only shown with a folder)

        self.nav_frame = QFrame()

        self.nav_frame.setStyleSheet("QFrame { border: none; background: transparent; }")

        nav_lay = QHBoxLayout(self.nav_frame)

        nav_lay.setContentsMargins(0, 0, 0, 0)

        nav_lay.setSpacing(6)

        btn_prev = QPushButton("←")

        btn_next = QPushButton("→")

        btn_prev.setFixedWidth(36)

        btn_next.setFixedWidth(36)

        btn_prev.clicked.connect(self.state.prev_image)

        btn_next.clicked.connect(self.state.next_image)

        self.lbl_nav = QLabel("1 / 1")

        self.lbl_nav.setAlignment(Qt.AlignCenter)

        nav_lay.addWidget(btn_prev)

        nav_lay.addWidget(self.lbl_nav, 1)

        nav_lay.addWidget(btn_next)

        self.nav_frame.setVisible(False)

        rv.addWidget(self.nav_frame)

        rv.addWidget(HLine())

        # ── Model ──

        lbl_mod = QLabel(_("Model"))

        lbl_mod.setStyleSheet("font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_mod)

        self.lbl_model = QLabel(_("No model"))

        self.lbl_model.setWordWrap(True)

        self.lbl_model.setStyleSheet(
            f"color: {T.INK3}; font-size: 9pt; background: {T.BG_SOFT}; "
            f"border: 1px solid {T.RULE}; border-radius: 4px; padding: 4px;"
        )

        rv.addWidget(self.lbl_model)

        btn_load_model = QPushButton(_("Load model…"))
        btn_load_model.setIcon(iconos.icon("dataset", 15, T.INK2))

        btn_load_model.clicked.connect(self._load_model)

        rv.addWidget(btn_load_model)

        rv.addWidget(QLabel(_("Minimum confidence:")))

        self.spin_conf = QDoubleSpinBox()

        self.spin_conf.setRange(0.01, 0.99)

        self.spin_conf.setSingleStep(0.05)

        self.spin_conf.setValue(0.25)

        self.spin_conf.setDecimals(2)

        rv.addWidget(self.spin_conf)

        # Inference resolution: decisive with tiny particles. At 640 px

        # a 12 px particle in a 4096 photo collapses to ~2 px and nothing is

        # detected. It must be possible to raise it.

        rv.addWidget(QLabel(_("Inference resolution (px):")))

        self.spin_imgsz = QSpinBox()

        self.spin_imgsz.setRange(320, 8192)

        self.spin_imgsz.setSingleStep(32)

        self.spin_imgsz.setValue(2080)

        self.spin_imgsz.setToolTip(
            _(
                "Longest side the image is resized to before inference.\nHigher = bigger particles for the network, but more memory.\nRounded to the nearest multiple of 32."
            )
        )

        rv.addWidget(self.spin_imgsz)

        self.chk_gpu = QCheckBox(_("Use the GPU if available"))

        self.chk_gpu.setChecked(True)

        rv.addWidget(self.chk_gpu)

        self.btn_detect = QPushButton(_("Detect"))
        self.btn_detect.setIcon(iconos.icon("ejecutar", 15, T.INK2))

        self.btn_detect.setObjectName("primary")

        self.btn_detect.clicked.connect(self._detect)

        rv.addWidget(self.btn_detect)

        self.btn_load_txt = QPushButton(_("Load labels (.txt)"))
        self.btn_load_txt.setIcon(iconos.icon("texto", 15, T.INK2))

        self.btn_load_txt.setToolTip(
            _(
                "Shows the YOLO annotations that come with the image.\nUseful to review the manual count over the plate, with sizes\nalready converted to µm by the calibration."
            )
        )

        self.btn_load_txt.clicked.connect(self._load_labels)

        rv.addWidget(self.btn_load_txt)

        self.btn_load_run = QPushButton(_("Load predictions from a run"))
        self.btn_load_run.setIcon(iconos.icon("dataset", 15, T.INK2))

        self.btn_load_run.setToolTip(
            _(
                "Reads the predictions the Detector left in runs/detect_.../ for this same photo. Use it to review a run you have already closed, without running the model again."
            )
        )
        self.btn_load_run.clicked.connect(self._run_load)

        rv.addWidget(self.btn_load_run)

        rv.addWidget(HLine())

        # ── Calibration ──

        lbl_calib = QLabel(_("µm/pixel calibration"))

        lbl_calib.setStyleSheet("font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_calib)

        row_calib = QHBoxLayout()

        self.btn_line = QPushButton(_("Line"))
        self.btn_line.setIcon(iconos.icon("linea", 15, T.INK2))

        self.btn_circ = QPushButton(_("Circle"))
        self.btn_circ.setIcon(iconos.icon("circulo", 15, T.INK2))

        self.btn_line.setCheckable(True)

        self.btn_circ.setCheckable(True)

        self.btn_line.clicked.connect(lambda: self._start_calib("linea"))

        self.btn_circ.clicked.connect(lambda: self._start_calib("circulo"))

        row_calib.addWidget(self.btn_line)

        row_calib.addWidget(self.btn_circ)

        rv.addLayout(row_calib)

        self.lbl_calib_hint = QLabel(_("Click on the image to place\nreference points."))

        self.lbl_calib_hint.setStyleSheet(f"color: {T.INK3}; font-size: 8.5pt;")

        self.lbl_calib_hint.setVisible(False)

        rv.addWidget(self.lbl_calib_hint)

        btn_cancel_calib = QPushButton(_("✕  Cancel calibration"))

        btn_cancel_calib.setStyleSheet(
            f"QPushButton {{ color: {T.WARN_TX}; border-color: {T.WARN_TX}; }}"
            f"QPushButton:hover {{ background: #fff8c5; }}"
        )

        btn_cancel_calib.clicked.connect(self.state.cancel_calib)

        self.btn_cancel_calib = btn_cancel_calib

        btn_cancel_calib.setVisible(False)

        rv.addWidget(btn_cancel_calib)

        self.lbl_calib_result = QLabel(_("Not calibrated"))

        self.lbl_calib_result.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt;")

        rv.addWidget(self.lbl_calib_result)

        rv.addWidget(HLine())

        # ── Results ──

        lbl_res = QLabel(_("Results"))

        lbl_res.setStyleSheet("font-weight: 600; font-size: 10pt;")

        rv.addWidget(lbl_res)

        rv.addWidget(QLabel(_("Filter by class:")))

        self.combo_filter = QComboBox()

        self.combo_filter.addItem(_("All classes"), None)

        self.combo_filter.currentIndexChanged.connect(self._on_filter_changed)

        rv.addWidget(self.combo_filter)

        self.lbl_det_count = QLabel(_("0 detections"))

        self.lbl_det_count.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")

        rv.addWidget(self.lbl_det_count)

        self.lbl_morph = QLabel("")

        self.lbl_morph.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt; font-weight: 600;")

        rv.addWidget(self.lbl_morph)

        # One row per particle, with its NUMBER in front: it is the same number

        # drawn on the image, so one can go from the row to the particle.

        self.table = QTableWidget(0, 6)

        self.table.setHorizontalHeaderLabels(
            ["#", _("Class"), _("Type"), _("Length μm"), _("Width μm"), _("Asp.")]
        )

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.table.verticalHeader().setDefaultSectionSize(22)

        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        self.table.setSelectionMode(QTableWidget.SingleSelection)

        self.table.setAlternatingRowColors(True)

        self.table.setFixedHeight(200)

        self.table.setStyleSheet("font-size: 9pt;")

        self.table.itemSelectionChanged.connect(self._on_particle_chosen)

        rv.addWidget(self.table)

        # ── Selected particle details ──

        # Here it is shown WHAT was measured: the mask contour and, on top of it, the

        # Feret line or the geodesic path. A size that cannot be seen

        # measured cannot be verified, and that is what a reviewer asks for.

        self.lbl_record_tit = QLabel(_("Select a particle to see how it was measured"))

        self.lbl_record_tit.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")

        self.lbl_record_tit.setWordWrap(True)

        rv.addWidget(self.lbl_record_tit)

        self.lbl_record = QLabel()

        self.lbl_record.setAlignment(Qt.AlignCenter)

        self.lbl_record.setMinimumHeight(150)

        self.lbl_record.setStyleSheet(
            "background: #0d1117; border: 1px solid #d0d7de; border-radius: 6px;"
        )

        rv.addWidget(self.lbl_record)

        self.lbl_record_data = QLabel("")

        self.lbl_record_data.setWordWrap(True)

        self.lbl_record_data.setStyleSheet(
            f"color: {T.INK3}; font-size: 9pt; font-family: Consolas, monospace;"
        )

        rv.addWidget(self.lbl_record_data)

        rv.addWidget(HLine())

        # ── Save ──

        btn_save = QPushButton(_("Save current info"))
        btn_save.setIcon(iconos.icon("guardar", 15, T.INK2))

        btn_save.setObjectName("primary")

        btn_save.clicked.connect(self._save)

        rv.addWidget(btn_save)

        self.lbl_status = QLabel(_("● Ready"))

        self.lbl_status.setAlignment(Qt.AlignCenter)

        self.lbl_status.setStyleSheet(f"color: {T.OK_TX}; font-size: 9pt;")

        rv.addWidget(self.lbl_status)

        rv.addStretch(1)

    # ── Connections and shortcuts ────────────────────────────

    def _connect_signals(self):

        self.state.image_loaded.connect(self._on_image_loaded)

        self.state.detections_changed.connect(self._on_detections_changed)

        self.state.calib_changed.connect(self._on_calib_changed)

        self.state.calib_pts_changed.connect(self._on_calib_pts_changed)

        self.state.calib_ready.connect(self._on_calib_ready)

    def _install_shortcuts(self):

        QShortcut(QKeySequence("Ctrl+D"), self, self._detect)

        QShortcut(QKeySequence(Qt.Key_Left), self, self.state.prev_image)

        QShortcut(QKeySequence(Qt.Key_Right), self, self.state.next_image)

        QShortcut(QKeySequence(Qt.Key_Escape), self, self.state.cancel_calib)

    # ── Open files ───────────────────────────────────────────

    def _open_image(self):

        path, _unused = QFileDialog.getOpenFileName(
            self, _("Open image"), "", _("Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)")
        )

        if path:
            self.state.load_single(Path(path))

            self.nav_frame.setVisible(False)

    def _open_folder(self):

        folder = QFileDialog.getExistingDirectory(self, _("Open image folder"))

        if folder:
            self.state.load_folder(Path(folder))

            total = len(self.state.images)

            if total > 1:
                self.nav_frame.setVisible(True)

                self._update_nav()

    def _on_image_loaded(self, path: Path):

        self.lbl_img_name.setText(path.name)

        self._update_nav()

        self._update_status_bar()

        self._set_status(_("● Image loaded"), T.OK)

    def _update_nav(self):

        total = len(self.state.images)

        idx = self.state.current_idx

        self.lbl_nav.setText(f"{idx + 1} / {total}")

    # ── Model and detection ──────────────────────────────────

    def _load_model(self):

        path, _unused = QFileDialog.getOpenFileName(
            self, _("Load YOLO model"), "", _("YOLO model (*.pt)")
        )

        if not path:
            return

        try:
            from polyx.core.detect.yolo import YoloModel

            self.state.model = YoloModel(path)

            self._model_name = Path(path).name

            self.lbl_model.setText(self._model_name)

            self._set_status(_("● Model ready"), T.OK)

        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, _("Error loading the model"), str(e))

    def _detect(self):

        if not self.state.model:
            QMessageBox.warning(self, _("No model"), _("Load a .pt model first."))

            return

        img = self.state.current_image

        if img is None:
            QMessageBox.warning(self, _("No image"), _("Open an image first."))

            return

        imgsz = int(round(self.spin_imgsz.value() / 32)) * 32  # multiple of stride  # noqa: RUF046

        device = "0" if (self.chk_gpu.isChecked() and _there_are_gpu()) else "cpu"

        try:
            self._set_status(
                _("● Detecting (imgsz %(imgsz)d, %(device)s)…")
                % {"imgsz": imgsz, "device": device},
                T.ACCENT,
            )

            QApplication.setOverrideCursor(Qt.WaitCursor)

            QApplication.processEvents()

            # Automatic tiling: the Viewer is used on the full plate, which is

            # exactly the case where a single pass makes the particles disappear.

            plan: dict = {}

            dets = self.state.model.predict_auto(
                str(img),
                conf=self.spin_conf.value(),
                imgsz=imgsz,
                device=device,
                tiling="auto",
                threshold_px=2000,
                record=plan,
            )

            self.state.detections = dets

            self.state.detections_changed.emit(dets)

            if plan.get("troceado"):
                p = plan["plan"]

                self._set_status(
                    ngettext(
                        "● %(count)d detection (tiled into %(tiles)d tiles of "
                        "%(tile)d px, imgsz %(imgsz)d)",
                        "● %(count)d detections (tiled into %(tiles)d tiles of "
                        "%(tile)d px, imgsz %(imgsz)d)",
                        len(dets),
                    )
                    % {
                        "count": len(dets),
                        "tiles": p["n_tiles"],
                        "tile": p["tile"],
                        "imgsz": imgsz,
                    },
                    T.OK,
                )

            else:
                self._set_status(
                    ngettext(
                        "● %(count)d detection (imgsz %(imgsz)d)",
                        "● %(count)d detections (imgsz %(imgsz)d)",
                        len(dets),
                    )
                    % {"count": len(dets), "imgsz": imgsz},
                    T.OK,
                )

        except Exception as e:  # noqa: BLE001
            # Without enough VRAM the failure is common when raising imgsz: it says

            # what to do instead of showing the raw exception.

            msg = str(e)

            if "out of memory" in msg.lower():
                msg = _(
                    "Out of GPU memory at imgsz %(imgsz)d.\n\n"
                    "Lower the inference resolution or untick “Use GPU”."
                ) % {"imgsz": imgsz}

            QMessageBox.warning(self, _("Detection error"), msg)

            self._set_status(_("● Detection error"), T.ERR)

        finally:
            QApplication.restoreOverrideCursor()

    def _run_load(self):
        """Loads into the Viewer the predictions a Detector run left behind.

        It is needed because the Detector's `Revisar` button only works
        while the run is still in memory: closing it loses the results. On
        disk they do remain, in runs/detect_*/<modelo>/labels/, so a run from
        last week can be reviewed again without repeating it.

        The ORIGINAL PHOTO is opened, never the run's annotated PNG: it has
        the boxes painted on top and measuring on it would put the rectangle
        stroke inside the particle's mask.
        """

        img = self.state.current_image

        if img is None:
            QMessageBox.warning(self, _("No image"), _("Open the original photo first."))

            return

        folder_2 = QFileDialog.getExistingDirectory(
            self,
            _("Run folder (runs/detect_...)"),
            str(RUNS_DIR),
        )

        if not folder_2:
            return

        root_2 = Path(folder_2)

        # The runner names each file NNNN_<photo name>.txt, with an

        # index in front so that two homonymous photos from different folders do not

        # collide. The naming owner parses it back, so the Viewer cannot desync.

        candidates = find_run_labels(root_2, img.stem)

        if not candidates:
            QMessageBox.information(
                self,
                _("No predictions"),
                _(
                    "That run has no predictions for %(n)s.\n\nThe 'labels' subfolders were searched."
                )
                % {"n": img.name},
            )

            return

        if len(candidates) > 1:
            # Several models ran on the same photo; choose which one.

            from PySide6.QtWidgets import QInputDialog

            options = [c.parent.parent.name for c in candidates]

            chosen, ok = QInputDialog.getItem(
                self,
                _("Several models"),
                _("This photo was analysed with several models. Which one do you want to review?"),
                options,
                0,
                False,
            )

            if not ok:
                return

            candidates = [c for c, o in zip(candidates, options) if o == chosen]  # noqa: B905

        txt = candidates[0]

        try:
            from polyx.core.detect.protocol import read_yolo_txt
            from polyx.core.image_io import read_image

            bgr = read_image(img)

            if bgr is None:
                raise OSError("no se pudo leer la imagen")

            height, width_2 = bgr.shape[:2]

            names = {i: n for i, n in enumerate(self._class_names(txt.parent))}  # noqa: C416

            dets = read_yolo_txt(txt, width_2, height, names)

        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(
                self,
                _("Error"),
                _("Could not read the predictions: %(error)s") % {"error": e},
            )

            return

        if not dets:
            QMessageBox.information(
                self,
                _("No predictions"),
                _("The file is empty: that photo had no detections."),
            )

            return

        self.state.detections = dets

        self.state.detections_changed.emit(dets)

        self._set_status(
            _("● %(n)s predictions loaded from %(c)s")
            % {"n": len(dets), "c": txt.parent.parent.parent.name},
            T.OK,
        )

    def _load_labels(self):
        """Loads the YOLO .txt of the current image and shows it as detections.

        It allows the manual count to be reviewed on the full plate and the
        sizes in µm to be seen without going back to the Labeller.
        """

        img = self.state.current_image

        if img is None:
            QMessageBox.warning(self, _("No image"), _("Open an image first."))

            return

        from polyx.core.labels import find_label_file, read_labels

        txt = find_label_file(img)

        if txt is None:
            QMessageBox.information(
                self,
                _("No labels"),
                _("There is no label file next to the image:\n%(name)s")
                % {"name": img.with_suffix(".txt").name},
            )

            return

        try:
            # Locate, size (EXIF-safe) and read in one call. The folder
            # convention is the codec's, so a Labeller dataset (labels/
            # sibling of images/) loads too, not only a .txt beside the photo.
            dets, malformed = read_labels(img)

        except ValueError:
            QMessageBox.warning(self, _("Error"), _("Could not read the image."))

            return

        self.state.detections = dets

        self.state.detections_changed.emit(dets)

        warning_2 = (
            ngettext(
                "  (%(count)d unreadable line)",
                "  (%(count)d unreadable lines)",
                malformed,
            )
            % {"count": malformed}
            if malformed
            else ""
        )

        self._set_status(
            ngettext(
                "● %(count)d label loaded from %(file)s%(warning)s",
                "● %(count)d labels loaded from %(file)s%(warning)s",
                len(dets),
            )
            % {"count": len(dets), "file": txt.name, "warning": warning_2},
            T.WARN if malformed else T.OK,
        )

    @staticmethod
    def _class_names(folder_2: Path) -> list[str]:
        """Reads classes.txt from the folder; if it is not there, it uses the study's classes."""

        f = folder_2 / "classes.txt"

        if f.exists():
            names = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]  # noqa: E741

            if names:
                return names

        return ["PET", "PP", "LDPE"]

    def _on_detections_changed(self, dets: list):

        self._measure_shapes(dets)

        self._rebuild_table(dets)

        self._update_filter_combo(dets)

        self.lbl_det_count.setText(
            ngettext("%(count)d detection", "%(count)d detections", len(dets))
            % {"count": len(dets)}
        )

        self._update_status_bar()

    def _measure_shapes(self, dets: list):
        """Numbers and measures the real shape of each particle.

        It is done here and not when drawing the table because measuring
        costs about 6 ms per particle and the table is rebuilt every time a
        filter changes.
        """

        self._bgr_current = None

        if not dets:
            self.lbl_morph.setText("")

            return

        path_2 = self.state.current_image

        if path_2 is None:
            return

        try:
            from polyx.core.measure.calibration import read_image
            from polyx.core.measure.shape import measure_detections

            bgr = read_image(path_2)

        except Exception:  # noqa: BLE001
            bgr = None

        if bgr is None:
            return

        self._bgr_current = bgr

        um = self.state.um_per_px if self.state.um_per_px > 0 else None

        # Top to bottom and left to right: the same criterion the

        # Detector uses, so that the number means the same in both modules.

        for i, d in enumerate(sorted(dets, key=lambda z: (round(z.y1 / 40), z.x1)), 1):
            d.number = i

        # Box size first as the declared fallback, real shape over it — the
        # same loop the Detector runs, so both modules report the same sizes.

        measure_detections(dets, bgr, um)

        fib = sum(1 for d in dets if d.morphotype == "fibra")

        sin = sum(1 for d in dets if d.morphotype is None)

        others = len(dets) - fib - sin

        txt = (
            ngettext("%(count)s fibre", "%(count)s fibres", fib) % {"count": fib}
            + " · "
            + ngettext("%(count)s particle", "%(count)s particles", others) % {"count": others}
        )

        if sin:
            txt += " · " + ngettext("%(count)s unmeasured", "%(count)s unmeasured", sin) % {
                "count": sin
            }

        if not um:
            txt += "   " + _("(uncalibrated: sizes in pixels only)")

        self.lbl_morph.setText(txt)

    def _rebuild_table(self, dets: list):

        self.table.setRowCount(0)

        for det in sorted(dets, key=lambda z: z.number or 0):
            row = self.table.rowCount()

            self.table.insertRow(row)

            it = QTableWidgetItem(str(det.number or row + 1))

            # The Detection travels inside the item: when selecting the row one must

            # be able to get back to it without looking it up by index, which drifts

            # as soon as the class filter is applied.

            it.setData(Qt.UserRole, det)

            self.table.setItem(row, 0, it)

            self.table.setItem(row, 1, QTableWidgetItem(det.class_name))

            self.table.setItem(row, 2, QTableWidgetItem(det.morphotype or "—"))

            self.table.setItem(
                row, 3, QTableWidgetItem(f"{det.length_um:.0f}" if det.length_um else "—")
            )

            self.table.setItem(
                row, 4, QTableWidgetItem(f"{det.width_um:.0f}" if det.width_um else "—")
            )

            self.table.setItem(row, 5, QTableWidgetItem(f"{det.aspect:.1f}" if det.aspect else "—"))

    def _on_particle_chosen(self):
        """Draws the measurement of the selected particle in the table."""

        rows = self.table.selectionModel().selectedRows()

        if not rows or getattr(self, "_bgr_current", None) is None:
            return

        item = self.table.item(rows[0].row(), 0)

        det = item.data(Qt.UserRole) if item else None

        if det is None:
            return

        try:
            from polyx.core.measure.drawing import draw_measurement

            img, m = draw_measurement(self._bgr_current, det, zoom=6)

        except Exception:  # noqa: BLE001
            img, m = None, None

        if img is None or m is None or not m.ok:
            self.lbl_record.setText(_("This particle could not be separated from the background"))

            self.lbl_record_data.setText("")

            return

        from PySide6.QtGui import QImage, QPixmap

        height, width_2 = img.shape[:2]

        qim = QImage(img.data, width_2, height, 3 * width_2, QImage.Format_BGR888).copy()

        pix = QPixmap.fromImage(qim)

        available = self.lbl_record.width() - 12

        if available > 0 and pix.width() > available:
            pix = pix.scaledToWidth(available, Qt.SmoothTransformation)

        self.lbl_record.setPixmap(pix)

        self.lbl_record_tit.setText(
            _(
                "Particle #%(number)s · %(class)s · %(morphotype)s"
                "   —   left: no marks · right: green = measured contour, "
                "yellow = Feret, magenta = geodesic"
            )
            % {
                "number": det.number,
                "class": det.class_name,
                "morphotype": det.morphotype or "—",
            }
        )

        um = self.state.um_per_px

        lines = [
            f"medido por  {m.method}",
            f"largo       {m.length_px:8.1f} px"
            + (f"  x {um:.4f} um/px = {det.length_um:.0f} um" if um > 0 and det.length_um else ""),
            f"ancho       {m.width_px:8.1f} px"
            + (f"  = {det.width_um:.0f} um" if um > 0 and det.width_um else ""),
            f"Feret       {m.feret_px:8.1f} px",
            f"geodesico   {m.geodesic_px:8.1f} px",
            f"aspecto     {m.aspect:8.1f}   curvatura {m.curvature:.2f}",
        ]

        if m.warning:
            lines.append(f"aviso: {m.warning}")

        self.lbl_record_data.setText("\n".join(lines))

    def _update_filter_combo(self, dets: list):

        self.combo_filter.blockSignals(True)

        prev = self.combo_filter.currentData()

        self.combo_filter.clear()

        self.combo_filter.addItem(_("All classes"), None)

        for _n in sorted({d.class_name for d in dets}):
            self.combo_filter.addItem(_n, _n)

        idx = self.combo_filter.findData(prev)

        if idx >= 0:
            self.combo_filter.setCurrentIndex(idx)

        self.combo_filter.blockSignals(False)

    def _on_filter_changed(self, _idx: int):

        # The class name goes in userData: if it were the shown text, translating

        # `Todas las clases` would leave the filter checking against a string that

        # would cease to exist and nothing would be shown again.

        self.canvas.set_filter_class(self.combo_filter.currentData())

    # ── Calibration ──────────────────────────────────────────

    def _start_calib(self, mode: str):

        self.state.start_calib(mode)

        self.btn_line.setChecked(mode == "linea")

        self.btn_circ.setChecked(mode == "circulo")

        self.lbl_calib_hint.setVisible(True)

        self.btn_cancel_calib.setVisible(True)

        if mode == "linea":
            self.lbl_calib_hint.setText(_("Click 2 points on\na known reference."))

        else:
            self.lbl_calib_hint.setText(
                _("Click 3 points on the edge\nof a circular object of known size.")
            )

        self._set_status(_("● Calibration mode active"), T.WARN)

    def _on_calib_pts_changed(self, pts: list):

        if not pts:
            self.btn_line.setChecked(False)

            self.btn_circ.setChecked(False)

            self.lbl_calib_hint.setVisible(False)

            self.btn_cancel_calib.setVisible(False)

        mode = self.state.calib_mode

        needed = 2 if mode == "linea" else 3

        self.lbl_calib_hint.setText(
            ngettext(
                "Click %(count)d more point.",
                "Click %(count)d more points.",
                needed - len(pts),
            )
            % {"count": needed - len(pts)}
            if pts
            else self.lbl_calib_hint.text()
        )

    def _on_calib_ready(self):
        """Enough points collected → ask the user for the real size.

        The positional arguments are mandatory: PySide6 does not accept
        ``min`` nor ``max`` as keywords in ``getDouble`` and it raised
        AttributeError, which left the calibration unusable.
        """

        mode = self.state.calib_mode

        if mode == "linea":
            val, ok = QInputDialog.getDouble(
                self,
                _("Calibration — Line"),
                _("Real length of the marked line (μm):"),
                1000.0,
                0.001,
                1_000_000.0,
                3,
            )

            if ok and val > 0:
                self.state.finish_calib_line(val)

            else:
                self.state.cancel_calib()

        elif mode == "circulo":
            # 100 mm by default: it is the diameter of the Petri dishes in the study

            val, ok = QInputDialog.getDouble(
                self,
                _("Calibration — Circle"),
                _("Real diameter of the circular object (μm):\n100 mm Petri dish → 100000"),
                100_000.0,
                0.001,
                10_000_000.0,
                3,
            )

            if ok and val > 0:
                self.state.finish_calib_circle(val)

            else:
                self.state.cancel_calib()

    def _on_calib_changed(self, um_per_px: float, mode_name: str):

        mode_label = {
            "línea": _("line"),
            "círculo": _("circle"),
        }.get(mode_name, mode_name)
        self.lbl_calib_result.setText(
            _("%(value).4f μm/px  (%(mode)s)") % {"value": um_per_px, "mode": mode_label}
        )

        self.lbl_calib_result.setStyleSheet(
            f"color: {T.OK_TX}; font-size: 9.5pt; font-weight: 600;"
        )

        self._set_status(_("● Calibration applied: %(value).4f μm/px") % {"value": um_per_px}, T.OK)

        self._update_status_bar()

        # Refresh the table with the new μm units

        if self.state.detections:
            self._rebuild_table(self.state.detections)

    def _update_status_bar(self):

        img = self.state.current_image

        img_txt = img.name if img else _("No image")

        if self.state.um_per_px > 0:
            calib_txt = f"{self.state.um_per_px:.4f} μm/px"

        else:
            calib_txt = _("Not calibrated")

        n_det = len(self.state.detections)

        det_txt = f"  —  {n_det} det." if n_det else ""

        self.status_bar.setText(f"  {img_txt}  —  {calib_txt}{det_txt}")

    # ── Save ───────────────────────────────────────────────

    def _save(self):

        img = self.state.current_image

        if img is None:
            QMessageBox.warning(self, _("No image"), _("No image is open."))

            return

        # Choose the destination folder

        out_dir = QFileDialog.getExistingDirectory(self, _("Output folder"))

        if not out_dir:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = Path(out_dir) / f"visor_{img.stem}_{ts}"

        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._save_annotated_image(img, save_dir)

            self._save_csv(save_dir)

            self._save_json(img, save_dir)

            self._set_status(_("● Saved to %(folder)s") % {"folder": save_dir.name}, T.OK)

            QMessageBox.information(
                self, _("Saved"), _("Files saved to:\n%(folder)s") % {"folder": save_dir}
            )

        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, _("Error while saving"), str(e))

            self._set_status(_("● Error while saving"), T.ERR)

    def _save_annotated_image(self, img: Path, save_dir: Path):

        import cv2

        from polyx.core.image_io import read_image

        # imdecode and not imread: on Windows imread fails with paths that contain

        # accents or non-ASCII characters, and returns None without warning.

        frame = read_image(img)

        if frame is None:
            return

        annotated = render_annotated_frame(frame, self.state.detections, self.state.um_per_px)

        out_path = save_dir / f"{img.stem}_anotada.png"

        ok, buf = cv2.imencode(".png", annotated)

        if ok:
            buf.tofile(str(out_path))

    def _save_csv(self, save_dir: Path):

        import csv

        out = save_dir / "detecciones.csv"

        rows = csv_rows(self.state.detections, self.state.um_per_px)

        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            w.writerows(rows)

    def _save_json(self, img: Path, save_dir: Path):

        um = self.state.um_per_px

        summary = {
            "imagen": img.name,
            "timestamp": datetime.now().isoformat(),
            "um_per_px": um if um > 0 else None,
            "n_detecciones": len(self.state.detections),
            "por_clase": {},
        }

        for det in self.state.detections:
            entry = summary["por_clase"].setdefault(det.class_name, {"n": 0})

            entry["n"] += 1

        out = save_dir / "resumen.json"

        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Drag & Drop ───────────────────────────────────────────

    def dragEnterEvent(self, ev):

        if ev.mimeData().hasUrls():
            for u in ev.mimeData().urls():
                p = u.toLocalFile().lower()

                if p.endswith(".pt") or any(
                    p.endswith(e) for e in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
                ):
                    ev.acceptProposedAction()

                    return

        ev.ignore()

    def dropEvent(self, ev):

        for url in ev.mimeData().urls():
            path = Path(url.toLocalFile())

            if path.suffix.lower() == ".pt":
                try:
                    from polyx.core.detect.yolo import YoloModel

                    self.state.model = YoloModel(str(path))

                    self.lbl_model.setText(path.name)

                    self._set_status(_("● Model loaded"), T.OK)

                except Exception as e:  # noqa: BLE001
                    self._set_status(_("● Error: %(error)s") % {"error": e}, T.ERR)

            elif path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
                self.state.load_single(path)

                self.nav_frame.setVisible(False)

        ev.acceptProposedAction()

    def _set_status(self, text: str, color: str = T.OK):

        self.lbl_status.setText(text)

        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 9pt;")
