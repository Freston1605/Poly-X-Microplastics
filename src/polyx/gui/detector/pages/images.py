"""Page 2 — `Imágenes`. File/folder selection + optional GT folder."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from polyx.core.detect.protocol import find_gt_for_image
from polyx.core.i18n import _, lazy, ngettext
from polyx.core.paths import IMAGE_EXTS
from polyx.gui import iconos
from polyx.gui import theme as T

from ._base import DetectorPage


class ImagesPage(DetectorPage):
    STEP_N = 2
    STEP_TITLE = lazy("Images")
    STEP_DESCRIPTION = lazy(
        "Select individual images or a folder. If you have ground truth labels in YOLO format (.txt) in the same folder or in a sibling 'labels/' folder, they are loaded and enable error analysis."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setAcceptDrops(True)

        # ── Source card ──
        c1, l1 = self.card(_("Source"), "dataset")
        row = QHBoxLayout()
        row.setSpacing(8)
        btn_files = QPushButton(_("Select images…"))
        btn_files.setIcon(iconos.icon("imagenes", 15, T.INK2))
        btn_files.clicked.connect(self._pick_files)
        btn_folder = QPushButton(_("Select folder…"))
        btn_folder.setIcon(iconos.icon("dataset", 15, T.INK2))
        btn_folder.clicked.connect(self._pick_folder)
        btn_clear = QPushButton(_("✕  Clear"))
        btn_clear.clicked.connect(self._clear)
        row.addWidget(btn_files)
        row.addWidget(btn_folder)
        row.addWidget(btn_clear)
        row.addStretch(1)
        l1.addLayout(row)

        # Optional GT folder
        gt_row = QHBoxLayout()
        gt_row.setSpacing(8)
        lbl_gt = QLabel(_("Ground-truth folder (optional):"))
        lbl_gt.setStyleSheet(f"color: {T.INK2}; border: none;")
        self.ed_gt = QLineEdit()
        self.ed_gt.setPlaceholderText(
            _("Path to the folder with YOLO .txt files (otherwise it looks next to the image)")
        )
        self.ed_gt.editingFinished.connect(self._gt_text_changed)
        btn_gt = QPushButton("…")
        btn_gt.setFixedWidth(36)
        btn_gt.clicked.connect(self._pick_gt_folder)
        gt_row.addWidget(lbl_gt)
        gt_row.addWidget(self.ed_gt, 1)
        gt_row.addWidget(btn_gt)
        l1.addLayout(gt_row)

        hint = QLabel(
            _(
                "If left empty, .txt files are looked up next to each image and in a sibling /labels/ folder. An image that has ground truth is included in the error analysis (True Positives, False Positives and False Negatives); otherwise only the model's detections are reported."
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l1.addWidget(hint)
        self.body.addWidget(c1)

        # ── Loaded images card ──
        c2, l2 = self.card(_("Loaded images"), "imagenes")
        self.lbl_count = QLabel(_("0 images"))
        self.lbl_count.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l2.addWidget(self.lbl_count)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([_("Image"), _("Folder"), "GT"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(280)
        l2.addWidget(self.table)
        self.body.addWidget(c2)

    # ──────────────────────────────────────────
    def _pick_files(self):
        ext_filter = _("Images (%(patterns)s)") % {
            "patterns": " ".join(f"*{extension}" for extension in IMAGE_EXTS)
        }
        files, _unused = QFileDialog.getOpenFileNames(self, _("Select images"), "", ext_filter)
        if files:
            self.state.images = [Path(f) for f in files]
            self._refresh_table()

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, _("Select a folder with images"))
        if d:
            root = Path(d)
            imgs: list[Path] = []
            for ext in IMAGE_EXTS:
                imgs.extend(root.rglob(f"*{ext}"))
            imgs = sorted(set(imgs))
            self.state.images = imgs
            self._refresh_table()

    def _clear(self):
        self.state.images = []
        self._refresh_table()

    def _pick_gt_folder(self):
        d = QFileDialog.getExistingDirectory(self, _("Select the GT folder"))
        if d:
            self.ed_gt.setText(d)
            self._gt_text_changed()

    def _gt_text_changed(self):
        txt = self.ed_gt.text().strip()
        self.state.gt_folder = Path(txt) if txt else None
        self._refresh_table()

    # ── Drag & Drop ───────────────────────────────────────────
    def dragEnterEvent(self, ev: QDragEnterEvent):
        if ev.mimeData().hasUrls():
            urls = ev.mimeData().urls()
            # Accepts folders or image files
            for u in urls:
                p = u.toLocalFile()
                from pathlib import Path as _P

                pp = _P(p)
                if pp.is_dir() or pp.suffix.lower() in IMAGE_EXTS:
                    ev.acceptProposedAction()
                    return
        ev.ignore()

    def dropEvent(self, ev: QDropEvent):
        imgs: list[Path] = []
        for url in ev.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                for ext in IMAGE_EXTS:
                    imgs.extend(p.rglob(f"*{ext}"))
            elif p.suffix.lower() in IMAGE_EXTS:
                imgs.append(p)
        if imgs:
            self.state.images = sorted(set(imgs))
            self._refresh_table()
        ev.acceptProposedAction()

    def _refresh_table(self):
        imgs = self.state.images
        self.lbl_count.setText(
            ngettext("%(count)d image", "%(count)d images", len(imgs)) % {"count": len(imgs)}
        )
        self.table.setRowCount(len(imgs))
        for r, p in enumerate(imgs):
            self.table.setItem(r, 0, QTableWidgetItem(p.name))
            self.table.setItem(r, 1, QTableWidgetItem(str(p.parent)))
            gt = find_gt_for_image(p, self.state.gt_folder)
            item = QTableWidgetItem("✓" if gt else "—")
            item.setTextAlignment(Qt.AlignCenter)
            if gt:
                item.setForeground(Qt.darkGreen)
            self.table.setItem(r, 2, item)
        self.state.images_changed.emit()
