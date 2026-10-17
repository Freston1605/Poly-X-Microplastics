"""Página 1 — Modelos. Hasta 3 slots .pt + carga rápida."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QFrame, QMessageBox,
)

from ._base import DetectorPage
from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.model.paths import DEFAULT_MODEL
from polyx.model.yolo_wrap import YoloModel
from polyx.model.i18n import tr


class _ModelSlotCard(QFrame):
    def __init__(self, idx: int, on_path_changed):
        super().__init__()
        self.idx = idx
        self._on_change = on_path_changed
        self.setAcceptDrops(True)
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(12)

        # Header del modelo
        head = QHBoxLayout()
        head.setSpacing(8)
        head_icon = QLabel()
        head_icon.setPixmap(iconos.pixmap("modelo", 17, T.INK2, 1.85))
        head_icon.setStyleSheet("font-size: 14pt; border: none;")
        head_title = QLabel(tr("Modelo {n}").format(n=idx + 1))
        head_title.setStyleSheet(f"color: {T.INK}; font-size: 12.5pt; font-weight: 600; border: none;")
        head.addWidget(head_icon)
        head.addWidget(head_title)
        head.addStretch(1)
        # Indicador de carga
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
        head.addWidget(self.lbl_status)
        lay.addLayout(head)

        # Alias
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        lbl1 = QLabel(tr("Alias:"))
        lbl1.setStyleSheet(f"color: {T.INK2}; border: none; min-width: 40px;")
        self.ed_alias = QLineEdit(tr("Modelo {n}").format(n=idx + 1))
        self.ed_alias.setMaximumWidth(220)
        self.ed_alias.editingFinished.connect(self._emit_change)
        row1.addWidget(lbl1)
        row1.addWidget(self.ed_alias)
        row1.addStretch(1)
        lay.addLayout(row1)

        # Path .pt
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        lbl2 = QLabel(".pt:")
        lbl2.setStyleSheet(f"color: {T.INK2}; border: none; min-width: 40px;")
        self.ed_path = QLineEdit("")
        self.ed_path.setPlaceholderText(tr("Ruta al archivo .pt del modelo entrenado…"))
        self.ed_path.editingFinished.connect(self._emit_change)
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(36)
        btn_browse.clicked.connect(self._browse)
        btn_clear = QPushButton("✕")
        btn_clear.setFixedWidth(36)
        btn_clear.clicked.connect(self._clear)
        row2.addWidget(lbl2)
        row2.addWidget(self.ed_path, 1)
        row2.addWidget(btn_browse)
        row2.addWidget(btn_clear)
        lay.addLayout(row2)

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar modelo YOLO (.pt)", "",
            "Pesos PyTorch (*.pt);;Todos (*.*)"
        )
        if f:
            self.ed_path.setText(f)
            self._emit_change()

    # ── Drag & Drop ──────────────────────────────────────────
    def dragEnterEvent(self, ev: QDragEnterEvent):
        if ev.mimeData().hasUrls():
            urls = ev.mimeData().urls()
            if any(u.toLocalFile().lower().endswith(".pt") for u in urls):
                ev.acceptProposedAction()
                self.setStyleSheet(
                    f"QFrame {{ background: #ddf4ff; border: 2px solid {T.ACCENT}; "
                    f"border-radius: 8px; }}"
                )
                return
        ev.ignore()

    def dragLeaveEvent(self, ev):
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )

    def dropEvent(self, ev: QDropEvent):
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        for url in ev.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".pt"):
                self.ed_path.setText(path)
                self._emit_change()
                break
        ev.acceptProposedAction()

    def _clear(self):
        self.ed_path.setText("")
        self._emit_change()

    def _emit_change(self):
        self._on_change(self.idx, self.ed_alias.text().strip(), self.ed_path.text().strip())

    def set_loaded_status(self, text: str, color: str):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 9pt; border: none;")


class ModelosPage(DetectorPage):
    STEP_N = 1
    STEP_TITLE = tr("Modelos")
    STEP_DESCRIPTION = (
        tr("Carga hasta 3 modelos .pt entrenados para detectar microplásticos. "
        "Si cargas más de uno, se compararán automáticamente en el reporte final.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # 3 slots
        self.slot_cards: list[_ModelSlotCard] = []
        for i in range(3):
            c = _ModelSlotCard(i, self._on_slot_changed)
            self.body.addWidget(c)
            self.slot_cards.append(c)

        # Carga rápida
        qf, qfl = self.card(tr("Carga rápida"), "rayo")
        info = QLabel(
            tr("Si tienes el modelo entrenado por el autor, úsalo como Modelo 1 con un solo clic.")
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        qfl.addWidget(info)

        row = QHBoxLayout()
        row.setSpacing(8)
        btn_default = QPushButton(tr("Usar bestdetectormedium.pt"))
        btn_default.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        btn_default.setCursor(Qt.PointingHandCursor)
        btn_default.clicked.connect(self._load_default)
        row.addWidget(btn_default)

        self.lbl_default = QLabel("")
        self.lbl_default.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
        row.addWidget(self.lbl_default)
        row.addStretch(1)
        qfl.addLayout(row)
        self.body.addWidget(qf)

        # Estado inicial
        if DEFAULT_MODEL.exists():
            self.lbl_default.setText(f"Encontrado en {DEFAULT_MODEL.name}")
        else:
            self.lbl_default.setText(tr("No encontrado en la raíz del proyecto."))

    # ──────────────────────────────────────────
    def _on_slot_changed(self, idx: int, alias: str, path_str: str):
        slot = self.state.model_slots[idx]
        slot.alias = alias or tr("Modelo {n}").format(n=idx + 1)
        if path_str:
            p = Path(path_str)
            if p.exists() and p.suffix.lower() == ".pt":
                slot.path = p
                slot.loaded = None  # se cargará perezosamente al ejecutar
                self.slot_cards[idx].set_loaded_status("✓ archivo válido", T.OK)
            else:
                slot.path = None
                slot.loaded = None
                self.slot_cards[idx].set_loaded_status("✗ archivo no encontrado", T.ERR)
        else:
            slot.path = None
            slot.loaded = None
            self.slot_cards[idx].set_loaded_status("", T.INK3)
        self.state.models_changed.emit()

    def _load_default(self):
        if not DEFAULT_MODEL.exists():
            QMessageBox.warning(
                self, tr("No encontrado"),
                f"No se encontró {DEFAULT_MODEL}.\n"
                "Cópialo a la raíz del proyecto e intenta de nuevo."
            )
            return
        # Cargar en slot 1
        card = self.slot_cards[0]
        card.ed_alias.setText(tr("bestdetectormedium"))
        card.ed_path.setText(str(DEFAULT_MODEL))
        card._emit_change()
