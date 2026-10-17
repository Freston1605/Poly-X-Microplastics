"""Dialogo de preferencias: tema, idioma y movimiento.

Las tres opciones se aplican de forma distinta, y el dialogo lo dice en vez de
dejar que el usuario lo descubra:

* **Tema** — se ve al instante en el launcher, que se rehace solo. Los modulos
  ya abiertos conservan el suyo, porque son procesos aparte y su hoja de
  estilo lleva los colores ya incrustados.
* **Idioma** — no se puede aplicar en caliente: los textos se fijan al
  construir cada widget. Los modulos que se abran despues salen traducidos.
* **Animacion** — inmediata en todas partes; solo hay que repintar.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)

from . import theme as T
from polyx.model.i18n import IDIOMAS, idioma, set_idioma, tr


# ──────────────────────────────────────────────────────────────────────
def muestra_tema(nombre: str, ancho: int = 104, alto: int = 62) -> QPixmap:
    """Miniatura de un tema: barra, tarjeta, texto y boton.

    Un cuadrado de color no dice como se vera el programa. Esto reproduce la
    estructura real —una barra superior, una superficie con texto y un boton de
    acento— que es lo que permite elegir sin tener que probar los cuatro.
    """
    p = T.PALETAS[nombre]
    px = QPixmap(ancho, alto)
    px.fill(QColor(p["BG_SOFT"]))
    q = QPainter(px)
    q.setRenderHint(QPainter.Antialiasing)

    # barra superior
    q.setPen(Qt.NoPen)
    q.setBrush(QColor(p["BG"]))
    q.drawRect(0, 0, ancho, 15)
    q.setPen(QColor(p["RULE"]))
    q.drawLine(0, 15, ancho, 15)
    q.setPen(Qt.NoPen)
    q.setBrush(QColor(p["ACCENT"]))
    q.drawRoundedRect(6, 4, 8, 8, 2, 2)
    q.setBrush(QColor(p["INK3"]))
    q.drawRoundedRect(19, 7, 26, 3, 1, 1)

    # tarjeta con dos lineas de texto
    q.setBrush(QColor(p["BG"]))
    q.setPen(QColor(p["RULE"]))
    q.drawRoundedRect(7, 22, ancho - 14, alto - 30, 4, 4)
    q.setPen(Qt.NoPen)
    q.setBrush(QColor(p["INK"]))
    q.drawRoundedRect(13, 29, 44, 4, 2, 2)
    q.setBrush(QColor(p["INK3"]))
    q.drawRoundedRect(13, 37, 60, 3, 1, 1)
    q.drawRoundedRect(13, 43, 34, 3, 1, 1)

    # boton de acento
    q.setBrush(QColor(p["ACCENT"]))
    q.drawRoundedRect(ancho - 38, 40, 28, 12, 3, 3)
    q.end()
    return px


class _TarjetaTema(QFrame):
    """Una opcion de tema: su miniatura, su nombre y su estado."""

    elegido = Signal(str)

    def __init__(self, nombre: str, etiqueta: str, activo: bool):
        super().__init__()
        self._nombre = nombre
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(etiqueta)
        self._pintar_estado(activo)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(9, 9, 9, 8)
        lay.setSpacing(7)

        vista = QLabel()
        vista.setPixmap(muestra_tema(nombre))
        vista.setStyleSheet("border: none;")
        lay.addWidget(vista, 0, Qt.AlignCenter)

        titulo = QLabel(etiqueta)
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet(
            f"color: {T.INK if activo else T.INK2}; font-size: 9.5pt; border: none;"
            f"font-weight: {'700' if activo else '500'};")
        lay.addWidget(titulo)

    def _pintar_estado(self, activo: bool) -> None:
        borde = T.ACCENT if activo else T.RULE
        grosor = 2 if activo else 1
        self.setStyleSheet(
            f"_TarjetaTema {{ background: {T.BG}; border: {grosor}px solid {borde};"
            f" border-radius: 9px; }}"
            f"_TarjetaTema:hover {{ border-color: {T.ACCENT}; }}")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.elegido.emit(self._nombre)
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        # Con el teclado la rejilla se recorre con Tab y se elige con Enter o
        # espacio, igual que cualquier control de Qt.
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.elegido.emit(self._nombre)
            return
        super().keyPressEvent(ev)


# ──────────────────────────────────────────────────────────────────────
class DialogoPreferencias(QDialog):
    """Preferencias de apariencia. El tema se aplica en vivo al elegirlo."""

    tema_cambiado = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Preferencias"))
        self.setMinimumWidth(520)
        self._construir()

    # ── Construccion ────────────────────────────────────────────────
    def _construir(self) -> None:
        # Se reconstruye entero al cambiar de tema: es un dialogo pequeno y
        # sale mas barato rehacerlo que ir tocando cada hoja de estilo.
        anterior = self.layout()
        if anterior is not None:
            QWidget().setLayout(anterior)

        self.setStyleSheet(T.GLOBAL_QSS + f"QDialog {{ background: {T.BG_SOFT}; }}")

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(22, 20, 22, 18)
        raiz.setSpacing(6)

        raiz.addWidget(self._titulo(tr("Tema")))
        raiz.addWidget(self._pista(
            tr("Se aplica al instante aquí. Los módulos ya abiertos conservan el suyo.")))

        rejilla = QGridLayout()
        rejilla.setSpacing(10)
        actual = T.tema()
        for i, nombre in enumerate(T.PALETAS):
            tarjeta = _TarjetaTema(nombre, tr(T.NOMBRES_TEMA[nombre]), nombre == actual)
            tarjeta.elegido.connect(self._elegir_tema)
            rejilla.addWidget(tarjeta, i // 4, i % 4)
        raiz.addLayout(rejilla)
        raiz.addSpacing(14)

        raiz.addWidget(self._titulo(tr("Idioma")))
        raiz.addWidget(self._pista(
            tr("Los módulos que abras a partir de ahora salen en el idioma nuevo.")))
        fila = QHBoxLayout()
        fila.setSpacing(8)
        self._botones_idioma: dict[str, QPushButton] = {}
        for cod, etiqueta in IDIOMAS.items():
            b = QPushButton(etiqueta)
            b.setCheckable(True)
            b.setChecked(cod == idioma())
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumWidth(112)
            b.setStyleSheet(self._qss_conmutador())
            b.clicked.connect(lambda _=False, c=cod: self._elegir_idioma(c))
            self._botones_idioma[cod] = b
            fila.addWidget(b)
        fila.addStretch(1)
        raiz.addLayout(fila)
        raiz.addSpacing(16)

        raiz.addWidget(self._titulo(tr("Movimiento")))
        self._chk_anim = QCheckBox(tr("Animar el panel del microscopio"))
        self._chk_anim.setChecked(T.animaciones())
        self._chk_anim.setCursor(Qt.PointingHandCursor)
        self._chk_anim.setStyleSheet(
            f"QCheckBox {{ color: {T.INK2}; background: transparent; spacing: 8px; }}")
        self._chk_anim.toggled.connect(T.set_animaciones)
        raiz.addWidget(self._chk_anim)
        raiz.addWidget(self._pista(
            tr("Apágalo si el movimiento te molesta o si el equipo va justo. "
               "El panel queda en un fotograma fijo.")))

        raiz.addSpacing(18)
        pie = QHBoxLayout()
        pie.addStretch(1)
        cerrar = QPushButton(tr("Cerrar"))
        cerrar.setObjectName("primary")
        cerrar.setCursor(Qt.PointingHandCursor)
        cerrar.setDefault(True)
        cerrar.clicked.connect(self.accept)
        pie.addWidget(cerrar)
        raiz.addLayout(pie)

    # ── Piezas ──────────────────────────────────────────────────────
    def _titulo(self, texto: str) -> QLabel:
        lbl = QLabel(texto)
        lbl.setStyleSheet(
            f"color: {T.INK}; font-size: 11.5pt; font-weight: 650;"
            f" border: none; background: transparent;")
        return lbl

    def _pista(self, texto: str) -> QLabel:
        lbl = QLabel(texto)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"color: {T.INK3}; font-size: 9pt; border: none;"
            f" background: transparent; margin-bottom: 6px;")
        return lbl

    def _qss_conmutador(self) -> str:
        return (
            f"QPushButton {{ background: {T.BG}; color: {T.INK2};"
            f" border: 1px solid {T.RULE}; border-radius: 6px; padding: 7px 14px; }}"
            f"QPushButton:hover {{ border-color: {T.ACCENT}; }}"
            f"QPushButton:checked {{ background: {T.ACCENT}; color: {T.ON_ACCENT};"
            f" border-color: {T.ACCENT}; font-weight: 600; }}")

    # ── Acciones ────────────────────────────────────────────────────
    def _elegir_tema(self, nombre: str) -> None:
        if nombre == T.tema():
            return
        T.set_tema(nombre)
        self._construir()          # el propio dialogo se repinta con la paleta nueva
        self.tema_cambiado.emit(nombre)

    def _elegir_idioma(self, codigo: str) -> None:
        for cod, boton in self._botones_idioma.items():
            boton.setChecked(cod == codigo)
        if codigo != idioma():
            set_idioma(codigo)
