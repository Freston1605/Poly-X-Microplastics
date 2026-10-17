"""Módulo 1 — Launcher (menú principal). Reproduce la Figura 1 del manual.

Hero con kicker, título 'Poly-X analytics', subtítulo, 4 KPI chips,
y 4 tarjetas grandes de módulo (Detector, Entrenador, Etiquetador, Visor).
"""
from __future__ import annotations
import logging
import math
import random
import sys
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QTimer, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QLinearGradient, QRadialGradient,
    QPainterPath, QFont, QKeySequence, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QFrame, QScrollArea, QMessageBox
)

from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.gui import marca
from polyx.model.i18n import tr, idioma
from polyx.gui.preferencias import DialogoPreferencias
from polyx.gui.widgets import LogoBadge, HLine
from polyx.gui.updates import ACTUALIZADOR, BuscadorActualizaciones, puede_actualizar
from polyx.model.plataforma import abrir_en_el_sistema, lanzar_actualizador
from polyx import __version__

# ──────────────────────────────────────────────────────────────────────
# Arranque instrumentado: el launcher deja rastro JSONL en app.jsonl.
# Los imports de observabilidad van perezosos, dentro de las funciones,
# para que una falla suya no impida ni retrase el arranque de la ventana.
# ──────────────────────────────────────────────────────────────────────


def _registrar_arranque() -> None:
    """Emite ``bootstrap.finish`` con version, python y sistema en app.jsonl.

    Nunca lanza: si HOME es read-only o el directorio de logs no se puede
    crear, el launcher abre igual que siempre. La observabilidad no puede
    ser peor que no tenerla.
    """
    try:
        from polyx.model import observabilidad
        from polyx.model.plataforma import nombre_sistema
        log = observabilidad.logger_para("app")
        observabilidad.evento(
            log,
            "bootstrap.finish",
            version=__version__,
            python=sys.version.split()[0],
            sistema=nombre_sistema(),
        )
    except Exception:
        pass


def _spawn_module(dotted: str) -> bool:
    """Lanza el modulo en un proceso aparte; True si arranco, False si no.

    Devuelve en vez de lanzar: asi la ventana decide como avisar al usuario
    y el arrastre de observabilidad (module.launch / module.launch.error en
    app.jsonl) no puede convertirse en un crash de la interfaz.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec(dotted)
        if spec is None:
            return False
        subprocess.Popen(
            [sys.executable, "-m", dotted],
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    except Exception as e:
        try:
            from polyx.model import observabilidad
            observabilidad.evento(
                observabilidad.logger_para("app"),
                "module.launch.error",
                level=logging.WARNING,
                modulo=dotted,
                error=str(e),
            )
        except Exception:
            pass
        return False
    try:
        from polyx.model import observabilidad
        observabilidad.evento(
            observabilidad.logger_para("app"), "module.launch", modulo=dotted
        )
    except Exception:
        pass
    return True


# ──────────────────────────────────────────────────────────────────────
# Panel del hero: campo de microscopio con una deteccion en curso
# ──────────────────────────────────────────────────────────────────────
class MicroscopePanel(QWidget):
    """Campo oscuro con particulas fluorescentes que un barrido va detectando.

    No es un adorno cualquiera: cuenta lo que hace el programa. Las particulas
    emiten como emite el Nile Red bajo UV —un nucleo saturado dentro de un halo
    que se apaga con la distancia, no un disco de color plano— y sobre ellas
    pasa un barrido que va dejando cajas con su clase y su confianza, que es
    exactamente la salida del Detector.

    Que se mueve, y por que:

    * **Deriva en dos ejes.** Cada particula recorre una figura de Lissajous
      propia, con su periodo y su fase. Antes todas subian y bajaban con el
      mismo seno y el conjunto latia como un solo bloque, que es justo lo que
      delata una animacion barata.
    * **Tres planos de profundidad.** Los del fondo son menores, mas apagados y
      mas lentos: asi se ve una preparacion que tiene espesor, y no una calca.
    * **Un barrido que detecta.** Recorre el campo cada ciclo y va marcando lo
      que encuentra, con la caja apareciendo cuando el barrido la cruza.

    Las dos particulas mas pequenas no reciben caja nunca. Es deliberado: son
    las que el detector real pierde, y dibujarlas todas marcadas prometeria un
    recall que el modelo no tiene.

    Con las animaciones apagadas (ver ``theme.animaciones``) se dibuja un solo
    fotograma compuesto, con el barrido terminado y todas las cajas puestas.
    """

    #: Duracion de un ciclo completo, en segundos.
    CICLO = 7.6
    #: Tramos del ciclo, en fraccion: espera, barrido, permanencia, salida.
    _INICIO_BARRIDO = 0.06
    _FIN_BARRIDO = 0.52
    _INICIO_SALIDA = 0.86

    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 240)
        self.setMaximumWidth(360)

        rnd = random.Random(11)
        clases = ("PET", "PP", "LDPE")
        planos = (0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2)   # 4 fondo, 5 medio, 4 frente

        # Reparto por anillos, no por sorteo libre. Trece posiciones al azar
        # dentro de un circulo dejan medio campo vacio y un grumo en la otra
        # mitad: es lo que pasa con la aleatoriedad en muestras pequenas.
        # Repartiendo 1 + 4 + 8 por anillos y sorteando solo el desvio, el
        # campo queda cubierto y aun asi no se ve una rejilla.
        sitios = [(0.0, 0.0)]
        for cuantos, radio in ((4, 0.215), (8, 0.355)):
            for k in range(cuantos):
                ang = 2 * math.pi * k / cuantos + rnd.uniform(-0.32, 0.32)
                rad = radio + rnd.uniform(-0.035, 0.035)
                sitios.append((rad * math.cos(ang), rad * math.sin(ang)))
        rnd.shuffle(sitios)

        # Las clases se reparten por turno y no por sorteo. Con trece sorteos
        # independientes salio un campo casi todo amarillo y un solo PET, que
        # da una impresion falsa: el programa distingue tres polimeros y este
        # panel es lo primero que se ve de el.
        turno = [clases[i % 3] for i in range(len(planos))]
        rnd.shuffle(turno)

        self._particulas: list[dict] = []
        for plano, (dx, dy), clase in zip(planos, sitios, turno):
            escala = (0.55, 0.82, 1.0)[plano]
            self._particulas.append({
                "x": 0.5 + dx,
                "y": 0.5 + dy,
                "r": rnd.uniform(5.5, 15.0) * escala,
                "clase": clase,
                "forma": rnd.choices((0, 1, 2), weights=(5, 3, 2))[0],
                "giro": rnd.uniform(0, 360),
                "plano": plano,
                "conf": rnd.uniform(0.71, 0.97),
                "ax": rnd.uniform(0.006, 0.020) * escala,
                "ay": rnd.uniform(0.006, 0.020) * escala,
                "wx": rnd.uniform(0.20, 0.44),
                "wy": rnd.uniform(0.20, 0.44),
                "fx": rnd.uniform(0, 2 * math.pi),
                "fy": rnd.uniform(0, 2 * math.pi),
            })

        # Las dos mas pequenas se quedan sin caja: son las que se pierden.
        for i, part in enumerate(sorted(self._particulas, key=lambda q: q["r"])):
            part["detectable"] = i >= 2
        # Solo las mayores llevan etiqueta, o el campo se vuelve ilegible.
        for i, part in enumerate(sorted(self._particulas, key=lambda q: -q["r"])):
            part["etiqueta"] = i < 3 and part["detectable"]

        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        # 33 ms ~ 30 fps: la deriva ya se ve continua y cuesta la mitad que 60,
        # en un panel al que nadie mira fijamente.
        self._timer.setInterval(33)

    # ── Ciclo de vida ───────────────────────────────────────────────
    def showEvent(self, e):
        super().showEvent(e)
        if T.animaciones():
            self._timer.start()

    def hideEvent(self, e):
        # Un panel tapado que sigue repintando gasta bateria sin que se vea.
        self._timer.stop()
        super().hideEvent(e)

    def _tick(self):
        self._t += self._timer.interval() / 1000.0
        self.update()

    # ── Geometria del ciclo ─────────────────────────────────────────
    def _fase(self) -> float:
        """Posicion dentro del ciclo, de 0 a 1."""
        if not T.animaciones():
            # Fotograma fijo: barrido terminado y cajas puestas.
            return (self._INICIO_SALIDA + self._FIN_BARRIDO) / 2
        return (self._t % self.CICLO) / self.CICLO

    def _avance_barrido(self, fase: float):
        """Altura del barrido (0 arriba, 1 abajo). None si no esta pasando."""
        if fase < self._INICIO_BARRIDO:
            return None
        if fase <= self._FIN_BARRIDO:
            return (fase - self._INICIO_BARRIDO) / (self._FIN_BARRIDO - self._INICIO_BARRIDO)
        return None

    def _pos(self, part: dict, t: float):
        return (part["x"] + part["ax"] * math.sin(part["wx"] * t + part["fx"]),
                part["y"] + part["ay"] * math.sin(part["wy"] * t + part["fy"]))

    def _alfa_caja(self, fase: float, y_rel: float) -> float:
        """Opacidad de la caja de una particula situada a la altura y_rel."""
        if fase < self._INICIO_BARRIDO:
            return 0.0
        if fase >= self._INICIO_SALIDA:
            return max(0.0, 1.0 - (fase - self._INICIO_SALIDA) / (1.0 - self._INICIO_SALIDA))
        avance = self._avance_barrido(fase)
        if avance is None:          # el barrido ya recorrio todo el campo
            return 1.0
        if avance < y_rel:
            return 0.0
        return min(1.0, (avance - y_rel) * 9.0)

    # ── Dibujo ──────────────────────────────────────────────────────
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        t = self._t
        fase = self._fase()
        W, H = self.width(), self.height()
        lado = min(W, H) - 18
        cx, cy = W / 2, H / 2
        campo = QRectF(cx - lado / 2, cy - lado / 2, lado, lado)

        # ── Objetivo: anillo con un realce arriba, para que parezca vidrio ──
        p.setPen(QPen(QColor(T.RING), 5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(campo.adjusted(-3, -3, 3, 3))
        realce = QLinearGradient(campo.topLeft(), campo.bottomLeft())
        realce.setColorAt(0.0, T.con_alfa("#ffffff", 45 if T.es_oscuro() else 60))
        realce.setColorAt(0.5, T.con_alfa("#ffffff", 0))
        p.setPen(QPen(QBrush(realce), 2))
        p.drawEllipse(campo.adjusted(-1, -1, 1, 1))

        # ── Campo: radial, mas claro al centro. La vineta sale sola ──
        fondo = QRadialGradient(campo.center(), lado / 2)
        fondo.setColorAt(0.0, QColor(T.FIELD_2))
        fondo.setColorAt(0.72, QColor(T.FIELD_1))
        fondo.setColorAt(1.0, QColor(T.FIELD_1).darker(140))
        p.setBrush(QBrush(fondo))
        p.setPen(Qt.NoPen)
        p.drawEllipse(campo)

        p.save()
        recorte = QPainterPath()
        recorte.addEllipse(campo)
        p.setClipPath(recorte)

        # ── Reticula tenue: da escala sin competir con las particulas ──
        p.setPen(QPen(T.con_alfa("#ffffff", 12), 1))
        paso = lado / 6
        for i in range(1, 6):
            p.drawLine(QPointF(campo.left() + i * paso, campo.top()),
                       QPointF(campo.left() + i * paso, campo.bottom()))
            p.drawLine(QPointF(campo.left(), campo.top() + i * paso),
                       QPointF(campo.right(), campo.top() + i * paso))

        # ── Particulas, de atras hacia adelante ──
        for part in sorted(self._particulas, key=lambda q: q["plano"]):
            self._dibujar_particula(p, part, campo, lado, t)

        # ── Barrido ──
        avance = self._avance_barrido(fase)
        if avance is not None:
            self._dibujar_barrido(p, campo, avance)

        # ── Cajas de deteccion ──
        # Las etiquetas se colocan de mayor a menor y se descarta la que caiga
        # encima de otra ya puesta: dos rotulos superpuestos no se leen, y en
        # un campo de 13 particulas eso pasa en cuanto la deriva las junta.
        ocupado: list[QRectF] = []
        for part in sorted(self._particulas, key=lambda q: -q["r"]):
            if not part["detectable"]:
                continue
            _, y_rel = self._pos(part, t)
            alfa = self._alfa_caja(fase, y_rel)
            if alfa > 0.01:
                self._dibujar_caja(p, part, campo, lado, t, alfa, ocupado)

        p.restore()
        self._dibujar_escala(p, cx, cy, lado)

    def _dibujar_particula(self, p, part, campo, lado, t):
        x_rel, y_rel = self._pos(part, t)
        x = campo.left() + x_rel * lado
        y = campo.top() + y_rel * lado
        r = part["r"] * (lado / 250.0)
        base = QColor(T.CLASS_COLOR_HEX[part["clase"]])

        # Halo: un radial que se apaga, no un disco de alfa constante. Es la
        # diferencia entre parecer fluorescencia y parecer una pegatina.
        intensidad = (115, 155, 195)[part["plano"]]
        radio_halo = r * 3.4
        halo = QRadialGradient(QPointF(x, y), radio_halo)
        centro = QColor(base); centro.setAlpha(intensidad)
        medio = QColor(base); medio.setAlpha(int(intensidad * 0.30))
        tenue = QColor(base); tenue.setAlpha(int(intensidad * 0.10))
        borde = QColor(base); borde.setAlpha(0)
        halo.setColorAt(0.00, centro)
        halo.setColorAt(0.30, medio)
        halo.setColorAt(0.62, tenue)
        halo.setColorAt(1.00, borde)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(QPointF(x, y), radio_halo, radio_halo)

        # Cuerpo. Los planos del fondo van apagados: profundidad de campo.
        cuerpo = QColor(base)
        if part["plano"] == 0:
            cuerpo.setAlpha(150)
        elif part["plano"] == 1:
            cuerpo.setAlpha(205)
        p.setBrush(QBrush(cuerpo))

        forma = part["forma"]
        if forma == 0:                                    # fragmento redondeado
            p.drawEllipse(QPointF(x, y), r, r * 0.92)
        elif forma == 1:                                  # fibra
            p.save()
            p.translate(x, y)
            p.rotate(part["giro"] + 9 * math.sin(0.35 * t + part["fx"]))
            p.drawRoundedRect(QRectF(-r * 1.9, -r * 0.24, r * 3.8, r * 0.48),
                              r * 0.24, r * 0.24)
            p.restore()
        else:                                             # fragmento irregular
            p.save()
            p.translate(x, y)
            p.rotate(part["giro"])
            camino = QPainterPath()
            camino.addEllipse(QPointF(0, 0), r * 1.15, r * 0.68)
            p.drawPath(camino)
            p.restore()

        # Nucleo: el punto donde la emision satura. Sin el, una particula
        # grande se lee como una mancha plana de color.
        if part["plano"] > 0 and forma != 1:
            nucleo = QRadialGradient(QPointF(x - r * 0.18, y - r * 0.18), r * 0.85)
            nucleo.setColorAt(0.0, T.con_alfa("#ffffff", 120))
            nucleo.setColorAt(1.0, T.con_alfa("#ffffff", 0))
            p.setBrush(QBrush(nucleo))
            p.drawEllipse(QPointF(x - r * 0.18, y - r * 0.18), r * 0.85, r * 0.85)

    def _dibujar_barrido(self, p, campo, avance):
        """Banda luminosa que recorre el campo de arriba abajo."""
        y = campo.top() + avance * campo.height()
        alto = campo.height() * 0.13
        banda = QLinearGradient(QPointF(0, y - alto), QPointF(0, y + 2))
        banda.setColorAt(0.0, T.con_alfa(T.ACCENT_TX, 0))
        banda.setColorAt(0.85, T.con_alfa(T.ACCENT_TX, 26))
        banda.setColorAt(1.0, T.con_alfa(T.ACCENT_TX, 52))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(banda))
        p.drawRect(QRectF(campo.left(), y - alto, campo.width(), alto))
        p.setPen(QPen(T.con_alfa(T.ACCENT_TX, 190), 1.4))
        p.drawLine(QPointF(campo.left(), y), QPointF(campo.right(), y))

    def _dibujar_caja(self, p, part, campo, lado, t, alfa, ocupado):
        x_rel, y_rel = self._pos(part, t)
        x = campo.left() + x_rel * lado
        y = campo.top() + y_rel * lado
        r = part["r"] * (lado / 250.0)
        margen = r * (2.0 if part["forma"] == 1 else 1.55)
        caja = QRectF(x - margen, y - margen * 0.85, margen * 2, margen * 1.7)

        color = QColor(T.CLASS_COLOR_HEX[part["clase"]])
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(T.con_alfa(color.name(), int(215 * alfa)), 1.5))
        p.drawRect(caja)

        # Esquinas marcadas: se lee como caja de deteccion aun a 1.5 px de trazo
        p.setPen(QPen(T.con_alfa(color.name(), int(255 * alfa)), 2.4))
        c = min(caja.width(), caja.height()) * 0.26
        for sx, sy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            px = caja.left() + sx * caja.width()
            py = caja.top() + sy * caja.height()
            p.drawLine(QPointF(px, py), QPointF(px + (c if sx == 0 else -c), py))
            p.drawLine(QPointF(px, py), QPointF(px, py + (c if sy == 0 else -c)))

        if not part["etiqueta"] or alfa < 0.5:
            return
        texto = f'{part["clase"]} {part["conf"]:.2f}'
        p.setFont(QFont(T.APP_FONT_FAMILY, 7, QFont.DemiBold))
        fm = p.fontMetrics()
        ancho = fm.horizontalAdvance(texto) + 8
        alto = fm.height() + 2
        etiqueta = QRectF(caja.left(), caja.top() - alto - 2, ancho, alto)
        if any(etiqueta.intersects(otra.adjusted(-3, -3, 3, 3)) for otra in ocupado):
            return
        ocupado.append(etiqueta)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(T.con_alfa(color.name(), int(230 * alfa))))
        p.drawRoundedRect(etiqueta, 2, 2)
        p.setPen(QPen(T.con_alfa("#0d1117", int(255 * alfa))))
        p.drawText(etiqueta, Qt.AlignCenter, texto)

    def _dibujar_escala(self, p, cx, cy, lado):
        # Dentro del circulo, no en la esquina de su caja envolvente: a
        # 0.355 del radio hacia abajo el borde ya se ha metido y una barra
        # puesta en la esquina queda flotando sobre la tarjeta.
        ancho = lado * 0.19
        y = cy + lado * 0.355
        x = cx + lado * 0.30 - ancho
        texto = "10 μm"
        p.setFont(QFont(T.APP_FONT_FAMILY, 8, QFont.Bold))
        fm = p.fontMetrics()

        # Placa translucida debajo. Las particulas derivan, asi que cualquier
        # posicion fija acaba tarde o temprano encima de una: en vez de buscar
        # un hueco que no existe, se asegura el contraste. Es ademas lo que
        # hace cualquier programa de microscopia con su barra de escala.
        alto_placa = fm.height() + 16
        placa = QRectF(x - 9, y - alto_placa + 8, ancho + 18, alto_placa)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(T.con_alfa("#000000", 125)))
        p.drawRoundedRect(placa, 4, 4)

        p.setPen(QPen(T.con_alfa("#ffffff", 230), 2))
        p.drawLine(QPointF(x, y), QPointF(x + ancho, y))
        for extremo in (x, x + ancho):
            p.drawLine(QPointF(extremo, y - 4), QPointF(extremo, y + 4))
        p.setPen(QPen(T.con_alfa("#ffffff", 240)))
        p.drawText(QPointF(x + ancho / 2 - fm.horizontalAdvance(texto) / 2, y - 7), texto)


# ──────────────────────────────────────────────────────────────────────
# Chips KPI superiores ('3 / POLÍMEROS / PET·PP·LDPE')
# ──────────────────────────────────────────────────────────────────────
class StatChip(QFrame):
    def __init__(self, big: str, label: str, sub: str, color: str):
        super().__init__()
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        self.setMinimumHeight(96)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)

        l_big = QLabel(big)
        l_big.setStyleSheet(f"color: {color}; font-size: 26pt; font-weight: 700; border: none;")
        l_label = QLabel(label.upper())
        l_label.setStyleSheet(
            f"color: {T.INK2}; font-size: 8.5pt; font-weight: 700; "
            f"letter-spacing: 1.5px; border: none;"
        )
        l_sub = QLabel(sub)
        l_sub.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
        l_sub.setWordWrap(True)

        lay.addWidget(l_big)
        lay.addWidget(l_label)
        lay.addWidget(l_sub)


# ──────────────────────────────────────────────────────────────────────
# Tarjeta de módulo
# ──────────────────────────────────────────────────────────────────────
class ModuleCard(QFrame):
    """Tarjeta de un modulo: icono, titulo, descripcion y accion de abrir.

    Toda la tarjeta es el area de clic, no solo el boton. Por eso necesita
    decir que es pulsable —borde de acento y elevacion al pasar por encima— y
    ser alcanzable con el teclado: sin ``StrongFocus`` la rejilla de modulos
    quedaba fuera del recorrido con Tab y solo existia para el raton.
    """

    def __init__(self, number: int, icono_nombre: str, title: str, description: str,
                 accent: str, on_open, atajo: str = ""):
        super().__init__()
        self._on_open = on_open
        self._accent = accent
        self.setStyleSheet(self._qss(activo=False))
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(title)
        self.setMinimumHeight(170)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(8)

        top = QHBoxLayout()
        tag = QLabel(f'{tr("MÓDULO")} 0{number}')
        tag.setStyleSheet(
            f"color: white; background: {accent}; padding: 3px 8px; "
            f"border-radius: 4px; font-size: 8.5pt; font-weight: 700; "
            f"letter-spacing: 1.2px; border: none;"
        )
        tag.setFixedHeight(20)
        tag.setAlignment(Qt.AlignCenter)
        top.addWidget(tag, 0, Qt.AlignLeft | Qt.AlignVCenter)
        top.addStretch(1)
        self._btn_open = QPushButton(tr("Abrir  →"))
        self._btn_open.setCursor(Qt.PointingHandCursor)
        self._btn_open.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {accent}; "
            f"border: none; font-weight: 600; padding: 4px 0; }}"
            f"QPushButton:hover {{ color: {T.ACCENT_D_TX}; }}"
        )
        self._btn_open.clicked.connect(self._abrir)
        top.addWidget(self._btn_open)
        lay.addLayout(top)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(iconos.pixmap(icono_nombre, 30, accent, 1.85))
        icon_lbl.setStyleSheet("border: none;")
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {T.INK}; font-size: 17pt; font-weight: 600; border: none;"
        )
        title_row.addWidget(icon_lbl)
        title_row.addWidget(title_lbl)
        title_row.addStretch(1)
        lay.addLayout(title_row)

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        lay.addWidget(desc)
        lay.addStretch(1)

        self.setCursor(Qt.PointingHandCursor)
        if atajo:
            self.setToolTip(f"{title}  ·  {atajo}")

    # ── Estado visual ───────────────────────────────────────────
    def _qss(self, activo: bool) -> str:
        borde = self._accent if activo else T.RULE
        fondo = T.sobre_fondo(self._accent, 0.05) if activo else T.BG
        return (f"ModuleCard {{ background: {fondo}; border: 1px solid {borde};"
                f" border-radius: 10px; }}")

    def enterEvent(self, ev):
        # Sin esto, una tarjeta entera pulsable no daba ninguna senal de serlo.
        self.setStyleSheet(self._qss(activo=True))
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self.setStyleSheet(self._qss(activo=False))
        super().leaveEvent(ev)

    def focusInEvent(self, ev):
        self.setStyleSheet(self._qss(activo=True))
        super().focusInEvent(ev)

    def focusOutEvent(self, ev):
        self.setStyleSheet(self._qss(activo=False))
        super().focusOutEvent(ev)

    # ── Accion ──────────────────────────────────────────────────
    def _abrir(self):
        """Abre el modulo y avisa de que se esta abriendo.

        Un modulo tarda dos o tres segundos en aparecer, porque su proceso
        importa torch antes de dibujar nada. Sin senal, ese silencio se lee
        como que el clic no llego y la gente vuelve a pulsar, con lo que
        acaban abriendose dos ventanas.
        """
        self._btn_open.setEnabled(False)
        self._btn_open.setText(tr("Abriendo…"))
        QTimer.singleShot(3200, self._restaurar_boton)
        self._on_open()

    def _restaurar_boton(self):
        self._btn_open.setEnabled(True)
        self._btn_open.setText(tr("Abrir  →"))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._abrir()
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._abrir()
            return
        super().keyPressEvent(ev)


# ──────────────────────────────────────────────────────────────────────
# Ventana principal Launcher
# ──────────────────────────────────────────────────────────────────────
class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        marca.titular(self, "launcher")
        self.resize(1180, 820)

        # Se guarda aparte porque la ventana se reconstruye al cambiar de tema
        # y el aviso de version no puede perderse en esa reconstruccion.
        self._sha_version_nueva: str = ""

        self._buscador = BuscadorActualizaciones(self)
        self._buscador.hay_version_nueva.connect(self._avisar_version_nueva)
        # Se dispara con la ventana ya dibujada: el arranque manda.
        QTimer.singleShot(1200, self._buscador.buscar)

        self._construir_ui()

    def _construir_ui(self):
        """Arma la ventana entera con la paleta activa.

        Rehacerla es la forma sencilla de aplicar un tema en caliente: los
        colores viven incrustados en el stylesheet de cada widget, asi que no
        hay forma de repintarlos sin volver a crearlos. En una ventana de
        cuatro tarjetas cuesta milisegundos y no se ve el parpadeo.
        """
        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Topbar ──
        topbar = QFrame()
        topbar.setStyleSheet(f"background: {T.BG}; border-bottom: 1px solid {T.RULE};")
        topbar.setFixedHeight(58)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(24, 0, 24, 0)
        tb.setSpacing(12)
        tb.addWidget(LogoBadge("POLY-X", "Microplastics analytics suite"))
        tb.addStretch(1)

        # Preferencias. El idioma vivia aqui como desplegable suelto; ahora
        # comparte dialogo con el tema y el movimiento, que son la misma
        # decision —como se ve y como se comporta el programa— y no tres
        # controles dispersos por la barra.
        self.btn_prefs = QPushButton()
        self.btn_prefs.setIcon(iconos.icono("ajustes", 18, T.INK2, 2.0))
        self.btn_prefs.setIconSize(QSize(18, 18))
        self.btn_prefs.setText(tr("Preferencias"))
        self.btn_prefs.setCursor(Qt.PointingHandCursor)
        self.btn_prefs.setToolTip(tr("Tema, idioma y movimiento"))
        self.btn_prefs.clicked.connect(self._abrir_preferencias)
        tb.addWidget(self.btn_prefs)

        version_lbl = QLabel(f"v{__version__}")
        version_lbl.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        tb.addWidget(version_lbl)

        # Aviso de version nueva. Nace oculto y solo aparece si la comprobacion
        # en segundo plano confirma que GitHub va por delante de lo instalado.
        self.btn_actualizar = QPushButton(tr("Actualizar"))
        self.btn_actualizar.setCursor(Qt.PointingHandCursor)
        self.btn_actualizar.setStyleSheet(
            f"QPushButton {{ background: {T.WARN}; color: white; border: none; "
            f"border-radius: 4px; padding: 4px 10px; font-size: 9pt; "
            f"font-weight: 600; }}"
            f"QPushButton:hover {{ background: {T.ERR}; }}")
        self.btn_actualizar.setVisible(False)
        self.btn_actualizar.clicked.connect(self._lanzar_actualizador)
        tb.addWidget(self.btn_actualizar)
        # Si la comprobacion ya respondio antes de esta reconstruccion, el
        # aviso se vuelve a poner en vez de perderse.
        if self._sha_version_nueva:
            self._avisar_version_nueva(self._sha_version_nueva)
        root.addWidget(topbar)

        # ── Scroll central ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        root.addWidget(scroll, 1)

        page = QWidget()
        scroll.setWidget(page)
        body = QVBoxLayout(page)
        body.setContentsMargins(48, 32, 48, 40)
        body.setSpacing(28)

        # ── Hero ──
        hero = QHBoxLayout()
        hero.setSpacing(36)

        left = QVBoxLayout()
        left.setSpacing(10)
        kicker = QLabel(tr("● PLATAFORMA DE ANÁLISIS"))
        kicker.setStyleSheet(
            f"color: {T.ACCENT_D_TX}; font-size: 9pt; font-weight: 700; "
            f"letter-spacing: 1.8px; border: none;"
        )
        left.addWidget(kicker)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        t1 = QLabel(tr("Poly-X"))
        t1.setStyleSheet(f"color: {T.INK}; font-size: 38pt; font-weight: 700; border: none;")
        t2 = QLabel(tr("analytics"))
        t2.setStyleSheet(
            f"color: {T.ACCENT_TX}; font-size: 38pt; font-weight: 300; border: none;"
        )
        title_row.addWidget(t1)
        title_row.addWidget(t2)
        title_row.addStretch(1)
        left.addLayout(title_row)

        subtitle = QLabel(tr("Plataforma de detección y clasificación de microplásticos"))
        subtitle.setStyleSheet(
            f"color: {T.INK2}; font-size: 13pt; font-weight: 500; border: none;"
        )
        left.addWidget(subtitle)

        descr = QLabel(
            tr("Detección automatizada de PET, PP y LDPE por fluorescencia Nile Red (254 nm) "
               "e inteligencia artificial. Entrenamiento, etiquetado, detección y reporte en un mismo flujo.")
        )
        descr.setWordWrap(True)
        descr.setStyleSheet(f"color: {T.INK3}; font-size: 10.5pt; border: none;")
        left.addWidget(descr)

        # Crédito de autoría (justo bajo el descriptor)
        author = QLabel(tr("Diseñado y desarrollado por <b>Cristofher Ferrada</b> · Dr (c) en Química, 2026"))
        author.setTextFormat(Qt.RichText)
        author.setStyleSheet(
            f"color: {T.ACCENT_D_TX}; font-size: 9.5pt; border: none; "
            f"background: {T.BG_SOFT}; padding: 6px 10px; border-radius: 4px;"
        )
        left.addWidget(author)
        left.addSpacing(4)

        # 4 chips
        chips = QHBoxLayout()
        chips.setSpacing(12)
        chips.addWidget(StatChip("3", tr("Polímeros"), "PET · PP · LDPE", T.ACCENT))
        chips.addWidget(StatChip("254", "nm", tr("fluorescencia Nile Red"), T.OK))
        chips.addWidget(StatChip("YOLO", "v8 / v11", "deep learning", T.VIO))
        chips.addWidget(StatChip("μm", tr("medición"), tr("calibración por píxel"), T.WARN))
        left.addLayout(chips)

        hero.addLayout(left, 3)
        hero.addWidget(MicroscopePanel(), 0, Qt.AlignTop)
        body.addLayout(hero)

        # ── Sección 'MÓDULOS' ──
        mod_kicker = QLabel(tr("MÓDULOS"))
        mod_kicker.setStyleSheet(
            f"color: {T.INK2}; font-size: 9pt; font-weight: 700; "
            f"letter-spacing: 1.8px; border: none;"
        )
        body.addWidget(mod_kicker)

        grid = QGridLayout()
        grid.setSpacing(16)
        grid.addWidget(ModuleCard(
            1, "detector", tr("Detector"),
            tr("Analiza imágenes con un modelo .pt entrenado. Genera salidas anotadas, "
               "CSV con centroides y diámetros, métricas globales e informe HTML de detección."),
            T.ACCENT, self.open_detector, "Alt+1"), 0, 0)
        grid.addWidget(ModuleCard(
            2, "entrenador", tr("Entrenador"),
            tr("Entrena modelos YOLO v8 / v11. Curvas en vivo, recomendaciones automáticas "
               "de calidad y comparación con runs anteriores."),
            T.OK, self.open_trainer, "Alt+2"), 0, 1)
        grid.addWidget(ModuleCard(
            3, "etiquetador", tr("Etiquetador"),
            tr("Anota imágenes en formato YOLO. Soporta pre-anotación con un modelo "
               "existente y atajos de teclado, ahorra ~80 % del tiempo manual."),
            T.WARN, self.open_labeler, "Alt+3"), 1, 0)
        grid.addWidget(ModuleCard(
            4, "visor", tr("Visor"),
            tr("Inspección de una imagen a la vez con calibración interactiva μm/píxel "
               "(línea o círculo) y medición precisa por partícula."),
            T.VIO, self.open_viewer, "Alt+4"), 1, 1)
        body.addLayout(grid)

        # ── Pie ──
        body.addStretch(1)
        body.addWidget(HLine())
        foot = QHBoxLayout()
        st = QLabel(tr("● Listo"))
        st.setStyleSheet(f"color: {T.OK_TX}; font-size: 9.5pt; font-weight: 600;")
        foot.addWidget(st)
        foot.addSpacing(20)
        copyr = QLabel("© Cristofher Ferrada · Poly-X v" + __version__)
        copyr.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        foot.addWidget(copyr)
        foot.addStretch(1)
        leeme_btn = QPushButton(tr("LÉAME"))
        leeme_btn.setIcon(iconos.icono("texto", 15, T.INK2, 2.0))
        leeme_btn.setCursor(Qt.PointingHandCursor)
        leeme_btn.clicked.connect(self.open_leeme)
        foot.addWidget(leeme_btn)
        manual_btn = QPushButton(tr("Manual de usuario"))
        manual_btn.setIcon(iconos.icono("manual", 15, T.INK2, 2.0))
        manual_btn.setCursor(Qt.PointingHandCursor)
        manual_btn.clicked.connect(self.open_manual)
        foot.addWidget(manual_btn)
        body.addLayout(foot)

        # Atajos de teclado para los cuatro modulos. Quien usa el programa a
        # diario abre siempre el mismo, y llegar con Alt+1 ahorra el viaje al
        # raton cada vez.
        for tecla, accion in (("Alt+1", self.open_detector),
                              ("Alt+2", self.open_trainer),
                              ("Alt+3", self.open_labeler),
                              ("Alt+4", self.open_viewer),
                              ("Ctrl+,", self._abrir_preferencias)):
            QShortcut(QKeySequence(tecla), self, activated=accion)

    # ── Preferencias ────────────────────────────────────────────
    def _abrir_preferencias(self):
        idioma_antes = idioma()
        dlg = DialogoPreferencias(self)
        dlg.tema_cambiado.connect(lambda _: self._construir_ui())
        dlg.exec()
        if idioma() != idioma_antes:
            # El idioma no se puede aplicar en caliente: cada widget fijo su
            # texto al construirse. Se avisa una vez, al cerrar, y no en cada
            # clic del dialogo.
            QMessageBox.information(
                self, tr("Idioma"),
                tr("El idioma se aplicará al reabrir Poly-X. Los módulos que "
                   "abras desde ahora ya salen en el idioma nuevo."))

    # ── Actualizaciones ─────────────────────────────────────────
    def _avisar_version_nueva(self, sha_corto: str):
        """Muestra el boton de actualizar cuando GitHub va por delante."""
        self._sha_version_nueva = sha_corto
        self.btn_actualizar.setText(tr("Actualizar") + f" · {sha_corto}")
        self.btn_actualizar.setToolTip(
            tr("Hay una versión nueva disponible en GitHub. "
               "Pulsa para descargarla e instalarla."))
        self.btn_actualizar.setVisible(True)

    def _lanzar_actualizador(self):
        if not puede_actualizar():
            QMessageBox.information(
                self, tr("Actualizar"),
                tr("No se encontró {} en la carpeta de instalación. Descarga "
                   "la versión nueva manualmente desde GitHub.").format(
                       ACTUALIZADOR.name))
            return
        if QMessageBox.question(
                self, tr("Actualizar"),
                tr("Se descargará la versión nueva y se cerrará Poly-X.\n\n"
                   "Tus modelos, resultados y datos no se tocan. ¿Continuar?"),
        ) != QMessageBox.Yes:
            return
        # El actualizador sobrescribe archivos del programa; hay que soltar
        # Poly-X antes para no chocar con los .py que estan en uso.
        if not lanzar_actualizador(ACTUALIZADOR):
            QMessageBox.warning(
                self, tr("Actualizar"),
                tr("No se pudo lanzar {}. Ejecútalo a mano desde la carpeta "
                   "de instalación.").format(ACTUALIZADOR.name))
            return
        self.close()

    # ── Lanzadores ──────────────────────────────────────────────
    def open_detector(self): self._launch_module("polyx.gui.detector")
    def open_trainer(self):  self._launch_module("polyx.gui.trainer")
    def open_labeler(self):  self._launch_module("polyx.gui.etiquetador")
    def open_viewer(self):   self._launch_module("polyx.gui.visor")

    def _launch_module(self, dotted: str, fallback: str = ""):
        """Lanza el módulo en un proceso aparte para no bloquear el launcher.

        Delega en _spawn_module, que no lanza: False significa que el módulo
        no existe o que su proceso no arrancó, y aquí toca avisar al usuario.
        """
        if not _spawn_module(dotted):
            self._toast(fallback or f"No se pudo abrir el módulo {dotted}")

    def open_manual(self):
        manual = Path(__file__).resolve().parents[2] / "Manual_PolyX.html"
        if manual.exists():
            import webbrowser
            webbrowser.open(manual.as_uri())
        else:
            self._toast("Manual_PolyX.html no encontrado")

    def open_leeme(self):
        leeme = Path(__file__).resolve().parents[2] / "LEEME.txt"
        if leeme.exists():
            abrir_en_el_sistema(str(leeme))
        else:
            self._toast("LEEME.txt no encontrado")

    def _toast(self, msg: str):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, tr("Poly-X"), msg)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    # Antes del primer show(): el boton de la barra de tareas se crea
    # con la identidad que haya en ese momento.
    marca.identificar(app, "launcher")
    # Rastro JSONL del arranque (app.jsonl); nunca impide abrir la ventana.
    _registrar_arranque()
    w = LauncherWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
