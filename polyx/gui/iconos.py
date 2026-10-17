"""Iconos vectoriales dibujados a mano, sin archivos ni dependencias.

Por que no emoji
----------------
La interfaz usaba 🔬 🎯 🏷 📐 como iconos de modulo. Un emoji lo dibuja la
fuente del sistema, no el programa: cambia de forma entre Windows 10 y 11,
entre Windows y macOS, y llega en color fijo, asi que no puede seguir al tema
ni al color de acento de su tarjeta. Ademas mezcla estilos —unos planos, otros
con degradado— y eso se nota junto en una rejilla de cuatro.

Estos se dibujan con ``QPainterPath`` sobre una rejilla de 24x24, con el trazo
proporcional al tamano pedido. Se ven iguales en los dos sistemas, toman el
color que se les pase y quedan nitidos a cualquier resolucion.

Uso:

    from polyx.gui import iconos
    lbl.setPixmap(iconos.pixmap("detector", 26, T.ACCENT))
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

#: Todos los caminos se definen en esta rejilla y se escalan al vuelo.
REJILLA = 24.0


# ════════════════════════════════════════════════════════════════════
#  Caminos
# ════════════════════════════════════════════════════════════════════
def _detector(p: QPainter) -> None:
    """Objetivo con retícula: mirar de cerca y marcar lo que hay."""
    p.drawEllipse(QPointF(11, 11), 7.0, 7.0)
    p.drawEllipse(QPointF(11, 11), 2.4, 2.4)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        p.drawLine(QPointF(11 + dx * 8.6, 11 + dy * 8.6),
                   QPointF(11 + dx * 10.4, 11 + dy * 10.4))
    p.drawLine(QPointF(16.2, 16.2), QPointF(20.6, 20.6))


def _entrenador(p: QPainter) -> None:
    """Curva que sube hacia una diana: el entrenamiento converge."""
    camino = QPainterPath()
    camino.moveTo(3.0, 19.5)
    camino.cubicTo(7.5, 19.0, 9.5, 13.0, 12.5, 9.0)
    p.drawPath(camino)
    p.drawEllipse(QPointF(17.0, 7.0), 4.6, 4.6)
    p.drawEllipse(QPointF(17.0, 7.0), 1.5, 1.5)
    p.drawLine(QPointF(3.0, 21.4), QPointF(21.0, 21.4))


def _etiquetador(p: QPainter) -> None:
    """Etiqueta con su ojal, y una caja de anotacion detras."""
    camino = QPainterPath()
    camino.moveTo(12.6, 2.8)
    camino.lineTo(21.2, 11.4)
    camino.lineTo(12.4, 20.2)
    camino.lineTo(3.8, 11.6)
    camino.lineTo(3.8, 3.0)
    camino.closeSubpath()
    p.drawPath(camino)
    p.drawEllipse(QPointF(8.0, 7.2), 1.9, 1.9)


def _visor(p: QPainter) -> None:
    """Regla en diagonal con sus marcas: medir sobre la imagen."""
    p.save()
    p.translate(12, 12)
    p.rotate(-38)
    p.drawRoundedRect(QRectF(-10.5, -3.8, 21.0, 7.6), 1.4, 1.4)
    for x in (-6.0, -2.0, 2.0, 6.0):
        largo = 3.4 if x in (-6.0, 2.0) else 2.2
        p.drawLine(QPointF(x, -3.8), QPointF(x, -3.8 + largo))
    p.restore()


def _ajustes(p: QPainter) -> None:
    """Deslizadores.

    Un engranaje de seis dientes dibujado con radios sueltos se lee como un
    sol, no como una rueda: para que parezca engranaje harian falta dientes
    con cuerpo, y a 18 px se emborronan. Los deslizadores no tienen esa
    ambiguedad y ademas describen mejor lo que hay detras del boton: no un
    motor que se configura, sino unas cuantas preferencias.
    """
    for y, x_mando in ((6.5, 15.5), (12.0, 9.0), (17.5, 14.0)):
        p.drawLine(QPointF(3.4, y), QPointF(20.6, y))
        p.drawEllipse(QPointF(x_mando, y), 2.5, 2.5)


def _manual(p: QPainter) -> None:
    """Libro abierto."""
    p.drawLine(QPointF(12, 6.4), QPointF(12, 19.2))
    camino = QPainterPath()
    camino.moveTo(12, 6.4)
    camino.cubicTo(9.0, 4.2, 6.0, 4.0, 3.2, 4.8)
    camino.lineTo(3.2, 17.6)
    camino.cubicTo(6.0, 16.8, 9.0, 17.0, 12, 19.2)
    p.drawPath(camino)
    camino2 = QPainterPath()
    camino2.moveTo(12, 6.4)
    camino2.cubicTo(15.0, 4.2, 18.0, 4.0, 20.8, 4.8)
    camino2.lineTo(20.8, 17.6)
    camino2.cubicTo(18.0, 16.8, 15.0, 17.0, 12, 19.2)
    p.drawPath(camino2)


def _texto(p: QPainter) -> None:
    """Hoja con lineas: el LEEME."""
    p.drawRoundedRect(QRectF(5.0, 3.2, 14.0, 17.6), 1.8, 1.8)
    for y in (8.0, 11.4, 14.8):
        p.drawLine(QPointF(8.2, y), QPointF(15.8, y))


# ── Barras laterales del Detector y del Entrenador ──────────────────
def _modelo(p: QPainter) -> None:
    """Capas apiladas: los pesos de un modelo."""
    tapa = QPainterPath()
    tapa.moveTo(12, 3.0)
    tapa.lineTo(20.8, 7.4)
    tapa.lineTo(12, 11.8)
    tapa.lineTo(3.2, 7.4)
    tapa.closeSubpath()
    p.drawPath(tapa)
    for dy in (4.6, 9.2):
        medio = QPainterPath()
        medio.moveTo(3.2, 7.4 + dy)
        medio.lineTo(12, 11.8 + dy)
        medio.lineTo(20.8, 7.4 + dy)
        p.drawPath(medio)


def _imagenes(p: QPainter) -> None:
    """Fotografia: marco, sol y horizonte."""
    p.drawRoundedRect(QRectF(3.2, 4.8, 17.6, 14.4), 2.0, 2.0)
    p.drawEllipse(QPointF(8.4, 9.4), 1.7, 1.7)
    monte = QPainterPath()
    monte.moveTo(4.6, 17.4)
    monte.lineTo(9.6, 12.2)
    monte.lineTo(13.0, 15.4)
    monte.lineTo(16.0, 12.6)
    monte.lineTo(19.6, 17.4)
    p.drawPath(monte)


def _editar(p: QPainter) -> None:
    """Lapiz: anotar a mano."""
    cuerpo = QPainterPath()
    cuerpo.moveTo(4.0, 20.0)
    cuerpo.lineTo(5.5, 15.3)
    cuerpo.lineTo(16.2, 4.6)
    cuerpo.lineTo(19.4, 7.8)
    cuerpo.lineTo(8.7, 18.5)
    cuerpo.closeSubpath()
    p.drawPath(cuerpo)
    p.drawLine(QPointF(14.4, 6.4), QPointF(17.6, 9.6))


def _ejecutar(p: QPainter) -> None:
    """Triangulo de reproduccion."""
    play = QPainterPath()
    play.moveTo(8.0, 5.4)
    play.lineTo(19.0, 12.0)
    play.lineTo(8.0, 18.6)
    play.closeSubpath()
    p.drawPath(play)


def _resultados(p: QPainter) -> None:
    """Barras: las cifras del analisis."""
    p.drawLine(QPointF(3.4, 20.2), QPointF(20.6, 20.2))
    for x, alto in ((5.6, 7.6), (10.3, 12.6), (15.0, 5.2)):
        p.drawRoundedRect(QRectF(x, 20.2 - alto, 3.4, alto), 1.0, 1.0)


def _errores(p: QPainter) -> None:
    """Triangulo de aviso."""
    tri = QPainterPath()
    tri.moveTo(12.0, 3.6)
    tri.lineTo(21.2, 19.8)
    tri.lineTo(2.8, 19.8)
    tri.closeSubpath()
    p.drawPath(tri)
    p.drawLine(QPointF(12.0, 10.0), QPointF(12.0, 14.4))
    p.drawEllipse(QPointF(12.0, 17.0), 0.55, 0.55)


def _comparar(p: QPainter) -> None:
    """Panel partido: lo mismo mirado por dos lados."""
    p.drawRoundedRect(QRectF(3.2, 5.0, 17.6, 14.0), 2.0, 2.0)
    p.drawLine(QPointF(12.0, 5.0), QPointF(12.0, 19.0))
    p.drawEllipse(QPointF(7.6, 12.0), 2.0, 2.0)
    p.drawRoundedRect(QRectF(14.4, 10.0, 4.0, 4.0), 0.8, 0.8)


def _reporte(p: QPainter) -> None:
    """Hoja con una curva: el informe."""
    p.drawRoundedRect(QRectF(4.8, 3.2, 14.4, 17.6), 1.8, 1.8)
    p.drawLine(QPointF(8.0, 7.6), QPointF(16.0, 7.6))
    curva = QPainterPath()
    curva.moveTo(8.0, 16.4)
    curva.lineTo(10.8, 13.2)
    curva.lineTo(13.2, 15.0)
    curva.lineTo(16.0, 10.8)
    p.drawPath(curva)


def _dataset(p: QPainter) -> None:
    """Carpeta: el conjunto de datos."""
    carpeta = QPainterPath()
    carpeta.moveTo(3.2, 19.0)
    carpeta.lineTo(3.2, 5.8)
    carpeta.lineTo(9.4, 5.8)
    carpeta.lineTo(11.4, 8.4)
    carpeta.lineTo(20.8, 8.4)
    carpeta.lineTo(20.8, 19.0)
    carpeta.closeSubpath()
    p.drawPath(carpeta)


def _augmentacion(p: QPainter) -> None:
    """Dos marcos desplazados: la misma imagen en variantes."""
    p.drawRoundedRect(QRectF(8.0, 3.4, 12.6, 12.6), 2.0, 2.0)
    p.drawRoundedRect(QRectF(3.4, 8.0, 12.6, 12.6), 2.0, 2.0)


def _evaluar(p: QPainter) -> None:
    """Portapapeles con visto: validar contra una referencia."""
    p.drawRoundedRect(QRectF(4.8, 4.6, 14.4, 16.0), 2.0, 2.0)
    p.drawRoundedRect(QRectF(9.0, 2.6, 6.0, 4.0), 1.0, 1.0)
    visto = QPainterPath()
    visto.moveTo(8.6, 13.4)
    visto.lineTo(11.0, 15.8)
    visto.lineTo(15.6, 10.4)
    p.drawPath(visto)


def _exportar(p: QPainter) -> None:
    """Salir de la caja: convertir a otro formato."""
    caja = QPainterPath()
    caja.moveTo(4.4, 14.2)
    caja.lineTo(4.4, 19.8)
    caja.lineTo(19.6, 19.8)
    caja.lineTo(19.6, 14.2)
    p.drawPath(caja)
    p.drawLine(QPointF(12.0, 15.4), QPointF(12.0, 4.4))
    flecha = QPainterPath()
    flecha.moveTo(7.9, 8.6)
    flecha.lineTo(12.0, 4.4)
    flecha.lineTo(16.1, 8.6)
    p.drawPath(flecha)


# ── Cabeceras de tarjeta ────────────────────────────────────────────
def _ver(p: QPainter) -> None:
    """Ojo: inspeccionar."""
    ojo = QPainterPath()
    ojo.moveTo(2.6, 12.0)
    ojo.cubicTo(6.5, 5.8, 17.5, 5.8, 21.4, 12.0)
    ojo.cubicTo(17.5, 18.2, 6.5, 18.2, 2.6, 12.0)
    p.drawPath(ojo)
    p.drawEllipse(QPointF(12, 12), 3.1, 3.1)


def _rayo(p: QPainter) -> None:
    """Rayo: lo rapido, lo automatico."""
    r = QPainterPath()
    r.moveTo(13.6, 2.6)
    r.lineTo(5.4, 13.4)
    r.lineTo(11.2, 13.4)
    r.lineTo(10.4, 21.4)
    r.lineTo(18.6, 10.6)
    r.lineTo(12.8, 10.6)
    r.closeSubpath()
    p.drawPath(r)


def _diana(p: QPainter) -> None:
    """Diana: el objetivo."""
    p.drawEllipse(QPointF(12, 12), 8.6, 8.6)
    p.drawEllipse(QPointF(12, 12), 4.8, 4.8)
    p.drawEllipse(QPointF(12, 12), 1.2, 1.2)


def _lista(p: QPainter) -> None:
    """Portapapeles con renglones."""
    p.drawRoundedRect(QRectF(4.8, 4.6, 14.4, 16.0), 2.0, 2.0)
    p.drawRoundedRect(QRectF(9.0, 2.6, 6.0, 4.0), 1.0, 1.0)
    for y in (11.0, 14.2, 17.4):
        p.drawLine(QPointF(8.2, y), QPointF(15.8, y))


def _guardar(p: QPainter) -> None:
    """Disquete: guardar en disco."""
    cuerpo = QPainterPath()
    cuerpo.moveTo(4.4, 4.4)
    cuerpo.lineTo(16.6, 4.4)
    cuerpo.lineTo(19.6, 7.4)
    cuerpo.lineTo(19.6, 19.6)
    cuerpo.lineTo(4.4, 19.6)
    cuerpo.closeSubpath()
    p.drawPath(cuerpo)
    p.drawRect(QRectF(8.2, 4.4, 7.6, 5.0))
    p.drawRect(QRectF(7.4, 13.4, 9.2, 6.2))


def _registro(p: QPainter) -> None:
    """Hoja con muchos renglones: el registro de la corrida."""
    p.drawRoundedRect(QRectF(4.6, 3.4, 14.8, 17.2), 1.8, 1.8)
    for y in (7.4, 10.2, 13.0, 15.8):
        p.drawLine(QPointF(7.8, y), QPointF(16.2, y))


def _paquete(p: QPainter) -> None:
    """Caja en perspectiva: un artefacto empaquetado."""
    caja = QPainterPath()
    caja.moveTo(12, 2.8)
    caja.lineTo(20.6, 7.2)
    caja.lineTo(20.6, 16.8)
    caja.lineTo(12, 21.2)
    caja.lineTo(3.4, 16.8)
    caja.lineTo(3.4, 7.2)
    caja.closeSubpath()
    p.drawPath(caja)
    p.drawLine(QPointF(3.4, 7.2), QPointF(12, 11.8))
    p.drawLine(QPointF(20.6, 7.2), QPointF(12, 11.8))
    p.drawLine(QPointF(12, 11.8), QPointF(12, 21.2))


def _espera(p: QPainter) -> None:
    """Reloj: lo que va a tardar."""
    p.drawEllipse(QPointF(12, 12), 8.6, 8.6)
    p.drawLine(QPointF(12, 6.8), QPointF(12, 12))
    p.drawLine(QPointF(12, 12), QPointF(15.8, 14.4))


def _buscar(p: QPainter) -> None:
    """Lupa."""
    p.drawEllipse(QPointF(10.6, 10.6), 6.4, 6.4)
    p.drawLine(QPointF(15.4, 15.4), QPointF(20.4, 20.4))


def _piezas(p: QPainter) -> None:
    """Rejilla: trocear en partes."""
    p.drawRoundedRect(QRectF(3.4, 3.4, 7.6, 7.6), 1.4, 1.4)
    p.drawRoundedRect(QRectF(13.0, 3.4, 7.6, 7.6), 1.4, 1.4)
    p.drawRoundedRect(QRectF(3.4, 13.0, 7.6, 7.6), 1.4, 1.4)
    p.drawRoundedRect(QRectF(13.0, 13.0, 7.6, 7.6), 1.4, 1.4)


def _ensayo(p: QPainter) -> None:
    """Matraz: la prueba."""
    matraz = QPainterPath()
    matraz.moveTo(9.4, 3.0)
    matraz.lineTo(9.4, 9.4)
    matraz.lineTo(4.2, 18.6)
    matraz.cubicTo(3.2, 20.4, 4.4, 21.2, 6.0, 21.2)
    matraz.lineTo(18.0, 21.2)
    matraz.cubicTo(19.6, 21.2, 20.8, 20.4, 19.8, 18.6)
    matraz.lineTo(14.6, 9.4)
    matraz.lineTo(14.6, 3.0)
    p.drawPath(matraz)
    p.drawLine(QPointF(8.2, 3.0), QPointF(15.8, 3.0))


def _recargar(p: QPainter) -> None:
    """Flecha circular: volver a leer."""
    arco = QPainterPath()
    arco.arcMoveTo(QRectF(4.0, 4.0, 16.0, 16.0), 65)
    arco.arcTo(QRectF(4.0, 4.0, 16.0, 16.0), 65, 285)
    p.drawPath(arco)
    punta = QPainterPath()
    punta.moveTo(11.6, 2.6)
    punta.lineTo(16.4, 6.0)
    punta.lineTo(11.4, 9.0)
    p.drawPath(punta)


def _idea(p: QPainter) -> None:
    """Bombilla: la recomendacion."""
    bulbo = QPainterPath()
    bulbo.moveTo(8.4, 14.6)
    bulbo.cubicTo(5.2, 11.0, 7.0, 4.2, 12.0, 4.2)
    bulbo.cubicTo(17.0, 4.2, 18.8, 11.0, 15.6, 14.6)
    p.drawPath(bulbo)
    p.drawLine(QPointF(9.2, 17.4), QPointF(14.8, 17.4))
    p.drawLine(QPointF(10.2, 20.2), QPointF(13.8, 20.2))


def _curva(p: QPainter) -> None:
    """Curva con puntos: la evolucion."""
    p.drawLine(QPointF(3.4, 20.2), QPointF(20.6, 20.2))
    linea = QPainterPath()
    linea.moveTo(4.8, 16.6)
    linea.lineTo(9.4, 11.4)
    linea.lineTo(13.6, 14.0)
    linea.lineTo(19.4, 5.6)
    p.drawPath(linea)
    for x, y in ((9.4, 11.4), (13.6, 14.0), (19.4, 5.6)):
        p.drawEllipse(QPointF(x, y), 1.15, 1.15)


def _equipo(p: QPainter) -> None:
    """Monitor: el hardware."""
    p.drawRoundedRect(QRectF(3.0, 4.6, 18.0, 12.4), 1.8, 1.8)
    p.drawLine(QPointF(9.0, 20.6), QPointF(15.0, 20.6))
    p.drawLine(QPointF(12.0, 17.0), QPointF(12.0, 20.6))


# ── Acciones ────────────────────────────────────────────────────────
def _detener(p: QPainter) -> None:
    """Cuadrado: parar."""
    p.drawRoundedRect(QRectF(6.4, 6.4, 11.2, 11.2), 1.6, 1.6)


def _circulo(p: QPainter) -> None:
    """Circunferencia con tres marcas: calibrar por tres puntos del borde."""
    p.drawEllipse(QPointF(12, 12), 8.2, 8.2)
    for x, y in ((12.0, 3.8), (19.1, 16.1), (4.9, 16.1)):
        p.drawEllipse(QPointF(x, y), 1.5, 1.5)


def _linea(p: QPainter) -> None:
    """Segmento con topes: calibrar por una longitud conocida."""
    p.drawLine(QPointF(4.4, 12.0), QPointF(19.6, 12.0))
    p.drawLine(QPointF(4.4, 8.2), QPointF(4.4, 15.8))
    p.drawLine(QPointF(19.6, 8.2), QPointF(19.6, 15.8))


def _papelera(p: QPainter) -> None:
    """Papelera: borrar."""
    p.drawLine(QPointF(3.6, 6.4), QPointF(20.4, 6.4))
    p.drawRoundedRect(QRectF(9.0, 3.0, 6.0, 3.4), 1.0, 1.0)
    cubo = QPainterPath()
    cubo.moveTo(5.8, 6.4)
    cubo.lineTo(7.0, 20.6)
    cubo.lineTo(17.0, 20.6)
    cubo.lineTo(18.2, 6.4)
    p.drawPath(cubo)
    p.drawLine(QPointF(10.2, 10.2), QPointF(10.6, 17.0))
    p.drawLine(QPointF(13.8, 10.2), QPointF(13.4, 17.0))


def _deshacer(p: QPainter) -> None:
    """Flecha que vuelve a la izquierda."""
    arco = QPainterPath()
    arco.moveTo(4.4, 10.4)
    arco.cubicTo(9.0, 4.6, 18.4, 5.6, 19.6, 12.4)
    arco.cubicTo(20.4, 16.8, 17.2, 19.8, 13.0, 20.0)
    p.drawPath(arco)
    punta = QPainterPath()
    punta.moveTo(9.4, 10.0)
    punta.lineTo(4.0, 10.6)
    punta.lineTo(5.6, 5.4)
    p.drawPath(punta)


def _rehacer(p: QPainter) -> None:
    """La misma flecha, reflejada."""
    p.save()
    p.translate(24, 0)
    p.scale(-1, 1)
    _deshacer(p)
    p.restore()


DIBUJOS = {
    "detector": _detector,
    "entrenador": _entrenador,
    "etiquetador": _etiquetador,
    "visor": _visor,
    "ajustes": _ajustes,
    "manual": _manual,
    "texto": _texto,
    # Barras laterales
    "modelo": _modelo,
    "imagenes": _imagenes,
    "editar": _editar,
    "ejecutar": _ejecutar,
    "resultados": _resultados,
    "errores": _errores,
    "comparar": _comparar,
    "reporte": _reporte,
    "dataset": _dataset,
    "augmentacion": _augmentacion,
    "evaluar": _evaluar,
    "exportar": _exportar,
    # Cabeceras de tarjeta
    "ver": _ver,
    "rayo": _rayo,
    "diana": _diana,
    "lista": _lista,
    "guardar": _guardar,
    "registro": _registro,
    "paquete": _paquete,
    "espera": _espera,
    "buscar": _buscar,
    "piezas": _piezas,
    "ensayo": _ensayo,
    "recargar": _recargar,
    "idea": _idea,
    "curva": _curva,
    "equipo": _equipo,
    # Acciones
    "detener": _detener,
    "circulo": _circulo,
    "linea": _linea,
    "papelera": _papelera,
    "deshacer": _deshacer,
    "rehacer": _rehacer,
}


# ════════════════════════════════════════════════════════════════════
#  Render
# ════════════════════════════════════════════════════════════════════
def pixmap(nombre: str, tam: int, color: str, grosor: float = 1.9) -> QPixmap:
    """Dibuja el icono ``nombre`` a ``tam`` px en el color pedido.

    ``grosor`` va en unidades de la rejilla de 24, no en pixeles, para que el
    trazo mantenga su peso relativo al escalar.
    """
    dibujo = DIBUJOS.get(nombre)
    px = QPixmap(tam, tam)
    px.fill(Qt.transparent)
    if dibujo is None:
        return px

    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(tam / REJILLA, tam / REJILLA)
    lapiz = QPen(QColor(color), grosor)
    lapiz.setCapStyle(Qt.RoundCap)
    lapiz.setJoinStyle(Qt.RoundJoin)
    p.setPen(lapiz)
    p.setBrush(Qt.NoBrush)
    dibujo(p)
    p.end()
    return px


def icono(nombre: str, tam: int, color: str, grosor: float = 1.9) -> QIcon:
    return QIcon(pixmap(nombre, tam, color, grosor))


def icono_conmutable(nombre: str, tam: int, color_off: str, color_on: str,
                     grosor: float = 1.85) -> QIcon:
    """Icono que cambia de color al marcarse el boton.

    Qt guarda un mapa por estado dentro del propio ``QIcon``: para un boton
    ``checkable`` usa ``QIcon.On`` cuando esta marcado y ``QIcon.Off`` cuando
    no. Asi el icono de la pagina activa se tine de acento junto con su texto,
    sin que haya que reconstruir el boton ni escuchar el ``toggled``.
    """
    ic = QIcon()
    ic.addPixmap(pixmap(nombre, tam, color_off, grosor), QIcon.Normal, QIcon.Off)
    ic.addPixmap(pixmap(nombre, tam, color_on, grosor), QIcon.Normal, QIcon.On)
    # Sin el par Active, Qt reutiliza el de Off al pasar el raton por encima de
    # un boton ya marcado y el icono parpadea al color apagado.
    ic.addPixmap(pixmap(nombre, tam, color_on, grosor), QIcon.Active, QIcon.On)
    ic.addPixmap(pixmap(nombre, tam, color_off, grosor), QIcon.Active, QIcon.Off)
    return ic
