"""Identidad de cada ventana: su icono, su titulo y su sitio en la barra.

Por que hacia falta
-------------------
Poly-X arranca como ``pythonw.exe -m polyx.<modulo>``. Windows no agrupa las
ventanas de la barra de tareas por el ejecutable sino por su **AppUserModelID**,
y un proceso de Python que no declara el suyo hereda el de la instalacion de
Python. De ahi que la barra mostrara el logotipo de Python: la ventana era de
Poly-X, pero para el shell el programa era Python.

Se arregla con dos cosas, y hacen falta las dos:

1. **Declarar un AppUserModelID propio** antes de crear la primera ventana
   (``plataforma.fijar_identidad_app``). Sin esto, Windows sigue usando el
   icono de Python por mucho icono de ventana que se ponga.
2. **Poner el icono de la ventana** (``QApplication.setWindowIcon``). Sin esto,
   Windows tiene una identidad propia pero ningun dibujo que asociarle.

Cada modulo declara un identificador distinto, de modo que el Detector y el
Entrenador aparecen en la barra como dos programas separados, con su icono y su
nombre. Es lo correcto aqui: son procesos independientes, se abren y se cierran
por separado, y a menudo se usan dos a la vez.

Los colores del icono NO siguen al tema
---------------------------------------
Un icono es identidad, no vista. Si el del Detector cambiara de azul a otro
tono porque alguien prefiere trabajar de noche, dejaria de reconocerse en la
barra de tareas -- que es justo para lo que sirve. Van fijos, igual que la
paleta de los informes (``theme.DOC``).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPixmap

from . import iconos
from polyx.model import plataforma
from polyx.model.i18n import tr

#: Raiz del proyecto (…/Microplasticos), dos niveles por encima de core/.
_RAIZ = Path(__file__).resolve().parents[2]

#: Prefijo del AppUserModelID. El formato que pide Windows es
#: Empresa.Producto.SubProducto.Version; lo importante es que sea estable
#: entre arranques, porque de el cuelgan el anclaje a la barra y los saltos.
_APP_ID = "CristofherFerrada.PolyX"


# ════════════════════════════════════════════════════════════════════
#  Identidad de cada modulo
# ════════════════════════════════════════════════════════════════════
# (icono de core/iconos.py, color del distintivo, titulo de ventana)
#
# Los colores son los mismos con que el launcher pinta la tarjeta de cada
# modulo, pero escritos aqui como literales: el icono tiene que verse igual
# con cualquier tema.
MODULOS: dict[str, dict] = {
    "launcher": {
        "icono": None,               # usa assets/polyx.ico, la marca de siempre
        "color": "#0969da",
        "titulo": "Poly-X · Suite de microplásticos",
        "id": _APP_ID,
    },
    "detector": {
        "icono": "detector",
        "color": "#0969da",
        "titulo": "Poly-X · Detector",
        "id": f"{_APP_ID}.Detector",
    },
    "entrenador": {
        "icono": "entrenador",
        "color": "#1a7f5f",
        "titulo": "Poly-X · Entrenador",
        "id": f"{_APP_ID}.Entrenador",
    },
    "etiquetador": {
        "icono": "etiquetador",
        "color": "#b07400",
        "titulo": "Poly-X · Etiquetador",
        "id": f"{_APP_ID}.Etiquetador",
    },
    "visor": {
        "icono": "visor",
        "color": "#6639ba",
        "titulo": "Poly-X · Visor",
        "id": f"{_APP_ID}.Visor",
    },
}

#: Tamanos que Windows pide de un icono, del que usa la barra al de 'Detalles'.
#: Se generan todos: si falta uno, el shell reescala el mas cercano y el
#: resultado se ve sucio justo en la barra de tareas, que es donde mas se mira.
_TAMANOS = (16, 20, 24, 32, 40, 48, 64, 128, 256)


# ════════════════════════════════════════════════════════════════════
#  Dibujo del distintivo
# ════════════════════════════════════════════════════════════════════
def _distintivo(tam: int, color: str, glifo: str | None) -> QPixmap:
    """Cuadrado redondeado del color del modulo, con su glifo en blanco.

    Reproduce el ``LogoBadge`` de la barra superior -- el cuadrado azul con la
    'P' -- para que el icono de la barra de tareas y el de dentro del programa
    se lean como la misma marca.
    """
    px = QPixmap(tam, tam)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)

    # El margen y el radio van en proporcion al tamano: a 16 px un margen fijo
    # de 2 px se come una octava parte del icono.
    margen = max(0.5, tam * 0.045)
    radio = tam * 0.22
    cuerpo = QRectF(margen, margen, tam - 2 * margen, tam - 2 * margen)

    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    camino = QPainterPath()
    camino.addRoundedRect(cuerpo, radio, radio)
    p.drawPath(camino)

    if glifo is None:
        # Launcher: la 'P' de la marca.
        p.setPen(QColor("#ffffff"))
        f = QFont("Segoe UI", 1)
        f.setPixelSize(int(tam * 0.60))
        f.setWeight(QFont.Bold)
        p.setFont(f)
        p.drawText(cuerpo, Qt.AlignCenter, "P")
    else:
        # El glifo crece en proporcion segun encoge el icono. A 16 px un dibujo
        # al 58 % deja 9 px de lado: con el trazo y el antialiasing encima, el
        # detalle se empasta y los cinco modulos se distinguen solo por color.
        # Dandole mas superficie y un trazo algo mas grueso, la silueta
        # sobrevive al tamano de la barra de tareas, que es donde mas se mira.
        if tam <= 24:
            fraccion, trazo = 0.76, 2.9
        elif tam <= 48:
            fraccion, trazo = 0.64, 2.5
        else:
            fraccion, trazo = 0.58, 2.2
        lado = max(8, int(tam * fraccion))
        dibujo = iconos.pixmap(glifo, lado, "#ffffff", grosor=trazo)
        p.drawPixmap(int((tam - lado) / 2), int((tam - lado) / 2), dibujo)

    p.end()
    return px


def icono_modulo(modulo: str) -> QIcon:
    """Icono multi-resolucion del modulo, listo para ventana y barra."""
    cfg = MODULOS.get(modulo, MODULOS["launcher"])

    # El launcher usa el .ico de siempre, que es el del acceso directo del
    # Escritorio: asi el icono anclado y el de la ventana son el mismo dibujo.
    if cfg["icono"] is None:
        for ruta in (_RAIZ / "assets" / "polyx.ico",
                     _RAIZ / "polyx" / "assets" / "polyx.ico"):
            if ruta.exists():
                return QIcon(str(ruta))

    ic = QIcon()
    for tam in _TAMANOS:
        ic.addPixmap(_distintivo(tam, cfg["color"], cfg["icono"]))
    return ic


# ════════════════════════════════════════════════════════════════════
#  Aplicacion
# ════════════════════════════════════════════════════════════════════
def identificar(app, modulo: str) -> None:
    """Da al proceso su identidad ante el sistema. Antes de la primera ventana.

    El orden importa: el AppUserModelID tiene que estar puesto antes de que
    Windows cree el boton de la barra de tareas, es decir antes del primer
    ``show()``. Ponerlo despues no tiene efecto hasta reabrir.
    """
    cfg = MODULOS.get(modulo, MODULOS["launcher"])
    plataforma.fijar_identidad_app(cfg["id"])
    app.setApplicationName("Poly-X")
    app.setApplicationDisplayName(tr(cfg["titulo"]))
    app.setOrganizationName("Cristofher Ferrada")
    app.setWindowIcon(icono_modulo(modulo))


def titular(ventana, modulo: str) -> None:
    """Pone titulo e icono a una ventana concreta.

    El icono de la aplicacion ya lo hereda cualquier ventana, pero se repite
    aqui a proposito: si el modulo se abre desde otro proceso -- como hace el
    launcher -- la ventana puede construirse antes de que ``identificar`` haya
    corrido, y entonces se queda sin icono.
    """
    cfg = MODULOS.get(modulo, MODULOS["launcher"])
    ventana.setWindowTitle(tr(cfg["titulo"]))
    ventana.setWindowIcon(icono_modulo(modulo))
