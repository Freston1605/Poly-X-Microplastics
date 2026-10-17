"""Cuándo una partícula detectada puede asignarse a un polímero, y cuándo no.

Detectar una partícula y decir de qué polímero es son dos preguntas distintas,
y la segunda es mucho más frágil que la primera. El Nile Red es solvatocrómico:
su emisión responde a la **polaridad** del entorno, no a la identidad química
del polímero. Por eso el PET —poliéster, polar— se separa limpio, mientras que
el PP y el LDPE, las dos poliolefinas apolares, comparten tono y solo difieren
en brillo; y el brillo depende de la exposición, del foco, del espesor de la
partícula y de cuánto tiñó el colorante. Nada de eso es el polímero.

La consecuencia práctica es que el modelo acierta la caja mucho más a menudo
que la clase. Este módulo permite decirlo en el informe en vez de esconderlo:
por debajo de un umbral de confianza la partícula **sigue contando** como
partícula detectada, pero se reporta como «no asignable» en lugar de inventar
un polímero.

Qué NO hace, y por qué
──────────────────────
Lo ideal sería abstenerse por *margen*: si la primera clase saca 0.42 y la
segunda 0.39, la elección es una moneda al aire aunque 0.42 supere el umbral.
No se puede con lo que hay: la salida de YOLO tras el NMS trae una sola
confianza y una sola clase por caja, y las puntuaciones del resto de clases se
pierden ahí dentro. Recuperarlas obliga a interceptar el tensor crudo de la
cabeza de detección, que es frágil y se rompe con cada versión de ultralytics.
La banda de confianza es lo que sí se puede sostener hoy.

Tampoco es una clase nueva del modelo. Una clase «desconocido» entrenada se
define por la duda de quien anota, no por lo que se ve en la imagen: la misma
partícula acabaría etiquetada LDPE un día y desconocido otro, y ese ruido caería
justo sobre PP y LDPE, que ya son las clases con peor recall.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List

# Etiqueta con la que se reporta una partícula cuyo polímero no se sostiene.
# Se deja en español y en minúsculas porque viaja a las tablas del informe tal
# cual; la traducción al inglés la hace i18n sobre esta cadena.
NO_ASIGNABLE = "no asignable"


def marcar_asignables(detecciones: Iterable, umbral: float) -> int:
    """Marca cada detección como asignable o no, según la confianza.

    Args:
        detecciones: objetos Detection ya producidos por el modelo.
        umbral: confianza mínima para sostener el polímero. Con 0 (o menos) la
            abstención queda desactivada y todo se reporta con su clase, que es
            el comportamiento histórico.

    Returns:
        Cuántas detecciones quedaron marcadas como no asignables.
    """
    if umbral <= 0:
        for d in detecciones:
            d.asignable = True
        return 0
    n = 0
    for d in detecciones:
        d.asignable = bool(d.conf >= umbral)
        if not d.asignable:
            n += 1
    return n


def clase_reportada(deteccion) -> str:
    """Nombre con el que la partícula debe aparecer en tablas y figuras.

    Se usa ``getattr`` con respaldo porque las detecciones leídas de un .txt de
    ground truth, y las de runs guardados antes de que existiera la abstención,
    no traen el campo: ahí la respuesta correcta es la clase de siempre.
    """
    if getattr(deteccion, "asignable", True):
        return deteccion.class_name
    return NO_ASIGNABLE


@dataclass
class ResumenAsignacion:
    """Cuánto del lote se pudo asignar a un polímero."""

    total: int = 0
    asignadas: int = 0
    no_asignables: int = 0
    umbral: float = 0.0

    @property
    def activa(self) -> bool:
        """La abstención estaba encendida en este análisis."""
        return self.umbral > 0

    @property
    def porcentaje_no_asignable(self) -> float:
        return 100.0 * self.no_asignables / self.total if self.total else 0.0

    def __str__(self) -> str:
        if not self.activa:
            return "abstención desactivada"
        return (f"{self.no_asignables} de {self.total} partículas sin polímero "
                f"asignable ({self.porcentaje_no_asignable:.1f} %), "
                f"umbral {self.umbral:.2f}")


def resumir(detecciones: Iterable, umbral: float) -> ResumenAsignacion:
    """Cuenta asignadas y no asignables sobre un lote ya marcado."""
    dets: List = list(detecciones)
    no_asig = sum(1 for d in dets if not getattr(d, "asignable", True))
    return ResumenAsignacion(total=len(dets), asignadas=len(dets) - no_asig,
                             no_asignables=no_asig, umbral=umbral)


def conteo_reportado(detecciones: Iterable) -> Counter:
    """Partículas por clase reportada, con «no asignable» como una categoría más.

    El total sigue siendo el número de partículas detectadas: abstenerse del
    polímero no descarta la partícula, que es justo lo que se quiere cuando el
    resultado se informa por cantidad de partículas.
    """
    c: Counter = Counter()
    for d in detecciones:
        c[clase_reportada(d)] += 1
    return c
