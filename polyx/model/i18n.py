"""Traduccion de la interfaz. Espanol es el idioma fuente; ingles, el destino.

Por que este esquema y no el de Qt (.ts + lupdate + .qm):

    La interfaz tiene ~2350 cadenas repartidas en 55 archivos. Con el flujo de
    Qt hay que envolver cada una en ``tr()``, ejecutar ``lupdate``, editar el
    ``.ts`` en Linguist y compilar a ``.qm``: cuatro pasos y dos herramientas
    externas cada vez que se toca un texto. Aqui la **cadena en espanol es la
    clave**, asi que traducir es agregar una entrada al diccionario y nada mas.
    Lo que no este traducido cae al espanol en vez de aparecer vacio o con la
    clave cruda, que es el modo de fallo habitual de los sistemas por clave.

Uso:

    from ..core.i18n import tr
    boton = QPushButton(tr("Abrir carpeta…"))

El idioma se resuelve una vez al arrancar: ``POLYX_IDIOMA`` si esta puesta, si
no lo guardado por el usuario, si no el idioma del sistema, y a falta de todo,
espanol.
"""
from __future__ import annotations

import json
import locale
import os
from pathlib import Path
from typing import Callable

from .traducciones_en import EN as _EN, NO_TRADUCIR

IDIOMAS = {"es": "Español", "en": "English"}
_ARCHIVO = Path.home() / ".polyx_idioma.json"

_idioma = "es"
_suscriptores: list[Callable[[str], None]] = []


# ── Diccionario es -> en ────────────────────────────────────────────
# Solo texto que ve el usuario. No se traducen nombres de clase de polimero
# (PET/PP/LDPE), unidades ni identificadores.
TRADUCCIONES: dict[str, str] = {
    # ── Launcher ──
    "Poly-X · Suite de microplásticos": "Poly-X · Microplastics suite",
    "● PLATAFORMA DE ANÁLISIS": "● ANALYSIS PLATFORM",
    "Plataforma de detección y clasificación de microplásticos":
        "Detection and classification platform for microplastics",
    "Abrir  →": "Open  →",
    "MÓDULO": "MODULE",
    "Detector": "Detector",
    "Entrenador": "Trainer",
    "Etiquetador": "Labeler",
    "Visor": "Viewer",
    "Idioma": "Language",
    "MÓDULOS": "MODULES",
    "● Listo": "● Ready",
    "Polímeros": "Polymers",
    "fluorescencia Nile Red": "Nile Red fluorescence",
    "medición": "measurement",
    "calibración por píxel": "per-pixel calibration",
    "Detección automatizada de PET, PP y LDPE por fluorescencia Nile Red (254 nm) "
    "e inteligencia artificial. Entrenamiento, etiquetado, detección y reporte en un mismo flujo.":
        "Automated detection of PET, PP and LDPE by Nile Red fluorescence (254 nm) "
        "and machine learning. Training, labeling, detection and reporting in a single workflow.",
    "Diseñado y desarrollado por <b>Cristofher Ferrada</b> · Dr (c) en Química, 2026":
        "Designed and developed by <b>Cristofher Ferrada</b> · PhD (c) in Chemistry, 2026",
    "Analiza imágenes con un modelo .pt entrenado. Genera salidas anotadas, "
    "CSV con centroides y diámetros, métricas globales y reporte HTML paper-quality.":
        "Analyzes images with a trained .pt model. Produces annotated output, a CSV "
        "of centroids and diameters, global metrics and a paper-quality HTML report.",
    "Entrena modelos YOLO v8 / v11. Curvas en vivo, recomendaciones automáticas "
    "de calidad y comparación con runs anteriores.":
        "Trains YOLO v8 / v11 models. Live curves, automatic quality recommendations "
        "and comparison against earlier runs.",
    "Anota imágenes en formato YOLO. Soporta pre-anotación con un modelo "
    "existente y atajos de teclado, ahorra ~80 % del tiempo manual.":
        "Annotates images in YOLO format. Supports pre-annotation with an existing "
        "model plus keyboard shortcuts, saving ~80% of the manual time.",
    "Inspección de una imagen a la vez con calibración interactiva μm/píxel "
    "(línea o círculo) y medición precisa por partícula.":
        "Inspects one image at a time with interactive µm/pixel calibration "
        "(line or circle) and precise per-particle measurement.",

    # ── Vocabulario compartido ──
    "Abrir carpeta…": "Open folder…",
    "Guardar": "Save",
    "Cancelar": "Cancel",
    "Aceptar": "OK",
    "Sin imágenes": "No images",
    "Sin modelo": "No model",
    "Sin GPU": "No GPU",
    "Error": "Error",
    "Aviso": "Warning",
    "Parámetros": "Parameters",
    "Modelos": "Models",
    "Imágenes": "Images",
    "Ejecutar": "Run",
    "Resultados": "Results",
    "Errores": "Errors",
    "Comparar": "Compare",
    "Reporte": "Report",
    "Dataset": "Dataset",
    "Modelo": "Model",
    "Entrenar": "Train",
    "Evaluar": "Evaluate",
    "Exportar": "Export",
    "Informe": "Summary",
    "Augmentación": "Augmentation",

    # ── Troceado automatico (Detector › Parametros) ──
    "Troceado automático (fotos grandes)": "Automatic tiling (large photos)",
    "Cuándo trocear:": "When to tile:",
    "Umbral (lado mayor, px):": "Threshold (longest side, px):",
    "Lado del tile (px):": "Tile size (px):",
    "Solape entre tiles:": "Tile overlap:",
    "auto": "auto",
    "siempre": "always",
    "nunca": "never",
    "Una foto de placa completa entra a la red reescalada a imgsz: a 4096 px "
    "reducidos a 2080, cada partícula encoge a la mitad y desaparece bajo el "
    "stride de la red. Troceando, cada tile entra a resolución nativa y las "
    "cajas vuelven a coordenadas de la foto completa, fusionadas con NMS "
    "global — no hay que cortar nada a mano ni sumar los conteos después.":
        "A whole-plate photo is rescaled to imgsz on its way into the network: "
        "going from 4096 px down to 2080 halves every particle, and it vanishes "
        "under the network stride. Tiling feeds each tile at native resolution "
        "and maps the boxes back to full-photo coordinates, merged with global "
        "NMS — nothing has to be cut by hand and no counts have to be added up "
        "afterwards.",

    # ── Entrenador › GPU y prioridades ──
    "Maximizar imgsz para mi GPU": "Maximize imgsz for my GPU",
    "Detectar GPU / VRAM": "Detect GPU / VRAM",
    "Sugerir imgsz MÁXIMO": "Suggest MAXIMUM imgsz",
    "Sugerir batch para mi GPU": "Suggest batch for my GPU",
    "Optimizar todo (imgsz → batch → velocidad)":
        "Optimize everything (imgsz → batch → speed)",
    "Detectando GPU…": "Detecting GPU…",
    "Sin GPU utilizable": "No usable GPU",
    "Configuración optimizada": "Optimized configuration",
    "Fija imgsz al máximo que aguanta la tarjeta (sin pasar de la resolución "
    "nativa del dataset), después sube el batch con lo que sobre, y al final "
    "ajusta AMP, workers y cache. En ese orden.":
        "Sets imgsz to the highest the card can take (without exceeding the "
        "dataset's native resolution), then raises batch with what is left, and "
        "finally tunes AMP, workers and cache. In that order.",
    "automático": "automatic",
}

# El grueso del diccionario vive en traducciones_en.py: aqui quedan las entradas
# que acompanan al codigo escrito en este modulo y las de uso transversal.
TRADUCCIONES.update(_EN)
# Lo que se deja igual a proposito se registra como identidad, no como ausencia:
# asi el auditor distingue "no hace falta traducir" de "falta traducir".
TRADUCCIONES.update({s: s for s in NO_TRADUCIR})


def _detectar_idioma_sistema() -> str:
    try:
        cod = (locale.getlocale()[0] or locale.getdefaultlocale()[0] or "")
    except Exception:
        cod = ""
    return "es" if cod.lower().startswith("es") else "en" if cod else "es"


def cargar() -> str:
    """Resuelve el idioma una vez, en orden de precedencia."""
    global _idioma
    env = os.environ.get("POLYX_IDIOMA", "").strip().lower()
    if env in IDIOMAS:
        _idioma = env
        return _idioma
    try:
        if _ARCHIVO.exists():
            guardado = json.loads(_ARCHIVO.read_text(encoding="utf-8")).get("idioma")
            if guardado in IDIOMAS:
                _idioma = guardado
                return _idioma
    except Exception:
        pass
    _idioma = _detectar_idioma_sistema()
    return _idioma


def idioma() -> str:
    return _idioma


def set_idioma(codigo: str) -> None:
    """Cambia el idioma y avisa a quien se haya suscrito.

    Los widgets ya construidos no se rehacen solos: el launcher pide reiniciar.
    Reconstruir cada ventana viva seria mas fragil que un reinicio de 2 s.
    """
    global _idioma
    if codigo not in IDIOMAS:
        return
    _idioma = codigo
    try:
        _ARCHIVO.write_text(json.dumps({"idioma": codigo}), encoding="utf-8")
    except Exception:
        pass
    for f in list(_suscriptores):
        try:
            f(codigo)
        except Exception:
            pass


def al_cambiar(funcion: Callable[[str], None]) -> None:
    _suscriptores.append(funcion)


def tr(texto: str) -> str:
    """Traduce si hay entrada; si no, devuelve el original en espanol."""
    if _idioma == "es":
        return texto
    return TRADUCCIONES.get(texto, texto)


def cobertura() -> dict:
    """Cuantas cadenas del diccionario hay, para saber cuanto falta.

    No mide el total de la interfaz (eso lo da `auditar_traduccion.py`), solo
    lo ya traducido.
    """
    return {"traducidas": len(TRADUCCIONES), "idioma": _idioma}


cargar()
