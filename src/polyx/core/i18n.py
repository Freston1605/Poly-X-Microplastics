"""Interface translation. Spanish is the source language; English, the target.

Why this scheme instead of Qt's (.ts + lupdate + .qm):

    The interface has ~2350 strings spread over 55 files. The Qt flow requires
    wrapping each one in ``tr()``, running ``lupdate``, editing the ``.ts`` in
    Linguist and compiling to ``.qm``: four steps and two external tools every
    time a text is touched. Here the **Spanish string is the key**, so
    translating is adding an entry to the dictionary and nothing more.
    Anything untranslated falls back to Spanish instead of showing empty or
    showing the raw key, which is the usual failure mode of key-based systems.

Usage:

    from ..core.i18n import tr
    boton = QPushButton(tr("Abrir carpeta…"))

The language is resolved once at startup: ``POLYX_IDIOMA`` if set, otherwise
the user's saved choice, otherwise the system language, and failing all of
that, Spanish."""
from __future__ import annotations

import json
import locale
import os
from pathlib import Path
from typing import Callable

from polyx.core.translations_en import EN as _EN, NO_TRANSLATE

LANGUAGES = {"es": "Español", "en": "English"}
_FILE = Path.home() / ".polyx_idioma.json"

_language = "es"
_subscribers: list[Callable[[str], None]] = []


# ── es -> en dictionary ──
# Text shown to the user only. Polymer class names are not translated
# (PET/PP/LDPE), units or identifiers.
TRANSLATIONS: dict[str, str] = {
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

    # ── Shared vocabulary ──
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

    # ── Automatic tiling (Detector › Parameters) ──
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

    # ── Trainer › GPU and priorities ──
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

# Most of the dictionary lives in `traducciones_en.py`: here remain the entries
# that go with the code written in this module and those used across modules.
TRANSLATIONS.update(_EN)
# What is intentionally left unchanged is recorded as identity, not as absence:
# that way the auditor distinguishes "no translation needed" from "translation missing".
TRANSLATIONS.update({s: s for s in NO_TRANSLATE})


def _detect_language_system() -> str:
    try:
        cod = (locale.getlocale()[0] or locale.getdefaultlocale()[0] or "")
    except Exception:
        cod = ""
    return "es" if cod.lower().startswith("es") else "en" if cod else "es"


def carry() -> str:
    """Resolves the language once, in precedence order."""
    global _language
    env = os.environ.get("POLYX_IDIOMA", "").strip().lower()
    if env in LANGUAGES:
        _language = env
        return _language
    try:
        if _FILE.exists():
            saved = json.loads(_FILE.read_text(encoding="utf-8")).get("idioma")
            if saved in LANGUAGES:
                _language = saved
                return _language
    except Exception:
        pass
    _language = _detect_language_system()
    return _language


def language() -> str:
    return _language


def set_language(code: str) -> None:
    """Changes the language and notifies whoever subscribed.

    Already-built widgets do not rebuild themselves: the launcher asks for a
    restart. Rebuilding every live window would be more fragile than a 2 s
    restart.
    """
    global _language
    if code not in LANGUAGES:
        return
    _language = code
    try:
        _FILE.write_text(json.dumps({"idioma": code}), encoding="utf-8")
    except Exception:
        pass
    for f in list(_subscribers):
        try:
            f(code)
        except Exception:
            pass


def when_changing(function: Callable[[str], None]) -> None:
    _subscribers.append(function)


def tr(text: str) -> str:
    """Translates when there is an entry; otherwise returns the Spanish original."""
    if _language == "es":
        return text
    return TRANSLATIONS.get(text, text)


def coverage() -> dict:
    """How many dictionary strings there are, to know how much is left.

    It does not measure the interface total (that is
    `scripts/audit_translations.py`'s job), only what is already translated.
    """
    return {"traducidas": len(TRANSLATIONS), "idioma": _language}


carry()
