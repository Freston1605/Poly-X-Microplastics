"""Captura las ventanas de Poly-X a PNG para el manual.

A diferencia de generar_manual.py, no parchea un HTML existente: vuelca las
capturas a una carpeta para que el manual se arme desde cero. Cubre las 21
pantallas (incluye la pestana 2 del Detector, "GT manual", que el generador
antiguo se saltaba).

Uso (desde la raiz del proyecto):
    .venv\Scripts\python.exe tools\manual\capturar_manual.py --salida docs/assets/manual_screenshots/es
    POLYX_IDIOMA=en .venv\Scripts\python.exe tools\manual\capturar_manual.py --salida docs/assets/manual_screenshots/en
"""
from __future__ import annotations
import sys, os, time, argparse, importlib.util
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from PySide6.QtWidgets import QApplication

# (nombre_archivo, modulo, clase, (w,h), tab, espera_ms)
CAPTURAS = [
    ("01_launcher",            "polyx.gui.launcher",    "LauncherWindow", (1180, 820), None, 1200),

    ("02_det_modelos",         "polyx.gui.detector",    "DetectorWindow", (1360, 860), 0, 700),
    ("03_det_imagenes",        "polyx.gui.detector",    "DetectorWindow", (1360, 860), 1, 700),
    ("04_det_gt_manual",       "polyx.gui.detector",    "DetectorWindow", (1360, 860), 2, 700),
    ("05_det_parametros",      "polyx.gui.detector",    "DetectorWindow", (1360, 860), 3, 700),
    ("06_det_ejecutar",        "polyx.gui.detector",    "DetectorWindow", (1360, 860), 4, 700),
    ("07_det_resultados",      "polyx.gui.detector",    "DetectorWindow", (1360, 860), 5, 700),
    ("08_det_errores",         "polyx.gui.detector",    "DetectorWindow", (1360, 860), 6, 700),
    ("09_det_comparar",        "polyx.gui.detector",    "DetectorWindow", (1360, 860), 7, 700),
    ("10_det_reporte",         "polyx.gui.detector",    "DetectorWindow", (1360, 860), 8, 900),

    ("11_ent_modelo",          "polyx.gui.trainer",  "TrainerWindow",  (1360, 880), 0, 700),
    ("12_ent_dataset",         "polyx.gui.trainer",  "TrainerWindow",  (1360, 880), 1, 700),
    ("13_ent_parametros",      "polyx.gui.trainer",  "TrainerWindow",  (1360, 880), 2, 700),
    ("14_ent_augmentacion",    "polyx.gui.trainer",  "TrainerWindow",  (1360, 880), 3, 700),
    ("15_ent_entrenar",        "polyx.gui.trainer",  "TrainerWindow",  (1360, 880), 4, 700),
    ("16_ent_evaluar",         "polyx.gui.trainer",  "TrainerWindow",  (1360, 880), 5, 700),
    ("17_ent_comparar",        "polyx.gui.trainer",  "TrainerWindow",  (1360, 880), 6, 700),
    ("18_ent_exportar",        "polyx.gui.trainer",  "TrainerWindow",  (1360, 880), 7, 700),
    ("19_ent_informe",         "polyx.gui.trainer",  "TrainerWindow",  (1360, 880), 8, 700),

    ("20_etiquetador",         "polyx.gui.labeler", "LabelerWindow",  (1420, 900), None, 1000),
    ("21_visor",               "polyx.gui.viewer",       "ViewerWindow",    (1440, 920), None, 1000),
]


def capturar(modulo: str, clase: str, size, tab, espera_ms: int, destino: Path) -> bool:
    if importlib.util.find_spec(modulo) is None:
        print(f"  [SKIP] {modulo} no existe")
        return False
    try:
        mod = importlib.import_module(modulo)
        klass = getattr(mod, clase, None)
        if klass is None:
            print(f"  [SKIP] {clase} no esta en {modulo}")
            return False
        w = klass()
        w.resize(*size)
        w.move(-3000, -3000)
        w.show()

        if tab is not None:
            btns = getattr(w, "sidebar_buttons", None)
            stack = getattr(w, "stack", None)
            if btns and stack and 0 <= tab < len(btns):
                btns[tab].setChecked(True)
                stack.setCurrentIndex(tab)

        deadline = time.time() + espera_ms / 1000.0
        while time.time() < deadline:
            QApplication.processEvents()
            time.sleep(0.02)

        pix = w.grab()
        w.close()
        destino.parent.mkdir(parents=True, exist_ok=True)
        ok = pix.save(str(destino), "PNG")
        print(f"  [{'OK' if ok else 'FAIL'}] {destino.name}  {pix.width()}x{pix.height()}")
        return ok
    except Exception as e:
        print(f"  [ERR] {modulo}.{clase}: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=None,
                    help="carpeta destino; por omision docs/assets/manual_screenshots/es")
    ap.add_argument("--solo", default=None, help="subcadena del nombre de captura")
    args = ap.parse_args()

    raiz = Path(__file__).resolve().parents[2]
    out = Path(args.salida) if args.salida else raiz / "docs" / "assets" / "manual_screenshots" / "es"
    print(f"Idioma UI : {os.environ.get('POLYX_IDIOMA', '(preferencia guardada)')}")
    print(f"Salida    : {out}")

    app = QApplication.instance() or QApplication(sys.argv)
    hechas = 0
    for nombre, modulo, clase, size, tab, espera in CAPTURAS:
        if args.solo and args.solo not in nombre:
            continue
        print(f"  -> {nombre}")
        if capturar(modulo, clase, size, tab, espera, out / f"{nombre}.png"):
            hechas += 1
    print(f"\nListo: {hechas} capturas en {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
