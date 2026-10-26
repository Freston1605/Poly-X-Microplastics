"""Laziness gate: deferred heavy imports of the report module stay deferred (M004/S01/T02).

Importing polyx.gui.report_html must NOT pull in the heavy scientific stack:
torch, ultralytics, polyx.core.morphology and polyx.core.calibration are all
imported lazily inside function bodies (cv2 L283/L1192/L1253/L2554,
torch L311/L335, psutil L305, calibracion L1167/L2144/L2216/L2566,
morfologia L1255/L2555, ultralytics L2093) so that every consumer of the
report module (GUI, trainer, tests) pays only for matplotlib + the light model
modules at import time.

The test runs in a fresh interpreter (subprocess) so suite-wide sys.modules
pollution cannot mask a leak: if any other test imported torch first, an
in-process check would always pass for the wrong reason.

Deliberate exception (decision D017, waiving cv2 from the laziness contract):
cv2 MAY end up in sys.modules, because report_html imports
``from polyx.core.detect.yolo import Detection`` at top level and yolo_wrap
does ``import cv2`` at module level (polyx/model/yolo_wrap.py:11). That
transitive load is pre-existing behavior of the package split, not something
the S02/S03 section-builder refactor introduces; forcing yolo_wrap to defer
is out of scope for M004. The test therefore asserts yolo_wrap IS importable
(keeping the waiver honest if the import topology moves) instead of asserting
cv2's absence.

Nota de idioma: el subproceso fuerza POLYX_IDIOMA=en (mismo contrato que el
baseline del render, ver tests/test_render_reporte_benigno.py) y
QT_QPA_PLATFORM=offscreen para hermeticidad; ninguno de los dos afecta la
topologia de imports, solo evitan dependencia del locale/entorno de la
maquina.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Modulos pesados que DEBEN quedar diferidos tras importar el informe.
LAZY_TARGETS = (
    "torch",
    "ultralytics",
    "polyx.core.morphology",
    "polyx.core.calibration",
)

# Contrato del gate segun la decision D017 (ver .gsd/DECISIONS.md):
# cv2 queda expresamente fuera del contrato de laziness porque entra de forma
# transitiva via polyx.core.detect.yolo (import a nivel de modulo en yolo_wrap).
MODULO_TRANSITIVO_ESPERADO = "polyx.core.detect.yolo"


def test_importar_informe_no_carga_los_pesados():
    """Fresh-interpreter gate: heavy imports stay lazy after report_html import."""
    script = "\n".join(
        [
            "import sys",
            "import polyx.gui.report_html  # noqa: F401 — importa y nada mas",
            # yolo_wrap debe seguir importable: es la via transitiva declarada
            # de cv2; si desaparece, la exencion de cv2 (D017) quedo obsoleta.
            "import polyx.core.detect.yolo",
            "filtrados = [m for m in sys.modules if m in %r]" % (LAZY_TARGETS,),
            "print('YoloWrapImportable:ok')",
            "print('FILTRADOS:' + ','.join(filtrados))",
        ]
    )
    # Se hereda el entorno del padre y solo se fijan las dos variables de
    # hermeticidad: el interprete hijo necesita PATH/locales del sistema.
    env = os.environ | {
        "QT_QPA_PLATFORM": "offscreen",
        "POLYX_IDIOMA": "en",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )

    # Si el propio import del informe revienta, el gate tiene que señalarlo.
    assert proc.returncode == 0, (
        "importar polyx.gui.report_html fallo en el interprete limpio:\n" + proc.stderr
    )
    assert "YoloWrapImportable:ok" in proc.stdout, proc.stdout + proc.stderr

    filtrados = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("FILTRADOS:")),
        "",
    ).removeprefix("FILTRADOS:")
    leak = filtrados.split(",")[0] if filtrados else ""
    assert not leak, (
        f"El modulo pesado '{leak}' quedo cargado en sys.modules tras importar "
        f"polyx.gui.report_html; se espera que quede diferido (imports "
        f"dentro del cuerpo de las funciones en report_html.py). Un import "
        f"eager de {leak} encarece el costo de arranque para TODO consumidor "
        f"del modulo de informe (GUI, trainer, tests). Vea decision D017 y "
        f"tests/test_importes_diferidos_informe.py."
    )
