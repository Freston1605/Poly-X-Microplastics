#!/usr/bin/env python3
"""Capture the M006/S05 English baseline reports (benign and hostile).

Builds the two shared DetectorState fixtures from tests/fixtures_informe.py
and renders each one through polyx.gui.report_html.generate_report exactly
as the app does (all sections, real pipeline, cronometro wrapper included).
The resulting HTML files are the M006/S05 English baseline used by the
byte-identity tests of the benign render and the injection tests.

The language is forced to English (``POLYX_IDIOMA=en``) before any polyx or
fixture module is imported, so the capture does not depend on the machine
locale or ``~/.polyx_idioma.json``. This is the same language the tests pin;
M006/S05 recaptured the snapshots after the twelve raw Spanish report
headings became English literals (see the README in the baseline directory).

Outputs (relative to the repo root):
- tests/fixtures/baseline_informe/benigna/informe.html  (+ informe.sha256)
- tests/fixtures/baseline_informe/hostil/informe.html   (+ informe.sha256)

Run from the repo root:  python tools/audit/capturar_fixtures_informe.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# The baseline is the English render (M006/S05): pin the language BEFORE
# importing tests.fixtures_informe or any polyx module, because i18n
# resolves and caches the language at import time.
os.environ["POLYX_IDIOMA"] = "en"

# Run as a raw file from tools/audit/, sys.path[0] points there: the repo
# root must go first so `import tests.fixtures_informe` and `import polyx`
# resolve exactly as they do under pytest.
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from tests.fixtures_informe import estado_benigno, estado_hostil  # noqa: E402

SALIDAS = {
    "benigna": RAIZ / "tests" / "fixtures" / "baseline_informe" / "benigna",
    "hostil": RAIZ / "tests" / "fixtures" / "baseline_informe" / "hostil",
}


def capturar() -> dict[str, str]:
    """Render both fixture states and persist each report with its sha256.

    Returns a dict variant -> sha256 hex digest, printed at the end so the
    M006/S05 English baseline hash lands in the task record.
    """
    import polyx.gui.report._common as report_common
    import polyx.gui.report_html as rh

    # The report embeds a machine-equipment table (_team) whose live values
    # (OS kernel, RAM, GPU, torch version) differ per runner. The byte
    # baseline is captured against the PINNED stack (CI: Python 3.11 +
    # matplotlib 3.9.2), so the equipment rows are pinned to canonical values
    # here and in tests/test_render_reporte_benigno.py (fixture
    # _idioma_del_baseline_y_logs_muertos). Never capture with live rows:
    # the fixture would not reproduce them on any other machine. The patch
    # lands on _common, the module _section_team reads the function from at
    # call time.
    report_common._team = lambda: [
        ("Operating system", "Linux"),
        ("Processor", "pinned"),
        ("RAM", "16.0 GB"),
        ("GPU", "not available"),
        ("PyTorch", "pinned"),
    ]

    # The gate compares figures by stable marker (see _normalizar_figuras in
    # tests/test_render_reporte_benigno.py): matplotlib PNG bytes differ per
    # renderer SO even at the same pin, so informe.sha256 registers the
    # NORMALIZED digest -- the same value the identity test compares.
    from tests.test_render_reporte_benigno import _normalizar_figuras

    # Same environmental pins as the test fixture: ultralytics version line
    # in the methods paragraph must not depend on which packages are
    # importable on the capturing machine.
    _ul_stub = type("ULStub", (), {"__version__": "8.3.40"})()
    sys.modules["ultralytics"] = _ul_stub

    estados = {
        "benigna": estado_benigno(),
        "hostil": estado_hostil(),
    }
    hashes: dict[str, str] = {}
    for variante, st in estados.items():
        directorio = SALIDAS[variante]
        directorio.mkdir(parents=True, exist_ok=True)
        destino = directorio / "informe.html"
        # Pipeline real: mismas secciones y firma que usan los tests de
        # identidad (T04) e inyeccion (T06), nada de un render recortado.
        rh.generate_report(st, destino, sections=list(rh.IDS_SECTIONS))
        datos = destino.read_text(encoding="utf-8")
        digest = hashlib.sha256(_normalizar_figuras(datos).encode("utf-8")).hexdigest()
        (directorio / "informe.sha256").write_text(digest + "\n", encoding="utf-8")
        hashes[variante] = digest
    return hashes


def main() -> int:
    """Entry point: capture both variants and report the digests."""
    hashes = capturar()
    for variante, digest in hashes.items():
        print(f"{variante}: {digest}  ({SALIDAS[variante] / 'informe.html'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
