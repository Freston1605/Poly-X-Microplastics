#!/usr/bin/env python3
"""Capture the pre-fix baseline reports (benign and hostile) of the real pipeline.

Builds the two shared DetectorState fixtures from tests/fixtures_informe.py
and renders each one through polyx.gui.report_html.generate_report exactly
as the app does (all sections, real pipeline, cronometro wrapper included).
The resulting HTML files are the pre-fix golden snapshots used by T04
(byte-identity of the benign render) and T06/T07 (dead injection).

Outputs (relative to the repo root):
- tests/fixtures/baseline_informe/benigna/informe.html  (+ informe.sha256)
- tests/fixtures/baseline_informe/hostil/informe.html   (+ informe.sha256)

Run from the repo root:  python scripts/capturar_fixtures_informe.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Ejecutado como archivo desde scripts/, sys.path[0] apunta a scripts/: hay
# que poner la raiz del repo primero para que `import tests.fixtures_informe`
# y `import polyx` resuelvan igual que en pytest.
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from tests.fixtures_informe import estado_benigno, estado_hostil  # noqa: E402

SALIDAS = {
    "benigna": RAIZ / "tests" / "fixtures" / "baseline_informe" / "benigna",
    "hostil": RAIZ / "tests" / "fixtures" / "baseline_informe" / "hostil",
}


def capturar() -> dict[str, str]:
    """Render both fixture states and persist each report with its sha256.

    Returns a dict variant -> sha256 hex digest, printed at the end so the
    benign pre-fix baseline hash lands in the task record.
    """
    import polyx.gui.report_html as rh

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
        rh.generate_report(st, destino, secciones=list(rh.IDS_SECCIONES))
        datos = destino.read_bytes()
        digest = hashlib.sha256(datos).hexdigest()
        (directorio / "informe.sha256").write_text(digest + "\n",
                                                   encoding="utf-8")
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
