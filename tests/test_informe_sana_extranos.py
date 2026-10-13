"""Que el informe neutralice texto ajeno en vez de renderizarlo.

Una imagen llamada ``<img src=x onerror=alert(1)>.png`` llega al HTML del
informe aqui, y el informe se abre en el navegador: si el nombre entra sin
escapar, en quien analice una carpeta compartida se ejecuta JS en el
contexto del navegador. El escape del nombre es lo que rompe el vector.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyx.core.yolo_wrap import Detection  # noqa: E402

NOMBRE_HOSTIL = "<img src=x onerror=alert(1)>.png"
IDAD_HOSTIL = "<b>peor-modelo</b>"


@pytest.mark.parametrize("lang", ["es", "en"])
def test_el_nombre_extrano_no_se_cuela_como_html(tmp_path, lang):
    """Nombre y alias hostiles solo aparecen escapados en el documento."""
    import os
    import importlib

    os.environ["POLYX_IDIOMA"] = lang
    import polyx.core.i18n as i18n
    importlib.reload(i18n)
    import polyx.core.report_html as rh
    importlib.reload(rh)
    from polyx.detector.state import DetectorState, ImageResult, ModelSlot

    ruta = tmp_path / NOMBRE_HOSTIL
    ruta.write_bytes(b"")
    st = DetectorState()
    st.params.conf_asignacion = 0.0
    alias_antes = st.model_slots[0].alias
    st.model_slots[0] = ModelSlot(alias=IDAD_HOSTIL, path=tmp_path / "fake.pt")
    st.results = {0: [ImageResult(image_path=ruta, model_idx=0,
                                  predictions=[
                                      Detection(class_id=0, class_name="PET",
                                                conf=0.9, x1=0, y1=0, x2=10, y2=10)],
                                  gt=[], has_gt=False)]}

    salida = tmp_path / "salida.html"
    rh.generate_report(st, salida, secciones=list(rh.IDS_SECCIONES))
    html = salida.read_text(encoding="utf-8")

    assert NOMBRE_HOSTIL not in html, "el nombre hostil entro tal cual"
    assert IDAD_HOSTIL not in html, "el alias hostil entro tal cual"
    assert NOMBRE_HOSTIL in html.replace("&lt;", "<").replace("&gt;", ">").replace(
        "&quot;", '"'), "el nombre debe seguir siendo legible, solo que escapado"

    # restaurar el idioma para los siguientes tests
    os.environ.pop("POLYX_IDIOMA", None)
    importlib.reload(i18n)
    importlib.reload(rh)
