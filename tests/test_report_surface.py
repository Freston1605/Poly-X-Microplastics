"""La superficie de exportacion del informe esta fijada por contrato.

``polyx.gui.report`` y su shim ``polyx.gui.report_html`` tienen que exportar
exactamente los 76 nombres del modulo monolitico original (el ultimo
``report_html.py`` del commit 26accdb): sus 77 defs de arbol superior menos
``REPORT_CSS``, que sustituyo la funcion perezosa ``_report_css()``. La lista
esta escrita a mano aqui a proposito y no se calcula en tiempo de ejecucion,
para que quitar un nombre del ``__all__`` o del shim rompa este test y no a
un consumidor en produccion.

El contrato se cruzo con los consumidores reales del arbol: todos usan un
subconjunto de estos 76 nombres.
"""

from __future__ import annotations

from pathlib import Path

# Los 76 nombres en orden ASCII. ES el contrato: no lo toques para hacer
# pasar el test; primero mira que consumidor sigue usando el nombre.
ESPERADOS: tuple[str, ...] = (
    "IDS_SECTIONS",
    "NUM",
    "PRESETS",
    "SECTIONS",
    "_ANCHORS",
    "_MAX_FOLDER_PER_PHOTO",
    "_MAX_GROUPS_BOXPLOT",
    "_NAME_MORPHOTYPE",
    "_ORDER_CLASSES",
    "_ORDER_STATION",
    "_PALETTE",
    "_PLATE",
    "_RE_PLATE",
    "_STATION",
    "_TABLE_T95",
    "_WIDTH_GALLERY",
    "_average_ic95",
    "_batch_metrics",
    "_chi2_sf",
    "_classes_views",
    "_clue_order",
    "_count_per_class",
    "_document_html",
    "_esc",
    "_fig_agreement_by_image",
    "_fig_boxplot_groups",
    "_fig_class_distribution",
    "_fig_comparison_metrics",
    "_fig_confidence_hist",
    "_fig_confusion_matrix",
    "_fig_metrics_by_class",
    "_fig_profile_depth",
    "_fig_size_hist",
    "_fig_sizes_by_section",
    "_fig_sweep",
    "_fig_to_b64",
    "_figure_calibration",
    "_generate_report_impl",
    "_img_b64",
    "_kruskal_wallis",
    "_lot_forms",
    "_morphotype",
    "_p_permutation",
    "_pr_f1",
    "_rankdata",
    "_record_particle",
    "_sample_of",
    "_section_calibration",
    "_section_comparison",
    "_section_count",
    "_section_errors",
    "_section_gallery",
    "_section_methods",
    "_section_models",
    "_section_records",
    "_section_references",
    "_section_results",
    "_section_shape",
    "_section_summary",
    "_section_team",
    "_section_verdict",
    "_size_per_folder",
    "_spearman",
    "_sweep_confidence",
    "_t_critical95",
    "_table_count",
    "_table_width",
    "_team",
    "_text_kruskal",
    "_text_trend",
    "_training_of",
    "_uri_gallery",
    "anchor",
    "choose_records",
    "generate_report",
    "section_depth",
)


def test_la_lista_del_contrato_esta_integra():
    """76 nombres, sin duplicados y ordenados: la lista no se degrada sola."""
    assert len(ESPERADOS) == 76
    assert len(set(ESPERADOS)) == len(ESPERADOS)
    assert list(ESPERADOS) == sorted(ESPERADOS)


def test_el_paquete_exporta_los_76_nombres():
    """``polyx.gui.report`` exporta el contrato completo, nombre por nombre."""
    import polyx.gui.report as paquete

    assert sorted(paquete.__all__) == list(ESPERADOS)
    assert len(paquete.__all__) == len(set(paquete.__all__))
    faltan = [n for n in ESPERADOS if not hasattr(paquete, n)]
    assert not faltan, f"el paquete no exporta: {faltan}"


def test_el_shim_exporta_los_mismos_76_nombres():
    """``polyx.gui.report_html`` sigue exponiendo la superficie vieja."""
    import polyx.gui.report_html as shim

    assert sorted(shim.__all__) == list(ESPERADOS)
    assert len(shim.__all__) == len(set(shim.__all__))
    faltan = [n for n in ESPERADOS if not hasattr(shim, n)]
    assert not faltan, f"el shim no exporta: {faltan}"


def test_report_css_queda_fuera_de_la_superficie():
    """``REPORT_CSS`` cae a proposito: lo reemplazo ``_report_css()``."""
    import polyx.gui.report as paquete
    import polyx.gui.report_html as shim

    assert "REPORT_CSS" not in ESPERADOS
    assert "REPORT_CSS" not in paquete.__all__
    assert "REPORT_CSS" not in shim.__all__
    assert not hasattr(paquete, "REPORT_CSS")
    assert not hasattr(shim, "REPORT_CSS")


def test_el_shim_apunta_al_fichero_de_la_implementacion():
    """``rh.__file__`` resuelve al paquete real: el CSS vive al lado de _common."""
    import polyx.gui.report._common as common
    import polyx.gui.report_html as shim

    assert Path(shim.__file__).resolve() == Path(common.__file__).resolve()
    assert Path(shim.__file__).name == "_common.py"
