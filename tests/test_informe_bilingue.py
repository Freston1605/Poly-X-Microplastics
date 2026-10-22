"""Que el informe salga de verdad en los dos idiomas.

Hay un modo de fallo que no se nota nunca al programar: si una cadena se
envuelve como ``{tr('...')}`` dentro de una cadena que NO es f-string, Python
no protesta y el documento sale con el codigo impreso tal cual:

    <h3>2.1 {tr('Equipo de computo')}</h3>

Se ve solo abriendo el HTML, y paso tres veces al traducir el informe. Aqui se
comprueba sobre el documento generado, que es donde el fallo existe.

Tambien se vigila lo contrario: que pedir ingles no devuelva un informe en
español porque la clave del diccionario dejo de coincidir con la del codigo
-- basta un espacio o un salto de linea de diferencia para que tr() caiga en
silencio al original.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from polyx.core.detect.protocol import Instance as Detection


def _estado_minimo(tmp_path):
    """Un DetectorState con una deteccion medida sobre una foto sintetica."""
    import cv2
    from polyx.gui.detector.state import DetectorState, ImageResult, ModelSlot
    from polyx.core import morphology

    img = np.full((260, 320, 3), 18, np.uint8)
    cv2.circle(img, (160, 130), 22, (60, 210, 240), -1)
    ruta = tmp_path / "placa.png"
    cv2.imwrite(str(ruta), img)

    d = Detection(class_id=0, class_name="PET", conf=0.9,
                  x1=138, y1=108, x2=182, y2=152)
    morphology.apply_to_detection(d, img, 30.0)
    d.number = 1

    st = DetectorState()
    st.model_slots[0] = ModelSlot(alias="m1", path=Path("fake.pt"))
    st.results = {0: [ImageResult(image_path=ruta, model_idx=0,
                                  predictions=[d], gt=[], has_gt=False)]}
    return st


def _informe(tmp_path, lang):
    """Genera el informe con el idioma pedido y devuelve su HTML."""
    os.environ["POLYX_IDIOMA"] = lang
    import polyx.core.i18n as i18n
    importlib.reload(i18n)
    import polyx.gui.report_html as rh
    importlib.reload(rh)
    salida = tmp_path / f"informe_{lang}.html"
    rh.generate_report(_estado_minimo(tmp_path), salida,
                       sections=list(rh.IDS_SECTIONS))
    return salida.read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _restaurar_idioma():
    previo = os.environ.get("POLYX_IDIOMA")
    yield
    if previo is None:
        os.environ.pop("POLYX_IDIOMA", None)
    else:
        os.environ["POLYX_IDIOMA"] = previo
    import polyx.core.i18n as i18n
    importlib.reload(i18n)
    import polyx.gui.report_html as rh
    importlib.reload(rh)


@pytest.mark.parametrize("lang", ["es", "en"])
def test_no_queda_codigo_sin_evaluar_en_el_documento(tmp_path, lang):
    """El fallo de envolver en tr() una cadena que no era f-string."""
    html = _informe(tmp_path, lang)
    assert "{tr(" not in html, (
        "el documento lleva un {tr(...)} impreso: esa cadena se envolvio "
        "dentro de una cadena normal, no de una f-string")


@pytest.mark.parametrize("lang", ["es", "en"])
def test_el_idioma_del_documento_es_el_pedido(tmp_path, lang):
    html = _informe(tmp_path, lang)
    assert f"<html lang='{lang}'" in html


def test_en_ingles_no_salen_los_rotulos_en_español(tmp_path):
    """Si una clave del diccionario deja de coincidir con la del codigo, tr()
    devuelve el español sin avisar y el informe queda mezclado."""
    html = _informe(tmp_path, "en")
    for rotulo_es, rotulo_en in [("<th>Parámetro</th>", "Parameter"),
                                 ("Morfotipo", "Morphotype"),
                                 ("<td>mínimo</td>", "minimum")]:
        assert rotulo_es not in html, f"quedo en español: {rotulo_es}"
        assert rotulo_en in html, f"falta la traduccion: {rotulo_en}"


def test_en_español_siguen_los_rotulos_en_español(tmp_path):
    """La otra direccion: traducir no debe romper el idioma original."""
    html = _informe(tmp_path, "es")
    assert "<th>Parámetro</th>" in html
    assert "Parameter" not in html


def _texto_visible(html: str) -> list[str]:
    """Lineas de texto del documento, sin marcado ni imagenes embebidas."""
    import re
    h = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    h = re.sub(r"data:image[^\"']+", "", h)
    h = re.sub(r"<[^>]+>", "\n", h)
    for a, b in [("&nbsp;", " "), ("&middot;", "·"), ("&aacute;", "á"),
                 ("&eacute;", "é"), ("&iacute;", "í"), ("&oacute;", "ó"),
                 ("&uacute;", "ú"), ("&ntilde;", "ñ")]:
        h = h.replace(a, b)
    return [l.strip() for l in h.split("\n") if l.strip()]


def test_el_informe_en_ingles_no_arrastra_frases_en_español(tmp_path):
    """Barrido ancho, no por sondas sueltas.

    Se generan los dos documentos y se comparan sus textos: una linea que
    aparece IGUAL en ambos y tiene pinta de español es una cadena que se quedo
    sin envolver en tr() o cuya clave dejo de coincidir. Es como se encontraron
    las ultimas ~100, y sin esta prueba vuelven a colarse a la primera edicion.
    """
    import re
    es = set(_texto_visible(_informe(tmp_path, "es")))
    en = set(_texto_visible(_informe(tmp_path, "en")))
    marca = re.compile(r"[áéíóúñÁÉÍÓÚÑ¿¡]|\b(el|la|los|las|del|que|con|para|"
                       r"por|una|un|se|es|su|sus|como|cada|sobre|entre|sin|no|"
                       r"clase|fibras|muestra|menor|mayor|mediana|verdaderos|"
                       r"falsos|clasificados|imagen)\b",
                       re.I)
    # Los apellidos de la bibliografia son nombres propios: no se traducen.
    sospechosas = [t for t in es & en
                   if marca.search(t) and len(t) > 2 and "(2024)" not in t]
    assert not sospechosas, (
        "texto sin traducir en el informe en ingles:\n  "
        + "\n  ".join(sorted(sospechosas)[:15]))


def test_ninguna_traduccion_se_congela_al_importar():
    """Un tr() en el cuerpo del modulo se evalua UNA vez, al importar.

    Con eso, cambiar el idioma y volver a generar el informe deja ese texto en
    el idioma anterior. Ya paso con el nombre del morfotipo. Se comprueba sobre
    el arbol sintactico porque en tiempo de ejecucion no se distingue.
    """
    import ast
    from pathlib import Path as _P

    ruta = _P(__file__).resolve().parents[1] / "polyx" / "gui" / "report_html.py"
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    dentro_de_funcion = set()
    for nodo in arbol.body:
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for sub in ast.walk(nodo):
                dentro_de_funcion.add(id(sub))
    congeladas = [
        n.lineno for n in ast.walk(arbol)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "tr" and id(n) not in dentro_de_funcion]
    assert not congeladas, (
        f"tr() en el cuerpo del modulo, lineas {congeladas}: se resuelve al "
        f"importar y no sigue al idioma elegido")


# ── Lote que obliga a construir todas las secciones ────────────────────
#
# El lote minimo de arriba deja media docena de secciones sin construir: sin
# dos carpetas no hay talla por carpeta, sin dos modelos no hay comparacion y
# sin ground truth no hay matriz de confusion. Ahi se escondio un `+ +` que
# Python acepta como mas unario y que solo reventaba al generar el informe de
# un lote real, en la exportacion a PDF:
#
#     TypeError: bad operand type for unary +: 'str'
#
# Sintacticamente valido, de modo que ni ast.parse ni el lote minimo lo veian.


def _estado_completo(tmp_path):
    """Dos carpetas, tres fotos cada una, dos modelos y ground truth."""
    import cv2
    from polyx.gui.detector.state import DetectorState, ImageResult, ModelSlot
    from polyx.core import morphology

    clases = ["PET", "PP", "LDPE"]
    st = DetectorState()
    st.model_slots[0] = ModelSlot(alias="yolov8n", path=Path("a.pt"))
    st.model_slots[1] = ModelSlot(alias="yolo11n", path=Path("b.pt"))
    st.results = {0: [], 1: []}

    n = 0
    for carpeta in ("Chiu Chiu", "Desembocadura"):
        for foto in range(3):
            rng = np.random.default_rng(n)
            img = np.full((520, 640, 3), 18, np.uint8)
            cajas = []
            for _ in range(5):
                cx, cy = int(rng.integers(70, 570)), int(rng.integers(70, 450))
                r = int(rng.integers(9, 26))
                cv2.circle(img, (cx, cy), r, (60, 210, 240), -1)
                cajas.append((cx - r, cy - r, cx + r, cy + r))
            # Una fibra por foto: las fichas reparten 6 fibras + 6 particulas,
            # y sin ninguna fibra esa rama no se ejercita.
            y = int(rng.integers(80, 440))
            cv2.line(img, (60, y), (300, y + 40), (60, 210, 240), 6)
            cajas.append((60, y - 8, 300, y + 48))

            ruta = tmp_path / carpeta / f"{foto + 1}.1.png"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ruta), img)

            escala = 30.0 + 4.0 * foto          # escala distinta en cada foto
            for mi in (0, 1):
                preds, gt = [], []
                for k, (x1, y1, x2, y2) in enumerate(cajas):
                    cid = k % 3
                    d = Detection(class_id=cid, class_name=clases[cid],
                                  conf=0.6 + 0.05 * k, x1=x1, y1=y1, x2=x2, y2=y2)
                    morphology.apply_to_detection(d, img, escala)
                    n += 1
                    d.number = n
                    # El segundo modelo se salta una caja: si los dos dieran lo
                    # mismo, la comparacion no tendria nada que enseñar.
                    if not (mi == 1 and k == 0):
                        preds.append(d)
                    # Y el ground truth confunde una clase, para que la matriz
                    # de confusion tenga algo fuera de la diagonal.
                    cid_gt = (cid + 1) % 3 if k == 2 else cid
                    gt.append(Detection(class_id=cid_gt, class_name=clases[cid_gt],
                                        conf=1.0, x1=x1, y1=y1, x2=x2, y2=y2))
                st.results[mi].append(ImageResult(
                    image_path=ruta, model_idx=mi, predictions=preds,
                    gt=gt, has_gt=True))
    return st


def _informe_completo(tmp_path, lang):
    os.environ["POLYX_IDIOMA"] = lang
    import polyx.core.i18n as i18n
    importlib.reload(i18n)
    import polyx.gui.report_html as rh
    importlib.reload(rh)
    salida = tmp_path / f"completo_{lang}.html"
    rh.generate_report(_estado_completo(tmp_path), salida,
                       sections=list(rh.IDS_SECTIONS))
    return salida.read_text(encoding="utf-8")


@pytest.mark.parametrize("lang", ["es", "en"])
def test_un_lote_con_todas_las_secciones_se_genera(tmp_path, lang):
    """Que ninguna seccion reviente al construirse, en ninguno de los dos
    idiomas. Es la prueba que habria cazado el mas unario."""
    html = _informe_completo(tmp_path, lang)
    assert html.count("<h2") >= 10, "faltan secciones por construir"
    assert "{tr(" not in html


def test_en_el_lote_completo_tampoco_se_cuela_español(tmp_path):
    """El mismo barrido de antes, pero sobre las secciones que el lote minimo
    no llega a construir: comparacion, errores, galeria y talla por carpeta."""
    import re
    es = set(_texto_visible(_informe_completo(tmp_path, "es")))
    en = set(_texto_visible(_informe_completo(tmp_path, "en")))
    marca = re.compile(r"[áéíóúñÁÉÍÓÚÑ¿¡]|\b(el|la|los|las|del|que|con|para|"
                       r"por|una|un|se|es|su|sus|como|cada|sobre|entre|sin|no|"
                       r"clase|fibras|muestra|menor|mayor|mediana|verdaderos|"
                       r"falsos|clasificados|imagen)\b",
                       re.I)
    sospechosas = [t for t in es & en
                   if marca.search(t) and len(t) > 2 and "(2024)" not in t]
    assert not sospechosas, (
        "texto sin traducir en el informe en ingles:\n  "
        + "\n  ".join(sorted(sospechosas)[:15]))
