"""The report really comes out in both languages.

The same batch is rendered twice, switching the language through
``set_language``, and the test reads the document the user receives: the
HTML ``lang``, the translated headings, and a sweep for sentences that stayed
untranslated. Nothing here inspects the catalogue, the translation call or the
``.po`` layout.

There is a failure mode that is easy to miss while coding: a string that never
reaches the translation layer makes the document come out in the wrong
language, and only opening the HTML shows it. The sweep below renders both
languages and flags lines that appear identical in both while reading Spanish.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyx.core.detect.protocol import Instance as Detection


def _estado_minimo(tmp_path):
    """A RunResult with one measured detection on a synthetic photo."""
    import cv2

    import polyx.core.measure as measure
    from polyx.core.detection_run import ImageResult, ModelInfo, RunParams, RunResult

    img = np.full((260, 320, 3), 18, np.uint8)
    cv2.circle(img, (160, 130), 22, (60, 210, 240), -1)
    image_path = tmp_path / "placa.png"
    ok, buffer = cv2.imencode(".png", img)
    assert ok
    image_path.write_bytes(buffer.tobytes())

    d = Detection(class_id=0, class_name="PET", conf=0.9, x1=138, y1=108, x2=182, y2=152)
    measure.apply_to_detection(d, img, 30.0)
    d.number = 1

    return RunResult(
        run_dir=tmp_path / "run",
        images=[image_path],
        models=[ModelInfo(alias="m1", path=Path("fake.pt"))],
        params=RunParams(),
        results={
            0: [
                ImageResult(
                    image_path=image_path,
                    model_idx=0,
                    predictions=[d],
                    gt=[],
                    has_gt=False,
                )
            ]
        },
    )


def _informe(tmp_path, lang):
    """Generate the report in the requested language and return its HTML."""
    import polyx.core.i18n as i18n
    import polyx.gui.report_html as rh

    assert i18n.set_language(lang)
    salida = tmp_path / f"informe_{lang}.html"
    rh.generate_report(_estado_minimo(tmp_path), salida, sections=list(rh.IDS_SECTIONS))
    return salida.read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _restaurar_idioma():
    """Leave the suite in the language conftest pins."""
    from polyx.core import i18n

    previous = i18n.language()
    yield
    i18n.set_language(previous)


@pytest.mark.parametrize("lang", ["es", "en"])
def test_the_document_keeps_no_unrendered_placeholders(tmp_path, lang):
    """An internal marker or an unresolved ``%(name)s`` reaches the user."""
    html = _informe(tmp_path, lang)
    assert "@@N@@" not in html, "an internal numbering marker was not substituted"
    assert "%(" not in html, "an unresolved placeholder is in the document"


@pytest.mark.parametrize("lang", ["es", "en"])
def test_el_idioma_del_documento_es_el_pedido(tmp_path, lang):
    html = _informe(tmp_path, lang)
    assert f"<html lang='{lang}'" in html


def test_en_english_no_salen_los_rotulos_en_español(tmp_path):
    """Asking for English must not return a mixed report."""
    html = _informe(tmp_path, "en")
    for rotulo_es, rotulo_en in [
        ("<th>Parámetro</th>", "Parameter"),
        ("Morfotipo", "Morphotype"),
        ("<td>mínimo</td>", "minimum"),
    ]:
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

    h = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL)
    h = re.sub(r"data:image[^\"']+", "", h)
    h = re.sub(r"<[^>]+>", "\n", h)
    for a, b in [
        ("&nbsp;", " "),
        ("&middot;", "·"),
        ("&aacute;", "á"),
        ("&eacute;", "é"),
        ("&iacute;", "í"),
        ("&oacute;", "ó"),
        ("&uacute;", "ú"),
        ("&ntilde;", "ñ"),
    ]:
        h = h.replace(a, b)
    return [l.strip() for l in h.split("\n") if l.strip()]  # noqa: E741


def test_el_informe_en_english_no_arrastra_frases_en_español(tmp_path):
    """Barrido ancho, no por sondas sueltas.

    Both documents are rendered and their texts compared: a line that appears
    identically in both and reads Spanish is a string that never reached the
    translation layer. This is how the last ~100 were found, and without this
    test they come back on the first edit.
    """
    import re

    es = set(_texto_visible(_informe(tmp_path, "es")))
    en = set(_texto_visible(_informe(tmp_path, "en")))
    marca = re.compile(
        r"[áéíóúñÁÉÍÓÚÑ¿¡]|\b(el|la|los|las|del|que|con|para|"
        r"por|una|un|se|es|su|sus|como|cada|sobre|entre|sin|no|"
        r"clase|fibras|muestra|menor|mayor|mediana|verdaderos|"
        r"falsos|clasificados|imagen)\b",
        re.IGNORECASE,
    )
    # Los apellidos de la bibliografia son nombres propios: no se traducen.
    sospechosas = [t for t in es & en if marca.search(t) and len(t) > 2 and "(2024)" not in t]
    assert not sospechosas, "texto sin traducir en el informe en ingles:\n  " + "\n  ".join(
        sorted(sospechosas)[:15]
    )


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

    import polyx.core.measure as measure
    from polyx.core.detection_run import ImageResult, ModelInfo, RunParams, RunResult

    clases = ["PET", "PP", "LDPE"]
    results: dict[int, list[ImageResult]] = {0: [], 1: []}

    n = 0
    for folder in ("Chiu Chiu", "Desembocadura"):
        for photo in range(3):
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

            path = tmp_path / folder / f"{photo + 1}.1.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, buffer = cv2.imencode(".png", img)
            assert ok
            path.write_bytes(buffer.tobytes())

            scale = 30.0 + 4.0 * photo  # each photo gets its own scale
            for mi in (0, 1):
                preds, gt = [], []
                for k, (x1, y1, x2, y2) in enumerate(cajas):
                    cid = k % 3
                    d = Detection(
                        class_id=cid,
                        class_name=clases[cid],
                        conf=0.6 + 0.05 * k,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                    measure.apply_to_detection(d, img, scale)
                    n += 1
                    d.number = n
                    # El segundo modelo se salta una caja: si los dos dieran lo
                    # mismo, la comparacion no tendria nada que enseñar.
                    if not (mi == 1 and k == 0):
                        preds.append(d)
                    # Y el ground truth confunde una clase, para que la matriz
                    # de confusion tenga algo fuera de la diagonal.
                    cid_gt = (cid + 1) % 3 if k == 2 else cid
                    gt.append(
                        Detection(
                            class_id=cid_gt,
                            class_name=clases[cid_gt],
                            conf=1.0,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    )
                results[mi].append(
                    ImageResult(
                        image_path=path, model_idx=mi, predictions=preds, gt=gt, has_gt=True
                    )
                )
    return RunResult(
        run_dir=tmp_path / "run",
        models=[
            ModelInfo(alias="yolov8n", path=Path("a.pt")),
            ModelInfo(alias="yolo11n", path=Path("b.pt")),
        ],
        params=RunParams(),
        results=results,
    )


def _informe_completo(tmp_path, lang):
    """Render the complete batch in the requested language and return its HTML."""
    import polyx.core.i18n as i18n
    import polyx.gui.report_html as rh

    assert i18n.set_language(lang)
    salida = tmp_path / f"completo_{lang}.html"
    rh.generate_report(_estado_completo(tmp_path), salida, sections=list(rh.IDS_SECTIONS))
    return salida.read_text(encoding="utf-8")


@pytest.mark.parametrize("lang", ["es", "en"])
def test_un_lote_con_todas_las_secciones_se_genera(tmp_path, lang):
    """Que ninguna seccion reviente al construirse, en ninguno de los dos
    idiomas. Es la prueba que habria cazado el mas unario."""
    html = _informe_completo(tmp_path, lang)
    assert html.count("<h2") >= 10, "faltan secciones por construir"
    assert "@@N@@" not in html
    assert "%(" not in html


def test_en_el_lote_completo_tampoco_se_cuela_español(tmp_path):
    """El mismo barrido de antes, pero sobre las secciones que el lote minimo
    no llega a construir: comparacion, errores, galeria y talla por carpeta."""
    import re

    es = set(_texto_visible(_informe_completo(tmp_path, "es")))
    en = set(_texto_visible(_informe_completo(tmp_path, "en")))
    marca = re.compile(
        r"[áéíóúñÁÉÍÓÚÑ¿¡]|\b(el|la|los|las|del|que|con|para|"
        r"por|una|un|se|es|su|sus|como|cada|sobre|entre|sin|no|"
        r"clase|fibras|muestra|menor|mayor|mediana|verdaderos|"
        r"falsos|clasificados|imagen)\b",
        re.IGNORECASE,
    )
    sospechosas = [t for t in es & en if marca.search(t) and len(t) > 2 and "(2024)" not in t]
    assert not sospechosas, "texto sin traducir en el informe en ingles:\n  " + "\n  ".join(
        sorted(sospechosas)[:15]
    )
