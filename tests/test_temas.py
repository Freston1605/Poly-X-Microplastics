"""Pruebas del sistema de temas.

Cada una fija una decision de diseno, no solo un comportamiento: si alguien
anade un tema bonito pero ilegible, o vuelve a usar un color de fondo como
color de texto, la suite lo dice antes de que llegue a una captura del manual.
"""
from __future__ import annotations

import json

import pytest

from polyx.gui import theme as T


# ────────────────────────────────────────────────────────────────────
#  Contraste
# ────────────────────────────────────────────────────────────────────
def _luminancia(hex_color: str) -> float:
    """Luminancia relativa segun WCAG 2.1."""
    h = hex_color.lstrip("#")
    canales = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lineal = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
              for c in canales]
    return 0.2126 * lineal[0] + 0.7152 * lineal[1] + 0.0722 * lineal[2]


def contraste(a: str, b: str) -> float:
    la, lb = _luminancia(a), _luminancia(b)
    claro, oscuro = max(la, lb), min(la, lb)
    return (claro + 0.05) / (oscuro + 0.05)


#: Pares que llevan texto encima y por tanto deben cumplir 4.5:1 (WCAG AA).
PARES_DE_TEXTO = [
    ("INK", "BG"), ("INK", "BG_SOFT"),
    ("INK2", "BG"), ("INK2", "BG_SOFT"),
    ("INK3", "BG"), ("INK3", "BG_SOFT"),
    ("ACCENT_TX", "BG"), ("ACCENT_D_TX", "BG"),
    ("OK_TX", "BG"), ("WARN_TX", "BG"), ("ERR_TX", "BG"), ("VIO_TX", "BG"),
    ("TIP_FG", "TIP_BG"),
    ("ON_ACCENT", "ACCENT"), ("ON_ACCENT", "ACCENT_D"),
]


@pytest.mark.parametrize("tema", sorted(T.PALETTES))
@pytest.mark.parametrize("frente,fondo", PARES_DE_TEXTO)
def test_texto_cumple_wcag_aa(tema: str, frente: str, fondo: str) -> None:
    """Todo texto de la interfaz llega a 4.5:1 en los cuatro temas.

    Es la razon de que cada estado tenga dos variantes: el tono que da 5:1 con
    blanco encima queda en 3.7:1 leido como texto sobre fondo oscuro.
    """
    p = T.PALETTES[tema]
    razon = contraste(p[frente], p[fondo])
    assert razon >= 4.5, (
        f"{tema}: {frente} sobre {fondo} da {razon:.2f}:1, por debajo de 4.5:1")


# ────────────────────────────────────────────────────────────────────
#  Integridad de las paletas
# ────────────────────────────────────────────────────────────────────
def test_todas_las_paletas_tienen_los_mismos_tokens() -> None:
    """Un token que falte en un tema sale como AttributeError al abrir la ventana.

    Se comprueba aqui y no en tiempo de ejecucion porque el fallo aparece en un
    `paintEvent` cualquiera, lejos del tema que lo causo.
    """
    referencia = set(T.PALETTES[T.DEFAULT_THEME])
    for nombre, tokens in T.PALETTES.items():
        faltan = referencia - set(tokens)
        sobran = set(tokens) - referencia
        assert not faltan, f"a '{nombre}' le faltan tokens: {sorted(faltan)}"
        assert not sobran, f"'{nombre}' tiene tokens de mas: {sorted(sobran)}"


def test_cada_tema_tiene_nombre_visible() -> None:
    assert set(T.NAMES_THEME) == set(T.PALETTES)


def test_los_colores_son_hex_de_seis_digitos() -> None:
    for nombre, tokens in T.PALETTES.items():
        for token, valor in tokens.items():
            assert valor.startswith("#") and len(valor) == 7, (
                f"{nombre}.{token} = {valor!r}: se espera #rrggbb")
            int(valor[1:], 16)


# ────────────────────────────────────────────────────────────────────
#  Cambio de tema
# ────────────────────────────────────────────────────────────────────
def test_cambiar_tema_republica_tokens_y_rehace_el_qss() -> None:
    """El QSS global se reconstruye: no puede quedarse con la paleta anterior.

    Fue una f-string a nivel de modulo, que se evalua una sola vez. Con temas,
    eso dejaba la hoja global con los colores del tema de arranque mientras el
    resto de la interfaz ya usaba los nuevos.
    """
    original = T.theme()
    try:
        T.set_theme("claro")
        qss_claro, ink_claro = T.GLOBAL_QSS, T.INK
        T.set_theme("oscuro")
        assert ink_claro != T.INK
        assert qss_claro != T.GLOBAL_QSS
        assert T.INK in T.GLOBAL_QSS
    finally:
        T.set_theme(original)


def test_tema_desconocido_no_hace_nada() -> None:
    original = T.theme()
    T.set_theme("no-existe")
    assert T.theme() == original


def test_es_oscuro_coincide_con_la_declaracion() -> None:
    original = T.theme()
    try:
        for nombre in T.PALETTES:
            T.set_theme(nombre)
            assert T.is_dark() == (nombre in T.DARK)
    finally:
        T.set_theme(original)


# ────────────────────────────────────────────────────────────────────
#  Lo que no debe seguir al tema
# ────────────────────────────────────────────────────────────────────
def test_los_informes_no_siguen_el_tema() -> None:
    """`T.DOC` se queda en la paleta clara pase lo que pase.

    Un informe se imprime y se archiva: si cambiara con la preferencia de quien
    lo genero, dos informes del mismo analisis no se parecerian.
    """
    original = T.theme()
    try:
        antes = (T.DOC.BG, T.DOC.INK, T.DOC.ACCENT)
        for nombre in T.PALETTES:
            T.set_theme(nombre)
            assert antes == (T.DOC.BG, T.DOC.INK, T.DOC.ACCENT)
        assert T.PALETTES["claro"]["BG"] == T.DOC.BG
    finally:
        T.set_theme(original)


def test_los_colores_de_polimero_no_siguen_el_tema() -> None:
    """PET/PP/LDPE son datos del dominio, no decoracion.

    Si cambiaran de tono con el tema, dos capturas del mismo analisis dejarian
    de ser comparables.
    """
    original = T.theme()
    try:
        esperado = dict(T.CLASS_COLOR_HEX)
        for nombre in T.PALETTES:
            T.set_theme(nombre)
            assert esperado == T.CLASS_COLOR_HEX
    finally:
        T.set_theme(original)


# ────────────────────────────────────────────────────────────────────
#  Persistencia
# ────────────────────────────────────────────────────────────────────
def test_se_guarda_tema_y_animaciones_juntos(tmp_path, monkeypatch) -> None:
    """Las dos preferencias comparten archivo: son la misma decision."""
    destino = tmp_path / "prefs.json"
    monkeypatch.setattr(T, "_FILE", destino)
    original = T.theme()
    try:
        T.set_theme("azul")
        T.set_animations(False)
        guardado = json.loads(destino.read_text(encoding="utf-8"))
        assert guardado == {"tema": "azul", "animaciones": False}
    finally:
        T.set_theme(original)
        T.set_animations(True)


def test_utilidades_de_color() -> None:
    assert T.mingle("#000000", "#ffffff", 0.0) == "#000000"
    assert T.mingle("#000000", "#ffffff", 1.0) == "#ffffff"
    assert T.mingle("#000000", "#ffffff", 0.5) == "#808080"
    assert T.with_alpha("#ff0000", 128).alpha() == 128
    # sobre_fondo mezcla contra BG, no contra blanco: en un tema oscuro mezclar
    # hacia blanco daria un pastel que no pertenece a la paleta.
    original = T.theme()
    try:
        T.set_theme("oscuro")
        tenue = T.on_background(T.ACCENT, 0.10)
        assert contraste(tenue, T.BG) < 1.6, "el fondo tenue debe quedar cerca de BG"
    finally:
        T.set_theme(original)
