"""Tests for the LibCST docstring harness (M006/S02).

Covers ``tests/translate_code_text.py`` (the list/check/apply/audit harness
that S03 and S04 will reuse) on top of the T01 transformer core in
``tools/translate_docstrings.py``. Fixtures are inline sources written into
``tmp_path``; the audit tests build a throwaway git repository there.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_RUTA_HARNESS = Path(__file__).resolve().parent / "translate_code_text.py"
_spec = importlib.util.spec_from_file_location("translate_code_text", _RUTA_HARNESS)
tct = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("translate_code_text", tct)
_spec.loader.exec_module(tct)

# Módulo de prueba: docstrings en español y código que debe sobrevivir intacto.
CODIGO_ES = '''\
"""Módulo de prueba del harness."""

# comentario en español que debe sobrevivir intacto
VARIABLE = "cadena de código que no es docstring"


class Cosa:
    """Docstring de clase, en español."""

    def metodo(self):
        """Docstring de método, en español."""
        return VARIABLE
'''

TRADUCCIONES = {
    0: "Test module for the harness.",
    1: "Class docstring, in English.",
    2: "Function docstring, in English.",
}


def _escribir(tmp_path: Path, contenido: str = CODIGO_ES) -> Path:
    ruta = tmp_path / "modulo_prueba.py"
    ruta.write_text(contenido, encoding="utf-8")
    return ruta


def _manifiesto(tmp_path: Path, ruta: Path) -> Path:
    manifiesto = tmp_path / "manifiesto.py"
    pares = ", ".join(f"{i}: {t!r}" for i, t in TRADUCCIONES.items())
    manifiesto.write_text(
        f"DOCSTRING_TRANSLATIONS = {{ {ruta.as_posix()!r}: {{ {pares} }} }}\n",
        encoding="utf-8",
    )
    return manifiesto


def test_list_enumerates_in_document_order(tmp_path, capsys):
    ruta = _escribir(tmp_path)
    assert tct.main(["list", "--json", str(ruta)]) == 0
    entradas = json.loads(capsys.readouterr().out)
    assert [(e["index"], e["file"]) for e in entradas] == [
        (0, ruta.as_posix()),
        (1, ruta.as_posix()),
        (2, ruta.as_posix()),
    ]
    assert "Módulo de prueba" in entradas[0]["firstline"]
    assert "Docstring de clase" in entradas[1]["firstline"]
    assert "Docstring de método" in entradas[2]["firstline"]


def test_detector_flags_spanish_and_spares_backticked_literals():
    """Anti-vacuity: the detector sees Spanish; backticked literals pass."""
    sintetica = "La detección de partículas genera informes del entrenamiento."
    hallazgos = tct.hallazgos_espanoles(sintetica)
    assert hallazgos, "the detector failed to flag a synthetic Spanish sentence"
    assert any(f.startswith("spanish-word:") for f in hallazgos)

    con_codigo = "Reports `«no asignable»` as one more category."
    assert tct.hallazgos_espanoles(tct._quitar_inline_code(con_codigo)) == []


def test_check_exits_1_on_spanish_and_0_on_clean(tmp_path):
    ruta = _escribir(tmp_path)
    assert tct.main(["check", str(ruta)]) == 1
    limpio = CODIGO_ES.replace(
        '"""Módulo de prueba del harness."""', '"""Test module for the harness."""'
    ).replace("Docstring de clase, en español.", "Class docstring, in English.").replace(
        "Docstring de método, en español.", "Function docstring, in English."
    )
    ruta_limpia = _escribir(tmp_path, limpio)
    assert tct.main(["check", str(ruta_limpia)]) == 0


def test_apply_rewrites_docstrings_and_is_idempotent(tmp_path, capsys):
    ruta = _escribir(tmp_path)
    manifiesto = _manifiesto(tmp_path, ruta)
    assert tct.main(["apply", "--manifest", str(manifiesto)]) == 0
    resultado = ruta.read_text(encoding="utf-8")
    assert '"""Test module for the harness."""' in resultado
    assert '"""Class docstring, in English."""' in resultado
    assert '"""Function docstring, in English."""' in resultado
    # nada fuera de los docstrings cambia: comentario, código y cadenas quedan
    assert "# comentario en español que debe sobrevivir intacto" in resultado
    assert 'VARIABLE = "cadena de código que no es docstring"' in resultado
    capsys.readouterr()
    assert tct.main(["apply", "--manifest", str(manifiesto)]) == 0
    assert "0 cambio(s) aplicados." in capsys.readouterr().out


def test_audit_passes_after_codemod(tmp_path, monkeypatch):
    """Token audit vs git HEAD: a pure docstring codemod leaves zero problems."""
    ruta = _escribir(tmp_path)
    _init_repo(tmp_path, ruta)
    manifiesto = _manifiesto(tmp_path, ruta)
    monkeypatch.chdir(tmp_path)
    assert tct.main(["apply", "--manifest", str(manifiesto)]) == 0
    assert tct.main(["audit", str(ruta)]) == 0


def test_audit_flags_non_docstring_corruption(tmp_path, monkeypatch, capsys):
    """Negative test (must-have 1): code churn outside docstrings is flagged."""
    ruta = _escribir(tmp_path)
    _init_repo(tmp_path, ruta)
    manifiesto = _manifiesto(tmp_path, ruta)
    monkeypatch.chdir(tmp_path)
    tct.main(["apply", "--manifest", str(manifiesto)])
    # corrupción de código real: la variable y su cadena cambian
    roto = ruta.read_text(encoding="utf-8").replace(
        'VARIABLE = "cadena de código que no es docstring"',
        'OTRA = "cadena distinta"',
    )
    ruta.write_text(roto, encoding="utf-8")
    capsys.readouterr()
    assert tct.main(["audit", str(ruta)]) == 1
    assert "token no-docstring cambia" in capsys.readouterr().err


def test_audit_flags_new_string_token_not_in_docstrings(tmp_path, monkeypatch):
    ruta = _escribir(tmp_path)
    _init_repo(tmp_path, ruta)
    monkeypatch.chdir(tmp_path)
    # una cadena de código nueva pero con el mismo TOKEN de un docstring
    # viejo: el tipo coincide y el valor estaba en docstrings, así que se
    # comprueba también que el token sea el MISMO token (por posición de
    # token), no solo el mismo valor
    roto = ruta.read_text(encoding="utf-8").replace(
        "return VARIABLE", 'return """Docstring de clase, en español."""'
    )
    ruta.write_text(roto, encoding="utf-8")
    assert tct.main(["audit", str(ruta)]) == 1


def _init_repo(tmp_path: Path, ruta: Path) -> None:
    """Crea un repo de juguete con el archivo ya confirmado en HEAD."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", ruta.name],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
    )


@pytest.mark.parametrize(
    "identificador",
    ["empaquetar_para_llevar.bat", "mi_variable_larga", "plantilla_v2"],
)
def test_word_boundaries_do_not_fire_inside_identifiers(identificador):
    """\b does not fire inside underscore tokens, so identifiers pass."""
    assert tct.hallazgos_espanoles(identificador) == []
