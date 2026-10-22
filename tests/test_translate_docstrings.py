"""Tests for the LibCST docstring codemod scaffolding (M006/S02/T01).

The driver under ``tools/translate_docstrings.py`` is loaded by file path
because ``tools/`` is not a package. Fixtures are inline sources written
into ``tmp_path``: no git-ignored or machine-local paths are involved.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_RUTA_DRIVER = (
    Path(__file__).resolve().parent.parent / "tools" / "translate_docstrings.py"
)
_spec = importlib.util.spec_from_file_location("translate_docstrings", _RUTA_DRIVER)
td = importlib.util.module_from_spec(_spec)
# registrar antes de ejecutar: @dataclass consulta sys.modules por __module__
sys.modules["translate_docstrings"] = td
_spec.loader.exec_module(td)

# Código de prueba: simula un módulo polyx con docstrings en español.
# Los comentarios y las cadenas NO docstring deben sobrevivir intactos.
CODIGO_ES = '''\
"""Módulo de prueba para el codemod."""

# comentario en español que debe sobrevivir intacto
VARIABLE_PUBLICA = "cadena española que no es docstring"


class Cosa:
    """Docstring de clase, en español."""

    def metodo(self, valor):
        """Docstring de función, en español."""
        texto_interno = "otra cadena española, tampoco docstring"
        return f"{valor}: {texto_interno}"
'''

# índices en orden de documento: módulo=0, Cosa=1, Cosa.metodo=2
TRADUCCIONES = {
    0: "Test module for the codemod.",
    1: "Class docstring, in English.",
    2: "Function docstring, in English.",
}


def _escribir(tmp_path: Path, contenido: str = CODIGO_ES) -> Path:
    ruta = tmp_path / "modulo_prueba.py"
    ruta.write_text(contenido, encoding="utf-8")
    return ruta


def test_traduce_docstrings_de_modulo_clase_y_funcion(tmp_path):
    ruta = _escribir(tmp_path)
    td.procesar_archivo(ruta, TRADUCCIONES, aplicar=True)
    resultado = ruta.read_text(encoding="utf-8")
    assert '"""Test module for the codemod."""' in resultado
    assert '"""Class docstring, in English."""' in resultado
    assert '"""Function docstring, in English."""' in resultado
    assert "Módulo de prueba" not in resultado


def test_no_docstring_strings_and_comments_untouched(tmp_path):
    ruta = _escribir(tmp_path)
    td.procesar_archivo(ruta, TRADUCCIONES, aplicar=True)
    resultado = ruta.read_text(encoding="utf-8")
    # cadenas de código (no docstring) y comentarios quedan tal cual
    assert 'VARIABLE_PUBLICA = "cadena española que no es docstring"' in resultado
    assert 'texto_interno = "otra cadena española, tampoco docstring"' in resultado
    assert "# comentario en español que debe sobrevivir intacto" in resultado
    # el f-string interno tampoco se toca
    assert 'return f"{valor}: {texto_interno}"' in resultado


def test_dry_run_no_escribe_nada(tmp_path):
    ruta = _escribir(tmp_path)
    original = ruta.read_text(encoding="utf-8")
    cambios = td.procesar_archivo(ruta, TRADUCCIONES, aplicar=False)
    assert len(cambios) == 3
    assert ruta.read_text(encoding="utf-8") == original


def test_idempotencia_segunda_pasada_sin_cambios(tmp_path):
    ruta = _escribir(tmp_path)
    td.procesar_archivo(ruta, TRADUCCIONES, aplicar=True)
    tras_primera = ruta.read_text(encoding="utf-8")
    cambios_segunda = td.procesar_archivo(ruta, TRADUCCIONES, aplicar=True)
    assert cambios_segunda == []
    assert ruta.read_text(encoding="utf-8") == tras_primera


def test_indices_sin_traduccion_quedan_intactos(tmp_path):
    ruta = _escribir(tmp_path)
    td.procesar_archivo(ruta, {1: "Only the class docstring changes."}, aplicar=True)
    resultado = ruta.read_text(encoding="utf-8")
    assert '"""Módulo de prueba para el codemod."""' in resultado
    assert '"""Only the class docstring changes."""' in resultado
    assert '"""Docstring de función, en español."""' in resultado


def test_enumera_docstrings_en_orden_de_documento():
    _, cambios, enumeracion = td.traducir_texto(CODIGO_ES, TRADUCCIONES)
    assert [par[0] for par in enumeracion] == [0, 1, 2]
    assert "Módulo de prueba" in enumeracion[0][1]
    assert "Docstring de clase" in enumeracion[1][1]
    assert "Docstring de función" in enumeracion[2][1]
    assert [c.indice for c in cambios] == [0, 1, 2]


def test_cli_dry_run_y_apply(tmp_path, capsys):
    ruta = _escribir(tmp_path)
    original = ruta.read_text(encoding="utf-8")
    # sin manifiesto no hay traducciones: cero cambios es lo correcto
    assert td.main(["--dry-run", str(ruta)]) == 0
    assert ruta.read_text(encoding="utf-8") == original
    assert "0 cambio(s) detectados" in capsys.readouterr().out
    # con manifiesto: informa los 3 cambios y no escribe nada en dry-run
    manifiesto = tmp_path / "manifiesto.py"
    manifiesto.write_text(
        f"DOCSTRING_TRANSLATIONS = {{ {ruta.as_posix()!r}: "
        f"{{0: 'CLI test module.', 1: 'CLI class docstring.', "
        f"2: 'CLI function docstring.'}} }}\n",
        encoding="utf-8",
    )
    argv = ["--manifest", str(manifiesto), str(ruta)]
    assert td.main(["--dry-run", *argv]) == 0
    assert "3 cambio(s) detectados" in capsys.readouterr().out
    assert ruta.read_text(encoding="utf-8") == original
    assert td.main(["--apply", *argv]) == 0
    assert '"""CLI test module."""' in ruta.read_text(encoding="utf-8")
    # segunda pasada con --apply: idempotente, cero cambios
    assert td.main(["--apply", *argv]) == 0
    assert "0 cambio(s) aplicados" in capsys.readouterr().out


def test_archivo_con_error_de_sintaxis_reporta_y_falla(tmp_path, capsys):
    ruta = tmp_path / "roto.py"
    ruta.write_text("def f(:\n", encoding="utf-8")
    with pytest.raises(td.cst.ParserSyntaxError):
        td.procesar_archivo(ruta, {0: "x"}, aplicar=False)
    assert td.main(["--dry-run", str(ruta)]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_contenido_que_rompe_comillas_de_una_linea_falla(tmp_path):
    ruta = _escribir(tmp_path, 'def f():\n    "doc"\n')
    # índice 1 = la función f (el índice 0 es el módulo, sin docstring)
    with pytest.raises(ValueError):
        td.procesar_archivo(ruta, {1: 'contenido con " dentro'}, aplicar=True)
