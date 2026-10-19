#!/usr/bin/env python3
"""LibCST codemod scaffolding to translate polyx docstrings to English.

Milestone M006, slice S02, task T01. The driver walks Python files,
enumerates every module/class/function docstring in document order
(pre-order traversal) and rewrites only the STRING tokens selected by a
translation manifest. Because LibCST round-trips the module, code,
comments, identifiers, blank lines and even original line endings
outside the touched tokens stay byte-identical.

CLI:
    python3 tools/translate_docstrings.py --dry-run polyx/model
    python3 tools/translate_docstrings.py --apply \
        --manifest <manifest_module> polyx/model

A manifest is a Python module defining ``DOCSTRING_TRANSLATIONS``:
``{file_path: {docstring_index: new_content}}`` with repository-relative
posix paths and 0-based indices in document order. ``new_content`` is
the docstring text WITHOUT quotes; the driver preserves the original
prefix and quote style of the token. Re-running ``--apply`` with the
same manifest is a no-op once the docstrings already match (idempotency).

This scaffolding ships no translation content; later S02 tasks add the
manifests and the check/audit modes on top of this driver.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import libcst as cst


@dataclass
class CambioDocstring:
    """One docstring rewrite (applied, or would-be under ``--dry-run``)."""

    indice: int
    antes: str  # first line of the old token, for the report
    despues: str  # first line of the new token


def _stmts_de(nodo: cst.CSTNode) -> Optional[Sequence[cst.BaseStatement]]:
    """Devuelve la secuencia de statements del cuerpo de un scope."""
    # Module ya expone .body como secuencia; ClassDef/FunctionDef usan bloques
    cuerpo = nodo.body
    if isinstance(cuerpo, (cst.IndentedBlock, cst.SimpleStatementSuite)):
        return cuerpo.body
    return cuerpo


def _docstring_de(stmts: Optional[Sequence[cst.BaseStatement]]) -> Optional[cst.SimpleString]:
    """Devuelve el nodo SimpleString si el primer statement es un docstring."""
    if not stmts:
        return None
    primero = stmts[0]
    if isinstance(primero, cst.SimpleStatementLine):
        # un docstring es una expresión suelta, única en su línea
        if len(primero.body) != 1:
            return None
        expr = primero.body[0]
    elif isinstance(primero, cst.Expr):
        # cuerpo en una sola línea: class X: "doc"
        expr = primero
    else:
        return None
    if not isinstance(expr, cst.Expr):
        # un import/assign/otra sentencia pequeña no es docstring
        return None
    if isinstance(expr.value, cst.SimpleString):
        return expr.value
    return None


def _separar_token(token: str) -> Tuple[str, str, str]:
    """Separa un token de cadena en (prefijo, comillas, contenido)."""
    # todo lo anterior a la primera comilla es el prefijo (r, b, f, rb, ...)
    i = 0
    while i < len(token) and token[i] not in "\"'":
        i += 1
    prefix, resto = token[:i], token[i:]
    if resto.startswith('"""') or resto.startswith("'''"):
        comillas = resto[:3]
    else:
        comillas = resto[:1]
    contenido = resto[len(comillas) : len(resto) - len(comillas)]
    return prefix, comillas, contenido


def _reconstruir_token(token_viejo: str, nuevo_contenido: str) -> str:
    """Reconstruye un token de cadena con contenido nuevo, mismo prefijo y comillas."""
    prefix, comillas, _ = _separar_token(token_viejo)
    if len(comillas) == 1:
        # en docstrings de una línea el contenido no puede romper las comillas
        if "\n" in nuevo_contenido or comillas in nuevo_contenido:
            raise ValueError(
                "el contenido traducido no cabe en un docstring de una sola línea"
            )
    return f"{prefix}{comillas}{nuevo_contenido}{comillas}"


class TraductorDocstrings(cst.CSTTransformer):
    """Translate the docstrings selected by an ``index -> content`` mapping.

    Indices are assigned in ``visit_*`` (pre-order, i.e. document order)
    while the rewrites happen in the ``leave_*`` overrides, the only
    place LibCST allows node replacement. Only Module, ClassDef and
    FunctionDef docstrings are touched: no other node type is rewritten,
    so nothing outside docstrings can change.
    """

    def __init__(self, traducciones: Dict[int, str]):
        self.traducciones = traducciones
        self.indice = 0
        self.cambios: List[CambioDocstring] = []
        # (índice, primera línea del token) para el inventario/listado
        self.enumeracion: List[Tuple[int, str]] = []
        # id(nodo) -> índice, asignado en visit y consultado en leave
        self._indice_por_id: Dict[int, int] = {}

    # --- numeración en pre-order (orden de documento) -----------------
    def _numerar(self, nodo: cst.CSTNode) -> None:
        self._indice_por_id[id(nodo)] = self.indice
        self.indice += 1

    def visit_Module(self, nodo: cst.Module) -> None:
        self._numerar(nodo)

    def visit_ClassDef(self, nodo: cst.ClassDef) -> None:
        self._numerar(nodo)

    def visit_FunctionDef(self, nodo: cst.FunctionDef) -> None:
        self._numerar(nodo)

    # --- reescritura en leave -----------------------------------------
    def leave_Module(self, nodo_original, nodo_actualizado):
        return self._revisar(nodo_original, nodo_actualizado)

    def leave_ClassDef(self, nodo_original, nodo_actualizado):
        return self._revisar(nodo_original, nodo_actualizado)

    def leave_FunctionDef(self, nodo_original, nodo_actualizado):
        return self._revisar(nodo_original, nodo_actualizado)

    def _revisar(self, nodo_original, nodo_actualizado):
        indice = self._indice_por_id.get(id(nodo_original))
        if indice is None:
            # defensivo: todo scope visitado debe tener índice asignado
            return nodo_actualizado
        string_node = _docstring_de(_stmts_de(nodo_actualizado))
        if string_node is None:
            self.enumeracion.append((indice, ""))
            return nodo_actualizado
        token_viejo = string_node.value
        primera_linea = token_viejo.splitlines()[0] if token_viejo else ""
        self.enumeracion.append((indice, primera_linea))
        if indice not in self.traducciones:
            return nodo_actualizado
        _, _, contenido = _separar_token(token_viejo)
        nuevo_contenido = self.traducciones[indice]
        if contenido == nuevo_contenido:
            # ya aplicado: segunda pasada sin cambios (idempotencia)
            return nodo_actualizado
        token_nuevo = _reconstruir_token(token_viejo, nuevo_contenido)
        self.cambios.append(
            CambioDocstring(
                indice=indice,
                antes=primera_linea,
                despues=token_nuevo.splitlines()[0] if token_nuevo else "",
            )
        )
        return self._con_docstring_reemplazado(nodo_actualizado, string_node, token_nuevo)

    def _con_docstring_reemplazado(
        self, nodo, string_node: cst.SimpleString, token_nuevo: str
    ):
        """Devuelve el nodo con su docstring (primer statement) reemplazado."""
        nuevo_string = string_node.with_changes(value=token_nuevo)
        stmts = list(_stmts_de(nodo))
        primero = stmts[0]
        if isinstance(primero, cst.SimpleStatementLine):
            nuevo_primero = primero.with_changes(
                body=[primero.body[0].with_changes(value=nuevo_string)]
            )
        else:  # expresión suelta en cuerpo de una línea
            nuevo_primero = primero.with_changes(value=nuevo_string)
        stmts[0] = nuevo_primero
        if isinstance(nodo, cst.Module):
            return nodo.with_changes(body=stmts)
        return nodo.with_changes(body=nodo.body.with_changes(body=stmts))


def traducir_texto(
    texto: str, traducciones: Dict[int, str]
) -> Tuple[str, List[CambioDocstring], List[Tuple[int, str]]]:
    """Aplica traducciones de docstrings a un módulo.

    Devuelve ``(nuevo_texto, cambios, enumeracion)`` donde la enumeración
    lista ``(índice, primera línea)`` de cada docstring en orden de documento.
    """
    modulo = cst.parse_module(texto)
    trafo = TraductorDocstrings(traducciones)
    nuevo = modulo.visit(trafo)
    # leave recorre en post-orden: se ordena por índice para el reporte
    cambios = sorted(trafo.cambios, key=lambda cambio: cambio.indice)
    return nuevo.code, cambios, sorted(trafo.enumeracion, key=lambda par: par[0])


def _archivos_py(rutas: Sequence[Path]) -> List[Path]:
    """Expande rutas (archivos o directorios) en una lista ordenada de .py."""
    archivos: List[Path] = []
    for ruta in rutas:
        if ruta.is_dir():
            archivos.extend(sorted(ruta.rglob("*.py")))
        else:
            archivos.append(ruta)
    return archivos


def procesar_archivo(
    ruta: Path, traducciones: Dict[int, str], aplicar: bool
) -> List[CambioDocstring]:
    """Procesa un archivo; con ``aplicar`` reescribe en sitio, si no solo informa."""
    # newline="" preserva los fin de línea originales (CRLF incluidos) byte a byte
    with ruta.open("r", encoding="utf-8", newline="") as manejador:
        texto = manejador.read()
    nuevo_texto, cambios, _ = traducir_texto(texto, traducciones)
    if aplicar and cambios:
        with ruta.open("w", encoding="utf-8", newline="") as manejador:
            manejador.write(nuevo_texto)
    return cambios


def cargar_manifest(ruta_manifest: Path) -> Dict[str, Dict[int, str]]:
    """Carga un módulo de manifiesto que define DOCSTRING_TRANSLATIONS."""
    spec = importlib.util.spec_from_file_location(ruta_manifest.stem, ruta_manifest)
    if spec is None or spec.loader is None:
        raise SystemExit(f"no se pudo cargar el manifiesto: {ruta_manifest}")
    modulo = importlib.util.module_from_spec(spec)
    # registrar antes de ejecutar: los decoradores consultan sys.modules
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    traducciones = getattr(modulo, "DOCSTRING_TRANSLATIONS", None)
    if not isinstance(traducciones, dict):
        raise SystemExit(
            f"el manifiesto {ruta_manifest} no define DOCSTRING_TRANSLATIONS"
        )
    return traducciones


def main(argv: Optional[List[str]] = None) -> int:
    """Punto de entrada del CLI del codemod."""
    parser = argparse.ArgumentParser(
        description="Traduce docstrings al inglés mediante LibCST (M006/S02)."
    )
    parser.add_argument(
        "rutas", nargs="+", type=Path, help="archivos o directorios a procesar"
    )
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument(
        "--dry-run", action="store_true", help="muestra los cambios sin escribir nada"
    )
    modo.add_argument(
        "--apply", action="store_true", help="reescribe los archivos en sitio"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="módulo Python que define DOCSTRING_TRANSLATIONS",
    )
    args = parser.parse_args(argv)

    por_archivo: Dict[str, Dict[int, str]] = {}
    if args.manifest is not None:
        por_archivo = cargar_manifest(args.manifest)

    total = 0
    for ruta in _archivos_py(args.rutas):
        clave = ruta.as_posix()
        try:
            cambios = procesar_archivo(ruta, por_archivo.get(clave, {}), args.apply)
        except (OSError, UnicodeDecodeError, ValueError, cst.ParserSyntaxError) as error:
            # falla de E/S, decodificación, traducción inválida o sintaxis:
            # se reporta y el CLI termina en error
            print(f"ERROR en {clave}: {error}", file=sys.stderr)
            return 1
        for cambio in cambios:
            print(f"{clave} #{cambio.indice}: {cambio.antes!r} -> {cambio.despues!r}")
        total += len(cambios)
    verbo = "aplicados" if args.apply else "detectados"
    print(f"{total} cambio(s) {verbo}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
