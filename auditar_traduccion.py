"""Dice que cadenas de la interfaz no tienen traduccion al ingles.

Recorre el arbol sintactico de cada modulo buscando llamadas a ``tr()`` con una
cadena literal, y las contrasta con el diccionario. Se hace con AST y no con una
expresion regular porque muchas cadenas van partidas en varios trozos entre
parentesis, y una regular las cortaria por la mitad.

Uso:
    python auditar_traduccion.py           # resumen
    python auditar_traduccion.py --listar  # ademas, las cadenas que faltan
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from polyx.model.traducciones_en import EN, NO_TRADUCIR  # noqa: E402


def cadenas_tr(archivo: Path) -> list[str]:
    """Todas las cadenas literales que pasan por tr() en ese archivo."""
    try:
        arbol = ast.parse(archivo.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    out = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        f = nodo.func
        nombre = f.id if isinstance(f, ast.Name) else (
            f.attr if isinstance(f, ast.Attribute) else "")
        if nombre != "tr" or not nodo.args:
            continue
        arg = nodo.args[0]
        # Solo literales: tr(variable) no se puede auditar en estatico, y
        # tr(f"...{x}") tampoco sirve como clave porque cambia en cada llamada.
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append(arg.value)
    return out


def main():
    # La consola de Windows va en cp1252 y muchas cadenas llevan flechas o
    # emoji; sin esto el auditor muere al imprimirlas en vez de auditar.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true",
                    help="imprime las cadenas sin traducir")
    args = ap.parse_args()

    faltan: dict[str, list[Path]] = {}
    total = 0
    for py in sorted((RAIZ / "polyx").rglob("*.py")):
        if "legacy" in py.parts or "__pycache__" in py.parts:
            continue
        for c in cadenas_tr(py):
            total += 1
            if c in EN or c in NO_TRADUCIR:
                continue
            faltan.setdefault(c, []).append(py.relative_to(RAIZ))

    unicas = len({c for py in sorted((RAIZ / "polyx").rglob("*.py"))
                  if "legacy" not in py.parts and "__pycache__" not in py.parts
                  for c in cadenas_tr(py)})
    print(f"cadenas en tr(): {total} ({unicas} distintas)")
    print(f"traducidas:      {unicas - len(faltan)}")
    print(f"SIN TRADUCIR:    {len(faltan)}"
          + (f"   ({100 * len(faltan) / unicas:.1f} %)" if unicas else ""))

    if faltan:
        por_archivo: dict[str, int] = {}
        for cs in faltan.values():
            for p in cs:
                por_archivo[str(p)] = por_archivo.get(str(p), 0) + 1
        print("\npor archivo:")
        for p, n in sorted(por_archivo.items(), key=lambda t: -t[1]):
            print(f"  {n:>4}  {p}")

    if args.listar and faltan:
        print("\ncadenas sin traducir:")
        for c in sorted(faltan):
            uno = c.replace("\n", " ")
            print(f"  {uno[:110]}")

    return 1 if faltan else 0


if __name__ == "__main__":
    raise SystemExit(main())
