#!/usr/bin/env python3
"""LibCST codemod core to rename polyx identifiers to English.

Milestone M006, slice S04, task T01. This module is the identifier sibling of
``tools/translate_docstrings.py`` (S02) and ``tools/translate_comments.py``
(S03): it rewrites ``cst.Name`` nodes from an exact ``{old: new}`` map and
nothing else. Because LibCST round-trips the module, comments, docstrings,
string literals, numbers, operators and line endings stay byte-identical
outside the renamed tokens.

Scope rules (D027/D028):

* module filenames and import paths are never renamed -- the module component
  of ``import polyx.gui.iconos`` and ``from .iconos import ...`` is skipped,
  and an imported symbol that is itself a module stem is left alone;
* the only string tokens that may change are the second argument of
  ``getattr``/``setattr``/``hasattr``/``delattr`` and the entries of
  ``__all__``;
* ``--scope consumers`` renames only names imported from polyx, their bound
  local aliases, renamed attribute accesses and keyword arguments of calls to
  imported polyx callables, so test-local Spanish definitions are untouched.

Preflight (D029) refuses a map when a target is not a valid identifier, is a
Python keyword or builtin, when two sources share a target, when a target is
also a source (that would make a second pass non-idempotent), when ``symtable``
finds the old and the new name in the same scope, or when the transformed
module does not compile.

CLI:
    python3 tools/translate_identifiers.py --dry-run --manifest map.py polyx
    python3 tools/translate_identifiers.py --apply --manifest map.py polyx
    python3 tools/translate_identifiers.py --apply --scope consumers \
        --manifest map.py tests

A manifest is a Python module defining ``IDENTIFIER_TRANSLATIONS`` as a flat
``{old_name: new_name}`` dict.
"""

from __future__ import annotations

import argparse
import builtins
import functools
from collections import Counter
import importlib.util
import io
import json
import keyword
import os
import re
import symtable
import sys
import time
import tokenize
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import libcst as cst

REPO_ROOT = Path(__file__).resolve().parents[1]
POLYX_ROOT = REPO_ROOT / "src" / "polyx"

# Dynamic string helpers whose second positional argument is an attribute name.
DYNAMIC_GETTERS = frozenset({"getattr", "setattr", "hasattr", "delattr"})

# Observability sinks whose keyword arguments are JSONL data keys, not Python
# symbols. The slice keeps data keys byte-identical, so the scanner must not
# report them and the codemod must not rename them (like ``str.format`` field
# keywords).
OBSERVABILITY_SINKS = frozenset({"event", "evento", "timer", "cronometro"})

# Builtins that are also plausible Spanish words. Used only to keep suggestion
# output valid; the rename preflight checks every builtin, not just these.
_BUILTIN_NAMES = frozenset(dir(builtins))

# ---------------------------------------------------------------------------
# Spanish identifier scanner.
# ---------------------------------------------------------------------------

# English/Spanish homographs (D029) that must never be reported: they are
# legitimate English words when they appear as identifiers. Units, formats and
# API acronyms are listed separately below.
HOMOGRAPH_ALLOWLIST = frozenset(
    {
        "animal",
        "area",
        "areas",
        "capital",
        "central",
        "color",
        "detector",
        "digital",
        "error",
        "final",
        "general",
        "global",
        "horizontal",
        "inferior",
        "local",
        "manual",
        "mental",
        "metal",
        "metric",
        "metrics",
        "model",
        "mortal",
        "normal",
        "original",
        "page",
        "panel",
        "par",
        "parseable",
        "precision",
        "principal",
        "real",
        "rural",
        "sexual",
        "simple",
        "social",
        "state",
        "superior",
        "trainer",
        "total",
        "union",
        "usual",
        "version",
        "vertical",
        "virtual",
        "visual",
        "vocal",
        "canvas",
        "indices",
        "fin",
    }
)

# Units, file formats and API/acronym fragments. They are single tokens that
# the scanner must never read as Spanish prose.
UNIT_ALLOWLIST = frozenset(
    {
        "api", "ap", "amp", "ascii", "b64", "bgr", "cm", "cpu", "crlf", "csv",
        "cuda", "cudnn", "cv", "dpi", "eof", "fps", "gpu", "gt", "gui", "hsv",
        "html", "http", "https", "id", "ids", "io", "jpeg", "jpg", "json",
        "jsonl", "llm", "mac", "map", "md5", "ml", "mm", "nms", "nm", "np",
        "os", "pc", "pdf", "png", "pt", "px", "qt", "qss", "ram", "rgb", "roi",
        "sep", "sha", "sql", "txt", "ui", "um", "uri", "url", "utf", "util",
        "uuid", "ws", "yaml", "yolo", "yolov8",
    }
)

ALLOWLIST = HOMOGRAPH_ALLOWLIST | UNIT_ALLOWLIST

# Third-party API identifiers whose spelling happens to contain a Spanish
# subword (Qt's ``QRadioButton`` contains "radio", ``SizeVerCursor`` contains
# "ver"). They are never renaming candidates: the symbol belongs to the
# library, not to polyx.
EXACT_ALLOWLIST = frozenset(
    {
        "QRadioButton",
        "SizeVerCursor",
        "ControlModifier",
        "mac_ver",
    }
)

# Curated Spanish-only subwords. Deliberately excludes English homographs and
# anything that reads as legitimate English technical prose. Extend this set
# when a real Spanish identifier survives; never loosen the detector.
SPANISH_SUBWORDS = frozenset(
    """
    abajo abierta abiertas abierto abiertos abriendo abrir accion acciones activa activas
    activo activos actual actuales actualiza actualizacion actualizaciones actualizada
    actualizado
    actualizador actualizando actualizar actualizo acuerdo acuerdos adelante adentro
    advertencia advertencias agrega agregada agregado agrupacion agrupadas agrupado
    agrupar ajustar ajuste ajustes alargada alargado alcance alcanza alineada alineado
    alinear alma almacen almacena almacenado almacenar alta altas alto altos altura
    alturas ambas ambos ancha anchas ancho anchos ancla anclas angulos anillo anillos
    animaciones anterior anteriores antes anula anulado aproximada aproximado aqui
    archivo archivos arquitectura arranca arrancado arrancar arranque arriba asignable
    asignables asignacion asignaciones asignada asignadas asignado asignados asignar
    aspecto atajo atajos atras augmentacion ausente ausentes avance avisar aviso avisos
    azul bajas bajo bajos barra barras barrido borde bordes borrada borrado borrados
    borrar boton botones bulbo buscar buscador buscadores caja cajas calcula calculada calculado
    calcular calculo calibra calibracion calibraciones calibrada calibrado calibrando
    calibrar cambiadas cambiado camino caminos campo campos cancelada cancelado
    camara
    candidatas candidatos capacidad cargar carpeta carpetas celda celdas centro centros cerrada
    cerrado cerrar chica chico ciclo ciclos circularidad circulo circulos clave clase clases
    clasificacion clasificaciones clasificado clasificar codigo codigos cola colores
    columna columnas comando comandos combinada compacta compacto comparacion comparaciones comparar
    comparativa comparativo completa completada completadas completo comun confianza confianzas
    conmutador conocidas conocido contenido contenidos contar conteo contorno contornos contraste control
    convertida convertido convertir convexa copiadas copias correcta correcto correctos corrida
    corridas corta corto criterio criterios critico cronometro crudo cuales cuantas
    cuantos cuenta cuentas cuerpo cuerpos curva curvas curvatura datos defecto dentro
    descripcion descripciones deshacer desplazamiento destino destinos detalle desvio deteccion
    detecciones detectada detectado detectar detener devuelta devuelto devuelve dialogo
    diametro dibujar dibujo dibujos dicha dichas dicho dichos direccion dispositivo
    dispositivos distinta distintas distintivo distintos disponible disponibles documento documentos dominio
    dominios ejecucion ejecuta ejecutada ejecutado ejecutar ejemplo ejemplos elaboracion
    elegida elegidas elegido elegidos elegir elemento elementos elimina eliminada
    eliminado eliminar elongacion enlazar ensayo entorno entrada entradas entrenador entrenadores
    entrenamiento entrenamientos entrenar entrenos equipo equipos errores escala escalas
    escena escenas escribir escrita escrito escalera espera espesor espesores estacion estado estados etiqueta
    etiquetador etiquetadores etiquetas etq evento eventos exacta exacto excluir exclusion
    extendida extendido externa externo extremo extremos facil falla fallas fallidas
    fallidos fallo fallos familia familias fibra fibras ficha fichas fijar figura figuras fila filas
    filtrada filtrado filtrar filtro filtros fina finitos fino fondo forzar forma formas formateador
    formato foto fotos fraccion fragmento fragmentos fuera funcion funcionamiento
    funciones galeria generada generadas generado generados generar geodesica geodesico glifo
    glifos grafica grafico grandes grosor grupo grupos guarda guardada guardadas guardado
    guardar hechas hechos historial holgura icono iconos identidad idioma idiomas imagen imagenes
    indeterminada indeterminado indice informe inicio instalada instalado instalar instancia instante
    intensidad intento interior interna interno interpolacion introducida introducido
    iteracion iteraciones izquierda lado lados lanzar larga largas largo largos lectura leer
    leida leidas leido leidos leyenda leyendas lienzo limitada limitado linea lineas
    llamada llamado llamar llegada llenado llenar lleva lote limite macizo malformada malformadas
    mando manejador manejadores mantener manuales maquina marca marcada marcadas marcado
    marcador marcadores marcar marcas margen margenes mascara mascaras maxima maximo
    mayor media mediana medianas medicion mediciones medida medidas medido medio medir
    mejor mejores menor mensaje mensajes metodo metodos metricas minima minimo misma
    mismas mismo mismos modelo modelos modulo modulos morfo morfotipo mostrada mostradas
    mostrado mostrados mostrar motivo motivos movida movido muestra muestras mueve nada
    nativa nativo nivel niveles ninguna ninguno nombre nombres notas nucleo nucleos nueva nuevas nuevo
    nuevos numero numeros nunca observabilidad obtenida obtenido ocupada ocupado omitida
    omitidas omitido opcion opciones optima optimizador optimo optimos orden ordenada
    ordenado ordenar ordenes origen origenes oscuro oscuros otras otros padre padres
    pagina paginas paleta paletas parada pared parametros parrafo parrafos parseable parseables
    parte partes particula particulas pasada pasado pedida pedidas pedido pedidos pegada
    pegadas perfil perfiles perimetro permutacion peso pesos peor pico pide pieza piezas pintar
    pista pistas placa placas plano planos poda podar politica porcentaje preannotar preanotacion preferencia
    preferencias prefijo presencia presente presentes previa previas previo primera
    primero problema problemas procedencia procesada procesado procesar propagar profundidad
    progreso propia propias propio propios puede pueden pulgadas punta puntas punto
    puntos radio radios raiz rango rangos rapida rapidez rapido razon razones reales recarga
    recargar rechazada rechazado reencuadrar recibe recibida recibido recogida recogido recomendacion
    recomendaciones recomendar recorrido recta recto recuento reemplace referencia referencias
    refrescar region regiones registro registros rehacer reiniciar rejilla relacion relaciones
    remota remoto reportada reportado reporte reportes resto resultado resultados resumen
    retorna revisada revisado revisar revision revolver robusta robusto rotulo ruta rutas
    salida salidas salto saltos seccion secciones segunda segundo segura seguro semilla separar
    senal sentido sesgo siguiente sistema sistemas sitio sitios sobrantes sobre solidez
    sufijo sugerencia sugerencias suma sumar suscriptor suscriptores tabla tablas
    tabulador tallas taller tamano tamanos tarea tareas tarjeta tarjetas techo tema temas
    tendencia termino terminos testigo texto textos tiempo tiempos titulo titulos toda
    todas todo todos tolerancia totales trabajo trabajos traduccion traducciones traducir
    tramo tramos trazo trazos troceado troceo trozo truncada truncado turno ultima
    ultimas ultimo ultimos umbral umbrales unica unico unidos usa usada usadas usado usados
    usuario usuarios util utiles valida validar valido valor valores varias varios vecina
    vecinas vecino vecinos ventana ventanas verdadera verdadero veredicto veredictos
    versiones vista vistas visto vistos viva vivas vivo vuelta vueltas
    alfa conmutable editar es evaluar exportar fase fija hay hilo informar lista
    nota optimizar paquete papelera rama rayo reelegir registrar restaurar ver visor
    asegurar estandar ir pendiente
    acum aplicar asig ayuda bloque cabeza cambiar cobertura coletilla construir corregir
    cubo delgadez distancia donde empate envolvente flecha fuga identificar lapiz leeme
    mapa matraz mej metr mezclar minuscula mitad morfologia ojo otra para paralaje
    publicar realce recorte resumir segmentar talla
    """.split()
)

# English words that trip the morphology heuristic but are not Spanish.
MORPHOLOGY_EXCEPTIONS = frozenset(
    {
        "avocado", "bonanza", "colorado", "coercion", "desperado", "metadata",
        "mikado", "suspicion", "tornado",
    }
)

# Spanish morphology heuristic. Accents are usually dropped from identifiers,
# so these endings catch words the vocabulary does not enumerate. The broad
# -os/-as/-es plurals are deliberately absent: they would flag English words
# such as "values", "images", "metrics" or "indices".
_MORPHOLOGY = re.compile(
    r"(?:cion|ciones|idad|idades|ador|adora|adores|adoras|amiento|amientos"
    r"|imiento|imientos|anza|anzas|encia|encias|oso|osa|osos|osas"
    r"|ivo|iva|ivos|ivas|ado|ada|ados|adas|ido|ida|idos|idas)$"
)

_ACCENTED = re.compile(r"[áéíóúüñÁÉÍÓÚÜÑ]")

# Subword splitter: acronym before CamelCase, CamelCase word, ALLCAPS run or
# digits. Digit runs are merged into the preceding subword so "g2" stays one
# token and its suffix survives translation.
_SUBWORD = re.compile(
    r"[A-ZÀ-ÖØ-Þ]+(?=[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ])"
    r"|[A-ZÀ-ÖØ-Þ]?[a-zà-öø-ÿ]+"
    r"|[A-ZÀ-ÖØ-Þ]+"
    r"|\d+"
)


def split_identifier(name: str) -> List[str]:
    """Split snake_case, CamelCase and ALLCAPS identifiers into subwords.

    Digit runs are merged into the preceding subword (``g2`` -> ``["g2"]``,
    ``N_ANGULOS`` -> ``["N", "ANGULOS"]``, ``DialogoPreferencias`` ->
    ``["Dialogo", "Preferencias"]``).
    """
    subwords: List[str] = []
    for part in name.split("_"):
        if not part:
            continue
        for match in _SUBWORD.finditer(part):
            token = match.group(0)
            if token.isdigit() and subwords:
                subwords[-1] += token
            else:
                subwords.append(token)
    return subwords


def _strip_digits(subword: str) -> str:
    """Return the alphabetic base of a subword (``angulos2`` -> ``angulos``)."""
    return re.sub(r"\d+$", "", subword)


def spanish_identifier_findings(name: str) -> List[str]:
    """Return Spanish markers for an identifier (``[]`` means clean).

    Findings use ``spanish-identifier:<subword>`` so a durable gate can
    filter on the token after the colon, exactly like the S02/S03 scanners.
    """
    if name in EXACT_ALLOWLIST:
        return []
    findings: List[str] = []
    for subword in split_identifier(name):
        base = _strip_digits(subword)
        if not base:
            continue
        lowered = base.lower()
        if lowered in ALLOWLIST:
            continue
        if _ACCENTED.search(base):
            findings.append(f"spanish-identifier:{lowered}")
            continue
        if lowered in SPANISH_SUBWORDS:
            findings.append(f"spanish-identifier:{lowered}")
            continue
        if (
            len(lowered) >= 5
            and lowered not in MORPHOLOGY_EXCEPTIONS
            and _MORPHOLOGY.search(lowered)
        ):
            findings.append(f"spanish-identifier:{lowered}")
    # Preserve order but report each subword once.
    return list(dict.fromkeys(findings))


@functools.lru_cache(maxsize=1)
def default_module_stems() -> frozenset:
    """Cached module stems of the checked-out polyx package."""
    return discover_module_stems()


def module_stem_bindings(text: str, stems: Iterable[str] = ()) -> Set[str]:
    """Return local names bound to an imported module stem in ``text``.

    ``from polyx.gui import iconos`` binds the module ``iconos``; the binding
    is module-path context, so it is neither renamed nor scanned (D027).
    """
    try:
        module = cst.parse_module(text)
    except cst.ParserSyntaxError:
        return set()
    return _module_stem_bindings(module, stems)


def _format_keyword_positions(text: str) -> Set[Tuple[int, int]]:
    """Positions of keyword names that are ``str.format`` field references.

    ``tr("... {metodo}").format(metodo=...)`` keeps the placeholder inside a
    string literal, which the codemod never rewrites. The keyword name is not
    a Python symbol, so the scanner must not treat it as a Spanish identifier.
    This is token-based because ``ast.keyword`` offsets inside f-strings are
    not the tokenizer's column positions.
    """
    ignored = {
        tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
        tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER,
    }
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type not in ignored
    ]
    positions: Set[Tuple[int, int]] = set()
    for index, token in enumerate(tokens):
        if (
            token.type != tokenize.NAME
            or token.string not in {"format", "format_map"}
            or index == 0
            or index + 1 >= len(tokens)
            or tokens[index - 1].string != "."
            or tokens[index + 1].string != "("
        ):
            continue
        depth = 0
        cursor = index + 1
        while cursor < len(tokens):
            current = tokens[cursor]
            if current.type == tokenize.OP:
                if current.string in "([{":
                    depth += 1
                elif current.string in ")]}":
                    depth -= 1
                    if depth == 0:
                        break
            if depth == 1 and current.type == tokenize.NAME:
                previous = tokens[cursor - 1]
                following = tokens[cursor + 1] if cursor + 1 < len(tokens) else None
                if (
                    following is not None
                    and following.type == tokenize.OP
                    and following.string == "="
                    and previous.type == tokenize.OP
                    and previous.string in {"(", ","}
                ):
                    positions.add(current.start)
            cursor += 1
    return positions


def _data_keyword_positions(text: str) -> Set[Tuple[int, int]]:
    """Positions of keyword names that are JSONL data keys.

    ``event(log, "x", sistema=...)`` forwards ``sistema`` into the JSONL
    payload; it is data, not a Python symbol, so the scanner and the codemod
    must leave it alone. Only calls to the observability sinks are considered.
    """
    ignored = {
        tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
        tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER,
    }
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type not in ignored
    ]
    positions: Set[Tuple[int, int]] = set()
    for index, token in enumerate(tokens):
        if (
            token.type != tokenize.NAME
            or token.string not in OBSERVABILITY_SINKS
            or index + 1 >= len(tokens)
            or tokens[index + 1].string != "("
            or (index > 0 and tokens[index - 1].string == "def")
        ):
            continue
        depth = 0
        cursor = index + 1
        while cursor < len(tokens):
            current = tokens[cursor]
            if current.type == tokenize.OP:
                if current.string in "([{":
                    depth += 1
                elif current.string in ")]}":
                    depth -= 1
                    if depth == 0:
                        break
            if depth == 1 and current.type == tokenize.NAME:
                previous = tokens[cursor - 1]
                following = tokens[cursor + 1] if cursor + 1 < len(tokens) else None
                if (
                    following is not None
                    and following.type == tokenize.OP
                    and following.string == "="
                    and previous.type == tokenize.OP
                    and previous.string in {"(", ","}
                ):
                    positions.add(current.start)
            cursor += 1
    return positions


def identifier_occurrences(
    text: str, stems: Optional[Iterable[str]] = None
) -> List[Tuple[int, str]]:
    """Yield ``(line, name)`` for every identifier outside import context.

    Module paths, imported symbols and the local bindings of imported module
    stems are skipped: the identifier they refer to is defined and scanned in
    its own module, and a module stem such as ``iconos`` must not be reported
    from ``from .iconos import ...`` or ``from polyx.gui import iconos``
    (D027). A local ``iconos`` variable in a module without that import is
    still an occurrence and is scanned normally.
    """
    if stems is None:
        stems = default_module_stems()
    protected = module_stem_bindings(text, stems)
    format_keywords = _format_keyword_positions(text)
    data_keywords = _data_keyword_positions(text)
    occurrences: List[Tuple[int, str]] = []
    in_import = False
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.NAME and token.string in {"import", "from"}:
            in_import = True
            continue
        if token.type == tokenize.NEWLINE:
            in_import = False
            continue
        if token.type == tokenize.NAME and not in_import:
            if (
                token.string in protected
                or token.start in format_keywords
                or token.start in data_keywords
            ):
                continue
            occurrences.append((token.start[0], token.string))
    return occurrences


def enumerate_identifiers(
    text: str, stems: Optional[Iterable[str]] = None
) -> List[Tuple[str, int]]:
    """Return ``(name, count)`` for every unique identifier, sorted by name.

    Python keywords and booleans are not identifiers, and names inside import
    statements are skipped for the same reason as in
    :func:`identifier_occurrences`: a module stem is not a symbol to rename.
    """
    counts: Dict[str, int] = {}
    for _line, name in identifier_occurrences(text, stems):
        if keyword.iskeyword(name) or name in {"True", "False", "None"}:
            continue
        counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items(), key=lambda item: item[0])


# ---------------------------------------------------------------------------
# Suggestion engine.
# ---------------------------------------------------------------------------

# Spanish subword -> English subword. Only Spanish-only words belong here;
# homographs are intentionally absent so suggestions never rename them.
SUBWORD_TRANSLATIONS: Dict[str, str] = {
    "abajo": "bottom",
    "abrir": "open",
    "accion": "action",
    "acum": "accumulated",
    "acuerdo": "agreement",
    "acuerdos": "agreements",
    "activa": "active",
    "activas": "active",
    "activo": "active",
    "activos": "active",
    "actualiza": "update",
    "actualizacion": "update",
    "actualizaciones": "updates",
    "actualizador": "updater",
    "actualizar": "update",
    "agrupar": "group",
    "agrupacion": "grouping",
    "agrupadas": "grouped",
    "ajustar": "adjust",
    "ajuste": "adjustment",
    "ajustes": "settings",
    "alargada": "elongated",
    "alto": "height",
    "altura": "height",
    "alturas": "heights",
    "ancho": "width",
    "ancha": "width",
    "anchos": "widths",
    "anclas": "anchors",
    "angulos": "angles",
    "anillo": "ring",
    "anillos": "rings",
    "anterior": "previous",
    "aproximado": "approximate",
    "archivo": "file",
    "archivos": "files",
    "arrancar": "start",
    "arquitectura": "architecture",
    "asignable": "assignable",
    "asignables": "assignables",
    "asignacion": "assignment",
    "asig": "assignment",
    "asignada": "assigned",
    "asignadas": "assigned",
    "aspecto": "aspect",
    "atajo": "shortcut",
    "atajos": "shortcuts",
    "augmentacion": "augmentation",
    "aviso": "warning",
    "avisar": "notify",
    "bajo": "low",
    "barras": "bars",
    "barra": "bar",
    "barrido": "sweep",
    "borde": "edge",
    "bordes": "edges",
    "boton": "button",
    "botones": "buttons",
    "buscador": "search",
    "caja": "box",
    "cajas": "boxes",
    "calibracion": "calibration",
    "calibraciones": "calibrations",
    "calibrar": "calibrate",
    "camara": "camera",
    "camino": "path",
    "campo": "field",
    "cancelado": "cancelled",
    "candidatos": "candidates",
    "capacidad": "capacity",
    "carpeta": "folder",
    "carpetas": "folders",
    "celdas": "cells",
    "centro": "center",
    "centros": "centers",
    "cerrar": "close",
    "chico": "small",
    "circularidad": "circularity",
    "circulo": "circle",
    "clase": "class_id",
    "clases": "classes",
    "clave": "key",
    "colores": "colors",
    "columna": "column",
    "comparacion": "comparison",
    "comparar": "compare",
    "comparativa": "comparison",
    "completo": "complete",
    "conmutador": "toggle",
    "confianza": "confidence",
    "conocidas": "known",
    "contar": "count",
    "conteo": "count",
    "contenido": "content",
    "contorno": "contour",
    "contornos": "contours",
    "convertir": "convert",
    "copiadas": "copied",
    "copias": "copies",
    "corrida": "run",
    "correcto": "correct",
    "crear": "create",
    "criterio": "criterion",
    "critico": "critical",
    "cronometro": "timer",
    "cuentas": "counts",
    "cuerpo": "body",
    "cuerpos": "bodies",
    "curvatura": "curvature",
    "datos": "data",
    "defecto": "default",
    "dentro": "inside",
    "descripcion": "description",
    "deshacer": "undo",
    "desplazamiento": "offset",
    "destino": "destination",
    "detalle": "detail",
    "detectar": "detect",
    "detectado": "detected",
    "deteccion": "detection",
    "detecciones": "detections",
    "detener": "stop",
    "diametro": "diameter",
    "dibujar": "draw",
    "dibujo": "drawing",
    "dibujos": "drawings",
    "dispositivo": "device",
    "dispositivos": "devices",
    "distintivo": "badge",
    "distintos": "distinct",
    "documento": "document",
    "dominio": "domain",
    "dominios": "domains",
    "ejecucion": "execution",
    "ejecutar": "run",
    "elegida": "chosen",
    "elegidas": "chosen",
    "elegido": "chosen",
    "elegir": "choose",
    "eliminar": "remove",
    "enlazar": "link",
    "elongacion": "elongation",
    "entrada": "input",
    "entrenador": "trainer",
    "entrenamiento": "training",
    "entrenar": "train",
    "entrenos": "training_runs",
    "equipo": "team",
    "errores": "errors",
    "evaluar": "evaluate",
    "escala": "scale",
    "escalera": "ladder",
    "escalas": "scales",
    "escena": "scene",
    "escribir": "write",
    "estacion": "station",
    "estado": "state",
    "etiqueta": "label",
    "etiquetas": "labels",
    "etq": "label",
    "etiquetador": "labeler",
    "evento": "event",
    "extremos": "extremes",
    "fallidos": "failed",
    "fallos": "failures",
    "familias": "families",
    "fibras": "fibers",
    "fibra": "fiber",
    "ficha": "record",
    "fichas": "records",
    "figura": "figure",
    "figuras": "figures",
    "filas": "rows",
    "fila": "row",
    "fino": "fine",
    "forzar": "force",
    "formateador": "formatter",
    "fragmento": "fragment",
    "fraccion": "fraction",
    "fuera": "outside",
    "funcion": "function",
    "generados": "generated",
    "generar": "generate",
    "geodesico": "geodesic",
    "guardado": "saved",
    "guardar": "save",
    "historial": "history",
    "icono": "icon",
    "iconos": "icons",
    "identidad": "identity",
    "idioma": "language",
    "idiomas": "languages",
    "imagen": "image",
    "imagenes": "images",
    "indeterminado": "indeterminate",
    "informar": "report",
    "informe": "report",
    "nota": "note",
    "notas": "notes",
    "inicio": "start",
    "instalada": "installed",
    "intensidad": "intensity",
    "interpolacion": "interpolation",
    "iteraciones": "iterations",
    "lado": "side",
    "largos": "lengths",
    "largo": "length",
    "leer": "read",
    "leeme": "readme",
    "limite": "limit",
    "leyenda": "legend",
    "lienzo": "canvas",
    "linea": "line",
    "lineas": "lines",
    "llenado": "filling",
    "malformadas": "malformed",
    "manejadores": "handlers",
    "mantener": "keep",
    "maquina": "machine",
    "marca": "marker",
    "marcadas": "marked",
    "marcador": "marker",
    "marcar": "mark",
    "marcas": "marks",
    "margen": "margin",
    "maximo": "maximum",
    "mediana": "median",
    "medianas": "medians",
    "medicion": "measurement",
    "mediciones": "measurements",
    "medidas": "measurements",
    "medir": "measure",
    "mej": "best",
    "mejor": "best",
    "metodo": "method",
    "metodos": "methods",
    "metr": "metrics",
    "metricas": "metrics",
    "minimo": "minimum",
    "mismas": "same",
    "mostradas": "shown",
    "muestra": "sample",
    "muestras": "samples",
    "nativo": "native",
    "niveles": "levels",
    "nombre": "name",
    "nombres": "names",
    "nucleo": "core",
    "nucleos": "cores",
    "nuevo": "new",
    "nueva": "new",
    "numero": "number",
    "observabilidad": "observability",
    "ocupado": "busy",
    "omitidas": "omitted",
    "opciones": "options",
    "opcion": "option",
    "optimizar": "optimize",
    "optimos": "optimal",
    "orden": "order",
    "ordenado": "sorted",
    "origen": "origin",
    "origenes": "origins",
    "oscuros": "dark",
    "paleta": "palette",
    "paletas": "palettes",
    "parrafo": "paragraph",
    "parseables": "parseable",
    "particula": "particle",
    "particulas": "particles",
    "pares": "pairs",
    "pedidas": "requested",
    "pedido": "request",
    "pegadas": "glued",
    "perfiles": "profiles",
    "perimetro": "perimeter",
    "permutacion": "permutation",
    "pesos": "weights",
    "peor": "worst",
    "pico": "peak",
    "pintar": "paint",
    "placas": "plates",
    "plano": "plane",
    "planos": "planes",
    "politica": "policy",
    "porcentaje": "percentage",
    "preannotar": "pre_annotate",
    "preferencias": "preferences",
    "prefijo": "prefix",
    "primero": "first",
    "problemas": "problems",
    "procedencia": "provenance",
    "procesar": "process",
    "profundidad": "depth",
    "propios": "own",
    "puede": "can",
    "punto": "point",
    "puntos": "points",
    "radio": "radius",
    "radios": "radii",
    "raiz": "root",
    "rapida": "fast",
    "rango": "range",
    "rangos": "ranges",
    "razon": "reason",
    "realizado": "performed",
    "reales": "real",
    "recargar": "reload",
    "rechazada": "rejected",
    "reencuadrar": "reframe",
    "recuento": "count",
    "reelegir": "reselect",
    "recomendacion": "recommendation",
    "recomendar": "recommend",
    "referencias": "references",
    "region": "region",
    "rehacer": "redo",
    "reiniciar": "restart",
    "rejilla": "grid",
    "reportada": "reported",
    "reporte": "report",
    "resumen": "summary",
    "resultado": "result",
    "resultados": "results",
    "revisada": "reviewed",
    "revisar": "review",
    "robusto": "robust",
    "rotulo": "label",
    "ruta": "path",
    "rutas": "paths",
    "salida": "output",
    "seccion": "section",
    "secciones": "sections",
    "segundo": "second",
    "semilla": "seed",
    "sentido": "direction",
    "senal": "signal",
    "siguiente": "next",
    "sistema": "system",
    "sitios": "sites",
    "solidez": "solidity",
    "sugerencia": "suggestion",
    "suscriptores": "subscribers",
    "tabla": "table",
    "tablas": "tables",
    "tallas": "sizes",
    "tamano": "size",
    "tamanos": "sizes",
    "tareas": "tasks",
    "tarjeta": "card",
    "tarjetas": "cards",
    "techo": "ceiling",
    "tema": "theme",
    "temas": "themes",
    "tendencia": "trend",
    "termino": "term",
    "texto": "text",
    "tiempo": "time",
    "titulo": "title",
    "trabajo": "work",
    "trabajos": "jobs",
    "traducciones": "translations",
    "traducir": "translate",
    "tramo": "range",
    "tramos": "ranges",
    "troceado": "tiling",
    "troceo": "tiling",
    "truncado": "truncated",
    "ultimo": "last",
    "umbral": "threshold",
    "umbrales": "thresholds",
    "usado": "used",
    "usadas": "used",
    "usuario": "user",
    "utiles": "useful",
    "valida": "valid",
    "validar": "validate",
    "valores": "values",
    "valor": "value",
    "varios": "several",
    "vecinas": "neighbours",
    "veredicto": "verdict",
    "versiones": "versions",
    "ventana": "window",
    "vista": "view",
    "vivas": "alive",
}

# Exact overrides for adjective-noun reordering, selected targets that would
# otherwise be builtins, and digit-suffixed widget names.
EXACT_OVERRIDES: Dict[str, str] = {
    "alto_caja": "box_height",
    "ancho_caja": "box_width",
    "largo_rect_eq_um": "equivalent_rect_length_um",
    "margen_de_caja": "box_margin",
    "mascara": "mask",
    "mascaras": "masks",
    "ruta_imagen": "image_path",
    "imagen_ruta": "image_path",
    "g2": "grid2",
    "lista": "items",
    "listas": "item_lists",
    "maximo": "maximum",
    "minimo": "minimum",
    "DialogoPreferencias": "PreferencesDialog",
    "Dialogo": "Dialog",
    # English noun phrases: the Spanish order is noun + modifiers, the English
    # one is modifiers + noun. The glossary can only translate subwords, so the
    # full identifiers are pinned here after deterministic review (T06).
    "NOMBRE_MEJOR_LAB": "BEST_LAB_NAME",
    "NOMBRE_MEJOR_REAL": "BEST_REAL_NAME",
    "PREFIJO_LAB": "LAB_PREFIX",
    "PREFIJO_REAL": "REAL_PREFIX",
    # Spanish grammar/particle phrases the subword composer cannot reorder.
    "_actualizar_aviso_comparar": "_update_compare_notice",
    "_etiqueta_de": "_label_for",
    "_imagenes_de_val": "_val_images",
    "se_puede_reelegir": "can_reselect",
    # Spanish-only words the subword composer would place in the wrong order.
    "ESCALERA_IMGSZ": "IMGSZ_LADDER",
    "crear_yaml_solo_real": "create_real_only_yaml",
    "tope": "max_size",
    "etq": "label_path",
    # English-homograph survivors: the word exists in English, but its use in
    # polyx is Spanish, so the scanner must not flag it and the override pins
    # the correct target (D029 homographs stay out of the detector vocabulary).
    "cab": "header_line",
    "patron": "pattern",
    "libre_gb": "free_gb",
    "ram_libre_gb": "ram_free_gb",
}


def _join_identifier(original: str, parts: Sequence[str]) -> str:
    """Rebuild an identifier from translated subwords in the original style.

    Leading and trailing underscores are preserved (``_ARCHIVO`` -> ``_FILE``,
    ``_SobreFormateador`` -> ``_SobreFormatter``) while the core keeps its
    snake_case, CamelCase or ALLCAPS convention.
    """
    leading = original[: len(original) - len(original.lstrip("_"))]
    trailing = original[len(original.rstrip("_")) :]
    core = original.strip("_")
    if not core:
        return original
    if "_" in core:
        if core.replace("_", "").isupper():
            joined = "_".join(part.upper() for part in parts)
        else:
            joined = "_".join(part.lower() for part in parts)
    elif core.isupper():
        joined = "_".join(part.upper() for part in parts)
    elif any(character.isupper() for character in core):
        joined = "".join(part[:1].upper() + part[1:] for part in parts)
    else:
        joined = "".join(parts)
    return leading + joined + trailing


def _is_valid_target(candidate: str) -> bool:
    """True when the candidate is a usable non-builtin, non-keyword name."""
    return (
        candidate.isidentifier()
        and not keyword.iskeyword(candidate)
        and candidate not in _BUILTIN_NAMES
    )


def suggest_identifier_map(names: Iterable[str]) -> Dict[str, str]:
    """Propose ``{old: new}`` for identifiers containing Spanish subwords.

    The result is a draft: callers must review reordering, abbreviations and
    same-scope collisions before using it as a manifest. Builtins, keywords and
    invalid identifiers are never emitted.
    """
    suggestions: Dict[str, str] = {}
    for name in names:
        if not name or name.startswith("__") and name.endswith("__"):
            continue
        if name in EXACT_OVERRIDES:
            candidate = EXACT_OVERRIDES[name]
            if candidate != name and _is_valid_target(candidate):
                suggestions[name] = candidate
            continue
        subwords = split_identifier(name)
        if not subwords:
            continue
        parts: List[str] = []
        translated_any = False
        for subword in subwords:
            base = _strip_digits(subword)
            digits = subword[len(base) :]
            replacement = SUBWORD_TRANSLATIONS.get(base.lower(), base)
            if replacement != base:
                translated_any = True
            parts.append(replacement + digits)
        if not translated_any:
            continue
        candidate = _join_identifier(name, parts)
        if candidate == name or not _is_valid_target(candidate):
            continue
        suggestions[name] = candidate
    return suggestions


# ---------------------------------------------------------------------------
# Translation-software bridge (D033).
# ---------------------------------------------------------------------------

DEFAULT_CACHE_PATH = (
    REPO_ROOT / "tests" / "identifier_manifests" / "translation_cache.json"
)

DEFAULT_HEADING_CACHE_PATH = (
    REPO_ROOT / "tests" / "heading_manifests" / "translation_cache.json"
)


class TranslationError(RuntimeError):
    """Base error for the translation-software bridge."""


class TranslationBackendError(TranslationError):
    """The configured backend is missing, unreachable or returned garbage."""


class OfflineCacheMiss(TranslationError):
    """``--offline`` was requested but a subword is not in the cache."""


class TranslationBackend:
    """Backend-neutral es -> en translator used by the identifier bridge."""

    name = "unknown"
    model = "unknown"

    def translate_many(self, texts: Sequence[str]) -> List[str]:
        """Translate each text; subclasses must return one result per input."""
        raise NotImplementedError


def _http_json(
    url: str,
    *,
    payload: Optional[Mapping[str, object]] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 30.0,
) -> object:
    """POST (or GET when ``payload`` is None) JSON and return the decoded body."""
    data = None
    request_headers = {
        "User-Agent": "polyx-identifier-translator/1.0",
        **(headers or {}),
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _google_web_translate(text: str, timeout: float = 20.0) -> str:
    """Translate through the public Google web endpoint (no API key)."""
    query = urllib.parse.urlencode(
        {"client": "gtx", "sl": "es", "tl": "en", "dt": "t", "q": text}
    )
    body = _http_json(
        "https://translate.googleapis.com/translate_a/single?" + query,
        timeout=timeout,
    )
    try:
        segments = body[0] or []
        translated = "".join(
            segment[0] for segment in segments if segment and segment[0]
        )
    except (IndexError, TypeError):
        raise TranslationBackendError(
            f"google-web returned an unexpected payload for {text!r}"
        )
    if not translated.strip():
        raise TranslationBackendError(
            f"google-web returned an empty translation for {text!r}"
        )
    return translated


class ArgosBackend(TranslationBackend):
    """Offline Argos Translate es -> en backend (preferred when importable)."""

    name = "argos"

    def __init__(self) -> None:
        try:
            import argostranslate.translate as argos_translate
            from argostranslate import package
        except ImportError as error:  # pragma: no cover - depends on install
            raise TranslationBackendError(
                "argostranslate is not installed; install it or select another "
                "backend with --backend"
            ) from error
        installed = [
            item
            for item in package.get_installed_packages()
            if getattr(item, "from_code", None) == "es"
            and getattr(item, "to_code", None) == "en"
        ]
        if not installed:  # pragma: no cover - depends on local models
            raise TranslationBackendError(
                "argostranslate has no installed es -> en model"
            )
        self._translate = argos_translate
        version = getattr(installed[0], "package_version", "unknown")
        self.model = f"argos:es-en:{version}"

    def translate_many(self, texts: Sequence[str]) -> List[str]:
        results: List[str] = []
        for text in texts:
            try:
                results.append(self._translate.translate(text, "es", "en"))
            except Exception as error:  # pragma: no cover - backend-specific
                raise TranslationBackendError(
                    f"argos failed for {text!r}: {error}"
                ) from error
        return results


class DeepTranslatorBackend(TranslationBackend):
    """``deep-translator`` backend with a no-key web fallback.

    The package is preferred. In environments where its Google transport is
    rate-limited, the same public Google endpoint is queried directly and the
    package's MyMemory transport is the final fallback. Every path is still
    machine translation; nothing is hand-authored.
    """

    name = "deep-translator"

    def __init__(self) -> None:
        self._google = None
        self._mymemory = None
        try:
            from deep_translator import GoogleTranslator

            self._google = GoogleTranslator(source="es", target="en")
        except ImportError:
            pass
        try:
            from deep_translator import MyMemoryTranslator

            self._mymemory = MyMemoryTranslator(
                source="spanish", target="english"
            )
        except ImportError:
            pass
        if self._google is None and self._mymemory is None:
            raise TranslationBackendError(
                "deep-translator is not installed; install it or select another "
                "backend with --backend"
            )
        self.model = "google-package"

    def _translate_package(self, texts: Sequence[str]) -> Optional[List[str]]:
        if self._google is None:
            return None
        try:
            result = self._google.translate_batch(list(texts))
        except Exception:
            return None
        if not result or any(not isinstance(item, str) or not item.strip() for item in result):
            return None
        return [str(item) for item in result]

    def translate_many(self, texts: Sequence[str]) -> List[str]:
        result = self._translate_package(texts)
        if result is not None:
            self.model = "google-package"
            return result
        web: List[str] = []
        web_failed = False
        for text in texts:
            try:
                web.append(_google_web_translate(text))
            except Exception:
                web_failed = True
                break
        if not web_failed:
            self.model = "google-web"
            return web
        if self._mymemory is None:
            raise TranslationBackendError(
                "deep-translator Google transport failed and MyMemory is not "
                "available"
            )
        try:
            result = self._mymemory.translate_batch(list(texts))
        except Exception as error:
            raise TranslationBackendError(
                f"deep-translator failed for {list(texts)!r}: {error}"
            ) from error
        if not result or len(result) != len(texts):
            raise TranslationBackendError(
                "deep-translator returned a malformed batch"
            )
        self.model = "mymemory"
        return [str(item) for item in result]


class LibreTranslateBackend(TranslationBackend):
    """``libretranslate:<url>`` backend (URL or API key via environment)."""

    name = "libretranslate"

    def __init__(self, url: str, api_key: Optional[str] = None) -> None:
        if not url:
            raise TranslationBackendError("libretranslate backend needs a URL")
        self.url = url.rstrip("/")
        if not self.url.endswith("/translate"):
            self.url += "/translate"
        self.api_key = api_key or os.environ.get("POLYX_LIBRETRANSLATE_API_KEY")
        self.model = self.url

    def translate_many(self, texts: Sequence[str]) -> List[str]:
        payload: Dict[str, object] = {
            "q": list(texts),
            "source": "es",
            "target": "en",
            "format": "text",
        }
        if self.api_key:
            payload["api_key"] = self.api_key
        try:
            body = _http_json(self.url, payload=payload)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise TranslationBackendError(
                f"libretranslate request failed: {error}"
            ) from error
        if not isinstance(body, Mapping):
            raise TranslationBackendError("libretranslate returned a non-object")
        translated = body.get("translatedText")
        if isinstance(translated, list):
            values = [str(item) for item in translated]
        elif isinstance(translated, str) and len(texts) == 1:
            values = [translated]
        else:
            raise TranslationBackendError(
                "libretranslate returned an unexpected translatedText"
            )
        if len(values) != len(texts):
            raise TranslationBackendError(
                "libretranslate returned the wrong number of translations"
            )
        return values


class DeepLBackend(TranslationBackend):
    """``deepl:<url>`` backend; the API key comes from the environment."""

    name = "deepl"

    def __init__(self, url: str, api_key: Optional[str] = None) -> None:
        if not url:
            raise TranslationBackendError("deepl backend needs a URL")
        self.url = url.rstrip("/")
        if not self.url.endswith("/translate"):
            self.url += "/v2/translate"
        self.api_key = api_key or os.environ.get("POLYX_DEEPL_API_KEY")
        if not self.api_key:
            raise TranslationBackendError(
                "deepl backend needs POLYX_DEEPL_API_KEY"
            )
        self.model = self.url

    def translate_many(self, texts: Sequence[str]) -> List[str]:
        data = urllib.parse.urlencode(
            [
                ("text", text)
                for text in texts
            ]
            + [("source_lang", "ES"), ("target_lang", "EN")]
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Authorization": f"DeepL-Auth-Key {self.api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "polyx-identifier-translator/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise TranslationBackendError(f"deepl request failed: {error}") from error
        translations = body.get("translations") if isinstance(body, Mapping) else None
        if not isinstance(translations, list) or len(translations) != len(texts):
            raise TranslationBackendError("deepl returned a malformed response")
        values: List[str] = []
        for item in translations:
            if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
                raise TranslationBackendError("deepl returned a malformed translation")
            values.append(str(item["text"]))
        return values


def make_backend(spec: Optional[str] = None) -> TranslationBackend:
    """Build a backend from ``--backend``, ``POLYX_TRANSLATE_BACKEND`` or auto."""
    selected = (spec or os.environ.get("POLYX_TRANSLATE_BACKEND") or "auto").strip()
    if not selected or selected == "auto":
        errors: List[str] = []
        for factory in (ArgosBackend, DeepTranslatorBackend):
            try:
                return factory()
            except TranslationBackendError as error:
                errors.append(str(error))
        raise TranslationBackendError(
            "no usable translation backend: " + "; ".join(errors)
        )
    if selected == "argos":
        return ArgosBackend()
    if selected == "deep-translator":
        return DeepTranslatorBackend()
    if selected.startswith("libretranslate:"):
        return LibreTranslateBackend(selected.split(":", 1)[1])
    if selected.startswith("deepl:"):
        return DeepLBackend(selected.split(":", 1)[1])
    raise TranslationBackendError(
        f"unknown translation backend {selected!r}; use argos, deep-translator, "
        "libretranslate:<url> or deepl:<url>"
    )


@dataclass(frozen=True)
class CacheEntry:
    """One cached machine translation with its provenance."""

    translation: str
    backend: str
    model: str
    timestamp: str
    usable: bool = True


class TranslationCache:
    """JSON cache of subword and full-identifier translations."""

    VERSION = 1

    def __init__(
        self,
        path: Path = DEFAULT_CACHE_PATH,
        entries: Optional[Mapping[str, CacheEntry]] = None,
    ) -> None:
        self.path = Path(path)
        self.entries: Dict[str, CacheEntry] = dict(entries or {})

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TranslationCache":
        cache_path = Path(path) if path is not None else DEFAULT_CACHE_PATH
        if not cache_path.exists():
            return cls(cache_path)
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TranslationError(
                f"cannot read translation cache {cache_path}: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise TranslationError(f"translation cache {cache_path} is not an object")
        entries: Dict[str, CacheEntry] = {}
        raw_entries = payload.get("entries", {})
        if not isinstance(raw_entries, Mapping):
            raise TranslationError(
                f"translation cache {cache_path} has no entries object"
            )
        for key, value in raw_entries.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                continue
            translation = value.get("translation")
            if not isinstance(translation, str):
                continue
            entries[key] = CacheEntry(
                translation=translation,
                backend=str(value.get("backend", "unknown")),
                model=str(value.get("model", "unknown")),
                timestamp=str(value.get("timestamp", "")),
                usable=bool(value.get("usable", True)),
            )
        return cls(cache_path, entries)

    def lookup(self, text: str) -> Optional[CacheEntry]:
        """Return the cached entry for ``text``, if any."""
        return self.entries.get(text)

    def store(
        self,
        text: str,
        translation: str,
        backend: str,
        model: str,
        *,
        usable: bool = True,
        timestamp: Optional[str] = None,
    ) -> CacheEntry:
        """Record one translation result with backend, model and timestamp."""
        entry = CacheEntry(
            translation=translation,
            backend=backend,
            model=model,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            usable=usable,
        )
        self.entries[text] = entry
        return entry

    def save(self, path: Optional[Path] = None) -> Path:
        """Write the cache atomically, sorted for stable diffs."""
        cache_path = Path(path) if path is not None else self.path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "entries": {
                key: {
                    "translation": entry.translation,
                    "backend": entry.backend,
                    "model": entry.model,
                    "timestamp": entry.timestamp,
                    "usable": entry.usable,
                }
                for key, entry in sorted(self.entries.items())
            },
        }
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)
        return cache_path


# Spanish grammar words that make a literal subword composition read as
# Spanish (``points_del_borde``). They are not translations: they only trigger
# a full-identifier machine query, which is where the English comes from.
SPANISH_PARTICLES = frozenset(
    {
        "a", "al", "con", "conmutable", "de", "del", "desde", "el", "en",
        "entre", "hacia", "hasta", "la", "las", "los", "o", "para", "por",
        "que", "sin", "sobre", "tras", "y",
    }
)

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "of",
        "on", "or", "the", "to", "with",
    }
)


@dataclass
class TranslationStats:
    """Deterministic counters reported after a manifest generation."""

    cache_hits: int = 0
    cache_misses: int = 0
    glossary_hits: int = 0
    phrase_hits: int = 0
    backend_calls: int = 0
    rejected_candidates: int = 0

    def summary(self) -> str:
        return (
            f"cache hits={self.cache_hits}, cache misses={self.cache_misses}, "
            f"glossary hits={self.glossary_hits}, phrase hits={self.phrase_hits}, "
            f"backend calls={self.backend_calls}, "
            f"rejected candidates={self.rejected_candidates}"
        )


@dataclass
class TranslationPlan:
    """A generated rename map with provenance and counters."""

    mapping: Dict[str, str]
    sources: Dict[str, str]
    stats: TranslationStats = field(default_factory=TranslationStats)


def _normalize_translation(text: str) -> str:
    """Turn a backend result into lower snake_case, dropping accents/commas."""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = normalized.strip().lower().split(",")[0].split(";")[0]
    words = re.findall(r"[a-z0-9]+", normalized)
    return "_".join(words)


def _is_usable_translation(text: str, source: Optional[str] = None) -> bool:
    """True when a backend result can be used as one identifier subword.

    A result equal to the source (after normalization) is rejected: emitting it
    would leave a Spanish token in the renamed identifier and the durable gate
    would still find it. Such a result triggers the full-identifier phrase
    query instead.
    """
    normalized = _normalize_translation(text)
    if not normalized:
        return False
    if source is not None and normalized == _normalize_translation(source):
        return False
    words = normalized.split("_")
    if len(words) != 1:
        return False
    return words[0] not in _STOPWORDS


def _is_usable_phrase(text: str, source: str) -> bool:
    """True when a full-identifier translation actually changed the phrase."""
    normalized = _normalize_translation(text)
    if not normalized:
        return False
    return normalized != _normalize_translation(source)


def _safe_candidate(candidate: str, stats: TranslationStats) -> str:
    """Return a valid target, suffixing builtins/keywords instead of emitting them."""
    if _is_valid_target(candidate):
        return candidate
    stats.rejected_candidates += 1
    for suffix in ("_value", "_item", "_name"):
        trial = candidate + suffix
        if _is_valid_target(trial):
            return trial
    raise TranslationError(
        f"backend produced no valid identifier from {candidate!r}"
    )


def _phrase_key(name: str) -> str:
    """Cache key for a full-identifier machine query."""
    return "@phrase:" + name.lower()


def _phrase_text(name: str) -> str:
    """Readable Spanish phrase for a full-identifier query."""
    return " ".join(split_identifier(name))


def _resolve_parts(
    name: str,
    subwords: Sequence[str],
    translations: Mapping[str, str],
) -> Optional[List[str]]:
    """Compose subword parts, returning None when one is still unresolved."""
    parts: List[str] = []
    for subword in subwords:
        base = _strip_digits(subword)
        digits = subword[len(base) :]
        lower = base.lower()
        if lower in SUBWORD_TRANSLATIONS:
            parts.append(SUBWORD_TRANSLATIONS[lower] + digits)
        elif not spanish_identifier_findings(base):
            parts.append(base + digits)
        elif lower in translations:
            parts.append(translations[lower] + digits)
        else:
            return None
    return parts


def _occupied_names(
    old: str, names_by_file: Mapping[Path, Set[str]]
) -> Set[str]:
    """Names that coexist with ``old`` in every file where it occurs."""
    occupied: Set[str] = set()
    for names in names_by_file.values():
        if old in names:
            occupied.update(names)
    return occupied


def build_identifier_map(
    names_by_file: Mapping[Path, Set[str]],
    cache: TranslationCache,
    *,
    backend: Optional[TranslationBackend] = None,
    backend_name: Optional[str] = None,
    offline: bool = False,
) -> TranslationPlan:
    """Build a collision-checked rename map through glossary + machine translation.

    The glossary and the cache are consulted before the backend; ``--offline``
    turns a missing cache entry into :class:`OfflineCacheMiss` instead of a
    network call. Same-scope and duplicate-target collisions are resolved with
    a deterministic ``_2`` suffix, never by inventing a new translation.
    """
    stats = TranslationStats()
    file_names = {Path(path): set(names) for path, names in names_by_file.items()}
    all_names: Set[str] = set()
    for names in file_names.values():
        all_names.update(names)

    backend_instance = backend

    def ensure_backend() -> TranslationBackend:
        nonlocal backend_instance
        if backend_instance is None:
            backend_instance = make_backend(backend_name)
        return backend_instance

    candidates: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    for name in sorted(all_names):
        if name.startswith("__") and name.endswith("__"):
            continue
        # Curated overrides are authoritative: they cover phrase reordering and
        # English-homograph survivors (``patron``, ``libre``, ``cab``) that the
        # Spanish scanner must not flag. They are consulted before the scanner
        # so pinning one never requires loosening the detector vocabulary.
        if name in EXACT_OVERRIDES:
            target = EXACT_OVERRIDES[name]
            if target != name and _is_valid_target(target):
                candidates[name] = target
                sources[name] = "override"
                continue
        if not spanish_identifier_findings(name):
            continue

        subwords = split_identifier(name)
        translations: Dict[str, str] = {}
        unresolved: List[str] = []
        needs_phrase = False
        for subword in subwords:
            base = _strip_digits(subword)
            lower = base.lower()
            # The glossary is authoritative even for a subword the scanner does
            # not flag (FIBRA, PLACA, RECTO): it is Spanish-only by construction.
            if lower in SUBWORD_TRANSLATIONS:
                stats.glossary_hits += 1
                continue
            if not spanish_identifier_findings(base):
                continue
            entry = cache.lookup(lower)
            if entry is not None:
                stats.cache_hits += 1
                if entry.usable and _is_usable_translation(entry.translation, lower):
                    translations[lower] = _normalize_translation(entry.translation)
                else:
                    needs_phrase = True
                continue
            stats.cache_misses += 1
            unresolved.append(lower)

        if unresolved:
            if offline:
                raise OfflineCacheMiss(
                    "offline mode but these subwords are not cached: "
                    + ", ".join(sorted(set(unresolved)))
                )
            active_backend = ensure_backend()
            try:
                results = active_backend.translate_many(unresolved)
            except TranslationBackendError:
                raise
            stats.backend_calls += 1
            if len(results) != len(unresolved):
                raise TranslationBackendError(
                    "backend returned the wrong number of translations"
                )
            for lower, raw in zip(unresolved, results):
                usable = _is_usable_translation(raw, lower)
                cache.store(
                    lower,
                    raw,
                    active_backend.name,
                    active_backend.model,
                    usable=usable,
                )
                if usable:
                    translations[lower] = _normalize_translation(raw)
                else:
                    needs_phrase = True

        if any(subword.lower() in SPANISH_PARTICLES for subword in subwords):
            needs_phrase = True

        phrase_words: Optional[List[str]] = None
        if needs_phrase:
            key = _phrase_key(name)
            phrase_source = _phrase_text(name)
            entry = cache.lookup(key)
            if entry is not None:
                stats.cache_hits += 1
                stats.phrase_hits += 1
                phrase_words = _normalize_translation(entry.translation).split("_")
            elif offline:
                raise OfflineCacheMiss(
                    f"offline mode but no phrase cache entry for {name!r}"
                )
            else:
                active_backend = ensure_backend()
                try:
                    result = active_backend.translate_many([phrase_source])[0]
                except TranslationBackendError:
                    raise
                stats.backend_calls += 1
                stats.phrase_hits += 1
                cache.store(
                    key,
                    result,
                    active_backend.name,
                    active_backend.model,
                )
                phrase_words = _normalize_translation(result).split("_")
            if not phrase_words or not any(phrase_words):
                raise TranslationError(
                    f"phrase translation for {name!r} was empty"
                )
            if _normalize_translation("_".join(phrase_words)) == _normalize_translation(
                phrase_source
            ):
                raise TranslationError(
                    f"phrase translation for {name!r} left the phrase in Spanish"
                )

        if phrase_words is not None:
            candidate = _join_identifier(name, phrase_words)
            provenance = "phrase"
        else:
            parts = _resolve_parts(name, subwords, translations)
            if parts is None:
                raise TranslationError(
                    f"could not resolve every subword of {name!r}"
                )
            candidate = _join_identifier(name, parts)
            provenance = "glossary" if translations == {} else "mt"
        if candidate == name:
            provenance = "phrase" if phrase_words is not None else provenance
        if not _is_valid_target(candidate):
            candidate = _safe_candidate(candidate, stats)
            provenance += "+sanitized"
        candidates[name] = candidate
        sources[name] = provenance

    # Resolve duplicate targets and same-scope coexistence deterministically.
    used: Set[str] = set()
    all_sources = set(candidates)
    resolved: Dict[str, str] = {}
    for old in sorted(candidates):
        target = candidates[old]
        base_target = target
        if target in all_sources or target == old:
            target = ""
        suffix = 2
        occupied = _occupied_names(old, file_names)
        while True:
            conflict = (
                not target
                or target in used
                or target in all_sources
                or (target in occupied and target not in all_sources)
            )
            if not conflict:
                break
            target = f"{base_target}_{suffix}"
            suffix += 1
            if suffix > 100:
                raise TranslationError(
                    f"could not find a free target for {old!r}"
                )
        if target != base_target:
            stats.rejected_candidates += 1
            sources[old] = sources[old] + "+collision"
        resolved[old] = target
        used.add(target)

    validate_rename_map(resolved)
    return TranslationPlan(mapping=resolved, sources=sources, stats=stats)


def write_manifest(
    path: Path,
    plan: TranslationPlan,
    *,
    command: str = "",
) -> Path:
    """Write a loadable ``IDENTIFIER_TRANSLATIONS`` manifest module."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '"""Generated identifier manifest -- do not edit by hand.',
        "",
        "Produced by the translation-software bridge in",
        "``tests/translate_code_text.py translate-identifiers`` (D033).",
    ]
    if command:
        lines.append(f"Command: {command}")
    lines.append(f"Provenance: {plan.stats.summary()}")
    lines.extend(
        [
            "",
            "Every entry comes from the curated glossary or from the cached",
            "machine-translation result recorded in translation_cache.json.",
            '"""',
            "",
            "IDENTIFIER_TRANSLATIONS = {",
        ]
    )
    for old in sorted(plan.mapping):
        lines.append(f"    {old!r}: {plan.mapping[old]!r},")
    lines.extend(["}", "", "IDENTIFIER_TRANSLATION_SOURCES = {"])
    for old in sorted(plan.mapping):
        lines.append(f"    {old!r}: {plan.sources.get(old, 'unknown')!r},")
    lines.append("}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


# ---------------------------------------------------------------------------
# Machine translation of user-visible report headings (D033, S05).
#
# The identifier bridge normalizes every result to one snake_case subword, so
# it cannot carry a report label with spaces, HTML tags or units. This prose
# path keeps the backend output verbatim (whitespace included) and validates
# that the result is actually English and structurally equivalent to the
# Spanish source before it can enter a manifest.
# ---------------------------------------------------------------------------

_HEADING_CACHE_PREFIX = "@heading:"

# Accented letters and inverted punctuation are the cheap, high-signal test
# for a result that is still Spanish. The word list catches short labels whose
# accents were stripped by the backend (``del``, ``para``, ``menos``).
_HEADING_SPANISH_LETTERS = re.compile(r"[áéíóúñüÁÉÍÓÚÑÜ¿¡]")

_HEADING_SPANISH_WORDS = frozenset(
    {
        "el", "la", "los", "las", "del", "que", "con", "para", "por",
        "una", "un", "se", "es", "su", "sus", "como", "cada", "sobre",
        "entre", "sin", "no", "de", "en", "al", "lo", "más", "menos",
    }
)

_HEADING_PLACEHOLDER = re.compile(r"\{[^{}]*\}")
_HEADING_TAG = re.compile(r"<[^<>]*>")
_HEADING_DIGITS = re.compile(r"\d+")
_HEADING_LEADING_WS = re.compile(r"^\s*")
_HEADING_TRAILING_WS = re.compile(r"\s*$")


def _heading_key(query: str) -> str:
    """Cache key for a report-heading machine query."""
    return _HEADING_CACHE_PREFIX + query


def _heading_contains_word(text: str, word: str) -> bool:
    """True when ``word`` appears as a whole word (unicode aware)."""
    return re.search(
        rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE
    ) is not None


def _is_usable_heading(text: str, source: str) -> bool:
    """True when ``text`` is an English report heading for ``source``.

    The result must be non-empty, must differ from the source, must not keep
    Spanish letters or whole-word Spanish grammar, and must preserve every
    ``{placeholder}``, ``<tag>``, digit run and the literal ``µm``. Leading
    and trailing whitespace are preserved as well, because report headings
    are often embedded in HTML with deliberate spacing.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if text.strip().casefold() == source.strip().casefold():
        return False
    if _HEADING_SPANISH_LETTERS.search(text):
        return False
    if any(_heading_contains_word(text, word) for word in _HEADING_SPANISH_WORDS):
        return False
    if _HEADING_LEADING_WS.match(text).group() != _HEADING_LEADING_WS.match(
        source
    ).group():
        return False
    if _HEADING_TRAILING_WS.search(text).group() != _HEADING_TRAILING_WS.search(
        source
    ).group():
        return False
    for pattern in (_HEADING_PLACEHOLDER, _HEADING_TAG, _HEADING_DIGITS):
        if Counter(pattern.findall(text)) != Counter(pattern.findall(source)):
            return False
    if "µm" in source and "µm" not in text:
        return False
    return True


def _default_heading_glossary() -> Mapping[str, str]:
    """Load the project EN dictionary, tolerating a harness launch from tests/.

    ``tests/translate_code_text.py`` runs with ``tests/`` as ``sys.path[0]``,
    so ``polyx`` may not be importable even though the repo root is the cwd.
    Adding the repo root and retrying is enough; the module has no logic.
    """
    try:
        from polyx.core.translations_en import EN
    except ModuleNotFoundError:
        root = str(REPO_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from polyx.core.translations_en import EN
    return EN


@dataclass
class HeadingTranslationPlan:
    """Generated report-heading translations with provenance and counters."""

    mapping: Dict[str, str]
    sources: Dict[str, str]
    stats: TranslationStats = field(default_factory=TranslationStats)


def translate_heading_queries(
    queries: Iterable[str],
    cache: TranslationCache,
    *,
    backend: Optional[TranslationBackend] = None,
    backend_name: Optional[str] = None,
    offline: bool = False,
    glossary: Optional[Mapping[str, str]] = None,
) -> HeadingTranslationPlan:
    """Translate report headings through the EN dictionary + cached MT.

    The curated EN dictionary (``polyx.core.translations_en.EN``) is an
    exact-match glossary: a hit never calls the backend. Every other query is
    looked up in the cache under ``@heading:<query>`` and the remaining ones
    are sent to the backend in a single batch. Results are stored verbatim
    with their backend/model/timestamp and validated with
    :func:`_is_usable_heading`; an unusable result raises
    :class:`TranslationError`, so a Spanish or corrupted heading can never be
    written into the manifest. A cache miss under ``--offline`` raises
    :class:`OfflineCacheMiss` instead of touching the network.
    """
    stats = TranslationStats()
    if glossary is None:
        # Lazy import: the tool is also used outside the repo checkout, and
        # importing polyx pulls the model package.
        glossary = _default_heading_glossary()

    ordered: List[str] = []
    seen: Set[str] = set()
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise TranslationError(
                f"heading query must be a non-empty string, got {query!r}"
            )
        if query not in seen:
            seen.add(query)
            ordered.append(query)

    mapping: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    pending: List[str] = []
    for query in ordered:
        if query in glossary:
            mapping[query] = glossary[query]
            sources[query] = "en-dictionary"
            stats.glossary_hits += 1
            continue
        entry = cache.lookup(_heading_key(query))
        if entry is not None:
            if entry.usable and _is_usable_heading(entry.translation, query):
                mapping[query] = entry.translation
                sources[query] = "cache"
                stats.cache_hits += 1
                continue
            # A stale or corrupt entry is a miss: re-query rather than pin
            # the bad text into the manifest.
            stats.rejected_candidates += 1
        stats.cache_misses += 1
        pending.append(query)

    if pending:
        if offline:
            raise OfflineCacheMiss(
                "offline mode but these headings are not cached: "
                + ", ".join(repr(query) for query in pending)
            )
        active_backend = backend
        if active_backend is None:
            active_backend = make_backend(backend_name)
        results = active_backend.translate_many(pending)
        stats.backend_calls += 1
        if len(results) != len(pending):
            raise TranslationBackendError(
                "backend returned the wrong number of translations"
            )
        for query, raw in zip(pending, results):
            usable = _is_usable_heading(raw, query)
            cache.store(
                _heading_key(query),
                raw,
                active_backend.name,
                active_backend.model,
                usable=usable,
            )
            if not usable:
                raise TranslationError(
                    f"unusable heading translation for {query!r}: {raw!r}"
                )
            mapping[query] = raw
            sources[query] = "mt"

    ordered_mapping = {query: mapping[query] for query in ordered}
    ordered_sources = {query: sources[query] for query in ordered}
    return HeadingTranslationPlan(
        mapping=ordered_mapping, sources=ordered_sources, stats=stats
    )


def write_heading_manifest(
    path: Path,
    plan: HeadingTranslationPlan,
    *,
    command: str = "",
) -> Path:
    """Write a loadable ``HEADING_TRANSLATIONS`` manifest module."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '"""Generated report-heading manifest -- do not edit by hand.',
        "",
        "Produced by the translation-software bridge in",
        "``tests/translate_code_text.py translate-headings`` (D033).",
    ]
    if command:
        lines.append(f"Command: {command}")
    lines.append(f"Provenance: {plan.stats.summary()}")
    lines.extend(
        [
            "",
            "Every entry comes from the project EN dictionary or from the cached",
            "machine-translation result recorded in translation_cache.json. No",
            "Spanish-to-English pair in this file was hand-authored.",
            '"""',
            "",
            "HEADING_TRANSLATIONS = {",
        ]
    )
    for query in sorted(plan.mapping):
        lines.append(f"    {query!r}: {plan.mapping[query]!r},")
    lines.extend(["}", "", "HEADING_TRANSLATION_SOURCES = {"])
    for query in sorted(plan.mapping):
        lines.append(f"    {query!r}: {plan.sources.get(query, 'unknown')!r},")
    lines.append("}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


# ---------------------------------------------------------------------------
# Module stem discovery.
# ---------------------------------------------------------------------------


def discover_module_stems(root: Path = POLYX_ROOT) -> frozenset:
    """Return every polyx module stem and package directory name.

    Stems are only used to skip imported symbols in import statements; a local
    variable with the same name is still renamed (D027).
    """
    stems: Set[str] = {root.name}
    if not root.is_dir():
        return frozenset(stems)
    for path in root.rglob("*.py"):
        stems.add(path.stem)
    for path in root.rglob("*"):
        if path.is_dir():
            stems.add(path.name)
    stems.discard("__init__")
    return frozenset(stems)


# ---------------------------------------------------------------------------
# Rename map validation and preflight.
# ---------------------------------------------------------------------------


def validate_rename_map(renames: Mapping[str, str]) -> None:
    """Validate a flat rename map, raising ``ValueError`` on a bad entry.

    Refuses empty/non-string entries, invalid identifiers, keywords, builtins,
    duplicate targets and targets that are also sources (a second apply would
    rename the first apply's output, so the codemod would not be idempotent).
    """
    if not isinstance(renames, Mapping):
        raise ValueError("identifier map must be a mapping of old -> new names")
    seen_targets: Dict[str, str] = {}
    for old, new in renames.items():
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError(f"identifier map entries must be strings: {old!r}")
        if not old:
            raise ValueError("identifier map contains an empty source name")
        if old == new:
            raise ValueError(f"{old!r}: source and target are identical")
        if not new.isidentifier():
            raise ValueError(f"{old!r} -> {new!r}: target is not a valid identifier")
        if keyword.iskeyword(new):
            raise ValueError(f"{old!r} -> {new!r}: target is a Python keyword")
        if new in _BUILTIN_NAMES:
            raise ValueError(f"{old!r} -> {new!r}: target is a Python builtin")
        if new in seen_targets:
            raise ValueError(
                f"{old!r} and {seen_targets[new]!r} both target {new!r}"
            )
        seen_targets[new] = old
    chained = sorted(set(renames.values()) & set(renames.keys()))
    if chained:
        raise ValueError(
            "targets are also sources, which would break idempotency: "
            + ", ".join(chained)
        )


def _symbol_tables(table: symtable.SymbolTable) -> Iterator[symtable.SymbolTable]:
    """Yield a symbol table and all of its nested child tables."""
    yield table
    for child in table.get_children():
        yield from _symbol_tables(child)


def same_scope_collisions(
    text: str, renames: Mapping[str, str], label: str = "<module>"
) -> List[str]:
    """Return scopes where an old name and its target already coexist."""
    problems: List[str] = []
    if not renames:
        return problems
    try:
        root = symtable.symtable(text, label, "exec")
    except SyntaxError as error:  # pragma: no cover - compile() reports too
        return [f"{label}: cannot build symtable: {error}"]
    for table in _symbol_tables(root):
        names = set(table.get_identifiers())
        for old, new in renames.items():
            if old in names and new in names:
                problems.append(
                    f"{label}: scope {table.get_name()!r} already has both "
                    f"{old!r} and its target {new!r}"
                )
    return problems


def preflight_module(
    source: str,
    renames: Mapping[str, str],
    label: str,
    transformed: Optional[str] = None,
) -> None:
    """Refuse scope collisions or uncompilable output.

    ``renames`` must be the map that actually applies to this module: for
    polyx scope the full map, for consumer scope only the imported bindings.
    """
    problems = same_scope_collisions(source, renames, label)
    if problems:
        raise ValueError("; ".join(problems))
    if transformed is not None:
        try:
            compile(transformed, label, "exec")
        except SyntaxError as error:
            raise ValueError(f"{label}: transformed module does not compile: {error}")


# ---------------------------------------------------------------------------
# Consumer import analysis.
# ---------------------------------------------------------------------------


@dataclass
class ConsumerImports:
    """Names imported from polyx and the local bindings they create.

    ``protected`` holds imported names that the consuming module also defines
    locally (a shadowing ``def``/``class``/assignment). Those stay completely
    untouched: the local definition wins and renaming the dead import would be
    a surprising edit of test-local source.
    """

    local_to_new: Dict[str, str]
    polyx_callables: Set[str]
    protected: Set[str]
    attribute_roots: Set[str] = field(default_factory=set)
    non_model_roots: Set[str] = field(default_factory=set)
    renamable_roots: Set[str] = field(default_factory=set)


def _dotted_name(node: Optional[cst.CSTNode]) -> Optional[str]:
    """Return the dotted text of a Name/Attribute chain, or None."""
    if node is None:
        return None
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        prefix = _dotted_name(node.value)
        if prefix is None:
            return None
        return f"{prefix}.{node.attr.value}"
    return None


def _is_polyx_module(module: Optional[str], relative: Sequence[cst.Dot]) -> bool:
    """True for absolute ``polyx`` imports and for relative package imports."""
    if relative:
        return True
    if not module:
        return False
    return module == "polyx" or module.startswith("polyx.")


def is_dynamic_attribute_call(func: cst.CSTNode) -> bool:
    """True for ``getattr``/``setattr`` and pytest's ``monkeypatch.setattr``.

    The second argument of these calls is an attribute name, so it is one of
    the few string tokens the codemod and the audit may rename.
    """
    name = _dotted_name(func)
    if name is None:
        return False
    return name in DYNAMIC_GETTERS or name.endswith("monkeypatch.setattr")


def analyze_consumer_imports(
    module: cst.Module,
    renames: Mapping[str, str],
    stems: Iterable[str] = (),
    label: str = "",
    attribute_modules: Iterable[str] = (),
) -> ConsumerImports:
    """Collect the local bindings a consumer file gets from polyx imports.

    ``from polyx.core.morphology import medir`` binds ``medir``, so every use
    of that binding may be renamed. A test-local ``def medir`` creates no
    import binding and therefore stays untouched. ``from ... import x as y``
    renames the imported symbol ``x`` but keeps the local alias ``y`` unless
    ``y`` itself appears in the map.
    """
    local_to_new: Dict[str, str] = {}
    callables: Set[str] = set()
    protected: Set[str] = set()
    non_model_roots: Set[str] = set()
    renamable_roots: Set[str] = set()
    shadowed = _module_local_definitions(module)
    stem_set = frozenset(stems)
    attribute_module_set = frozenset(attribute_modules)
    is_model_file = label.replace("\\", "/").startswith("polyx/model/")
    for node in module.body:
        if not isinstance(node, cst.SimpleStatementLine):
            continue
        for small in node.body:
            if isinstance(small, cst.ImportFrom):
                _collect_import_from(
                    small, renames, local_to_new, callables, protected, shadowed,
                    stem_set, non_model_roots, is_model_file,
                    attribute_module_set, renamable_roots,
                )
            elif isinstance(small, cst.Import):
                _collect_import(
                    small, renames, local_to_new, non_model_roots, is_model_file,
                    attribute_module_set, renamable_roots, callables,
                )
    _collect_nested_imports(
        module, renames, local_to_new, callables, protected, stem_set,
        non_model_roots, is_model_file, attribute_module_set, renamable_roots,
    )
    for name in protected:
        local_to_new.pop(name, None)
    return ConsumerImports(
        local_to_new=local_to_new,
        polyx_callables=callables,
        protected=protected,
        attribute_roots=_argparse_roots(module),
        non_model_roots=non_model_roots,
        renamable_roots=renamable_roots,
    )


def _collect_nested_imports(
    module: cst.Module,
    renames: Mapping[str, str],
    local_to_new: Dict[str, str],
    callables: Set[str],
    protected: Set[str],
    stems: frozenset,
    non_model_roots: Set[str],
    is_model_file: bool,
    attribute_modules: frozenset = frozenset(),
    renamable_roots: Optional[Set[str]] = None,
) -> None:
    """Collect polyx imports that live inside a function or method body.

    ``report_html.generate_report`` imports ``cronometro`` inside the function;
    a top-level-only scan would rename the import and leave the call behind.
    Nested imports are explicit polyx bindings, so no module-level shadow
    analysis applies to them.
    """
    if renamable_roots is None:
        renamable_roots = set()

    class _NestedImportVisitor(cst.CSTVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            self.depth += 1

        def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
            self.depth -= 1

        def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> None:
            self.depth += 1

        def leave_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> None:
            self.depth -= 1

        def visit_ClassDef(self, node: cst.ClassDef) -> None:
            self.depth += 1

        def leave_ClassDef(self, node: cst.ClassDef) -> None:
            self.depth -= 1

        def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
            if self.depth:
                _collect_import_from(
                    node, renames, local_to_new, callables, protected, set(),
                    stems, non_model_roots, is_model_file,
                    attribute_modules, renamable_roots,
                )

        def visit_Import(self, node: cst.Import) -> None:
            if self.depth:
                _collect_import(
                    node, renames, local_to_new, non_model_roots, is_model_file,
                    attribute_modules, renamable_roots, callables,
                )

    module.visit(_NestedImportVisitor())


def _argparse_roots(module: cst.Module) -> Set[str]:
    """Names that hold an argparse namespace in this module.

    ``args.salida`` is an argparse destination, not a polyx model attribute;
    the consumer transformer must not rename it even when ``salida`` is in the
    manifest. The same applies to ``parser.parse_known_args()`` and an explicit
    ``argparse.Namespace(...)``.
    """
    roots: Set[str] = set()

    def collect_target(target: cst.CSTNode) -> None:
        if isinstance(target, cst.Name):
            roots.add(target.value)
        elif isinstance(target, (cst.Tuple, cst.List)):
            for element in target.elements:
                collect_target(element.value)

    class _Visitor(cst.CSTVisitor):
        def visit_Assign(self, node: cst.Assign) -> None:
            if _is_argparse_call(node.value):
                for target in node.targets:
                    collect_target(target.target)

        def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
            if node.value is not None and _is_argparse_call(node.value):
                collect_target(node.target)

    module.visit(_Visitor())
    return roots


def _is_argparse_call(node: cst.CSTNode) -> bool:
    """True for ``...parse_args()``, ``parse_known_args()`` and Namespace()."""
    if not isinstance(node, cst.Call):
        return False
    name = _dotted_name(node.func) or ""
    return name.endswith(
        ("parse_args", "parse_known_args", "Namespace")
    )


def _module_local_definitions(module: cst.Module) -> Set[str]:
    """Names bound by top-level defs, classes and assignments in a module."""
    names: Set[str] = set()
    for node in module.body:
        if isinstance(node, (cst.FunctionDef, cst.ClassDef)):
            names.add(node.name.value)
        if not isinstance(node, cst.SimpleStatementLine):
            continue
        for small in node.body:
            if isinstance(small, (cst.FunctionDef, cst.ClassDef)):
                names.add(small.name.value)
            elif isinstance(small, cst.Assign):
                for target in small.targets:
                    if isinstance(target.target, cst.Name):
                        names.add(target.target.value)
    return names


def _module_stem_bindings(module: cst.Module, stems: Iterable[str]) -> Set[str]:
    """Local names bound to an imported module stem.

    ``from polyx.gui import iconos`` binds the module ``iconos``. The import
    path and the binding must both stay: the plan excludes module stems in
    import context, and renaming only the uses would break the module
    reference. Nested imports count too: ``from polyx.core import
    observabilidad`` inside a function binds the same stem and must not be
    renamed while the import stays.
    """
    stem_set = frozenset(stems)

    class _StemBindingVisitor(cst.CSTVisitor):
        def __init__(self) -> None:
            self.bindings: Set[str] = set()

        def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
            if isinstance(node.names, cst.ImportStar):
                return False
            for alias in node.names:
                if alias.asname is not None:
                    continue
                imported = _dotted_name(alias.name)
                if imported in stem_set:
                    self.bindings.add(imported)
            return False

    visitor = _StemBindingVisitor()
    module.visit(visitor)
    return visitor.bindings


def _collect_import_from(
    node: cst.ImportFrom,
    renames: Mapping[str, str],
    local_to_new: Dict[str, str],
    callables: Set[str],
    protected: Set[str],
    shadowed: Set[str],
    stems: frozenset,
    non_model_roots: Set[str],
    is_model_file: bool,
    attribute_modules: frozenset = frozenset(),
    renamable_roots: Optional[Set[str]] = None,
) -> None:
    module_name = _dotted_name(node.module)
    if not _is_polyx_module(module_name, node.relative):
        return
    attribute_module = bool(module_name and module_name in attribute_modules)
    roots = renamable_roots if renamable_roots is not None else set()
    if node.relative:
        model_module = is_model_file
    else:
        model_module = bool(
            module_name
            and (module_name == "polyx.core" or module_name.startswith("polyx.core."))
        )
    if isinstance(node.names, cst.ImportStar):
        return
    for alias in node.names:
        imported = _dotted_name(alias.name)
        if imported is None:
            continue
        full_module = f"{module_name}.{imported}" if module_name else imported
        alias_attribute_module = attribute_module or full_module in attribute_modules
        if alias.asname is not None:
            local = alias.asname.name.value
            if not model_module:
                non_model_roots.add(local)
            if alias_attribute_module:
                roots.add(local)
            if local in shadowed:
                # The local definition shadows the alias; never treat it as a
                # polyx callable for keyword renaming.
                protected.add(local)
                continue
            target = renames.get(local)
            if target is not None:
                local_to_new[local] = target
            callables.add(local)
        else:
            if not model_module:
                non_model_roots.add(imported)
            if alias_attribute_module:
                roots.add(imported)
            if imported in shadowed or (
                imported in stems and not alias_attribute_module
            ):
                # A local definition shadows the import, or the imported name
                # is a module stem (``from polyx.gui import iconos``): leave
                # both the import and its uses untouched. An explicitly
                # renamable attribute module overrides the stem protection.
                protected.add(imported)
                continue
            target = renames.get(imported)
            if target is not None:
                local_to_new[imported] = target
            callables.add(imported)


def _collect_import(
    node: cst.Import,
    renames: Mapping[str, str],
    local_to_new: Dict[str, str],
    non_model_roots: Set[str],
    is_model_file: bool,
    attribute_modules: frozenset = frozenset(),
    renamable_roots: Optional[Set[str]] = None,
    callables: Optional[Set[str]] = None,
) -> None:
    """Handle ``import polyx.core.morphology as m``; the path never changes."""
    roots = renamable_roots if renamable_roots is not None else set()
    for alias in node.names:
        if alias.asname is None:
            continue
        local = alias.asname.name.value
        imported = _dotted_name(alias.name) or ""
        model_module = imported == "polyx.core" or imported.startswith("polyx.core.")
        if not model_module:
            non_model_roots.add(local)
        if imported in attribute_modules:
            roots.add(local)
            if callables is not None:
                callables.add(local)
        target = renames.get(local)
        if target is not None:
            local_to_new[local] = target


# ---------------------------------------------------------------------------
# Transformer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentifierChange:
    """One identifier rewrite (applied, or would-be under ``--dry-run``)."""

    old: str
    new: str
    kind: str  # name, import, attribute, keyword, getattr, __all__


_STRING_PREFIX = re.compile(r"^(?P<prefix>[A-Za-z]*)(?P<quotes>\"\"\"|'''|\"|')")


def _string_token(old_token: str, new_value: str) -> str:
    """Rebuild a string token with a new body, keeping prefix and quotes."""
    match = _STRING_PREFIX.match(old_token)
    if match is None:
        return old_token
    return f"{match.group('prefix')}{match.group('quotes')}{new_value}{match.group('quotes')}"


def _renamed_string(node: cst.SimpleString, renames: Mapping[str, str]) -> Optional[cst.SimpleString]:
    """Return a SimpleString with a renamed body, or None when unchanged."""
    value = node.evaluated_value
    if not isinstance(value, str):
        return None
    target = renames.get(value)
    if target is None or target == value:
        return None
    try:
        token = _string_token(node.value, target)
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        return None
    return node.with_changes(value=token)


def _rewrite_all_entries(
    node: cst.CSTNode, renames: Mapping[str, str], changes: List[IdentifierChange]
) -> cst.CSTNode:
    """Rewrite the string entries of an ``__all__`` value node."""
    if isinstance(node, cst.SimpleString):
        renamed = _renamed_string(node, renames)
        if renamed is None:
            return node
        changes.append(
            IdentifierChange(
                old=node.evaluated_value,
                new=renamed.evaluated_value,
                kind="__all__",
            )
        )
        return renamed
    if isinstance(node, (cst.List, cst.Tuple, cst.Set)):
        elements = []
        for element in node.elements:
            elements.append(element.with_changes(value=_rewrite_all_entries(element.value, renames, changes)))
        return node.with_changes(elements=elements)
    return node


class IdentifierTranslator(cst.CSTTransformer):
    """Rename identifiers from an exact map, leaving every other token alone.

    ``scope="polyx"`` renames every matching ``cst.Name``. ``scope="consumers"``
    renames only names imported from polyx, attribute accesses in the map and
    keyword arguments passed to imported polyx callables.
    """

    def __init__(
        self,
        renames: Mapping[str, str],
        *,
        scope: str = "polyx",
        stems: Iterable[str] = (),
        imported: Optional[Mapping[str, str]] = None,
        polyx_callables: Optional[Iterable[str]] = None,
        protected: Optional[Iterable[str]] = None,
        attribute_roots: Optional[Iterable[str]] = None,
        non_model_roots: Optional[Iterable[str]] = None,
        renamable_roots: Optional[Iterable[str]] = None,
    ) -> None:
        if scope not in {"polyx", "consumers"}:
            raise ValueError(f"unknown scope: {scope!r}")
        self.renames = dict(renames)
        self.scope = scope
        self.stems = frozenset(stems)
        self.imported = dict(imported or {})
        self.polyx_callables = frozenset(polyx_callables or ())
        self.protected = frozenset(protected or ())
        self.attribute_roots = frozenset(attribute_roots or ())
        self.non_model_roots = frozenset(non_model_roots or ())
        self.renamable_roots = frozenset(renamable_roots or ())
        self.changes: List[IdentifierChange] = []
        # ``str.format`` field names live inside string literals, which this
        # codemod never rewrites. Renaming their keyword arguments would break
        # the placeholder contract, so those Name nodes stay untouched.
        self._protected_keywords: Set[int] = set()

    # -- imports ---------------------------------------------------------

    def visit_Import(self, node: cst.Import) -> bool:
        # Do not descend into the module path; aliases are handled in leave.
        return False

    def leave_Import(
        self, original_node: cst.Import, updated_node: cst.Import
    ) -> cst.Import:
        aliases = []
        for alias in updated_node.names:
            asname = alias.asname
            if asname is not None:
                new = self.renames.get(asname.name.value)
                if new is not None and new != asname.name.value:
                    self.changes.append(
                        IdentifierChange(asname.name.value, new, "import")
                    )
                    alias = alias.with_changes(
                        asname=asname.with_changes(
                            name=asname.name.with_changes(value=new)
                        )
                    )
            aliases.append(alias)
        return updated_node.with_changes(names=aliases)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        # Skip both the module path and the imported names; leave rebuilds the
        # aliases with module stems preserved.
        return False

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        if isinstance(updated_node.names, cst.ImportStar):
            return updated_node
        module_name = _dotted_name(updated_node.module)
        polyx_import = _is_polyx_module(module_name, updated_node.relative)
        if self.scope == "consumers" and not polyx_import:
            return updated_node
        aliases = []
        for alias in updated_node.names:
            imported = _dotted_name(alias.name)
            new_alias = alias
            shadowed = (
                imported is not None
                and imported in self.protected
                and alias.asname is None
            )
            if (
                imported is not None
                and imported not in self.stems
                and not shadowed
            ):
                new_imported = self.renames.get(imported)
                if new_imported is not None and new_imported != imported:
                    self.changes.append(
                        IdentifierChange(imported, new_imported, "import")
                    )
                    new_alias = new_alias.with_changes(
                        name=new_alias.name.with_changes(value=new_imported)
                    )
            asname = new_alias.asname
            if asname is not None:
                new_local = self.renames.get(asname.name.value)
                if new_local is not None and new_local != asname.name.value:
                    self.changes.append(
                        IdentifierChange(asname.name.value, new_local, "import")
                    )
                    new_alias = new_alias.with_changes(
                        asname=asname.with_changes(
                            name=asname.name.with_changes(value=new_local)
                        )
                    )
            aliases.append(new_alias)
        return updated_node.with_changes(names=aliases)

    # -- names, attributes, keyword arguments ----------------------------

    def visit_Call(self, node: cst.Call) -> bool:
        """Protect ``str.format`` field names and JSONL data keys.

        Neither is a Python symbol: the placeholder lives in an untouched
        string literal and the data key lives in the JSONL payload.
        """
        func = node.func
        if isinstance(func, cst.Attribute):
            callee = func.attr.value
        elif isinstance(func, cst.Name):
            callee = func.value
        else:
            callee = None
        if (
            isinstance(func, cst.Attribute)
            and callee in {"format", "format_map"}
        ) or callee in OBSERVABILITY_SINKS:
            for arg in node.args:
                if arg.keyword is not None:
                    self._protected_keywords.add(id(arg.keyword))
        return True

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.Name:
        old = original_node.value
        if old in self.protected or id(original_node) in self._protected_keywords:
            return updated_node
        new = (
            self.renames.get(old)
            if self.scope == "polyx"
            else self.imported.get(old)
        )
        if new is None or new == old:
            return updated_node
        self.changes.append(IdentifierChange(old, new, "name"))
        return updated_node.with_changes(value=new)

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.Attribute:
        if self.scope != "consumers":
            return updated_node
        root = _call_root_name(original_node)
        if root in self.attribute_roots:
            return updated_node
        if root in self.non_model_roots and root not in self.renamable_roots:
            return updated_node
        old = original_node.attr.value
        new = self.renames.get(old)
        if new is None or new == old:
            return updated_node
        self.changes.append(IdentifierChange(old, new, "attribute"))
        return updated_node.with_changes(
            attr=updated_node.attr.with_changes(value=new)
        )

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.Call:
        call = updated_node
        # 1. Dynamic attribute strings: getattr(obj, "ruta") and friends.
        func = original_node.func
        if is_dynamic_attribute_call(func):
            args = list(call.args)
            first_root = _call_root_name(args[0].value) if args else None
            if (
                first_root in self.non_model_roots
                and first_root not in self.renamable_roots
            ):
                args = []
            if len(args) >= 2:
                second = args[1].value
                if isinstance(second, cst.SimpleString):
                    renamed = _renamed_string(second, self.renames)
                    if renamed is not None:
                        self.changes.append(
                            IdentifierChange(
                                second.evaluated_value,
                                renamed.evaluated_value,
                                "getattr",
                            )
                        )
                        args[1] = args[1].with_changes(value=renamed)
                        call = call.with_changes(args=args)
        # 2. Keyword arguments of calls to imported polyx callables.
        if self.scope == "consumers":
            root = _call_root_name(original_node.func)
            if root in self.polyx_callables:
                args = list(call.args)
                original_args = list(original_node.args)
                for position, arg in enumerate(args):
                    keyword_node = arg.keyword
                    if keyword_node is None:
                        continue
                    # Protection is keyed by the original node id; the
                    # updated node carries a rebuilt keyword Name.
                    original_keyword = (
                        original_args[position].keyword
                        if position < len(original_args)
                        else None
                    )
                    if (
                        original_keyword is not None
                        and id(original_keyword) in self._protected_keywords
                    ):
                        continue
                    new_keyword = self.renames.get(keyword_node.value)
                    if new_keyword is None or new_keyword == keyword_node.value:
                        continue
                    self.changes.append(
                        IdentifierChange(keyword_node.value, new_keyword, "keyword")
                    )
                    args[position] = arg.with_changes(
                        keyword=keyword_node.with_changes(value=new_keyword)
                    )
                call = call.with_changes(args=args)
        return call

    # -- __all__ ---------------------------------------------------------

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.Assign:
        if not _targets_all(original_node.targets):
            return updated_node
        return updated_node.with_changes(
            value=_rewrite_all_entries(updated_node.value, self.renames, self.changes)
        )

    def leave_AugAssign(
        self, original_node: cst.AugAssign, updated_node: cst.AugAssign
    ) -> cst.AugAssign:
        if not isinstance(original_node.target, cst.Name):
            return updated_node
        if original_node.target.value != "__all__":
            return updated_node
        return updated_node.with_changes(
            value=_rewrite_all_entries(updated_node.value, self.renames, self.changes)
        )


def _targets_all(targets: Sequence[cst.AssignTarget]) -> bool:
    """True when every assign target is the ``__all__`` name."""
    return bool(targets) and all(
        isinstance(target.target, cst.Name) and target.target.value == "__all__"
        for target in targets
    )


def _call_root_name(node: cst.CSTNode) -> Optional[str]:
    """Return the base Name of a call expression (``a.b.c()`` -> ``a``)."""
    current = node
    while isinstance(current, cst.Attribute):
        current = current.value
    if isinstance(current, cst.Name):
        return current.value
    return None


# ---------------------------------------------------------------------------
# Public translation API.
# ---------------------------------------------------------------------------


def translate_source(
    text: str,
    renames: Mapping[str, str],
    *,
    scope: str = "polyx",
    stems: Iterable[str] = (),
    label: str = "<module>",
    attribute_modules: Iterable[str] = (),
) -> Tuple[str, List[IdentifierChange]]:
    """Apply an identifier map to a module and return ``(new_text, changes)``.

    The map is validated first, then consumer imports are analysed, then the
    transformed module is preflighted for same-scope collisions and compiled.
    ``renames`` may be empty, in which case the source is returned unchanged.
    """
    validate_rename_map(renames)
    module = cst.parse_module(text)
    imported: Mapping[str, str] = {}
    callables: Iterable[str] = ()
    protected: Iterable[str] = ()
    attribute_roots: Iterable[str] = ()
    non_model_roots: Iterable[str] = ()
    renamable_roots: Iterable[str] = ()
    if scope == "consumers":
        analysis = analyze_consumer_imports(
            module, renames, stems, label, attribute_modules
        )
        imported = analysis.local_to_new
        callables = analysis.polyx_callables
        protected = analysis.protected
        attribute_roots = analysis.attribute_roots
        non_model_roots = analysis.non_model_roots
        renamable_roots = analysis.renamable_roots
    else:
        protected = _module_stem_bindings(module, stems)
    translator = IdentifierTranslator(
        renames, scope=scope, stems=stems, imported=imported,
        polyx_callables=callables, protected=protected,
        attribute_roots=attribute_roots, non_model_roots=non_model_roots,
        renamable_roots=renamable_roots,
    )
    new_module = module.visit(translator)
    changes = list(translator.changes)
    if changes:
        effective = dict(imported) if scope == "consumers" else dict(renames)
        preflight_module(text, effective, label, new_module.code)
    return new_module.code, changes


def process_file(
    path: Path,
    renames: Mapping[str, str],
    apply: bool,
    *,
    scope: str = "polyx",
    stems: Iterable[str] = (),
    attribute_modules: Iterable[str] = (),
) -> List[IdentifierChange]:
    """Process one file; with ``apply`` rewrite in place, otherwise report only.

    ``newline=""`` preserves the original line endings (CRLF included) byte for
    byte.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    new_text, changes = translate_source(
        text, renames, scope=scope, stems=stems, label=path.as_posix(),
        attribute_modules=attribute_modules,
    )
    if apply and changes:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(new_text)
    return changes


def load_manifest(path: Path) -> Dict[str, str]:
    """Load a manifest module that defines ``IDENTIFIER_TRANSLATIONS``."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load manifest: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass consults sys.modules (MEM105).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    translation = getattr(module, "IDENTIFIER_TRANSLATIONS", None)
    if not isinstance(translation, dict):
        raise SystemExit(f"manifest {path} does not define IDENTIFIER_TRANSLATIONS")
    try:
        validate_rename_map(translation)
    except ValueError as error:
        raise SystemExit(f"manifest {path}: {error}") from error
    return dict(translation)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the identifier codemod CLI."""
    parser = argparse.ArgumentParser(
        description="Rename polyx identifiers to English through LibCST (M006/S04)."
    )
    parser.add_argument(
        "paths", nargs="+", type=Path, help="files or directories to process"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="report changes without writing"
    )
    mode.add_argument(
        "--apply", action="store_true", help="rewrite the files in place"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Python module that defines IDENTIFIER_TRANSLATIONS",
    )
    parser.add_argument(
        "--scope",
        choices=["polyx", "consumers"],
        default="polyx",
        help="rename every symbol (polyx) or only imported polyx bindings",
    )
    args = parser.parse_args(argv)

    renames = load_manifest(args.manifest) if args.manifest else {}
    stems = discover_module_stems()
    total = 0
    for path in _python_files(args.paths):
        try:
            changes = process_file(
                path, renames, args.apply, scope=args.scope, stems=stems
            )
        except (OSError, UnicodeDecodeError, ValueError, cst.ParserSyntaxError) as error:
            print(f"ERROR in {path.as_posix()}: {error}", file=sys.stderr)
            return 1
        for change in changes:
            print(f"{path.as_posix()}: {change.old} -> {change.new} ({change.kind})")
        total += len(changes)
    verb = "applied" if args.apply else "detected"
    print(f"{total} change(s) {verb}.")
    return 0


def _python_files(paths: Sequence[Path]) -> List[Path]:
    """Expand files and directories into a sorted list of ``*.py`` files."""
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        else:
            files.append(path)
    return files


if __name__ == "__main__":
    sys.exit(main())
