#!/usr/bin/env python3
"""LibCST codemod harness to translate polyx docstrings to English.

Milestone M006, slice S02. This is the shared harness that S02 (docstrings),
S03 (comments) and S04 (identifiers) reuse; the transformer core lives in
``tools/translate_docstrings.py`` (built in T01) and is loaded from file
path because ``tools/`` is not a package.

Modes (subcommands):

- ``list [--json] PATH...``: enumerate every module/class/function
  docstring with its document-order index and first line.
- ``check PATH...``: scan every docstring for Spanish remnants (curated
  word regex + accented-letter heuristic, inline-code backtick spans
  stripped first). Exit 0 when clean, 1 when findings exist.
- ``apply --manifest MODULE PATH...``: rewrite the docstrings selected by
  the manifest (``{file_path: {index: new_content}}``, repo-relative
  posix paths). Paths default to the manifest keys.
- ``audit [PATH...]``: token-level audit against git HEAD — every token
  that differs from HEAD must be a docstring STRING token of the old
  file. Paths default to every ``*.py`` modified vs HEAD. Exit 0 proves
  zero non-docstring churn in the touched files.

The Spanish runtime tr() layer and all non-docstring string literals are
never touched by this harness.

Comment modes (S03) reuse the same file-path loading and manifest
conventions through ``tools/translate_comments.py``:

- ``list-comments [--json] PATH...``: enumerate every COMMENT token
  with its document-order index, tokenize line number and protected
  flag.
- ``check-comments PATH...``: scan every non-protected comment and
  every contiguous comment block with the comment-specific D023
  detector. Exit 0 when clean, 1 when findings exist.
- ``apply-comments --manifest MODULE [PATH...]``: rewrite the comments
  selected by a ``COMMENT_TRANSLATIONS`` manifest (paths default to the
  manifest keys).
- ``audit-comments [PATH...]``: token-level audit against git HEAD;
  every changed token must be a COMMENT whose old value existed in HEAD.
  Paths default to every ``*.py`` modified vs HEAD.

Identifier modes (S04) reuse the same conventions through
``tools/translate_identifiers.py``:

- ``list-identifiers [--json] PATH...``: enumerate unique identifier names
  with occurrence counts and Spanish findings.
- ``check-identifiers PATH...``: scan identifier occurrences outside import
  statements. Exit 0 when clean, 1 when findings exist.
- ``suggest-identifiers [--json] PATH...``: propose an old -> new map from the
  Spanish subword dictionary.
- ``translate-identifiers [--write-manifest PATH] [--backend NAME] [--offline]
  PATH...``: build the map through the translation-software bridge (D033),
  caching every subword result in ``tests/identifier_manifests/
  translation_cache.json``. The glossary/cache are consulted before the
  backend; ``--offline`` fails loudly instead of calling the network.
- ``apply-identifiers --manifest MODULE [--scope consumers] [--dry-run]
  PATH...``: apply a flat ``IDENTIFIER_TRANSLATIONS`` map; it writes by
  default and reports only under ``--dry-run``.
- ``audit-identifiers --manifest MODULE [PATH...]``: token audit against git
  HEAD; every changed token must be a NAME mapped by the manifest or an
  approved dynamic STRING (getattr family or ``__all__``).

Report-heading mode (S05) uses the same bridge without the identifier
normalizer:

- ``translate-headings --input FILE [--write-manifest PATH] [--backend NAME]
  [--offline] [--cache PATH]``: translate user-visible report labels through
  the project EN dictionary (exact match) and cached machine translation.
  FILE is a JSON array or ``{"queries": [...]}``; every result keeps its
  spaces, tags and units, and is validated before it reaches the manifest.
  The cache lands in ``tests/heading_manifests/translation_cache.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
import subprocess
import sys
import tokenize
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import libcst as cst

# Núcleo del codemod construido en T01: cargamos el módulo por ruta
# porque tools/ no es un paquete.
_RUTA_NUCLEO = Path(__file__).resolve().parent.parent / "tools" / "translate_docstrings.py"
_spec_nucleo = importlib.util.spec_from_file_location(
    "translate_docstrings", _RUTA_NUCLEO
)
td = importlib.util.module_from_spec(_spec_nucleo)
sys.modules.setdefault("translate_docstrings", td)
_spec_nucleo.loader.exec_module(td)

# Comment core built in S03/T01: same file-path loading trick, with the
# sys.modules registration BEFORE exec_module so @dataclass works (MEM105).
_RUTA_COMMENT_CORE = (
    Path(__file__).resolve().parent.parent / "tools" / "translate_comments.py"
)
_spec_comment_core = importlib.util.spec_from_file_location(
    "translate_comments", _RUTA_COMMENT_CORE
)
tc = importlib.util.module_from_spec(_spec_comment_core)
sys.modules["translate_comments"] = tc
_spec_comment_core.loader.exec_module(tc)

# Identifier core built in S04/T01: same file-path loading trick.
_RUTA_IDENTIFIER_CORE = (
    Path(__file__).resolve().parent.parent / "tools" / "translate_identifiers.py"
)
_spec_identifier_core = importlib.util.spec_from_file_location(
    "translate_identifiers", _RUTA_IDENTIFIER_CORE
)
ti = importlib.util.module_from_spec(_spec_identifier_core)
sys.modules["translate_identifiers"] = ti
_spec_identifier_core.loader.exec_module(ti)

# ---------------------------------------------------------------------------
# Detector de restos de español (misma base que tests/test_docs_ingles.py:
# palabras curadas de alta señal + heurística de letras acentuadas).
# ---------------------------------------------------------------------------

# Palabras españolas curadas, ancladas a límites de palabra. Deliberadamente
# estrecha: solo palabras que no aparecen en prosa técnica inglesa legítima.
PALABRAS_ES = re.compile(
    r"\b(?:"
    r"detecci\w*|instalaci\w*|configuraci\w*|caracterizaci\w*|descripci\w*"
    r"|m[óo]dulos?|archivos?|part[íi]culas?|pol[íi]meros?|informes?"
    r"|im[áa]genes?|caracter[íi]sticas"
    r"|etiquet(?:a|as|ado|ados|ada|adas|ar|adora|adoras)"
    r"|captur(?:a|as|ar|ado|ada|ados|adas|ando|ura|uras)"
    r"|tama[ñn]os?|ventanas?|directorios?|pruebas|c[óo]digos?"
    r"|comentarios|resultados|ejecut\w*"
    r"|entren(?:a|as|ar|ando|ado|ada|ados|adas|o|amiento|amientos)"
    r"|biling[üu]es?"
    r"|espa[ñn]ol|p[áa]ginas?|funci[óo]n|funciones"
    r"|devuelve|retorna|genera|calcula|convierte|agrega|a[ñn]ade|crea"
    r"|permite|evita|muestra|guarda|carga|actualiza|define|representa"
    r"|contiene|proporciona|necesita|requiere|utiliza|rutas?|historial"
    r"|cada|para|como|donde|cuando|porque|aunque|tras|muy|por|que|del"
    r"|entre|desde|hasta|sobre|tambi[ée]n|adem[áa]s|seg[úu]n"
    r"|siempre|nunca|debe|pueden|puede|valores?|cadenas?|clases?"
    r"|listas?|defecto|[óo]rden|m[áa]ximo|m[íi]nimo"
    r"|ajust\w*|puntos?|bordes?|cajas?|umbrales?|escalas?|tallas?|anillos?"
    r")\b",
    re.IGNORECASE,
)

# Heurística de letras acentuadas: cualquier palabra alfabética que contenga
# uno de los acentos del español se trata como resto de español, salvo la
# lista de excepciones (nombres propios van en minúsculas, sin mayúsculas).
_ACENTUADA = re.compile(r"[A-Za-zÀ-ÿ]*[áéíóúñüÁÉÍÓÚÑÜ][A-Za-zÀ-ÿ]*")

# Tokens legítimos con acentos que el escáner debe tolerar (extender aquí,
# nunca aflojar la regex).
ALLOWLIST: frozenset = frozenset(
    {
        "detecciones",  # nombre de parámetro en Args: de asignacion.asignar
        # los literales de UI en español van dentro de comillas invertidas y
        # se quitan antes del escaneo
        "normalized",
        "normalize",
        "normalization",
        "normalizer",
    }
)


# ---------------------------------------------------------------------------
# Comment-specific Spanish detector (D023).
#
# Comments are mostly ASCII Spanish, so the docstring detector alone misses
# whole sentences such as 'la mala clasificacion penaliza en ambos lados'.
# After stripping backticked inline code, path-like tokens and code-shaped
# identifiers, a comment is flagged by ANY of:
#   - the shared curated regex plus accented heuristic,
#   - an unaccented Spanish content-word vocabulary,
#   - a Spanish-vs-English stopword ratio (es >= 3 and es >= en + 2).
# ---------------------------------------------------------------------------

# Unaccented Spanish content words that do not appear in legitimate English
# technical prose. Deliberately content-only: the stopword ratio below covers
# function words, so these regexes do not need to carry the whole language.
SPANISH_CONTENT_WORDS = re.compile(
    r"\b(?:"
    r"resumen(?:es)?|clasificaci\w*|clasificacion(?:es)?"
    r"|filtros?|filtrad[oa]s?|filtrar|confianza|parametros?"
    r"|troceado|trocear|troceador|atajos?|atajo|guardad[oa]s?|guardar"
    r"|cargad[oa]s?|cargar|listad[oa]s?|listar|mostrad[oa]s?|mostrar"
    r"|selecci\w*|seleccionad[oa]s?|seleccionar|elegid[oa]s?|elegir"
    r"|usad[oa]s?|usar|utilizad[oa]s?|utilizar|hech[oa]s?|hacer"
    r"|dich[oa]s?|dado|dada|dados|dadas|puest[oa]s?|poner"
    r"|vuelt[oa]s?|volver|abiert[oa]s?|abrir|cerrad[oa]s?|cerrar"
    r"|activ[oa]s?|inactiv[oa]s?|habilitad[oa]s?|deshabilitad[oa]s?"
    r"|visibl\w*|ocult[oa]s?|ocultar|izquierda|derecha|arriba|abajo"
    r"|dentro|fuera|encima|debajo|delante|detras|calcul\w*|obten\w*"
    r"|obtiene|devuelt\w*|devolver|generad[oa]s?|generar|cread[oa]s?"
    r"|crear|constru\w*|construid[oa]s?|escrib\w*|escribir|escrit[oa]s?"
    r"|leer|leid[oa]s?|proces\w*|procesad[oa]s?|procesar|aplic\w*"
    r"|aplicad[oa]s?|aplicar|convert\w*|convertid[oa]s?|convertir"
    r"|normaliz\w*|normalizad[oa]s?|asign\w*|asignad[oa]s?|asignar"
    r"|agrup\w*|agrupad[oa]s?|orden\w*|ordenad[oa]s?|ordenar"
    r"|compar\w*|comparad[oa]s?|actualiz\w*|actualizad[oa]s?"
    r"|elimin\w*|eliminad[oa]s?|borr\w*|borrad[oa]s?"
    r"|reinici\w*|reiniciad[oa]s?|arranc\w*|arrancad[oa]s?"
    r"|funcion(?:es)?|metodos?|parametros?|opcion(?:es)?"
    r"|nombre(?:s)?|carpeta(?:s)?|campo(?:s)?|tabla(?:s)?"
    r"|columna(?:s)?|fila(?:s)?|lienzo|barra(?:s)?|boton(?:es)?"
    r"|aviso(?:s)?|mensaje(?:s)?|texto(?:s)?|contenido(?:s)?"
    r"|documento(?:s)?|reporte(?:s)?|ejemplo(?:s)?|caso(?:s)?"
    r"|cambio(?:s)?|diferencia(?:s)?|correcto|incorrecto"
    r"|verdadero|falso|nuevo|nueva|nuevos|nuevas|mismo|misma"
    r"|mismos|mismas|otro|otra|otros|otras|primero|primera"
    r"|segundo|segunda|tercero|tercera|ultimo|ultima|rojo|roja"
    r"|azul|verde|amarillo|amarilla|naranja|naranjo|negro|negra"
    r"|blanco|blanca|gris|segundos?|minutos?|horas?|dias?|noche"
    r"|semana(?:s)?|alta|altos|altas|bajo|baja|bajos|bajas"
    r")\b",
    re.IGNORECASE,
)

# Function words used only for the ratio test. Words that are identical in
# both languages are harmless: the ratio needs a clear Spanish margin.
SPANISH_STOPWORDS = frozenset(
    {
        "a", "al", "algo", "algunos", "ambos", "ante", "antes", "aqui",
        "asi", "aun", "cada", "como", "con", "contra", "cual", "cuando",
        "de", "del", "desde", "donde", "dos", "durante", "e", "el", "ella",
        "ellas", "ellos", "en", "entre", "era", "eran", "es", "esa", "ese",
        "eso", "esos", "esta", "estas", "este", "esto", "estos", "fue",
        "ha", "han", "hasta", "hay", "la", "las", "le", "les", "lo", "los",
        "mas", "me", "mi", "muy", "no", "nos", "o", "otra", "otro", "para",
        "pero", "por", "porque", "que", "se", "ser", "si", "sin", "sobre",
        "son", "su", "sus", "te", "tiene", "tienen", "todo", "todos", "tras",
        "tu", "un", "una", "uno", "unos", "y", "ya",
    }
)

ENGLISH_STOPWORDS = frozenset(
    {
        "a", "after", "all", "also", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "between", "both", "but",
        "by", "can", "could", "did", "do", "does", "each", "for", "from",
        "had", "has", "have", "here", "how", "if", "in", "into", "is", "it",
        "its", "may", "might", "more", "most", "must", "new", "not", "of",
        "off", "on", "once", "one", "only", "or", "other", "out", "over",
        "per", "same", "should", "so", "some", "such", "than", "that", "the",
        "their", "then", "there", "these", "they", "this", "those", "three",
        "to", "two", "under", "up", "use", "used", "using", "via", "was",
        "were", "what", "when", "where", "which", "who", "why", "will", "with",
        "would",
    }
)

_CODE_SHAPED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_PATH_LIKE = re.compile(r"\b[\w./\\-]*[/\\][\w./\\-]*\b")
_DOTTED_PATH = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_WORD = re.compile(r"[A-Za-z\u00c0-\u00ff]+")


def _is_code_shaped(token: str) -> bool:
    """True for identifiers such as TAMANO_TOTAL, ImageResult or x2."""
    if "_" in token:
        return True
    if any(character.isdigit() for character in token):
        return True
    if len(token) > 1 and token[0].islower() and any(
        character.isupper() for character in token[1:]
    ):
        return True
    # All-caps constants such as PET or JSONL are code-shaped too.
    if len(token) > 1 and token.isupper():
        return True
    return False


def _strip_code_like(text: str, identifiers: Sequence[str]) -> str:
    """Remove backticked literals, paths and code-shaped identifiers."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = _PATH_LIKE.sub(" ", text)
    text = _DOTTED_PATH.sub(" ", text)
    for name in sorted(set(identifiers), key=len, reverse=True):
        if name:
            text = re.sub(rf"\b{re.escape(name)}\b", " ", text)
    return _CODE_SHAPED.sub(
        lambda match: " " if _is_code_shaped(match.group(0)) else match.group(0),
        text,
    )


def spanish_comment_findings(
    text: str, identifiers: Sequence[str] = ()
) -> List[str]:
    """Return Spanish markers found in a comment body ([] = clean)."""
    cleaned = _strip_code_like(text, identifiers)
    findings = hallazgos_espanoles(cleaned)
    for match in SPANISH_CONTENT_WORDS.finditer(cleaned):
        word = match.group(0).lower()
        if word not in ALLOWLIST:
            findings.append(f"comment-word:{word}")
    words = [word.lower() for word in _WORD.findall(cleaned)]
    spanish_hits = sum(1 for word in words if word in SPANISH_STOPWORDS)
    english_hits = sum(1 for word in words if word in ENGLISH_STOPWORDS)
    if spanish_hits >= 3 and spanish_hits >= english_hits + 2:
        findings.append(
            f"stopword-ratio:es={spanish_hits},en={english_hits}"
        )
    # Preserve order but avoid reporting the same marker twice.
    return list(dict.fromkeys(findings))


def _comment_tokens(text: str) -> List[tokenize.TokenInfo]:
    """Return every COMMENT token of a Python source in tokenize order."""
    return [
        token
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type == tokenize.COMMENT
    ]


def _identifiers_from_source(text: str) -> set:
    """Return every identifier name in a Python source."""
    return {
        token.string
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type == tokenize.NAME
    }


def comment_blocks(
    tokens: Sequence[tokenize.TokenInfo],
) -> List[List[tokenize.TokenInfo]]:
    """Group consecutive comment lines that start at the same column."""
    blocks: List[List[tokenize.TokenInfo]] = []
    current: List[tokenize.TokenInfo] = []
    for token in tokens:
        if (
            current
            and token.start[0] == current[-1].start[0] + 1
            and token.start[1] == current[-1].start[1]
        ):
            current.append(token)
        else:
            if current:
                blocks.append(current)
            current = [token]
    if current:
        blocks.append(current)
    return blocks


def _quitar_inline_code(texto: str) -> str:
    """Quita los tramos de código entre comillas invertidas del docstring."""
    texto = unicodedata.normalize("NFC", texto)
    # se quitan primero los bloques ``` ... ``` y luego los tramos `...`
    texto = re.sub(r"```.*?```", " ", texto, flags=re.DOTALL)
    texto = re.sub(r"`[^`\n]+`", " ", texto)
    return texto


def hallazgos_espanoles(prosa: str) -> List[str]:
    """Devuelve los marcadores de español hallados en prosa ([] = limpia)."""
    hallazgos: List[str] = []
    for match in PALABRAS_ES.finditer(prosa):
        palabra = match.group(0).lower()
        if palabra not in ALLOWLIST:
            hallazgos.append(f"spanish-word:{palabra}")
    for match in _ACENTUADA.finditer(prosa):
        palabra = match.group(0)
        if palabra.lower() not in ALLOWLIST:
            hallazgos.append(f"accented:{palabra}")
    return hallazgos


# ---------------------------------------------------------------------------
# Enumeración de docstrings: mismo truco del núcleo (numerar en visit_*,
# extraer en leave_*) pero conservando el texto completo del token.
# ---------------------------------------------------------------------------


class _Enumerador(cst.CSTVisitor):
    """Lista (índice, contenido) de cada docstring en orden de documento."""

    def __init__(self) -> None:
        self.docs: List[Tuple[int, str]] = []
        self._indice = 0
        self._indice_por_id: Dict[int, int] = {}

    def _asignar(self) -> int:
        indice = self._indice
        self._indice += 1
        return indice

    def visit_Module(self, nodo: cst.Module) -> None:
        self._indice_por_id[id(nodo)] = self._asignar()

    def visit_ClassDef(self, nodo: cst.ClassDef) -> None:
        self._indice_por_id[id(nodo)] = self._asignar()

    def visit_FunctionDef(self, nodo: cst.FunctionDef) -> None:
        self._indice_por_id[id(nodo)] = self._asignar()

    def leave_Module(self, nodo_original):
        self._extraer(nodo_original)

    def leave_ClassDef(self, nodo_original):
        self._extraer(nodo_original)

    def leave_FunctionDef(self, nodo_original):
        self._extraer(nodo_original)

    def _extraer(self, nodo) -> None:
        indice = self._indice_por_id[id(nodo)]
        string_node = td._docstring_de(td._stmts_de(nodo))
        if string_node is None:
            return
        _, _, contenido = td._separar_token(string_node.value)
        self.docs.append((indice, contenido))


def _ordenado(docs: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    return sorted(docs, key=lambda par: par[0])


def enumerar_docstrings(texto: str) -> List[Tuple[int, str]]:
    """Devuelve [(índice, contenido)] de cada docstring en orden documento."""
    visitante = _Enumerador()
    cst.parse_module(texto).visit(visitante)
    return _ordenado(visitante.docs)


def enumerar_archivo(ruta: Path) -> List[Tuple[int, str]]:
    """Enumera los docstrings de un archivo .py del repositorio."""
    return enumerar_docstrings(ruta.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Auditoría de tokens contra git HEAD.
# ---------------------------------------------------------------------------


def _tokens_de(texto: str) -> List[Tuple[int, str]]:
    """Secuencia (tipo, valor) de todos los tokens de un texto Python."""
    salida: List[Tuple[int, str]] = []
    for tok in tokenize.generate_tokens(io.StringIO(texto).readline):
        salida.append((tok.type, tok.string))
    return salida


def _texto_head(ruta: Path) -> Optional[str]:
    """Contenido del archivo en git HEAD, o None si es nuevo sin seguimiento.

    La ruta se hace relativa a la raíz del repo: ``git show HEAD:<ruta>``
    interpreta las rutas absolutas respecto a la raíz y falla fuera de él.
    """
    ruta_absoluta = ruta.resolve()
    raiz = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    )
    try:
        clave = ruta_absoluta.relative_to(raiz).as_posix()
    except ValueError:
        return None
    resultado = subprocess.run(
        ["git", "show", f"HEAD:{clave}"],
        capture_output=True,
        check=False,
    )
    if resultado.returncode != 0:
        return None
    return resultado.stdout.decode("utf-8")


def _valores_docstring_head(texto: str) -> set:
    """Valores de token (con comillas) de los docstrings del texto HEAD."""
    valores: set = set()

    # reutiliza el visitante pero guardando el token completo
    class _Coleccionista(_Enumerador):
        def _extraer(self, nodo) -> None:
            string_node = td._docstring_de(td._stmts_de(nodo))
            if string_node is None:
                return
            valores.add(string_node.value)

    coleccionista = _Coleccionista()
    cst.parse_module(texto).visit(coleccionista)
    return valores


def auditar_archivo(ruta: Path) -> List[str]:
    """Verifica que solo tokens de docstring difieran de git HEAD.

    Devuelve la lista de problemas ([] = limpio).
    """
    texto_actual = ruta.read_text(encoding="utf-8")
    texto_head = _texto_head(ruta)
    if texto_head is None:
        # archivo nuevo: no hay base contra la cual comparar
        return []
    if texto_actual == texto_head:
        return []
    valores_docstring = _valores_docstring_head(texto_head)
    tokens_head = _tokens_de(texto_head)
    tokens_actual = _tokens_de(texto_actual)
    problemas: List[str] = []
    if len(tokens_head) != len(tokens_actual):
        problemas.append(
            f"{ruta.as_posix()}: cantidad de tokens cambia "
            f"({len(tokens_head)} -> {len(tokens_actual)})"
        )
    for pos, (viejo, nuevo) in enumerate(zip(tokens_head, tokens_actual)):
        if viejo == nuevo:
            continue
        if viejo[0] != nuevo[0]:
            problemas.append(
                f"{ruta.as_posix()}: token {pos} cambia de tipo "
                f"({viejo[0]} -> {nuevo[0]})"
            )
            continue
        if viejo[1] != nuevo[1]:
            if viejo[0] != tokenize.STRING:
                problemas.append(
                    f"{ruta.as_posix()}: token no-docstring cambia: "
                    f"{viejo[1]!r} -> {nuevo[1]!r}"
                )
            elif viejo[1] not in valores_docstring:
                problemas.append(
                    f"{ruta.as_posix()}: token STRING no docstring cambia: "
                    f"{viejo[1]!r}"
                )
    return problemas


def _archivos_modificados() -> List[Path]:
    """Archivos .py modificados respecto a git HEAD, bajo polyx/.

    El alcance del audit es el código del paquete: el propio driver y los
    tests se editan a mano y no participan del codemod.
    """
    resultado = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "polyx/"],
        capture_output=True,
        check=True,
        text=True,
    )
    return [Path(linea) for linea in resultado.stdout.splitlines() if linea.strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _expandir(rutas: Sequence[Path]) -> List[Path]:
    return td._archivos_py(rutas)


def cmd_list(args: argparse.Namespace) -> int:
    """Enumera los docstrings de las rutas dadas."""
    entradas = []
    for ruta in _expandir(args.rutas):
        for indice, primera in enumerar_archivo(ruta):
            entradas.append(
                {
                    "file": ruta.as_posix(),
                    "index": indice,
                    "firstline": primera,
                }
            )
    if args.json:
        print(json.dumps(entradas, indent=1, ensure_ascii=False))
    else:
        for entrada in entradas:
            print(
                f"{entrada['file']} #{entrada['index']}: {entrada['firstline']}"
            )
    print(f"{len(entradas)} docstring(s).", file=sys.stderr)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Escanea los docstrings de las rutas dadas buscando restos de español."""
    total_docs = 0
    total_hallazgos = 0
    for ruta in _expandir(args.rutas):
        contenido = ruta.read_text(encoding="utf-8")
        docs = enumerar_docstrings(contenido)
        total_docs += len(docs)
        for indice, contenido_doc in docs:
            hallazgos = hallazgos_espanoles(_quitar_inline_code(contenido_doc))
            if hallazgos:
                total_hallazgos += len(hallazgos)
                primera = contenido_doc.splitlines()[0] if contenido_doc else ""
                print(
                    f"{ruta.as_posix()} #{indice}: {primera!r} -> {hallazgos}",
                    file=sys.stderr,
                )
    print(f"{total_docs} docstring(s) escaneados, {total_hallazgos} hallazgo(s).")
    return 1 if total_hallazgos else 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Aplica el manifiesto sobre los archivos (por omisión, sus claves)."""
    por_archivo = td.cargar_manifest(args.manifest)
    rutas = args.rutas or [Path(clave) for clave in sorted(por_archivo)]
    return _ejecutar_aplicacion(rutas, por_archivo)


def _ejecutar_aplicacion(
    rutas: Sequence[Path], por_archivo: Dict[str, Dict[int, str]]
) -> int:
    total = 0
    for ruta in _expandir(rutas):
        clave = ruta.as_posix()
        try:
            cambios = td.procesar_archivo(ruta, por_archivo.get(clave, {}), True)
        except (OSError, UnicodeDecodeError, ValueError, cst.ParserSyntaxError) as error:
            print(f"ERROR en {clave}: {error}", file=sys.stderr)
            return 1
        for cambio in cambios:
            print(f"{clave} #{cambio.indice}: {cambio.antes!r} -> {cambio.despues!r}")
        total += len(cambios)
    print(f"{total} cambio(s) aplicados.")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Audita que los cambios vs HEAD estén confinados a tokens de docstring."""
    rutas = _expandir(args.rutas or _archivos_modificados())
    if not rutas:
        print("audit: no hay archivos .py modificados respecto a HEAD.")
        return 0
    problemas: List[str] = []
    for ruta in rutas:
        problemas.extend(auditar_archivo(ruta))
    for problema in problemas:
        print(problema, file=sys.stderr)
    print(f"{len(rutas)} archivo(s) auditado(s), {len(problemas)} problema(s).")
    return 1 if problemas else 0


# ---------------------------------------------------------------------------
# Comment modes (S03).
# ---------------------------------------------------------------------------


def scan_comments(text: str) -> List[Tuple[str, List[str]]]:
    """Return ``(line_label, findings)`` for every Spanish comment finding."""
    tokens = _comment_tokens(text)
    identifiers = _identifiers_from_source(text)
    results: List[Tuple[str, List[str]]] = []
    for token in tokens:
        if tc.is_protected_comment(token.string):
            continue
        body = tc.split_comment(token.string)[1]
        findings = spanish_comment_findings(body, identifiers)
        if findings:
            results.append((str(token.start[0]), findings))
    for block in comment_blocks(tokens):
        bodies = [
            tc.split_comment(token.string)[1]
            for token in block
            if not tc.is_protected_comment(token.string)
        ]
        if len(bodies) < 2:
            continue
        findings = spanish_comment_findings(" ".join(bodies), identifiers)
        if findings:
            results.append(
                (f"{block[0].start[0]}-{block[-1].start[0]}", findings)
            )
    return results


def _head_text_comments(path: Path) -> Optional[str]:
    """Return the file content at git HEAD, or None when it is untracked.

    ``git -C`` is used so the audit also works in a temporary toy repo
    whose files live outside the current working directory.
    """
    try:
        root = Path(
            subprocess.run(
                ["git", "-C", str(path.resolve().parent), "rev-parse", "--show-toplevel"],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        key = path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{key}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8")


def audit_comment_file(path: Path) -> List[str]:
    """Verify that only COMMENT tokens differ from git HEAD.

    Requires equal token counts and compares ``(type, string)`` pairs, not
    token positions: a longer comment shifts column offsets on its line
    but never the line count. Returns the list of problems ([] = clean).
    """
    current_text = path.read_text(encoding="utf-8")
    head_text = _head_text_comments(path)
    if head_text is None or current_text == head_text:
        return []
    head_tokens = _tokens_de(head_text)
    current_tokens = _tokens_de(current_text)
    head_comments = {
        value for token_type, value in head_tokens if token_type == tokenize.COMMENT
    }
    problems: List[str] = []
    if len(head_tokens) != len(current_tokens):
        problems.append(
            f"{path.as_posix()}: token count changes "
            f"({len(head_tokens)} -> {len(current_tokens)})"
        )
    for position, (old, new) in enumerate(zip(head_tokens, current_tokens)):
        if old == new:
            continue
        if old[0] != new[0]:
            problems.append(
                f"{path.as_posix()}: token {position} changes type "
                f"({old[0]} -> {new[0]})"
            )
            continue
        if old[0] != tokenize.COMMENT:
            problems.append(
                f"{path.as_posix()}: non-comment token changes: {old[1]!r} -> {new[1]!r}"
            )
            continue
        if old[1] not in head_comments:
            problems.append(
                f"{path.as_posix()}: changed COMMENT did not exist in HEAD: {old[1]!r}"
            )
    return problems


def cmd_list_comments(args: argparse.Namespace) -> int:
    """Enumerate comments with line numbers paired from tokenize order."""
    entries = []
    for path in _expandir(args.rutas):
        source = path.read_text(encoding="utf-8")
        tokens = _comment_tokens(source)
        enumeration = tc.enumerate_comments(source)
        if len(tokens) != len(enumeration):
            print(
                f"ERROR in {path.as_posix()}: LibCST found "
                f"{len(enumeration)} comments, tokenize found {len(tokens)}",
                file=sys.stderr,
            )
            return 1
        for (index, raw), token in zip(enumeration, tokens):
            if raw != token.string:
                print(
                    f"ERROR in {path.as_posix()}: comment #{index} order mismatch",
                    file=sys.stderr,
                )
                return 1
            entries.append(
                {
                    "file": path.as_posix(),
                    "index": index,
                    "line": token.start[0],
                    "text": raw,
                    "protected": tc.is_protected_comment(raw),
                }
            )
    if args.json:
        print(json.dumps(entries, indent=1, ensure_ascii=False))
    else:
        for entry in entries:
            suffix = " [protected]" if entry["protected"] else ""
            print(
                f"{entry['file']} #{entry['index']} line {entry['line']}: "
                f"{entry['text']!r}{suffix}"
            )
    print(f"{len(entries)} comment(s).", file=sys.stderr)
    return 0


def cmd_check_comments(args: argparse.Namespace) -> int:
    """Scan comments and contiguous blocks for Spanish remnants."""
    total_comments = 0
    total_findings = 0
    for path in _expandir(args.rutas):
        source = path.read_text(encoding="utf-8")
        total_comments += sum(
            1
            for token in _comment_tokens(source)
            if not tc.is_protected_comment(token.string)
        )
        for line_label, findings in scan_comments(source):
            total_findings += len(findings)
            print(
                f"{path.as_posix()}#{line_label}: {findings}",
                file=sys.stderr,
            )
    print(
        f"{total_comments} comment(s) scanned, {total_findings} finding(s)."
    )
    return 1 if total_findings else 0


def cmd_apply_comments(args: argparse.Namespace) -> int:
    """Apply a COMMENT_TRANSLATIONS manifest (paths default to its keys)."""
    by_file = tc.load_manifest(args.manifest)
    paths = args.rutas or [Path(key) for key in sorted(by_file)]
    total = 0
    for path in _expandir(paths):
        key = path.as_posix()
        try:
            changes = tc.process_file(path, by_file.get(key, {}), True)
        except (OSError, UnicodeDecodeError, ValueError, cst.ParserSyntaxError) as error:
            print(f"ERROR in {key}: {error}", file=sys.stderr)
            return 1
        for change in changes:
            print(
                f"{key} #{change.index}: {change.before!r} -> {change.after!r}"
            )
        total += len(changes)
    print(f"{total} change(s) applied.")
    return 0


def cmd_audit_comments(args: argparse.Namespace) -> int:
    """Audit that changes against HEAD are confined to COMMENT tokens."""
    paths = _expandir(args.rutas or _archivos_modificados())
    if not paths:
        print("audit-comments: no .py files modified against HEAD.")
        return 0
    problems: List[str] = []
    for path in paths:
        problems.extend(audit_comment_file(path))
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"{len(paths)} file(s) audited, {len(problems)} problem(s).")
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# Identifier modes (S04).
# ---------------------------------------------------------------------------


def list_identifiers(paths: Sequence[Path]) -> List[dict]:
    """Return one ``{file, name, count, findings}`` entry per unique name."""
    entries: List[dict] = []
    for path in _expandir(paths):
        source = path.read_text(encoding="utf-8")
        for name, count in ti.enumerate_identifiers(source):
            entries.append(
                {
                    "file": path.as_posix(),
                    "name": name,
                    "count": count,
                    "findings": ti.spanish_identifier_findings(name),
                }
            )
    return entries


def scan_identifiers(text: str) -> List[Tuple[int, str, List[str]]]:
    """Return ``(line, name, findings)`` for every Spanish identifier use.

    Import statements are skipped in full: the module path must never be
    renamed and the imported symbol is scanned where it is defined.
    """
    results: List[Tuple[int, str, List[str]]] = []
    for line, name in ti.identifier_occurrences(text):
        findings = ti.spanish_identifier_findings(name)
        if findings:
            results.append((line, name, findings))
    return results


def _approved_dynamic_positions(text: str) -> set:
    """Return the ``(line, column)`` starts of approved dynamic strings.

    Approved contexts are the second argument of ``getattr``/``setattr``/
    ``hasattr``/``delattr`` and the string entries of ``__all__`` (D030).
    """
    module = cst.parse_module(text)
    wrapper = cst.MetadataWrapper(module)
    positions: set = set()

    def collect(node) -> None:
        if isinstance(node, cst.SimpleString):
            position = visitor.get_metadata(cst.metadata.PositionProvider, node).start
            positions.add((position.line, position.column))
        elif isinstance(node, (cst.List, cst.Tuple, cst.Set)):
            for element in node.elements:
                collect(element.value)

    class _Approved(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

        def visit_Call(self, node: cst.Call) -> None:
            if not ti.is_dynamic_attribute_call(node.func):
                return
            if len(node.args) < 2:
                return
            second = node.args[1].value
            if not isinstance(second, cst.SimpleString):
                return
            position = self.get_metadata(
                cst.metadata.PositionProvider, second
            ).start
            positions.add((position.line, position.column))

        def visit_Assign(self, node: cst.Assign) -> None:
            if any(
                isinstance(target.target, cst.Name)
                and target.target.value == "__all__"
                for target in node.targets
            ):
                collect(node.value)

        def visit_AugAssign(self, node: cst.AugAssign) -> None:
            if isinstance(node.target, cst.Name) and node.target.value == "__all__":
                collect(node.value)

    visitor = _Approved()
    wrapper.visit(visitor)
    return positions


def _literal_value(token_value: str) -> Optional[str]:
    """Evaluate a simple string token, returning None when unsupported."""
    import ast

    try:
        value = ast.literal_eval(token_value)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def audit_identifier_file(
    path: Path, mapping: Dict[str, str], extra_mappings: Sequence[Dict[str, str]] = ()
) -> List[str]:
    """Verify that only mapped NAME and approved STRING tokens differ.

    Token count and type sequence must stay identical. Every changed NAME must
    map old -> new through the manifest; every changed STRING must sit in an
    approved dynamic context or be an identifier-shaped structural reference
    to a renamed name (``"_seccion_resumen"`` in a test's constructor list).
    COMMENT, NUMBER, OP and any other token change is a problem.

    ``extra_mappings`` let a task audit a working tree that still carries the
    previous task's uncommitted consumer renames: those changes are legitimate
    and traceable to their own generated manifest.
    """
    mappings = [mapping, *extra_mappings]

    def maps(old: str, new: str) -> bool:
        return any(candidate.get(old) == new for candidate in mappings)

    current_text = path.read_text(encoding="utf-8")
    # ``-C`` on the file's own parent so the audit also works in a throwaway
    # toy repository (the docstring audit reads HEAD from the process cwd).
    head_text = _head_text_comments(path)
    if head_text is None or current_text == head_text:
        return []
    head_tokens = _tokens_de(head_text)
    current_tokens = _tokens_de(current_text)
    approved = _approved_dynamic_positions(current_text)
    current_positions = _token_positions(current_text)
    problems: List[str] = []
    if len(head_tokens) != len(current_tokens):
        problems.append(
            f"{path.as_posix()}: token count changes "
            f"({len(head_tokens)} -> {len(current_tokens)})"
        )
    for position, (old, new) in enumerate(zip(head_tokens, current_tokens)):
        if old == new:
            continue
        if old[0] != new[0]:
            problems.append(
                f"{path.as_posix()}: token {position} changes type "
                f"({old[0]} -> {new[0]})"
            )
            continue
        if old[0] == tokenize.NAME:
            if not maps(old[1], new[1]):
                problems.append(
                    f"{path.as_posix()}: NAME change not in manifest: "
                    f"{old[1]!r} -> {new[1]!r}"
                )
            continue
        if old[0] == tokenize.STRING:
            old_value = _literal_value(old[1])
            new_value = _literal_value(new[1])
            if old_value is not None and new_value is not None and maps(old_value, new_value):
                # Structural references to a renamed callable/constant, e.g.
                # ``"_seccion_resumen"`` in a test's constructor table.
                continue
            token_position = current_positions[position]
            if token_position not in approved:
                problems.append(
                    f"{path.as_posix()}: STRING change outside an approved "
                    f"dynamic context: {old[1]!r} -> {new[1]!r}"
                )
                continue
            if old_value is not None and new_value is not None:
                if not maps(old_value, new_value):
                    problems.append(
                        f"{path.as_posix()}: dynamic STRING change not in "
                        f"manifest: {old_value!r} -> {new_value!r}"
                    )
            continue
        problems.append(
            f"{path.as_posix()}: non-name token changes: {old[1]!r} -> {new[1]!r}"
        )
    return problems


def _token_positions(text: str) -> List[Tuple[int, int]]:
    """Token start positions in the same order as ``_tokens_de``."""
    return [
        token.start
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
    ]


AUDIT_EXCLUDED = frozenset(
    {
        "tools/translate_identifiers.py",
        "tests/translate_code_text.py",
        "tests/test_translate_identifiers.py",
    }
)

# The translation harness and its tests have their own ``indice``/``salida``
# fields and argparse attributes. Consumer mode must not reinterpret those as
# polyx model attributes, so they are never rewritten by an in-repo sweep.
CONSUMER_EXCLUDED = frozenset(
    {
        "tools/translate_docstrings.py",
        "tools/translate_comments.py",
        "tools/translate_identifiers.py",
        "tests/translate_code_text.py",
        "tests/test_translate_identifiers.py",
        "tests/test_translate_docstrings.py",
        "tests/test_translate_comments.py",
        "tests/test_comments_ingles.py",
        "tests/test_docs_ingles.py",
        "tests/test_docstrings_ingles.py",
    }
)


def _modified_python_files() -> List[Path]:
    """Every ``*.py`` file modified against git HEAD (audit default set).

    The identifier tooling and its focused tests are excluded: they are the
    audit machinery, not a translation target, so their own source edits must
    not be judged as manifest churn.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    )
    return [
        Path(line)
        for line in result.stdout.splitlines()
        if line.strip()
        and line.endswith(".py")
        and Path(line).as_posix() not in AUDIT_EXCLUDED
        and not Path(line).as_posix().startswith("tests/identifier_manifests/")
    ]


def cmd_list_identifiers(args: argparse.Namespace) -> int:
    """Enumerate every unique identifier of the given paths."""
    entries = list_identifiers(args.rutas)
    if args.json:
        print(json.dumps(entries, indent=1, ensure_ascii=False))
    else:
        for entry in entries:
            suffix = f" -> {entry['findings']}" if entry["findings"] else ""
            print(f"{entry['file']}: {entry['name']} ({entry['count']}){suffix}")
    print(f"{len(entries)} identifier(s).", file=sys.stderr)
    return 0


def cmd_check_identifiers(args: argparse.Namespace) -> int:
    """Scan identifier occurrences for Spanish remnants."""
    total_names = 0
    total_findings = 0
    for path in _expandir(args.rutas):
        source = path.read_text(encoding="utf-8")
        total_names += len({name for _line, name in ti.identifier_occurrences(source)})
        for line, name, findings in scan_identifiers(source):
            total_findings += len(findings)
            print(
                f"{path.as_posix()}:{line}: {name!r} -> {findings}",
                file=sys.stderr,
            )
    print(
        f"{total_names} unique identifier(s) scanned, "
        f"{total_findings} finding(s)."
    )
    return 1 if total_findings else 0


def cmd_suggest_identifiers(args: argparse.Namespace) -> int:
    """Propose an old -> new map for the Spanish identifiers found."""
    names: List[str] = []
    seen: set = set()
    for path in _expandir(args.rutas):
        source = path.read_text(encoding="utf-8")
        for name, _count in ti.enumerate_identifiers(source):
            if name not in seen:
                seen.add(name)
                names.append(name)
    suggestions = ti.suggest_identifier_map(names)
    if args.json:
        print(json.dumps(suggestions, indent=1, ensure_ascii=False, sort_keys=True))
    else:
        for old in sorted(suggestions):
            print(f"{old} -> {suggestions[old]}")
    print(f"{len(suggestions)} suggestion(s).", file=sys.stderr)
    return 0


def cmd_translate_identifiers(args: argparse.Namespace) -> int:
    """Generate a rename manifest through the translation-software bridge."""
    cache = ti.TranslationCache.load(args.cache)
    names_by_file: Dict[Path, set] = {}
    for path in _expandir(args.rutas):
        source = path.read_text(encoding="utf-8")
        names_by_file[path] = {
            name for name, _count in ti.enumerate_identifiers(source)
        }
    try:
        plan = ti.build_identifier_map(
            names_by_file,
            cache,
            backend_name=args.backend,
            offline=args.offline,
        )
    except ti.TranslationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.write_manifest:
        ti.write_manifest(
            args.write_manifest,
            plan,
            command=" ".join(["translate-identifiers", *map(str, args.rutas)]),
        )
        print(f"manifest written to {args.write_manifest.as_posix()}")
    else:
        for old in sorted(plan.mapping):
            print(f"{old} -> {plan.mapping[old]}")
    cache.save(args.cache)
    print(plan.stats.summary())
    print(f"{len(plan.mapping)} identifier(s).")
    return 0


def cmd_translate_headings(args: argparse.Namespace) -> int:
    """Translate report headings through the translation-software bridge.

    ``--input`` is a JSON array of query strings or an object with a
    ``queries`` list. Every resolution comes from the project EN dictionary,
    the local cache or the machine backend; nothing is typed by hand.
    """
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: cannot read {args.input}: {error}", file=sys.stderr)
        return 1
    if isinstance(payload, dict):
        queries = payload.get("queries")
    else:
        queries = payload
    if not isinstance(queries, list) or not all(
        isinstance(query, str) for query in queries
    ):
        print(
            "ERROR: the input must be a JSON array of strings or an object "
            "with a 'queries' array",
            file=sys.stderr,
        )
        return 1
    cache = ti.TranslationCache.load(args.cache)
    try:
        plan = ti.translate_heading_queries(
            queries,
            cache,
            backend_name=args.backend,
            offline=args.offline,
        )
    except ti.TranslationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for query in plan.mapping:
        print(f"{query} -> {plan.mapping[query]}")
    if args.write_manifest:
        ti.write_heading_manifest(
            args.write_manifest,
            plan,
            command=" ".join(
                ["translate-headings", "--input", str(args.input)]
            ),
        )
        print(f"manifest written to {args.write_manifest.as_posix()}")
    cache.save(args.cache)
    print(plan.stats.summary())
    print(f"{len(plan.mapping)} heading(s).")
    return 0


def cmd_apply_identifiers(args: argparse.Namespace) -> int:
    """Apply an IDENTIFIER_TRANSLATIONS manifest to the given paths."""
    mapping = ti.load_manifest(args.manifest)
    stems = ti.discover_module_stems()
    total = 0
    paths = _expandir(args.rutas)
    if args.scope == "consumers":
        paths = [
            path
            for path in paths
            if path.as_posix() not in CONSUMER_EXCLUDED
            and not path.as_posix().startswith("tests/identifier_manifests/")
        ]
    for path in paths:
        try:
            changes = ti.process_file(
                path,
                mapping,
                not args.dry_run,
                scope=args.scope,
                stems=stems,
                attribute_modules=args.attribute_module,
            )
        except (OSError, UnicodeDecodeError, ValueError, cst.ParserSyntaxError) as error:
            print(f"ERROR in {path.as_posix()}: {error}", file=sys.stderr)
            return 1
        for change in changes:
            print(
                f"{path.as_posix()}: {change.old} -> {change.new} "
                f"({change.kind})"
            )
        total += len(changes)
    verb = "detected" if args.dry_run else "applied"
    print(f"{total} change(s) {verb}.")
    return 0


def _sibling_identifier_manifests(manifest: Path) -> List[Dict[str, str]]:
    """Load the other generated manifests next to ``manifest``.

    Prior tasks' consumer renames can still be uncommitted when the next task
    audits the tree. Those changes are traceable to their own machine-generated
    manifest, so the audit accepts them without weakening the current map.
    """
    mappings: List[Dict[str, str]] = []
    for candidate in sorted(manifest.parent.glob("*.py")):
        if candidate.resolve() == manifest.resolve():
            continue
        try:
            mappings.append(ti.load_manifest(candidate))
        except (SystemExit, Exception):  # pragma: no cover - defensive
            continue
    return mappings


def cmd_audit_identifiers(args: argparse.Namespace) -> int:
    """Audit that changes against HEAD are mapped names or dynamic strings."""
    mapping = ti.load_manifest(args.manifest)
    extra_mappings = _sibling_identifier_manifests(args.manifest)
    paths = _expandir(args.rutas or _modified_python_files())
    if not paths:
        print("audit-identifiers: no .py files modified against HEAD.")
        return 0
    problems: List[str] = []
    for path in paths:
        problems.extend(audit_identifier_file(path, mapping, extra_mappings))
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"{len(paths)} file(s) audited, {len(problems)} problem(s).")
    return 1 if problems else 0


def main(argv: Optional[List[str]] = None) -> int:
    """Punto de entrada del CLI del harness."""
    parser = argparse.ArgumentParser(
        description=(
            "Translate docstrings and comments to English through LibCST "
            "(M006/S02-S03)."
        )
    )
    sub = parser.add_subparsers(dest="modo", required=True)

    p_list = sub.add_parser("list", help="enumerate docstrings")
    p_list.add_argument("--json", action="store_true")
    p_list.add_argument("rutas", nargs="+", type=Path)
    p_list.set_defaults(func=cmd_list)

    p_check = sub.add_parser("check", help="scan docstrings for Spanish")
    p_check.add_argument("rutas", nargs="+", type=Path)
    p_check.set_defaults(func=cmd_check)

    p_apply = sub.add_parser("apply", help="apply a docstring manifest")
    p_apply.add_argument("--manifest", type=Path, required=True)
    p_apply.add_argument("rutas", nargs="*", type=Path)
    p_apply.set_defaults(func=cmd_apply)

    p_audit = sub.add_parser("audit", help="audit docstring tokens against git HEAD")
    p_audit.add_argument("rutas", nargs="*", type=Path)
    p_audit.set_defaults(func=cmd_audit)

    p_list_comments = sub.add_parser(
        "list-comments", help="enumerate comments with line numbers"
    )
    p_list_comments.add_argument("--json", action="store_true")
    p_list_comments.add_argument("rutas", nargs="+", type=Path)
    p_list_comments.set_defaults(func=cmd_list_comments)

    p_check_comments = sub.add_parser(
        "check-comments", help="scan comments for Spanish remnants"
    )
    p_check_comments.add_argument("rutas", nargs="+", type=Path)
    p_check_comments.set_defaults(func=cmd_check_comments)

    p_apply_comments = sub.add_parser(
        "apply-comments", help="apply a COMMENT_TRANSLATIONS manifest"
    )
    p_apply_comments.add_argument("--manifest", type=Path, required=True)
    p_apply_comments.add_argument("rutas", nargs="*", type=Path)
    p_apply_comments.set_defaults(func=cmd_apply_comments)

    p_audit_comments = sub.add_parser(
        "audit-comments", help="audit COMMENT tokens against git HEAD"
    )
    p_audit_comments.add_argument("rutas", nargs="*", type=Path)
    p_audit_comments.set_defaults(func=cmd_audit_comments)

    p_list_identifiers = sub.add_parser(
        "list-identifiers", help="enumerate unique identifier names"
    )
    p_list_identifiers.add_argument("--json", action="store_true")
    p_list_identifiers.add_argument("rutas", nargs="+", type=Path)
    p_list_identifiers.set_defaults(func=cmd_list_identifiers)

    p_check_identifiers = sub.add_parser(
        "check-identifiers", help="scan identifiers for Spanish remnants"
    )
    p_check_identifiers.add_argument("rutas", nargs="+", type=Path)
    p_check_identifiers.set_defaults(func=cmd_check_identifiers)

    p_suggest_identifiers = sub.add_parser(
        "suggest-identifiers", help="propose an old -> new rename map"
    )
    p_suggest_identifiers.add_argument("--json", action="store_true")
    p_suggest_identifiers.add_argument("rutas", nargs="+", type=Path)
    p_suggest_identifiers.set_defaults(func=cmd_suggest_identifiers)

    p_translate_identifiers = sub.add_parser(
        "translate-identifiers",
        help="generate a manifest through the translation-software bridge",
    )
    p_translate_identifiers.add_argument(
        "--write-manifest", type=Path, default=None
    )
    p_translate_identifiers.add_argument(
        "--backend",
        default=None,
        help="argos, deep-translator, libretranslate:<url> or deepl:<url>",
    )
    p_translate_identifiers.add_argument(
        "--offline",
        action="store_true",
        help="use only the glossary and cache; fail on a cache miss",
    )
    p_translate_identifiers.add_argument(
        "--cache", type=Path, default=ti.DEFAULT_CACHE_PATH
    )
    p_translate_identifiers.add_argument("rutas", nargs="+", type=Path)
    p_translate_identifiers.set_defaults(func=cmd_translate_identifiers)

    p_translate_headings = sub.add_parser(
        "translate-headings",
        help="translate report headings through the translation-software bridge",
    )
    p_translate_headings.add_argument(
        "--input", type=Path, required=True, help="JSON array or {queries: [...]}"
    )
    p_translate_headings.add_argument(
        "--write-manifest", type=Path, default=None
    )
    p_translate_headings.add_argument(
        "--backend",
        default=None,
        help="argos, deep-translator, libretranslate:<url> or deepl:<url>",
    )
    p_translate_headings.add_argument(
        "--offline",
        action="store_true",
        help="use only the EN dictionary and cache; fail on a cache miss",
    )
    p_translate_headings.add_argument(
        "--cache", type=Path, default=ti.DEFAULT_HEADING_CACHE_PATH
    )
    p_translate_headings.set_defaults(func=cmd_translate_headings)

    p_apply_identifiers = sub.add_parser(
        "apply-identifiers", help="apply an IDENTIFIER_TRANSLATIONS manifest"
    )
    p_apply_identifiers.add_argument("--manifest", type=Path, required=True)
    p_apply_identifiers.add_argument(
        "--scope", choices=["polyx", "consumers"], default="polyx"
    )
    p_apply_identifiers.add_argument(
        "--dry-run", action="store_true", help="report changes without writing"
    )
    p_apply_identifiers.add_argument(
        "--attribute-module",
        action="append",
        default=[],
        metavar="MODULE",
        help=(
            "dotted polyx module whose imported module aliases may have "
            "attributes renamed in consumer scope"
        ),
    )
    p_apply_identifiers.add_argument("rutas", nargs="+", type=Path)
    p_apply_identifiers.set_defaults(func=cmd_apply_identifiers)

    p_audit_identifiers = sub.add_parser(
        "audit-identifiers", help="audit NAME/dynamic STRING changes against git HEAD"
    )
    p_audit_identifiers.add_argument("--manifest", type=Path, required=True)
    p_audit_identifiers.add_argument("rutas", nargs="*", type=Path)
    p_audit_identifiers.set_defaults(func=cmd_audit_identifiers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
