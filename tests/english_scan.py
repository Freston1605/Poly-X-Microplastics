"""Self-contained Spanish-remnant scanner for the English-convention gates.

This is the durable half of the retired translation tooling: the four
``*_ingles`` guards still need to recognise Spanish prose and identifiers, but
they no longer need a codemod to do it. Everything here is standard-library
``ast``/``tokenize`` plus the curated vocabulary that was already proven
against the package; nothing imports LibCST.

Three scanners live side by side:

- comments: ``comment_blocks``, ``_comment_tokens``, ``_identifiers_from_source``,
  ``spanish_comment_findings`` and ``scan_comments``;
- docstrings: ``enumerate_docstrings``, ``spanish_findings`` and
  ``_strip_inline_code``;
- identifiers: ``identifier_occurrences`` and ``spanish_identifier_findings``.

The vocabulary is data, not logic: extend a word list when a real Spanish term
survives, never loosen a regex.
"""
from __future__ import annotations

import ast
import functools
import io
import re
import tokenize
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
POLYX_ROOT = REPO_ROOT / "src" / "polyx"

# ---------------------------------------------------------------------------
# Comment splitting and protection (formerly the codemod's comment core).
# ---------------------------------------------------------------------------

# A comment starts with one or more hashes, an optional ':' (the '#:'
# doc-comment style), and the full run of spaces/tabs that follows. The
# whitespace run belongs to the prefix so aligned tables keep their margin.
_PREFIX = re.compile(r"^(?P<hashes>#+)(?P<colon>:?)(?P<space>[ \t]*)")

# Directives that must never be read as prose.
_SHEBANG = re.compile(r"^#!")
_CODING_COOKIE = re.compile(r"\bcoding[:=]\s*[-.\w]+", re.IGNORECASE)
_DIRECTIVE = re.compile(
    r"^#\s*(?:"
    r"noqa\b|type\s*:|pragma\s*:|fmt\s*:|isort\s*:|pylint\s*:|mypy\s*:|nosec\b"
    r")",
    re.IGNORECASE,
)
_CELL_MARKER = re.compile(r"^#\s*%%")

# Any Unicode alphanumeric (letters and digits) makes a comment prose; a body
# made only of punctuation or box-drawing characters is a separator line.
_PROSE_CHARACTER = re.compile(r"[^\W_]", re.UNICODE)


def split_comment(raw: str) -> Tuple[str, str, str]:
    """Split a raw comment into ``(prefix, body, trailing_whitespace)``.

    The prefix is the leading hash run, an optional ``:`` and the full
    following run of spaces/tabs. The trailing whitespace is the run of
    spaces/tabs at the end of the comment. Only ``body`` is prose.
    """
    match = _PREFIX.match(raw)
    if match is None:
        # Defensive: Python comments always start with '#'. Treat the whole
        # value as the body so nothing is silently dropped.
        return "", raw, ""
    prefix = match.group(0)
    rest = raw[len(prefix):]
    body = rest.rstrip(" \t")
    trailing = rest[len(body):]
    return prefix, body, trailing


def is_protected_comment(raw: str) -> bool:
    """Return True when a raw comment must never be treated as prose.

    Protected comments are shebangs, PEP 263 coding cookies, tool directives
    (``# noqa``, ``# type:``, ``# pragma:``, ``# fmt:``, ``# isort:``,
    ``# pylint:``, ``# mypy:``, ``# nosec``), ``# %%`` cell markers, and
    empty or separator-only comments (bodies with no letters or digits).
    """
    if _SHEBANG.match(raw):
        return True
    if _CODING_COOKIE.search(raw):
        return True
    if _DIRECTIVE.match(raw):
        return True
    if _CELL_MARKER.match(raw):
        return True
    _, body, _ = split_comment(raw)
    if not body.strip():
        return True
    if _PROSE_CHARACTER.search(body) is None:
        return True
    return False


# ---------------------------------------------------------------------------
# Shared prose detector: curated high-signal words plus accented-letter
# heuristic (the same base the docs gate has always used).
# ---------------------------------------------------------------------------

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

_ACENTUADA = re.compile(r"[A-Za-zÀ-ÿ]*[áéíóúñüÁÉÍÓÚÑÜ][A-Za-zÀ-ÿ]*")

PROSE_ALLOWLIST: frozenset = frozenset(
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
    findings = spanish_findings(cleaned)
    for match in SPANISH_CONTENT_WORDS.finditer(cleaned):
        word = match.group(0).lower()
        if word not in PROSE_ALLOWLIST:
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


def _identifiers_from_source(text: str) -> Set[str]:
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


def scan_comments(text: str) -> List[Tuple[str, List[str]]]:
    """Return ``(line_label, findings)`` for every Spanish comment finding."""
    tokens = _comment_tokens(text)
    identifiers = _identifiers_from_source(text)
    results: List[Tuple[str, List[str]]] = []
    for token in tokens:
        if is_protected_comment(token.string):
            continue
        body = split_comment(token.string)[1]
        findings = spanish_comment_findings(body, identifiers)
        if findings:
            results.append((str(token.start[0]), findings))
    for block in comment_blocks(tokens):
        bodies = [
            split_comment(token.string)[1]
            for token in block
            if not is_protected_comment(token.string)
        ]
        if len(bodies) < 2:
            continue
        findings = spanish_comment_findings(" ".join(bodies), identifiers)
        if findings:
            results.append(
                (f"{block[0].start[0]}-{block[-1].start[0]}", findings)
            )
    return results


# ---------------------------------------------------------------------------
# Docstrings.
# ---------------------------------------------------------------------------


def _strip_inline_code(text: str) -> str:
    """Remove backticked code spans from docstring prose."""
    text = unicodedata.normalize("NFC", text)
    # fenced blocks first, then single spans.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", " ", text)
    return text


def spanish_findings(prose: str) -> List[str]:
    """Return the Spanish markers found in prose ([] = clean)."""
    findings: List[str] = []
    for match in PALABRAS_ES.finditer(prose):
        word = match.group(0).lower()
        if word not in PROSE_ALLOWLIST:
            findings.append(f"spanish-word:{word}")
    for match in _ACENTUADA.finditer(prose):
        word = match.group(0)
        if word.lower() not in PROSE_ALLOWLIST:
            findings.append(f"accented:{word}")
    return findings


def enumerate_docstrings(text: str) -> List[Tuple[int, str]]:
    """Return ``[(index, content)]`` for each docstring in document order.

    The index is the pre-order position of the owning node among modules,
    classes and functions, matching the numbering the retired LibCST visitor
    produced.
    """
    tree = ast.parse(text)
    docs: List[Tuple[int, str]] = []
    counter = 0

    def visit(node: ast.AST) -> None:
        nonlocal counter
        index: Optional[int] = None
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            index = counter
            counter += 1
            body = getattr(node, "body", None)
            if body:
                first = body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    docs.append((index, first.value.value))
        # Descend through every node, not only definitions: a class or
        # function nested in an ``if``/``try`` block still owns a docstring.
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return docs


# ---------------------------------------------------------------------------
# Identifiers.
# ---------------------------------------------------------------------------

# Observability sinks whose keyword arguments are JSONL data keys, not Python
# symbols: the scanner must leave them alone.
OBSERVABILITY_SINKS = frozenset({"event", "evento", "timer", "cronometro"})

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

EXACT_ALLOWLIST = frozenset(
    {
        "QRadioButton",
        "SizeVerCursor",
        "ControlModifier",
        "mac_ver",
    }
)

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

MORPHOLOGY_EXCEPTIONS = frozenset(
    {
        "avocado", "bonanza", "colorado", "coercion", "desperado", "metadata",
        "mikado", "suspicion", "tornado",
    }
)

_MORPHOLOGY = re.compile(
    r"(?:cion|ciones|idad|idades|ador|adora|adores|adoras|amiento|amientos"
    r"|imiento|imientos|anza|anzas|encia|encias|oso|osa|osos|osas"
    r"|ivo|iva|ivos|ivas|ado|ada|ados|adas|ido|ida|idos|idas)$"
)

_ACCENTED_LETTER = re.compile(r"[áéíóúüñÁÉÍÓÚÜÑ]")

_SUBWORD = re.compile(
    r"[A-ZÀ-ÖØ-Þ]+(?=[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ])"
    r"|[A-ZÀ-ÖØ-Þ]?[a-zà-öø-ÿ]+"
    r"|[A-ZÀ-ÖØ-Þ]+"
    r"|\d+"
)

IDENTIFIER_ALLOWLIST = HOMOGRAPH_ALLOWLIST | UNIT_ALLOWLIST


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
    filter on the token after the colon, exactly like the prose scanners.
    """
    if name in EXACT_ALLOWLIST:
        return []
    findings: List[str] = []
    for subword in split_identifier(name):
        base = _strip_digits(subword)
        if not base:
            continue
        lowered = base.lower()
        if lowered in IDENTIFIER_ALLOWLIST:
            continue
        if _ACCENTED_LETTER.search(base):
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
    stems: Set[str] = {POLYX_ROOT.name}
    if not POLYX_ROOT.is_dir():
        return frozenset(stems)
    for path in POLYX_ROOT.rglob("*.py"):
        stems.add(path.stem)
    for path in POLYX_ROOT.rglob("*"):
        if path.is_dir():
            stems.add(path.name)
    stems.discard("__init__")
    return frozenset(stems)


def module_stem_bindings(text: str, stems: Iterable[str] = ()) -> Set[str]:
    """Return local names bound to an imported module stem in ``text``.

    ``from polyx.gui import iconos`` binds the module ``iconos``; the binding
    is module-path context, so it is neither renamed nor scanned. Nested
    imports count too.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    stem_set = frozenset(stems)
    bindings: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.names:
            for alias in node.names:
                if alias.asname is not None:
                    continue
                if alias.name in stem_set:
                    bindings.add(alias.name)
    return bindings


def _ignored_token(token: tokenize.TokenInfo) -> bool:
    return token.type in {
        tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
        tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER,
    }


def _keyword_positions_after(
    text: str, heads: Set[str], *, skip_definitions: bool, require_dot: bool = False
) -> Set[Tuple[int, int]]:
    """Positions of keyword names in calls whose head is in ``heads``.

    Used for two sets of non-symbols: ``str.format`` field references (which
    must be called as a method, hence ``require_dot``), and the keyword
    arguments of the observability sinks (JSONL data keys).
    """
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if not _ignored_token(token)
    ]
    positions: Set[Tuple[int, int]] = set()
    for index, token in enumerate(tokens):
        if (
            token.type != tokenize.NAME
            or token.string not in heads
            or index + 1 >= len(tokens)
            or tokens[index + 1].string != "("
        ):
            continue
        if require_dot and (index == 0 or tokens[index - 1].string != "."):
            continue
        if skip_definitions and index > 0 and tokens[index - 1].string == "def":
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


def _format_keyword_positions(text: str) -> Set[Tuple[int, int]]:
    """Positions of keyword names that are ``str.format`` field references."""
    return _keyword_positions_after(
        text, {"format", "format_map"}, skip_definitions=False, require_dot=True
    )


def _data_keyword_positions(text: str) -> Set[Tuple[int, int]]:
    """Positions of keyword names that are JSONL data keys."""
    return _keyword_positions_after(text, set(OBSERVABILITY_SINKS), skip_definitions=True)


def identifier_occurrences(
    text: str, stems: Optional[Iterable[str]] = None
) -> List[Tuple[int, str]]:
    """Yield ``(line, name)`` for every identifier outside import context.

    Module paths, imported symbols and the local bindings of imported module
    stems are skipped: the identifier they refer to is defined and scanned in
    its own module. A local ``iconos`` variable in a module without that
    import is still an occurrence and is scanned normally.
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
