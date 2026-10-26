"""Texto del manual de Poly-X en español.

Solo contenido: el motor que lo convierte en HTML está en generar_manual.py.

Tokens disponibles dentro de cada sección:
    [[fig:ui/01_launcher|Pie de figura]]      captura de la interfaz (por idioma)
    [[fig:web/w01_github_repo|Pie]]           captura de una página web
    [[fig:win/p01_zip_descargado|Pie]]        captura de Windows
    [[esquema:python_installer|Pie]]          dibujo, no fotografía
"""

TITULO_DOC = "Manual de Poly-X — Suite de detección de microplásticos"
DESCRIPCION = ("Manual completo de Poly-X: instalación paso a paso desde GitHub en "
               "Windows y macOS, y guia de los cuatro módulos.")
SELLO = "Manual de usuario · v2.0.0"
TITULO_H1 = 'Poly-X <em>analytics</em>'
BAJADA = ("Detección, medida y clasificación de microplásticos por fluorescencia "
          "Nile Red bajo luz UV, con modelos YOLO v8/v11. De la descarga en GitHub "
          "al informe listo para publicar.")
META = ("<b>Cristofher Ferrada</b> · Dr (c) en Ciencias mención Química<br>"
        "Laboratorio de Química Ambiental · Pontificia Universidad Católica de Valparaíso<br>"
        "Windows 10/11 y macOS · Python 3.11 · 2026")

ETIQUETA_INDICE = "Contenido"
ETIQUETA_FIGURA = "Figura"
PALABRA_ESQUEMA = "Esquema"

PIE = ("<b>Poly-X v2.0.0</b> — Cristofher Ferrada, 2026. "
       "Repositorio: <a href='https://github.com/CrissFerrada/Poly-X-Microplastics'>"
       "github.com/CrissFerrada/Poly-X-Microplastics</a> · "
       "Contacto: <a href='mailto:cristofher.ferrada@pucv.cl'>cristofher.ferrada@pucv.cl</a><br>"
       "Las capturas de la interfaz se regeneran contra el programa instalado, de modo "
       "que este manual muestra la versión que tienes delante y no una anterior.")


# ════════════════════════════════════════════════════════════════════
#  Esquemas: dibujos de los diálogos que no admiten fotografía honesta
# ════════════════════════════════════════════════════════════════════
_SVG_PY = """
<svg viewBox="0 0 760 470" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Pantalla inicial del instalador de Python para Windows">
  <rect width="760" height="470" fill="#f6f8fa"/>
  <rect x="40" y="26" width="680" height="418" rx="8" fill="#fff" stroke="#d0d7de"/>
  <rect x="40" y="26" width="680" height="40" rx="8" fill="#eaeef2"/>
  <rect x="40" y="56" width="680" height="10" fill="#eaeef2"/>
  <text x="60" y="52" font-family="Segoe UI,sans-serif" font-size="14" fill="#1f2328">
    Python 3.11.9 (64-bit) Setup</text>
  <text x="694" y="52" font-family="Segoe UI,sans-serif" font-size="15" fill="#656d76">✕</text>

  <rect x="40" y="66" width="200" height="290" fill="#2b5b84"/>
  <text x="140" y="196" text-anchor="middle" font-family="Segoe UI,sans-serif"
        font-size="17" font-weight="600" fill="#ffd43b">python</text>
  <text x="140" y="220" text-anchor="middle" font-family="Segoe UI,sans-serif"
        font-size="12" fill="#9fc3e0">windows</text>

  <text x="266" y="106" font-family="Segoe UI,sans-serif" font-size="19"
        font-weight="600" fill="#1f2328">Install Python 3.11.9 (64-bit)</text>
  <text x="266" y="132" font-family="Segoe UI,sans-serif" font-size="13" fill="#424a53">
    Select Install Now to install Python with default settings.</text>

  <rect x="266" y="152" width="424" height="46" rx="5" fill="#f6f8fa" stroke="#d0d7de"/>
  <text x="284" y="173" font-family="Segoe UI,sans-serif" font-size="14"
        font-weight="600" fill="#0969da">▸ Install Now</text>
  <text x="284" y="190" font-family="Segoe UI,sans-serif" font-size="11.5" fill="#656d76">
    C:\\Users\\...\\AppData\\Local\\Programs\\Python\\Python311</text>

  <rect x="266" y="208" width="424" height="34" rx="5" fill="#f6f8fa" stroke="#d0d7de"/>
  <text x="284" y="230" font-family="Segoe UI,sans-serif" font-size="14" fill="#424a53">
    ▸ Customize installation</text>

  <!-- Las dos casillas que importan -->
  <rect x="258" y="262" width="440" height="76" rx="7" fill="#fffbf0" stroke="#9a6700" stroke-width="2"/>
  <rect x="274" y="276" width="15" height="15" rx="2.5" fill="#0969da" stroke="#0550ae"/>
  <path d="M277.5 283.5 l3.4 3.6 l6.4 -7.2" stroke="#fff" stroke-width="2.2" fill="none"
        stroke-linecap="round" stroke-linejoin="round"/>
  <text x="300" y="289" font-family="Segoe UI,sans-serif" font-size="13.5"
        font-weight="700" fill="#1f2328">Add python.exe to PATH</text>
  <rect x="274" y="304" width="15" height="15" rx="2.5" fill="#0969da" stroke="#0550ae"/>
  <path d="M277.5 311.5 l3.4 3.6 l6.4 -7.2" stroke="#fff" stroke-width="2.2" fill="none"
        stroke-linecap="round" stroke-linejoin="round"/>
  <text x="300" y="317" font-family="Segoe UI,sans-serif" font-size="13.5"
        font-weight="700" fill="#1f2328">Use admin privileges when installing py.exe</text>

  <path d="M712 300 L706 300" stroke="#9a6700" stroke-width="2"/>
  <text x="716" y="292" font-family="Segoe UI,sans-serif" font-size="12"
        font-weight="700" fill="#9a6700">marca</text>
  <text x="716" y="308" font-family="Segoe UI,sans-serif" font-size="12"
        font-weight="700" fill="#9a6700">las dos</text>

  <rect x="40" y="356" width="680" height="88" fill="#f6f8fa"/>
  <line x1="40" y1="356" x2="720" y2="356" stroke="#d0d7de"/>
  <rect x="560" y="394" width="132" height="32" rx="4" fill="#0969da"/>
  <text x="626" y="415" text-anchor="middle" font-family="Segoe UI,sans-serif"
        font-size="13.5" font-weight="600" fill="#fff">Install Now</text>
  <rect x="428" y="394" width="120" height="32" rx="4" fill="#fff" stroke="#d0d7de"/>
  <text x="488" y="415" text-anchor="middle" font-family="Segoe UI,sans-serif"
        font-size="13.5" fill="#424a53">Cancel</text>
</svg>
"""

_SVG_GATEKEEPER = """
<svg viewBox="0 0 760 400" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Aviso de macOS: desarrollador no identificado">
  <rect width="760" height="400" fill="#f6f8fa"/>
  <rect x="215" y="40" width="330" height="320" rx="13" fill="#fff" stroke="#d0d7de"/>
  <circle cx="380" cy="104" r="34" fill="#fef3c7" stroke="#9a6700" stroke-width="2"/>
  <text x="380" y="118" text-anchor="middle" font-size="36" fill="#9a6700">⚠</text>
  <text x="380" y="176" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="15" font-weight="700" fill="#1f2328">No se puede abrir</text>
  <text x="380" y="197" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="15" font-weight="700" fill="#1f2328">«Lanzar_macOS.command»</text>
  <text x="380" y="222" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="12.5" fill="#424a53">porque es de un desarrollador</text>
  <text x="380" y="240" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="12.5" fill="#424a53">no identificado.</text>
  <line x1="215" y1="268" x2="545" y2="268" stroke="#eaeef2"/>
  <rect x="243" y="288" width="126" height="34" rx="7" fill="#fff" stroke="#d0d7de"/>
  <text x="306" y="310" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="13" fill="#424a53">Mover a la papelera</text>
  <rect x="391" y="288" width="126" height="34" rx="7" fill="#0969da"/>
  <text x="454" y="310" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="13" font-weight="600" fill="#fff">Cancelar</text>
  <text x="380" y="345" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="11.5" fill="#cf222e" font-weight="600">Ni una ni otra: cancela y abre con clic derecho</text>
</svg>
"""

_SVG_MAC_ABRIR = """
<svg viewBox="0 0 760 400" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Menú contextual de Finder con la opción Abrir">
  <rect width="760" height="400" fill="#f6f8fa"/>
  <rect x="46" y="34" width="668" height="332" rx="10" fill="#fff" stroke="#d0d7de"/>
  <rect x="46" y="34" width="668" height="38" rx="10" fill="#eaeef2"/>
  <rect x="46" y="62" width="668" height="10" fill="#eaeef2"/>
  <circle cx="70" cy="53" r="6" fill="#ff5f57"/><circle cx="90" cy="53" r="6" fill="#febc2e"/>
  <circle cx="110" cy="53" r="6" fill="#28c840"/>
  <text x="380" y="58" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="13" font-weight="600" fill="#1f2328">Poly-X-Microplastics-main</text>

  <g font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#1f2328">
    <text x="82" y="106">📄  actualizar_macOS.command</text>
    <text x="82" y="140">📄  construir_app_macOS.command</text>
    <rect x="66" y="152" width="300" height="26" rx="5" fill="#dbeafe"/>
    <text x="82" y="170" font-weight="700">📄  Lanzar_macOS.command</text>
    <text x="82" y="204">📄  LEEME_macOS.md</text>
    <text x="82" y="238">📁  polyx</text>
    <text x="82" y="272">📄  README.md</text>
  </g>

  <!-- menú contextual -->
  <rect x="286" y="150" width="238" height="196" rx="9" fill="#fff" stroke="#d0d7de"/>
  <rect x="286" y="150" width="238" height="196" rx="9" fill="none" stroke="#00000012"/>
  <rect x="292" y="176" width="226" height="28" rx="5" fill="#0969da"/>
  <text x="308" y="171" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Abrir con</text>
  <text x="308" y="195" font-family="-apple-system,Segoe UI,sans-serif" font-size="13.5"
        font-weight="700" fill="#fff">Abrir</text>
  <text x="308" y="226" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Mover a la papelera</text>
  <line x1="292" y1="240" x2="518" y2="240" stroke="#eaeef2"/>
  <text x="308" y="262" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Obtener información</text>
  <text x="308" y="288" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Renombrar</text>
  <text x="308" y="314" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Comprimir</text>
  <text x="308" y="338" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Duplicar</text>

  <path d="M560 190 L536 190" stroke="#1f6b5e" stroke-width="2.4" marker-end="url(#fl)"/>
  <defs><marker id="fl" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
    <path d="M0 0 L9 4.5 L0 9 z" fill="#1f6b5e"/></marker></defs>
  <text x="566" y="186" font-family="-apple-system,Segoe UI,sans-serif" font-size="12.5"
        font-weight="700" fill="#1f6b5e">clic DERECHO</text>
  <text x="566" y="203" font-family="-apple-system,Segoe UI,sans-serif" font-size="12.5"
        font-weight="700" fill="#1f6b5e">y luego Abrir</text>
</svg>
"""

ESQUEMAS = {
    "python_installer": _SVG_PY,
    "gatekeeper": _SVG_GATEKEEPER,
    "mac_abrir": _SVG_MAC_ABRIR,
}


# ════════════════════════════════════════════════════════════════════
#  Secciones
# ════════════════════════════════════════════════════════════════════
SECCIONES = [

# ── 1 ───────────────────────────────────────────────────────────────
{
"id": "que-es",
"titulo": "Qué es Poly-X",
"sub": "Qué problema resuelve, qué hace y qué no hace.",
"html": r"""
<p>Poly-X es un programa de escritorio para <b>contar, medir y clasificar
microplásticos</b> en fotografías de microscopio. Las partículas se tiñen con
<b>rojo Nilo</b> (Nile Red) y se fotografian bajo luz <b>UV de 254 nm</b>: cada
polímero emite con un color y un brillo distintos, y sobre esa emisión trabaja un
modelo de detección <b>YOLO v8/v11</b> que localiza cada partícula, le asigna una
clase y devuelve su talla en micrómetros.</p>

<p>Cubre el ciclo entero sin salir del programa: se anotan imágenes, se entrena un
modelo con ellas, se analiza un lote completo y se emite un informe con las
tablas y figuras ya armadas.</p>

<div class="modulos">
  <div class="mod"><span class="ic">🔬</span><b>Detector</b>
    <span>Analiza carpetas enteras con un modelo entrenado y produce el informe.</span></div>
  <div class="mod"><span class="ic">🎯</span><b>Entrenador</b>
    <span>Entrena modelos YOLO v8 y v11 con curvas en vivo y auditoría del dataset.</span></div>
  <div class="mod"><span class="ic">🏷</span><b>Etiquetador</b>
    <span>Anota partículas en formato YOLO, con pre-anotación automática.</span></div>
  <div class="mod"><span class="ic">📐</span><b>Visor</b>
    <span>Inspecciona una imagen, calibra micrómetros por píxel y revisa cada medida.</span></div>
</div>

<h3>Los tres polímeros</h3>
<p>El modelo distingue tres clases, que son las del método publicado del
laboratorio:</p>
<div class="tabla-env">
<table>
  <thead><tr><th>Clase</th><th>Nombre</th><th>Emisión observada bajo UV</th></tr></thead>
  <tbody>
    <tr><td><span class="tag pet">PET</span></td><td>Tereftalato de polietileno</td>
        <td>Rojo–salmon</td></tr>
    <tr><td><span class="tag pp">PP</span></td><td>Polipropileno</td>
        <td>Amarillo <b>verdoso y apagado</b></td></tr>
    <tr><td><span class="tag ldpe">LDPE</span></td><td>Polietileno de baja densidad</td>
        <td>Amarillo franco y <b>más brillante</b></td></tr>
  </tbody>
</table>
</div>

<div class="aviso warn">
  <span class="et">Lo que conviene saber antes de confiar en una clase</span>
  <p>PP y LDPE <b>no se separan por tono sino por brillo</b>: los dos son
  amarillentos. Es la confusión más frecuente al anotar y la razón de que el
  recall caiga en esas dos clases. Está explicado con las cifras en
  <a href="#polimeros">la sección 12</a>.</p>
</div>

<h3>El flujo completo</h3>
<pre><code><span class="c">Fotografías de microscopio (UV 254 nm, tinción Nile Red)</span>
        ↓
  🏷 <span class="p">Etiquetador</span>  → anota PET / PP / LDPE en formato YOLO
        ↓
  🎯 <span class="p">Entrenador</span>   → entrena el modelo con esas anotaciones
        ↓
  🔬 <span class="p">Detector</span>     → analiza el lote y emite el informe
        ↓
  📐 <span class="p">Visor</span>        → revisa partícula a partícula y calibra</code></pre>

<p>Si ya tienes un modelo entrenado, el camino corto es <b>Detector solo</b>: los
otros tres módulos existen para construir ese modelo y para verificarlo.</p>

<div class="aviso">
  <span class="et">Alcance de este manual</span>
  <p>Las secciones <b>2 a 6</b> son la instalación: descargar de GitHub, instalar
  y arrancar por primera vez, con una fotografía por paso. Las secciones
  <b>7 a 15</b> recorren todo lo que el programa sabe hacer hoy, pantalla por
  pantalla. Si solo vienes a instalarlo, con las primeras seis basta.</p>
</div>
"""
},

# ── 2 ───────────────────────────────────────────────────────────────
{
"id": "requisitos",
"titulo": "Antes de empezar",
"sub": "Qué necesita el equipo, y cuanto va a ocupar.",
"html": r"""
<div class="tabla-env">
<table>
  <thead><tr><th>Componente</th><th>Windows</th><th>macOS</th></tr></thead>
  <tbody>
    <tr><td><b>Sistema</b></td><td>Windows 10 u 11</td><td>11 Big Sur o posterior</td></tr>
    <tr><td><b>Python</b></td><td><b>3.11.x</b> — no 3.12 ni superior</td><td>3.9 o superior</td></tr>
    <tr><td><b>RAM</b></td><td>8 GB mínimo</td><td>8 GB mínimo</td></tr>
    <tr><td><b>Disco</b></td><td colspan="2">Unos <b>6 GB</b> libres: el entorno con
        PyTorch para GPU ocupa cerca de 5 GB</td></tr>
    <tr><td><b>Aceleración</b></td>
        <td>GPU NVIDIA opcional. Con ella el entrenamiento va 20–30× más rápido;
            el instalador elige la versión de CUDA que corresponde a la tarjeta</td>
        <td><b>Apple Silicon:</b> GPU integrada por MPS.<br>
            <b>Intel:</b> solo CPU, ~1 min por foto</td></tr>
    <tr><td><b>Internet</b></td><td colspan="2">Solo durante la instalación y las
        actualizaciones. Después funciona sin conexión</td></tr>
  </tbody>
</table>
</div>

<div class="aviso err">
  <span class="et">Python 3.12 no sirve</span>
  <p>En Windows hay que instalar <b>Python 3.11.x</b>. Las versiones 3.12 y
  posteriores no tienen ruedas compatibles con la combinación de PyTorch,
  NumPy 1.26 y Ultralytics 8.3 que Poly-X fija, y la instalación falla a mitad
  con errores de compilación que no dicen cuál es la causa real.</p>
</div>

<h3>Cuánto tarda</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Paso</th><th>Tiempo típico</th><th>Se hace</th></tr></thead>
  <tbody>
    <tr><td>Instalar Python 3.11</td><td>2–4 min</td><td>una vez por equipo</td></tr>
    <tr><td>Descargar Poly-X de GitHub</td><td>menos de 1 min (≈ 5 MB)</td><td>una vez</td></tr>
    <tr><td>Ejecutar <code>SETUP.bat</code></td><td>5–15 min</td><td>una vez</td></tr>
    <tr><td>Arrancar el programa</td><td>2–4 s</td><td>cada día</td></tr>
  </tbody>
</table>
</div>

<div class="aviso vio">
  <span class="et">Un dato que sorprende</span>
  <p>La descarga de GitHub pesa <b>menos de 5 MB</b>, pero la instalación termina
  ocupando varios gigabytes. No es un error: lo que pesa es <b>PyTorch con
  soporte de GPU</b>, que se descarga aparte y ronda los <b>2,8 GB</b>
  comprimidos. El programa en sí es pequeño; la biblioteca de cálculo no.</p>
</div>
"""
},

# ── 3 ───────────────────────────────────────────────────────────────
{
"id": "instalar-windows",
"titulo": "Instalación en Windows, paso a paso",
"sub": "Cinco pasos, con una fotografía de cada uno. Se hace una sola vez.",
"anclas": [("w-python", "Paso 1 · Python 3.11"),
           ("w-descargar", "Paso 2 · Descargar de GitHub"),
           ("w-extraer", "Paso 3 · Descomprimir"),
           ("w-setup", "Paso 4 · Ejecutar SETUP.bat"),
           ("w-arrancar", "Paso 5 · Primer arranque")],
"html": r"""
<ol class="pasos">

<li id="w-python"><span class="t">Instalar Python 3.11.9</span>
<p>Si el equipo ya tiene Python 3.11, salta al paso 2. Para comprobarlo, abre el
menú Inicio, escribe <code>cmd</code>, y en la ventana negra escribe:</p>
<pre><code><span class="p">python --version</span>
<span class="o">Python 3.11.9</span>   <span class="c">← si sale esto, ya está; ve al paso 2</span></code></pre>

<p>Si no lo tienes, entra a
<a href="https://www.python.org/downloads/release/python-3119/">python.org/downloads/release/python-3119</a>.</p>

[[fig:web/w04_python_release|La página de la versión 3.11.9 en python.org. Baja hasta el final,
donde está la tabla de archivos.]]

<p>Al final de esa página hay una tabla con todos los instaladores. El que
necesitas es <b>Windows installer (64-bit)</b>, el que dice <i>Recommended</i>.</p>

[[fig:web/w06_python_tabla_recorte|Tabla de archivos de la versión 3.11.9. En Windows se descarga
<b>Windows installer (64-bit)</b>; en un Mac, <b>macOS 64-bit universal2 installer</b>.]]

<div class="aviso err">
  <span class="et">La casilla que decide si todo lo demás funciona</span>
  <p>Al abrir el instalador, <b>antes</b> de pulsar <i>Install Now</i>, marca
  <b>Add python.exe to PATH</b> abajo del todo. Sin esa casilla,
  <code>SETUP.bat</code> no encuentra Python y se detiene. Es, con diferencia,
  el fallo más común de esta instalación.</p>
</div>

[[esquema:python_installer|Pantalla inicial del instalador de Python en Windows. Marca las dos
casillas de abajo y recien entonces pulsa <i>Install Now</i>. Va como dibujo y no como
fotografía porque en un equipo que ya tiene Python el instalador muestra la pantalla de
mantenimiento, no esta.]]

<p>Marca también <b>tcl/tk and IDLE</b> si el instalador te ofrece elegir
componentes: algunos diálogos del programa lo usan.</p>
</li>

<li id="w-descargar"><span class="t">Descargar Poly-X desde GitHub</span>
<p>Abre <a href="https://github.com/CrissFerrada/Poly-X-Microplastics">github.com/CrissFerrada/Poly-X-Microplastics</a>.
No hace falta tener cuenta ni iniciar sesión: el repositorio es público.</p>

[[fig:web/w01_github_repo|Página del repositorio. El botón verde <b>Code</b> está arriba a la
derecha del listado de archivos.]]

<p>Pulsa el botón verde <b>Code</b> y, en el menú que se despliega, la última
opción: <b>Download ZIP</b>.</p>

[[fig:web/w03_github_download_zip|El menú <b>Code</b> desplegado. La opción que buscas es
<b>Download ZIP</b>, al final. Las de arriba son para quien use Git o GitHub Desktop.]]

<div class="aviso">
  <span class="et">Si prefieres usar Git</span>
  <p>Si ya tienes Git instalado, esto es equivalente y además facilita
  actualizar:</p>
  <pre><code><span class="p">git clone https://github.com/CrissFerrada/Poly-X-Microplastics.git</span></code></pre>
</div>
</li>

<li id="w-extraer"><span class="t">Descomprimir el ZIP</span>
<p>El archivo llega a tu carpeta de <b>Descargas</b> y pesa alrededor de
<b>5 MB</b>. Haz clic derecho sobre el y elige <b>Extraer todo</b>
(o selecciona el archivo y usa el botón <b>Extraer todo</b> de la barra
superior del Explorador).</p>

[[fig:win/p01_zip_descargado|El ZIP recien descargado, con el botón <b>Extraer todo</b> visible
en la barra del Explorador de archivos.]]

<div class="aviso warn">
  <span class="et">No lo ejecutes desde dentro del ZIP</span>
  <p>Windows deja abrir archivos dentro de un ZIP como si fuera una carpeta, pero
  <b>lo descomprime a una carpeta temporal</b> que borra después. Si ejecutas
  <code>SETUP.bat</code> desde ahí, la instalación se pierde. Descomprime
  primero, de verdad.</p>
</div>

<p>Elige una ruta <b>sin acentos ni caracteres raros</b> y con permisos de
escritura. <code>C:\PolyX\</code> o tu carpeta de Documentos sirven; el
Escritorio sincronizado con OneDrive puede dar problemas si la sincronización
bloquea archivos mientras se instala.</p>

<p>Al terminar tendras una carpeta llamada
<code>Poly-X-Microplastics-main</code> con todo dentro.</p>

[[fig:win/p02_carpeta_extraida|La carpeta ya descomprimida. El archivo que hay que ejecutar es
<b>SETUP.bat</b>, seleccionado aquí abajo del todo.]]
</li>

<li id="w-setup"><span class="t">Ejecutar SETUP.bat</span>
<p>Doble clic en <b>SETUP.bat</b>. Se abre una ventana negra: eso es normal y es
donde va a ocurrir todo. Lo primero que hace es preguntarte <b>donde instalar</b>.</p>

[[fig:win/p03_setup_pregunta_carpeta|Primera pantalla de <code>SETUP.bat</code>. Pulsa
<kbd>Enter</kbd> para instalar en la misma carpeta que descomprimiste, que es lo recomendado.]]

<p>Pulsa <kbd>Enter</kbd> sin escribir nada y se instala ahí mismo. Si prefieres
otra ubicación, pega la ruta y pulsa <kbd>Enter</kbd>.</p>

<p>A partir de ahí el instalador trabaja solo. Detecta Python, crea el entorno
<code>.venv</code>, mira qué tarjeta gráfica tienes y descarga <b>la versión de
PyTorch que corresponde a esa tarjeta</b>.</p>

[[fig:win/p04_setup_python_gpu|El instalador detectando la GPU y descargando PyTorch. Aquí
reconoció una NVIDIA de <i>compute capability</i> 7.5 y eligió CUDA 11.8: son 2,8 GB, y es
la parte que más tarda.]]

<div class="aviso vio">
  <span class="et">Por qué mira la generación de la tarjeta y no solo si hay GPU</span>
  <p>Saber que hay una NVIDIA no basta. Las tarjetas <b>RTX 50xx</b>
  (Blackwell, <code>sm_120</code>) necesitan CUDA 12.8: si se les instalan las
  ruedas de CUDA 11.8, <code>torch.cuda.is_available()</code> devuelve
  <code>True</code>, todo parece correcto, y el fallo aparece recien al entrenar
  con un <i>no kernel image is available for execution on the device</i> que no
  se parece en nada a su causa. Por eso el instalador comprueba, al final, que
  PyTorch traiga kernels para <b>tu</b> tarjeta concreta.</p>
</div>

<p>Cuando termina, ofrece crear un acceso directo en el Escritorio y te dice como
arrancar el programa.</p>

[[fig:win/p05_setup_completado|Comprobación final. No basta con que haya GPU: el instalador contrasta la arquitectura de la tarjeta (<code>sm_75</code>) contra la lista de kernels que trae PyTorch. Debajo, la búsqueda de instalaciones anteriores — las carpetas con <code>.git</code> se omiten siempre.]]

[[fig:win/p06_setup_final|Instalación completada. Dice dónde quedó instalado, como iniciarlo y que falta: el modelo <code>.pt</code>, que no viene en la descarga.]]

<div class="aviso">
  <span class="et">Si el equipo ya tenia una versión anterior de Poly-X</span>
  <p>Al final, <code>SETUP.bat</code> busca instalaciones antiguas. Si encuentra
  alguna, te muestra que contiene —modelos, entrenamientos, detecciones— y pide
  confirmación antes de tocar nada. El orden es siempre
  <b>copiar → verificar → retirar</b>: solo si la copia salió bien, la carpeta
  vieja va a la <b>papelera</b>, nunca al borrado directo. Las carpetas con
  <code>.git</code> se omiten, porque un repositorio de desarrollo no es una
  instalación vieja.</p>
</div>
</li>

<li id="w-arrancar"><span class="t">Arrancar Poly-X</span>
<p>Doble clic en el acceso directo <b>Poly-X</b> del Escritorio. Si preferiste no
crearlo, dentro de la carpeta tienes tres formas equivalentes:</p>

<div class="tabla-env">
<table>
  <thead><tr><th>Archivo</th><th>Qué hace</th><th>Cuándo usarlo</th></tr></thead>
  <tbody>
    <tr><td><code>Poly-X.vbs</code></td><td>Abre el programa <b>sin ventana negra</b></td>
        <td>El uso normal, día a día</td></tr>
    <tr><td><code>iniciar_polyx.bat</code></td><td>Lo mismo, pero mostrando la consola</td>
        <td>Cuando algo falla y quieres ver el mensaje de error</td></tr>
    <tr><td><code>.venv\Scripts\python.exe -m polyx.launcher</code></td>
        <td>Arranque manual desde la terminal</td><td>Para depurar</td></tr>
  </tbody>
</table>
</div>

[[fig:ui/01_launcher|El Launcher recien abierto. Desde aquí se entra a los cuatro módulos, y
el selector de arriba a la derecha cambia el idioma de todo el programa.]]

<div class="aviso ok">
  <span class="et">Instalación terminada</span>
  <p>Si ves esta pantalla, ya esta. Falta una sola cosa antes de poder detectar:
  copiar un modelo entrenado dentro de <code>models\</code>, porque
  <b>los modelos no vienen en la descarga</b>. Es la
  <a href="#modelo-pt">sección 5</a>.</p>
</div>
</li>

</ol>

<h3>Si algo sale mal en la instalación</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Lo que ves</th><th>Qué pasa</th><th>Qué hacer</th></tr></thead>
  <tbody>
    <tr><td><code>Python no encontrado</code></td>
        <td>Python no esta, o se instalo sin marcar <i>Add to PATH</i></td>
        <td>Reinstala Python 3.11.9 marcando la casilla <b>Add python.exe to PATH</b></td></tr>
    <tr><td>La ventana negra se cierra sola al instante</td>
        <td>Lo ejecutaste desde dentro del ZIP</td>
        <td>Descomprime de verdad y vuelve a intentarlo</td></tr>
    <tr><td><code>No module named tkinter</code></td>
        <td>Python se instalo sin tcl/tk</td>
        <td>Reinstala Python marcando <b>tcl/tk and IDLE</b></td></tr>
    <tr><td>Se queda parado en <i>Downloading torch</i></td>
        <td>Son 2,8 GB; puede tardar varios minutos</td>
        <td>Esperar. Solo cancela si no avanza en 15 minutos</td></tr>
    <tr><td><code>CUDA not available</code></td>
        <td>No es un error: no hay GPU NVIDIA utilizable</td>
        <td>Funciona igual con CPU, más lento. Se puede ignorar</td></tr>
    <tr><td>Windows avisa de un archivo peligroso</td>
        <td>SmartScreen desconfia de los <code>.bat</code> descargados</td>
        <td><i>Más información</i> → <i>Ejecutar de todas formas</i></td></tr>
    <tr><td>Errores de permisos al escribir</td>
        <td>La carpeta elegida es de solo lectura o está sincronizando</td>
        <td>Instala en <code>C:\PolyX\</code> o en Documentos</td></tr>
  </tbody>
</table>
</div>
"""
},

# ── 4 ───────────────────────────────────────────────────────────────
{
"id": "instalar-macos",
"titulo": "Instalación en macOS, paso a paso",
"sub": "Los mismos pasos, con una diferencia importante: el aviso de seguridad de Apple.",
"html": r"""
<div class="aviso vio">
  <span class="et">Sobre las imágenes de esta sección</span>
  <p>Las páginas web son fotografías reales, iguales para los dos sistemas. Los
  <b>diálogos propios de macOS van como dibujo</b>, marcados como <i>Esquema</i>,
  porque este manual se preparo en un equipo Windows. Reproducen fielmente lo que
  vas a ver, pero no son capturas de pantalla y no pretenden serlo.</p>
</div>

<ol class="pasos">

<li><span class="t">Comprobar Python</span>
<p>macOS trae Python, pero conviene confirmar la versión. Abre <b>Terminal</b>
(Aplicaciones → Utilidades) y escribe:</p>
<pre><code><span class="p">python3 --version</span>
<span class="o">Python 3.11.9</span></code></pre>
<p>Con <b>3.9 o superior</b> es suficiente. Si no lo tienes, descarga el
<b>macOS 64-bit universal2 installer</b> de la misma página de python.org que
aparece en la <a href="#instalar-windows">sección 3</a>.</p>
</li>

<li><span class="t">Descargar Poly-X desde GitHub</span>
<p>Identico a Windows: abre
<a href="https://github.com/CrissFerrada/Poly-X-Microplastics">el repositorio</a>,
botón verde <b>Code</b> → <b>Download ZIP</b>. Safari descomprime el ZIP solo al
terminar la descarga; si no lo hace, doble clic sobre el archivo.</p>
</li>

<li><span class="t">Abrir Lanzar_macOS.command con clic derecho</span>
<p>En macOS <b>no hay un instalador aparte</b>. Un solo archivo,
<code>Lanzar_macOS.command</code>, instala la primera vez y arranca el programa
todas las siguientes.</p>

<div class="aviso vio">
  <span class="et">Por qué un solo archivo y no dos</span>
  <p>Cada <code>.command</code> descargado necesita su propia aprobación de
  seguridad la primera vez que se abre. Con instalador y lanzador separados
  habria que pasar dos veces por ese aviso, y la segunda —con el programa ya
  instalado— es justo cuando más parece que algo se rompio.</p>
</div>

<p><b>Clic derecho</b> sobre <code>Lanzar_macOS.command</code> → <b>Abrir</b> →
confirmar <b>Abrir</b> en el diálogo.</p>

[[esquema:mac_abrir|El menú contextual de Finder. La primera vez hay que abrirlo con clic
<b>derecho</b> y elegir <b>Abrir</b>; el doble clic normal no funciona todavia.]]
</li>

<li><span class="t">Pasar el aviso de «desarrollador no identificado»</span>
<p>Si haces doble clic normal, macOS muestra este aviso:</p>

[[esquema:gatekeeper|El aviso de Gatekeeper. <b>No está roto:</b> macOS bloquea por defecto
cualquier script descargado que no venga firmado con una cuenta de desarrollador de Apple.
Cancela y abrelo con clic derecho.]]

<div class="aviso ok">
  <span class="et">Es lo esperable, no un fallo</span>
  <p>macOS bloquea todo script descargado que no este firmado con una cuenta de
  desarrollador de Apple, que cuesta 99 USD al año. <b>Clic derecho → Abrir →
  Abrir</b> resuelve el aviso, y solo hay qué hacerlo <b>la primera vez</b>:
  después el doble clic normal funciona.</p>
  <p>Si aun así se resiste, en Terminal:</p>
  <pre><code><span class="p">xattr -d com.apple.quarantine Lanzar_macOS.command</span></code></pre>
</div>
</li>

<li><span class="t">Esperar la instalación y arrancar</span>
<p>El mismo archivo instala durante <b>10–15 minutos</b> y después abre el
programa. Detecta solo el tipo de procesador:</p>

<div class="tabla-env">
<table>
  <thead><tr><th>Mac</th><th>Aceleración</th><th>Rendimiento</th></tr></thead>
  <tbody>
    <tr><td><b>Apple Silicon</b> (M1/M2/M3/M4)</td><td>GPU integrada por <b>MPS</b></td>
        <td>Rápido</td></tr>
    <tr><td><b>Intel</b></td><td>Solo CPU</td>
        <td>Cerca de <b>1 minuto por foto</b> en lotes con troceo</td></tr>
  </tbody>
</table>
</div>

<div class="aviso warn">
  <span class="et">Mac con procesador Intel</span>
  <p>PyTorch dejo de publicar versiones para Intel a partir de la 2.3, así que el
  instalador fija la <b>2.2.2</b>, la última con soporte x86_64. Funciona
  correctamente, pero sin aceleración por GPU.</p>
</div>
</li>
</ol>

<h3>Problemas propios de macOS</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Lo que ves</th><th>Qué hacer</th></tr></thead>
  <tbody>
    <tr><td><i>No se puede abrir porque es de un desarrollador no identificado</i></td>
        <td>Clic <b>derecho</b> sobre el archivo → <b>Abrir</b>. Solo la primera vez</td></tr>
    <tr><td><code>bad interpreter: /bin/bash^M</code></td>
        <td>El archivo llego con finales de línea de Windows. <b>Descargalo otra vez
            desde GitHub</b>; no lo copies por correo ni WhatsApp</td></tr>
    <tr><td>Doble clic y no pasa nada</td>
        <td>Falta el permiso de ejecución: <code>chmod +x Lanzar_macOS.command</code></td></tr>
  </tbody>
</table>
</div>

<div class="aviso">
  <span class="et">Empaquetar como aplicación</span>
  <p>Opcionalmente, <code>construir_app_macOS.command</code> genera un
  <code>Poly-X.app</code> normal, que se puede arrastrar al Dock. Detalle completo
  en <code>LEEME_macOS.md</code>.</p>
</div>
"""
},

# ── 5 ───────────────────────────────────────────────────────────────
{
"id": "modelo-pt",
"titulo": "El modelo entrenado: lo único que no viene en la descarga",
"sub": "Sin un archivo .pt el Detector y el Visor no tienen con qué detectar.",
"html": r"""
<div class="aviso err">
  <span class="et">Importante</span>
  <p>Los modelos entrenados (<code>*.pt</code>) <b>no se incluyen en la descarga
  de GitHub</b>, por su tamaño. Un modelo mediano pesa decenas de megabytes y
  GitHub no es el sitio para distribuirlo. El programa se instala igual, pero
  el Detector y el Visor no podran analizar nada hasta que copies uno.</p>
</div>

<h3>Qué hacer</h3>
<p>Copia tu archivo <code>.pt</code> dentro de la carpeta <code>models\</code> de
la instalación. El programa la crea sola durante la instalación, aunque quede
vacía:</p>

<pre><code>Poly-X-Microplastics-main\
├── polyx\
├── <span class="p">models\</span>              <span class="c">← el archivo .pt va aquí</span>
│   └── bestdetectormedium.pt
├── runs\
└── SETUP.bat</code></pre>

<p>No hay que renombrarlo ni configurar nada: el Detector lista todos los
<code>.pt</code> que encuentre ahí.</p>

<h3>De dónde sale un modelo</h3>
<div class="dl">
<dl>
  <dt>Te lo pasa quien ya lo entreno</dt>
  <dd>Es lo normal en un laboratorio: un modelo se entrena una vez y lo usa todo
      el grupo. Un <code>.pt</code> se copia por pendrive o por disco compartido
      como cualquier archivo.</dd>
  <dt>Lo entrenas tu</dt>
  <dd>Con el <a href="#entrenador">Entrenador</a>, a partir de imágenes anotadas
      con el <a href="#etiquetador">Etiquetador</a>. Es el camino largo, y es el
      que hay que recorrer si tus fotografías vienen de otro microscopio, otra
      tinción u otra iluminación.</dd>
</dl>
</div>

<div class="aviso warn">
  <span class="et">Un modelo no es universal</span>
  <p>Un modelo aprende de las fotografías con que se entreno: su microscopio, su
  iluminación, su tinción y su rango de tallas. Aplicado a fotografías tomadas de
  otro modo puede fallar sin avisar —detectando de menos, que es el error que no
  se nota— y por eso conviene verificar con conteo manual antes de confiar en
  las cifras. La <a href="#flujos">sección 14</a> propone como.</p>
</div>
"""
},

# ── 6 ───────────────────────────────────────────────────────────────
{
"id": "launcher",
"titulo": "El Launcher: la pantalla de entrada",
"sub": "Desde aquí se abre cada módulo, y se cambia el idioma.",
"html": r"""
[[fig:ui/01_launcher|El Launcher. Cada módulo se abre en su propia ventana, independiente de
las demás.]]

<p>Cada módulo arranca como un <b>proceso aparte</b>. Eso significa que puedes
tener el Etiquetador anotando y el Detector analizando a la vez, y que si uno
falla no se lleva a los otros por delante.</p>

<h3>El selector de idioma</h3>
<p>Arriba a la derecha. Cambia entre <b>español</b> e <b>inglés</b> y la elección
se recuerda entre sesiones; la primera vez toma el idioma del sistema.</p>

<div class="aviso">
  <span class="et">Cuando surte efecto</span>
  <p>Como los módulos son procesos separados, leen el idioma <b>al abrirse</b>.
  El cambio se nota en cuanto abras el siguiente módulo, no en las ventanas que
  ya están abiertas.</p>
  <p>El <b>informe de detección también sale en el idioma elegido</b>: títulos,
  tablas, pies de figura, ejes de los gráficos y la prosa de métodos. Incluso el
  atributo <code>lang</code> del HTML se ajusta, para que el corrector del
  navegador y los lectores de pantalla lo traten bien.</p>
  <p>Para forzarlo sin tocar la interfaz, la variable de entorno
  <code>POLYX_IDIOMA=en</code>.</p>
</div>

<h3>El aviso de versión nueva</h3>
<p>Al abrir el Launcher se comprueba en segundo plano si GitHub va por delante.
Si lo esta, aparece un botón con el identificador de la versión nueva. La
comprobación <b>no retrasa el arranque</b> y se calla ante cualquier fallo: sin
internet, o con GitHub caido, simplemente no aparece el aviso.</p>
"""
},

# ── 7 ───────────────────────────────────────────────────────────────
{
"id": "detector",
"titulo": "Módulo Detector",
"sub": "Análisis en lote de una carpeta entera, y el informe final. Nueve pantallas.",
"anclas": [("d-modelos", "Modelos"), ("d-imagenes", "Imágenes"),
           ("d-gt", "GT manual"), ("d-parametros", "Parámetros"),
           ("d-ejecutar", "Ejecutar"), ("d-resultados", "Resultados"),
           ("d-errores", "Errores"), ("d-comparar", "Comparar"),
           ("d-reporte", "Reporte")],
"html": r"""
<p>Es el módulo principal: recibe una carpeta de fotografías y un modelo, y
devuelve el conteo, las tallas y un informe. Las nueve pantallas de la izquierda
se recorren de arriba abajo, en ese orden.</p>

<h3 id="d-modelos">Modelos</h3>
[[fig:ui/02_det_modelos|Pestaña <b>Modelos</b>. Tres ranuras, para comparar hasta tres modelos
sobre exactamente las mismas fotografías.]]
<p>Se cargan hasta <b>tres modelos <code>.pt</code> a la vez</b>. No es un
capricho: comparar dos arquitecturas sobre el mismo lote, con los mismos
parámetros, es lo único que permite atribuir una diferencia de métricas al modelo
y no a las condiciones. También acepta <b>arrastrar y soltar</b> el archivo
directamente sobre la ventana.</p>

<h3 id="d-imagenes">Imágenes</h3>
[[fig:ui/03_det_imagenes|Pestaña <b>Imágenes</b>. La columna GT indica, para cada fotografía,
si existe una anotación manual con la que contrastar.]]
<p>Se elige una carpeta y se leen todas las imágenes, incluidas las de las
subcarpetas. La columna <b>GT</b> avisa de cuáles tienen <i>ground truth</i>: un
archivo <code>.txt</code> con el conteo manual. Sin GT el programa detecta igual,
pero no puede calcular aciertos ni errores, porque no hay contra que compararse.</p>

<h3 id="d-gt">GT manual</h3>
[[fig:ui/04_det_gt_manual|Pestaña <b>GT manual</b>. Un anotador completo dentro del propio
Detector, para marcar la verdad de referencia sin cambiar de módulo.]]
<p>Anotador de cajas integrado: dibujar arrastrando, seleccionar, mover,
redimensionar por las esquinas, zoom con la rueda y desplazamiento con el botón
central. Guarda <code>.txt</code> en formato YOLO junto a la imagen.</p>
<p>Existe para no tener que salir del Detector cuando faltan unas pocas
anotaciones. Para una campaña de cientos de imágenes, el
<a href="#etiquetador">Etiquetador</a> es la herramienta adecuada, porque lleva
la cuenta del avance entre sesiones.</p>

<h3 id="d-parametros">Parámetros</h3>
[[fig:ui/05_det_parametros|Pestaña <b>Parámetros</b>. Confianza, IoU, escala en micrómetros por
píxel y resolución de inferencia.]]
<div class="tabla-env">
<table>
  <thead><tr><th>Parámetro</th><th>Qué controla</th><th>Valor de partida</th></tr></thead>
  <tbody>
    <tr><td><b>Confianza</b></td><td>Umbral por debajo del cual una detección se descarta</td>
        <td>0,25 — bajar a 0,10 si no detecta nada</td></tr>
    <tr><td><b>IoU (NMS)</b></td><td>Cuanto se pueden solapar dos cajas antes de fusionarse</td>
        <td>0,45</td></tr>
    <tr><td><b>IoU (aciertos)</b></td><td>Solape mínimo para contar una detección como acierto</td>
        <td>0,50</td></tr>
    <tr><td><b>μm/píxel</b></td><td>La escala. Convierte píxeles en micrómetros</td>
        <td>Automática contra la placa</td></tr>
    <tr><td><b>Resolución</b></td><td>Tamaño al que la red ve la imagen</td>
        <td>Rápido 1280 · Equilibrado 2560 · Máxima 4096</td></tr>
  </tbody>
</table>
</div>

<div class="aviso ok">
  <span class="et">Calibración automática contra la placa Petri</span>
  <p>El borde de la placa se localiza solo y su diámetro conocido fija los
  <b>μm/px de cada fotografía</b>, sin marcar nada a mano. Importa porque la
  distancia de disparo varia entre tomas: en el material del estudio la escala
  real va de <b>31 a 50 μm/px</b>, un factor 1,6. Un valor único para todo el
  lote daría tallas con hasta un <b>50 % de error</b>.</p>
</div>

<p>Hay además un botón que <b>mide cuanta resolución aguanta tu GPU</b> antes de
quedarse sin memoria, para no descubrirlo a mitad de un lote.</p>

<div class="aviso vio">
  <span class="et">Hasta dónde se sostiene el polímero</span>
  <p>Detectar una partícula y decir de qué polímero es son dos preguntas
  distintas, y la segunda es mucho más frágil. El Nile Red es
  <b>solvatocrómico</b>: su emisión responde a la <b>polaridad</b> del entorno,
  no a la identidad química. Por eso el PET —poliéster, polar— se separa limpio,
  mientras que el PP y el LDPE, las dos poliolefinas apolares, comparten tono y
  solo difieren en <i>brillo</i>; y el brillo depende de la exposición, del foco,
  del espesor de la partícula y de cuánto tiñó el colorante.</p>
  <p>El campo <b>Confianza mínima para asignar</b> permite decirlo en el informe
  en vez de esconderlo: por debajo de ese valor la partícula <b>sigue contando</b>
  como partícula detectada, pero se reporta como <b>«no asignable»</b> en lugar de
  atribuirle un polímero. En 0 queda desactivado. El informe declara en Métodos
  qué porcentaje del lote quedó sin asignar, que es exactamente el dato que un
  revisor va a pedir.</p>
  <p>Para confirmar el polímero de verdad hace falta espectroscopía —FTIR o
  Raman—. Ninguna tinción fluorescente sustituye eso.</p>
</div>

<h3 id="d-ejecutar">Ejecutar</h3>
[[fig:ui/06_det_ejecutar|Pestaña <b>Ejecutar</b>. Antes de empezar avisa de si el lote se va a
trocear y de cuánto va a tardar.]]

<div class="aviso vio">
  <span class="et">El troceado de fotografías grandes</span>
  <p>Por encima de cierto umbral, la fotografía se analiza en <b>recortes
  solapados</b> en vez de entera. La razón es concreta: a resolución completa las
  partículas caen por debajo del <i>stride</i> de la red y desaparecen —una
  partícula de 12 px en una foto de 4096 px se reduce a 2 px cuando la red la
  ve a 640—. Las cajas se devuelven después a coordenadas de la foto original y
  los solapes se fusionan con NMS, de modo que <b>los resultados y el informe
  hablan siempre de la fotografía completa, nunca de los recortes</b>.</p>
</div>

<h3 id="d-resultados">Resultados</h3>
[[fig:ui/07_det_resultados|Pestaña <b>Resultados</b>. Métricas globales y la tabla de partículas,
con su talla ya en micrómetros.]]
<p>Métricas globales (aciertos, falsos positivos, falsos negativos, F1, precisión
y recall) y la tabla completa: una fila por partícula, con clase, confianza,
largo, ancho, área, relación de aspecto y si es <b>fibra</b> o <b>fragmento</b>.
Incluye <b>histograma de tallas</b> por clase y por tramos, apilado por polímero,
y <b>exportación a CSV</b> directa.</p>

<div class="aviso warn">
  <span class="et">La talla se mide sobre la partícula, no sobre su caja</span>
  <p>La caja de una partícula alargada está casi vacía y depende de como haya
  caido: una fibra tumbada en diagonal tiene caja <b>cuadrada</b>. Medido sobre
  7.129 partículas anotadas, la caja <b>sobreestima el área 1,87×</b>. Por eso se
  segmenta dentro de la caja y se mide sobre la máscara. El detalle está en la
  <a href="#medida">sección 11</a>.</p>
</div>

<h3 id="d-errores">Errores</h3>
[[fig:ui/08_det_errores|Pestaña <b>Errores</b>. Matriz de confusión y galería de los fallos, con
filtro por tipo.]]
<p>Lista cada caja equivocada y la clasifica en <b>falso positivo</b> (detecto
algo que no estaba), <b>falso negativo</b> (se le paso una partícula real) o
<b>mal clasificada</b> (la encontro pero le puso la clase equivocada). Con la
matriz de confusión al lado, se ve de un vistazo si el problema es de detección o
de clase — que se corrigen de maneras distintas.</p>

<h3 id="d-comparar">Comparar</h3>
[[fig:ui/09_det_comparar|Pestaña <b>Comparar</b>. Los modelos cargados, enfrentados sobre el
mismo lote.]]
<p>Tabla comparativa de los modelos cargados, fotografía por fotografía, con las
detecciones de cada uno y —si hay ground truth— sus aciertos, errores y el F1
global.</p>

<h3 id="d-reporte">Reporte</h3>
[[fig:ui/10_det_reporte|Pestaña <b>Reporte</b>. Trece secciones elegibles y tres presets. Es
donde el análisis se convierte en documento.]]
<p>Es la salida del módulo y merece sección propia: está detallada en la
<a href="#informe">sección 13</a>.</p>
"""
},

# ── 8 ───────────────────────────────────────────────────────────────
{
"id": "entrenador",
"titulo": "Módulo Entrenador",
"sub": "Entrenar un modelo YOLO propio, con curvas en vivo. Nueve pantallas.",
"html": r"""
<p>Se usa cuando el modelo disponible no sirve para tus fotografías: otro
microscopio, otra tinción, otro rango de tallas. Necesita un dataset ya anotado
—el que produce el <a href="#etiquetador">Etiquetador</a>— y bastante paciencia.</p>

<h3>Modelo</h3>
[[fig:ui/11_ent_modelo|Pestaña <b>Modelo</b>. Familia YOLO v8 o v11, tamaño de nano a xlarge,
o unos pesos propios de partida.]]
<p>Soporta <b>YOLO v8 y v11</b>, en todos los tamaños. Hay una casilla para
<b>entrenar v8 y v11 con la misma configuración</b>: ambas corridas reutilizan
idénticos <code>imgsz</code>, <code>batch</code>, épocas, semilla y augmentación.
Es lo que permite atribuir la diferencia de métricas a la arquitectura y no a los
hiperparametros. Van en secuencia porque comparten GPU, y al terminar el registro
resume la comparación.</p>

<h3>Dataset</h3>
[[fig:ui/12_ent_dataset|Pestaña <b>Dataset</b>. Carga del <code>data.yaml</code> con
auto-detección de particiones y vista previa de las cajas.]]
<p>Carga el <code>data.yaml</code>, detecta las particiones train/val/test y
valida el dataset con indicadores <b>✓/✗</b>: imágenes presentes, etiquetas
correspondientes, clases coherentes. Incluye una <b>auditoría previa</b> con la
distribución de clases y un veredicto de si el dataset es apto para entrenar.</p>

<h3>Parámetros</h3>
[[fig:ui/13_ent_parametros|Pestaña <b>Parámetros</b>. Épocas, batch, resolución y optimizador,
con recomendaciones automaticas.]]

<h3>Augmentación</h3>
[[fig:ui/14_ent_augmentacion|Pestaña <b>Augmentación</b>. Volteos, rotación, mosaico, HSV,
mixup y copy-paste.]]

<h3>Entrenar</h3>
[[fig:ui/15_ent_entrenar|Pestaña <b>Entrenar</b>. Curvas en vivo de mAP, pérdida, precisión y
recall mientras corre.]]
<p>Al arrancar declara explícitamente si está usando GPU o CPU, para que nadie
descubra tres horas después que estaba entrenando en procesador.</p>

<div class="aviso ok">
  <span class="et"><code>best_real.pt</code>: el peso que sirve para tus fotos</span>
  <p>Ultralytics guarda <code>best.pt</code> según el mAP de la validación
  completa, y en un dataset mixto ese número lo dominan las placas dopadas de
  laboratorio: en el dataset del Loa había <b>1191 cajas de laboratorio frente a
  47 de sedimento real</b>. El modelo que gana ese promedio no es
  necesariamente el que mejor funciona sobre sedimento.</p>
  <p>Al terminar, Poly-X evalua todos los puntos de control contra <b>solo</b> el
  sedimento real y guarda el ganador aparte, como <code>best_real.pt</code>.
  <code>best.pt</code> se conserva intacto.</p>
</div>

<h3>Evaluar</h3>
[[fig:ui/16_ent_evaluar|Pestaña <b>Evaluar</b>. Valida cualquier <code>.pt</code> contra un
dataset YOLO, con métricas por clase.]]

<h3>Comparar</h3>
[[fig:ui/17_ent_comparar|Pestaña <b>Comparar</b>. Todas las corridas anteriores con sus
métricas, para no perder el hilo entre experimentos.]]

<h3>Exportar</h3>
[[fig:ui/18_ent_exportar|Pestaña <b>Exportar</b>. Convierte el modelo a ONNX, TensorRT o CoreML.]]

<h3>Informe</h3>
[[fig:ui/19_ent_informe|Pestaña <b>Informe</b>. Genera el HTML del entrenamiento, con curvas y
configuración completa.]]

<div class="aviso warn">
  <span class="et">Si se queda sin memoria</span>
  <p>El error <code>CUDA out of memory</code> es el más común al entrenar. Baja el
  <b>batch</b> a 4 o 2, y la resolución a 640. Es preferible a bajar el tamaño del
  modelo, que cuesta más métricas.</p>
</div>
"""
},

# ── 9 ───────────────────────────────────────────────────────────────
{
"id": "etiquetador",
"titulo": "Módulo Etiquetador",
"sub": "Anotar cientos de imágenes sin perder la cuenta entre sesiones.",
"html": r"""
[[fig:ui/20_etiquetador|El Etiquetador. La lista de la izquierda distingue lo revisado de lo
pendiente, y ese estado se recupera del disco al reabrir la carpeta.]]

<p>Anotador de cajas en formato YOLO. Se dibuja arrastrando, se asigna clase con
clic derecho o con las teclas <kbd>1</kbd>–<kbd>9</kbd>, y hay
<b>deshacer/rehacer</b> por imagen.</p>

<h3>El seguimiento del avance</h3>
<p>Está pensado para campañas de cientos de imágenes repartidas en varias
sesiones. La lista distingue tres casos:</p>

<div class="tabla-env">
<table>
  <thead><tr><th>Marca</th><th>Significado</th></tr></thead>
  <tbody>
    <tr><td><code>✓ nombre (n)</code></td><td>revisada, con n partículas</td></tr>
    <tr><td><code>· nombre (0)</code></td><td>revisada, sin partículas — <b>es un dato</b></td></tr>
    <tr><td><code>○ nombre</code></td><td>todavia sin revisar</td></tr>
  </tbody>
</table>
</div>

<div class="aviso vio">
  <span class="et">Por qué pasar de largo no crea archivo</span>
  <p>Un <code>.txt</code> solo se escribe al <b>marcar la imagen como revisada</b>
  o al <b>dibujar una caja</b>. Registrar como «revisada con cero» una imagen que
  apenas se ojeó falsearía un conteo censal: el cero de una placa mirada y el
  cero de una placa saltada valen cosas distintas, y el archivo no puede
  distinguirlos después.</p>
</div>

<h3>Pre-anotación automática</h3>
<p>Se puede cargar un modelo <code>.pt</code> y dejar que proponga las cajas, con
<code>conf</code> e <code>imgsz</code> ajustables y GPU si está disponible.
Después se corrigen a mano. Sobre el material del estudio ahorra alrededor del
<b>80 % del tiempo</b> frente a anotar desde cero.</p>

<h3>Atajos</h3>
<div class="atajos">
<dl>
  <dt><kbd>Espacio</kbd></dt><dd>marcar revisada y avanzar</dd>
  <dt><kbd>Tab</kbd></dt><dd>saltar a la siguiente sin revisar</dd>
  <dt><kbd>←</kbd> <kbd>→</kbd></dt><dd>navegar entre imágenes</dd>
  <dt><kbd>1</kbd>–<kbd>9</kbd></dt><dd>cambiar la clase activa</dd>
</dl>
<dl>
  <dt><kbd>F</kbd></dt><dd>reencuadrar la imagen</dd>
  <dt><kbd>Supr</kbd></dt><dd>borrar la caja seleccionada</dd>
  <dt><kbd>Ctrl</kbd>+<kbd>Z</kbd></dt><dd>deshacer</dd>
  <dt><kbd>Ctrl</kbd>+<kbd>Y</kbd></dt><dd>rehacer</dd>
</dl>
</div>

<h3>Detalles que evitan perder trabajo</h3>
<ul>
  <li><b>Auto-guardado silencioso</b> cada 60 segundos.</li>
  <li><b>El zoom se conserva</b> entre imágenes, con una casilla en el panel
      derecho. Con tolerancia de tamaño, porque los recortes de una rejilla
      difieren en 1 px por redondeo y exigir igualdad exacta hacia perder el zoom
      en cada cambio.</li>
  <li><b>Lado mínimo de caja: 2 px.</b> Estaba en 5 px y descartaba
      <b>en silencio</b> marcas legítimas: las partículas más pequeñas del estudio
      miden unos 8 px de lado. Ahora, si una caja se rechaza, se avisa en la barra
      de estado.</li>
  <li>Guarda el <code>.txt</code> junto a la imagen —o en <code>labels/</code> si
      la imagen está en <code>images/</code>— y genera <code>classes.txt</code>.</li>
  <li>Las miniaturas se generan en segundo plano, para que la ventana responda de
      inmediato.</li>
</ul>
"""
},

# ── 10 ──────────────────────────────────────────────────────────────
{
"id": "visor",
"titulo": "Módulo Visor",
"sub": "Una imagen a la vez: calibrar la escala y verificar cada medida.",
"html": r"""
[[fig:ui/21_visor|El Visor. La tabla lista cada partícula, y al seleccionar una fila se ve
<b>sobre que se midió</b>: el recorte limpio y, al lado, la máscara con la medida dibujada.]]

<h3>Calibración interactiva μm/píxel</h3>
<p>Dos modos, y en ambos el programa pide después la medida real:</p>
<div class="dl">
<dl>
  <dt>📏 Línea — dos clics</dt>
  <dd>Se marca una referencia de longitud conocida y se escribe cuanto mide.</dd>
  <dt>⭕ Círculo — tres clics</dt>
  <dd>Se marcan tres puntos del borde y se escribe el diámetro real. Es el modo
      útil con placas Petri, cuyo diámetro se conoce; el diálogo viene con
      100.000 μm por defecto, que es la placa del estudio.</dd>
</dl>
</div>
<p>La barra inferior muestra la escala en tiempo real, del estilo
<code>📐 0.4880 μm/px (línea)</code>.</p>

<h3>Revisión partícula a partícula</h3>
<p>Es la función que hace del Visor una herramienta de verificación y no solo de
visualización. La tabla lista cada partícula con su número, clase, tipo, largo,
ancho y aspecto. Al seleccionar una fila se ve <b>sobre que se midió</b>:</p>
<ul>
  <li>a la izquierda, el recorte <b>sin marcas</b>;</li>
  <li>a la derecha, el contorno de la máscara con la medida dibujada encima:
      <span style="color:#9a6700;font-weight:650">en amarillo la recta de Feret</span>,
      <span style="color:#6639ba;font-weight:650">en magenta el camino geodésico</span>,
      <span style="color:#1f6b5e;font-weight:650">en verde la máscara</span>;</li>
  <li>y la cuenta completa de píxeles a micrómetros.</li>
</ul>

<div class="aviso ok">
  <span class="et">El criterio</span>
  <p>Una talla que no se puede ver medida no se puede verificar. Por eso el Visor
  no muestra solo el número: muestra la partícula, la máscara de la que salió y
  la recta que se midió sobre ella.</p>
</div>

<h3>Lo demás que hace</h3>
<ul>
  <li><b>Detectar</b> con un modelo cargado, con resolución configurable de 320 a
      8192 y GPU si está disponible. Con objetos diminutos en fotografías grandes
      el <code>imgsz</code> es determinante: a valores bajos las partículas caen
      por debajo del <i>stride</i> de la red y no se detecta nada.</li>
  <li><b>Cargar etiquetas <code>.txt</code></b> ya existentes y verlas sobre la
      imagen con las tallas convertidas a μm. Sirve para revisar un conteo manual
      sin volver al Etiquetador.</li>
  <li><b>Cargar las predicciones de una corrida ya cerrada</b> desde
      <code>runs/detect_.../</code>, sin volver a pasar el modelo. Abre siempre la
      fotografía original y nunca el PNG anotado, que lleva las cajas pintadas
      encima.</li>
  <li><b>Arrastrar y soltar</b> una imagen o un modelo directamente sobre el
      lienzo.</li>
  <li><b>Exportar</b> la imagen anotada, <code>detecciones.csv</code> y
      <code>resumen.json</code>.</li>
</ul>
"""
},

# ── 11 ──────────────────────────────────────────────────────────────
{
"id": "medida",
"titulo": "Cómo se mide la talla de una partícula",
"sub": "El criterio, sus excepciones, su exactitud y sus límites declarados.",
"html": r"""
<p>El criterio general es <b>la línea recta más larga que cabe en la
partícula</b>: la mayor distancia entre dos puntos de su contorno, o
<i>diámetro de Feret máximo</i>. No depende de la orientación con que la
partícula haya caido, y un borde dentado no la altera.</p>

<p>Esa recta deja de servir cuando la partícula esta <b>contorsionada</b>: en una
fibra doblada, la distancia entre extremos es la cuerda, y en un arco de media
circunferencia se queda un <b>35 % corta</b>. Para esos casos se mide el
<i>diámetro geodésico</i>: el camino más largo que cabe <b>dentro</b> de la
partícula, que al no poder salirse de la máscara rodea la curva.</p>

<div class="tabla-env">
<table>
  <thead><tr><th>Forma de la partícula</th><th>Qué se reporta como largo</th></tr></thead>
  <tbody>
    <tr><td>Compacta o irregular, pero no doblada</td>
        <td><b>Feret máximo</b> — la recta más larga</td></tr>
    <tr><td>Alargada y contorsionada (fibra)</td>
        <td><b>Diámetro geodésico</b> — sigue la curva</td></tr>
  </tbody>
</table>
</div>

<p>El geodésico solo se aplica si la partícula es <b>delgada</b> (largo ≥ 4 ×
grosor) y <b>no convexa</b> (solidez &lt; 0,90). Cada condición viene de un fallo
observado: sin la primera, cualquier concavidad hace que el camino rodee la
partícula en vez de atravesarla —un grumo real de 44 px recibía 73—; sin la
segunda, el largo pasaria a depender del ángulo de giro, que es justo el defecto
que se quería eliminar.</p>

<div class="aviso ok">
  <span class="et">Exactitud</span>
  <p>Contra formas sintéticas de talla conocida —rectas, rectas giradas, arcos de
  60°, 120° y 180°, un círculo, una recta de borde dentado y un grumo con
  muesca— el largo así medido da <b>0,6 % de error mediano y 4,7 % en el peor
  caso</b>. Está fijado en <code>tests/test_morfologia.py</code>, de modo que un
  cambio bienintencionado que lo empeore hace fallar la suite.</p>
</div>

<div class="aviso warn">
  <span class="et">El rectángulo equivalente no es una talla</span>
  <p>La fórmula <i>L</i> = (<i>P</i> + √(<i>P</i>²−16<i>A</i>))/4 da el largo de un
  rectángulo con la misma área y el mismo perímetro, que es otra cosa. Depende del
  perímetro, así que un borde dentado la infla un <b>22,5 %</b>, y no esta
  definida para partículas compactas, en las que <i>P</i>² &lt; 16<i>A</i>. Se
  reporta como descriptor, porque comparada con las otras dos delata bordes
  irregulares, pero <b>no se usa como talla</b>.</p>
  <p>Una corrección de agosto de 2026 nació justo de ahí: se había reportado
  «22 % de fibras y aspecto hasta 21,1», que era un artefacto de usar esta
  fórmula como largo. Medido con Feret y geodésico sobre 6.638 partículas, el
  aspecto mediano es <b>1,58</b>, el máximo <b>8,7</b>, y las fibras son el
  <b>1,1 %</b>. El material es mayoritariamente fragmentos compactos.</p>
</div>

<h3>De píxeles a micrómetros: la escala</h3>
<p>Todo lo anterior se mide en píxeles. La conversión <b>no es un factor único
para el lote</b>: depende de la distancia de disparo, y en el material de este
estudio la escala real va de <b>31 a 50 μm/px</b>, un factor 1,6.</p>

<p>Por eso cada fotografía se calibra con la suya, contra el anillo de la placa
Petri: se localiza el centro aproximado con Hough, se muestrea el borde en
<b>720 direcciones</b> y se ajusta una circunferencia por mínimos cuadrados con
rechazo de atípicos. El radio de Hough <b>no se usa</b>, porque llega a errar un
12 % y ese error entraría entero en todos los tamaños.</p>

<div class="aviso vio">
  <span class="et">Qué borde son los 100 mm, y hacia donde puede fallar</span>
  <p>El anillo tiene una pared de unos <b>2 mm</b> —medido sobre las fotografías
  del estudio: el borde interno cae en 0,960 del radio ajustado y el externo en
  1,000—. El diámetro nominal de una placa Petri es ambiguo a ese nivel: puede
  referirse al externo o al útil interior. Aquí se toma el <b>externo</b>, que es
  el borde al que ajusta el círculo. Si el nominal se refiriera al interior, la
  escala correcta sería un <b>4,2 % mayor</b> y todas las tallas estarian
  <b>subestimadas</b> en esa cifra. El sesgo solo puede ir en ese sentido, porque
  el externo es el mayor de los dos bordes posibles. Queda declarado en el propio
  informe.</p>
</div>

<h3>Partículas que se tocan</h3>
<p>Dos partículas en contacto forman una sola mancha, y medirlas juntas sumaría
sus tallas. Se separan por <i>watershed</i> sobre la transformada de distancia:
el centro de cada una queda lejos del fondo y el cuello que las une queda cerca,
de modo que el corte cae por el cuello. Sobre círculos de talla conocida las
separa hasta un <b>27 % de solapamiento del diámetro</b>, sin partir ninguna
partícula de una sola pieza.</p>

<h3>La fibra se mide entera</h3>
<p>La máscara <b>no</b> se recorta a la caja del detector cuando la partícula es
alargada, ni se pasa por el separador de partículas pegadas: las dos cosas
cortaban fibras reales. En la que lo destapó se perdía un <b>53 % del largo</b>
—369 px de componente conexa quedaban en 174—, y una talla subestimada no se nota
en las cifras, solo en la imagen. Está fijado en
<code>tests/test_fibra_no_se_trunca.py</code>.</p>

<div class="aviso err">
  <span class="et">Limitaciones declaradas</span>
  <p>Dos partículas solapadas <b>más alla de un 40 % de su diámetro</b> se siguen
  midiendo como una sola: a esa altura ya no hay un cuello por el que cortar.</p>
  <p>En una <b>fibra muy enroscada</b> el camino geodésico ataja por el interior
  de cada codo, subestimando hasta un <b>19 %</b> en el caso más cerrado
  ensayado.</p>
</div>
"""
},

# ── 12 ──────────────────────────────────────────────────────────────
{
"id": "polimeros",
"titulo": "Los tres polímeros, y por qué dos se confunden",
"sub": "Lo que hay que saber para interpretar el recall por clase.",
"html": r"""
<div class="tabla-env">
<table>
  <thead><tr><th>ID</th><th>Clase</th><th>Fluorescencia observada</th><th>Color de la caja en pantalla</th></tr></thead>
  <tbody>
    <tr><td>0</td><td><span class="tag pet">PET</span></td><td>Rojo–salmon</td><td>🔴 rojo</td></tr>
    <tr><td>1</td><td><span class="tag pp">PP</span></td><td>Amarillo <b>verdoso, apagado</b></td><td>🟠 naranjo</td></tr>
    <tr><td>2</td><td><span class="tag ldpe">LDPE</span></td><td>Amarillo franco, <b>más brillante</b></td><td>🟡 amarillo</td></tr>
  </tbody>
</table>
</div>

<p>Los colores de la derecha son solo los de las cajas en la interfaz:
<b>no describen la emisión real</b>. Medido sobre el interior de las cajas del
dataset de entrenamiento (media RGB, n = 30 por clase):</p>

<div class="tabla-env">
<table>
  <thead><tr><th>Clase</th><th class="num-col">R</th><th class="num-col">G</th><th class="num-col">B</th></tr></thead>
  <tbody>
    <tr><td><b>PET</b></td><td class="num-col">116</td><td class="num-col">58</td><td class="num-col">65</td></tr>
    <tr><td><b>PP</b></td><td class="num-col">122</td><td class="num-col"><b>125</b></td><td class="num-col">32</td></tr>
    <tr><td><b>LDPE</b></td><td class="num-col">181</td><td class="num-col">162</td><td class="num-col">57</td></tr>
  </tbody>
</table>
</div>

<div class="aviso warn">
  <span class="et">PP y LDPE se separan por brillo, no por tono</span>
  <p>Los dos son amarillentos, pero en PP el verde <b>iguala o supera</b> al rojo
  (122 frente a 125) y la emisión es bastante más apagada; en LDPE el rojo sube a
  181. Es la confusión más habitual al anotar, y la razón de que el recall por
  clase se hunda ahí: <b>PET 0,98 · PP 0,70 · LDPE 0,54</b>.</p>
  <p>Si tu conteo separa PP de LDPE, esas dos cifras son el límite de lo que el
  modelo puede sostener hoy. El total de partículas es mucho más fiable que el
  reparto entre esas dos clases.</p>
</div>

<div class="aviso">
  <span class="et">Nota histórica</span>
  <p>La documentación antigua describía el PP como «naranjo», que no corresponde a
  la emisión observada. Se corrigió en agosto de 2026 a partir de la medición RGB
  y de la observación directa.</p>
</div>
"""
},

# ── 13 ──────────────────────────────────────────────────────────────
{
"id": "informe",
"titulo": "El informe de detección",
"sub": "Un HTML autocontenido con calidad de publicación, y su PDF.",
"html": r"""
[[fig:ui/10_det_reporte|La pestaña <b>Reporte</b>. Trece casillas de sección y tres presets:
Completo, Resumen breve y Metodológico.]]

<p>El informe es la salida real del trabajo: un <b>HTML autocontenido</b> con
todas las imágenes incrustadas en base64, que se puede enviar por correo sin que
se rompa nada, y que se exporta a <b>PDF con un clic</b>.</p>

<h3>Qué puede incluir</h3>
<div class="dl">
<dl>
  <dt>Galería comparativa Predicción vs Ground Truth</dt>
  <dd>Lado a lado. Las imágenes de galería se recodifican y se limita su número
      para que el archivo siga siendo abrible en un navegador; <b>las métricas,
      en cambio, cubren todas las imágenes</b>.</dd>

  <dt>Sección de calibración</dt>
  <dd>De donde salió la escala de cada fotografía, su mínimo, mediana y máximo, la
      <b>media con intervalo de confianza al 95 %</b>, y una figura sobre una placa
      real con el círculo ajustado y su diámetro dibujados encima.</dd>

  <dt>Talla por carpeta y por foto</dt>
  <dd>Compara la distribución de tallas entre carpetas —cada carpeta como sitio de
      muestreo, estación o condición— con diagramas de caja y una prueba de
      <b>Kruskal-Wallis</b>.</dd>

  <dt>Perfil en profundidad del testigo</dt>
  <dd>Cuando las fotografías se llaman <code>tramo.testigo</code>, el tramo deja de
      ser una carpeta cualquiera y pasa a ser una variable <b>ordenada</b>. El
      informe dibuja el perfil con la profundidad en el eje vertical —la convención
      de cualquier testigo de sedimento— y responde si el número de partículas y la
      talla mediana crecen o decrecen con la profundidad, con <b>Spearman</b> y un
      valor de <i>p</i> por <b>permutación</b>.</dd>

  <dt>Ficha de partículas medidas</dt>
  <dd>Las <b>6 fibras y las 6 partículas mayores</b>, cada una con su recorte al
      lado y la medida dibujada encima. El reparto es deliberado: las fibras son
      minoría y son justo donde actúa el método geodésico.</dd>

  <dt>Comparación entre modelos</dt>
  <dd>Tabla foto por foto con las detecciones de cada modelo y, si hay ground
      truth, sus aciertos, errores y el F1 global.</dd>

  <dt>Histograma de tallas</dt>
  <dd>Por clase y por tramos, apilado por polímero.</dd>
</dl>
</div>

<div class="aviso vio">
  <span class="et">Por qué la correlación va sobre los tramos y no sobre las partículas</span>
  <p>Dos partículas de la misma placa <b>no son observaciones independientes</b> de
  la profundidad: comparten placa, toma y condiciones. Correlacionar partícula a
  partícula inflaría el tamaño muestral con repeticiones de la misma medida y
  produciría valores de <i>p</i> que no significan nada. La correlación se calcula
  <b>sobre los tramos</b>.</p>
</div>

<h3>Secciones elegibles</h3>
<p>Trece casillas y tres presets: <b>Completo</b>, <b>Resumen breve</b> y
<b>Metodológico</b>. Al desmarcar una sección, las demás se <b>renumeran solas</b>
y el índice se ajusta; una sección marcada pero sin datos se omite igualmente, en
vez de aparecer vacía.</p>

<h3>Alcance elegible</h3>
<p>El informe puede cubrir el trabajo completo, <b>solo las fotografías que
marques</b>, o ambos de una vez. Las cifras, los gráficos y la matriz de confusión
se <b>recalculan sobre lo elegido</b>, de modo que el informe siempre describe las
fotografías que muestra.</p>

<div class="aviso ok">
  <span class="et">Y sale en tu idioma</span>
  <p>Títulos, tablas, pies de figura, ejes de los gráficos y la prosa de métodos
  siguen el idioma elegido en el Launcher — no solo la interfaz.</p>
</div>
"""
},

# ── 14 ──────────────────────────────────────────────────────────────
{
"id": "flujos",
"titulo": "Flujos de trabajo recomendados",
"sub": "Tres escenarios, según de dónde partas.",
"html": r"""
<h3>A · Ya tienes un modelo y solo quieres analizar fotografías</h3>
<p>Es el caso más frecuente y el más corto.</p>
<ol>
  <li>Copia el <code>.pt</code> en <code>models\</code>.</li>
  <li>Abre el <b>Detector</b> desde el Launcher.</li>
  <li><b>Modelos</b> → carga el modelo.</li>
  <li><b>Imágenes</b> → elige la carpeta.</li>
  <li><b>Parámetros</b> → deja la calibración automática si tus fotos incluyen la
      placa completa; si no, fija los μm/px a mano.</li>
  <li><b>Ejecutar</b> → ▶ iniciar. Lee el aviso de troceado antes de empezar.</li>
  <li><b>Reporte</b> → generar HTML, y exportar a PDF si hace falta enviarlo.</li>
</ol>

<h3>B · Quieres entrenar un modelo para tus propias fotografías</h3>
<ol>
  <li><b>Etiquetador</b> → anota. Empieza por la estación más densa: es la que
      sostiene el resultado principal y la que más enseña al modelo.</li>
  <li>Recuenta un <b>~10 % al azar</b> para estimar tu error intra-observador. Los
      revisores lo piden, y sin esa cifra no hay con que comparar el error del
      modelo.</li>
  <li><b>Entrenador</b> → <b>Dataset</b>: pasa la auditoría antes de entrenar.</li>
  <li><b>Entrenar</b>. Si comparas arquitecturas, usa la casilla de <b>v8 y v11 con
      la misma configuración</b>.</li>
  <li>Quédate con <code>best_real.pt</code> si tu dataset mezcla laboratorio y
      material real.</li>
  <li><b>Detector</b> → analiza y compara contra tu conteo manual.</li>
</ol>

<div class="aviso warn">
  <span class="et">Una sola semilla no permite afirmar que una arquitectura gana</span>
  <p>Dos corridas con semillas distintas del mismo modelo pueden diferir más que
  dos arquitecturas entre si. Si vas a sostener «v11 supera a v8» en un
  documento, repite con <b>varias semillas</b> y reporta la dispersión.</p>
</div>

<h3>C · Quieres verificar una medida concreta</h3>
<ol>
  <li><b>Visor</b> → abre la fotografía.</li>
  <li><b>Cargar predicciones</b> de la corrida ya cerrada, desde
      <code>runs/detect_.../</code>. No hace falta volver a pasar el modelo.</li>
  <li>Selecciona la fila de la partícula en la tabla.</li>
  <li>Mira el recorte y la máscara con la medida dibujada encima: ahí se ve si la
      talla salió de una segmentación correcta o de una máscara rota.</li>
</ol>

<div class="aviso">
  <span class="et">Lo que el programa no puede darte</span>
  <p>Poly-X cuenta y mide partículas. Para reportar <b>partículas por kilogramo</b>
  —la unidad comparable con la literatura— hace falta la <b>masa seca de cada
  tramo</b>, que se mide en el laboratorio y no sale de las fotografías. El
  programa no la puede inventar.</p>
</div>
"""
},

# ── 15 ──────────────────────────────────────────────────────────────
{
"id": "atajos",
"titulo": "Atajos de teclado",
"sub": "Los del Etiquetador y el GT manual son los que más tiempo ahorran.",
"html": r"""
<h3>Etiquetador y GT manual</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Tecla</th><th>Qué hace</th></tr></thead>
  <tbody>
    <tr><td><kbd>Espacio</kbd></td><td>marcar la imagen como revisada y avanzar</td></tr>
    <tr><td><kbd>Tab</kbd></td><td>saltar a la siguiente imagen sin revisar</td></tr>
    <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>imagen anterior / siguiente (autoguarda)</td></tr>
    <tr><td><kbd>1</kbd> … <kbd>9</kbd></td><td>cambiar la clase activa, o la de la caja seleccionada</td></tr>
    <tr><td><kbd>F</kbd></td><td>reencuadrar la imagen a la ventana</td></tr>
    <tr><td><kbd>Supr</kbd></td><td>borrar la caja seleccionada</td></tr>
    <tr><td><kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd></td><td>deshacer / rehacer</td></tr>
    <tr><td>Rueda del ratón</td><td>zoom</td></tr>
    <tr><td>Botón central, o <kbd>Espacio</kbd>+arrastrar</td><td>desplazar la imagen</td></tr>
    <tr><td>Clic izquierdo + arrastrar</td><td>dibujar una caja</td></tr>
    <tr><td>Clic derecho sobre una caja</td><td>asignarle clase</td></tr>
  </tbody>
</table>
</div>

<h3>Visor</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Acción</th><th>Qué hace</th></tr></thead>
  <tbody>
    <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>navegar por la carpeta</td></tr>
    <tr><td>Rueda del ratón</td><td>zoom</td></tr>
    <tr><td>Arrastrar y soltar</td><td>abrir una imagen o cargar un modelo <code>.pt</code></td></tr>
    <tr><td>2 clics en modo línea</td><td>calibrar por longitud conocida</td></tr>
    <tr><td>3 clics en modo círculo</td><td>calibrar por diámetro conocido</td></tr>
  </tbody>
</table>
</div>
"""
},

# ── 16 ──────────────────────────────────────────────────────────────
{
"id": "actualizar",
"titulo": "Actualizar y desinstalar",
"sub": "Traer lo nuevo sin reinstalar, y retirar el programa sin dejar restos.",
"html": r"""
<h3>Actualizar</h3>
<p>Doble clic en el actualizador que corresponda:</p>
<div class="tabla-env">
<table>
  <thead><tr><th>Sistema</th><th>Archivo</th></tr></thead>
  <tbody>
    <tr><td><span class="tag win">Windows</span></td><td><code>actualizar.bat</code></td></tr>
    <tr><td><span class="tag mac">macOS</span></td><td><code>actualizar_macOS.command</code></td></tr>
  </tbody>
</table>
</div>

<p>Comprueba si hay un commit nuevo en <code>main</code> y, si lo hay, descarga y
reemplaza <b>solo los archivos del programa</b>. <b>Conserva</b> tu entorno
<code>.venv</code>, tus modelos <code>models/*.pt</code>, tus <code>runs/</code> y
cualquier dato local. No necesita tener Git instalado —descarga por HTTPS—, solo
conexión a internet.</p>

<div class="aviso">
  <span class="et">Da igual desde qué sistema actualices</span>
  <p>Cada actualizador se protege a si mismo mientras corre, pero <b>si actualiza
  los archivos de la otra plataforma</b>. El proyecto queda completo para ambas
  vengas de donde vengas.</p>
</div>

<p>No hace falta acordarse de ejecutarlo: el <a href="#launcher">Launcher</a> avisa
solo cuando GitHub va por delante.</p>

<h3>Desinstalar</h3>
<p>En Windows, <code>DESINSTALAR.bat</code>. Retira el entorno y los archivos del
programa. Revisa antes que no queden dentro de la carpeta modelos o resultados que
quieras conservar: <code>models\</code> y <code>runs\</code> son tuyos, no del
programa.</p>

<div class="aviso warn">
  <span class="et">Antes de desinstalar</span>
  <p>Saca de la carpeta tus <code>.pt</code> y tus <code>runs/</code>. Un modelo
  entrenado puede representar días de trabajo y no se recupera de GitHub, porque
  nunca estuvo ahí.</p>
</div>
"""
},

# ── 17 ──────────────────────────────────────────────────────────────
{
"id": "problemas",
"titulo": "Solución de problemas",
"sub": "Lo que suele fallar, por qué, y qué hacer.",
"html": r"""
<h3>Al instalar</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Síntoma</th><th>Causa</th><th>Solución</th></tr></thead>
  <tbody>
    <tr><td><code>Python no encontrado</code></td><td>Falta Python, o se instalo sin PATH</td>
        <td>Reinstalar Python 3.11.9 marcando <b>Add python.exe to PATH</b></td></tr>
    <tr><td><code>No module named tkinter</code></td><td>Python sin tcl/tk</td>
        <td>Reinstalar marcando <b>tcl/tk and IDLE</b></td></tr>
    <tr><td>La consola se cierra sola</td><td>Se ejecuto desde dentro del ZIP</td>
        <td>Descomprimir de verdad y reintentar</td></tr>
    <tr><td><code>bad interpreter: /bin/bash^M</code> <span class="tag mac">macOS</span></td>
        <td>El archivo llego con finales de línea de Windows</td>
        <td>Descargar otra vez desde GitHub, sin pasarlo por correo o WhatsApp</td></tr>
    <tr><td><i>Desarrollador no identificado</i> <span class="tag mac">macOS</span></td>
        <td>Gatekeeper bloquea scripts sin firmar</td>
        <td>Clic <b>derecho</b> → <b>Abrir</b>, solo la primera vez</td></tr>
  </tbody>
</table>
</div>

<h3>Al detectar</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Síntoma</th><th>Causa</th><th>Solución</th></tr></thead>
  <tbody>
    <tr><td><b>No detecta nada</b></td><td>Confianza demasiado alta, o resolución demasiado baja</td>
        <td>Bajar confianza a 0,10. Y subir la resolución: con partículas de ~12 px
            en fotos de 4096, a 640 colapsan a ~2 px y desaparecen</td></tr>
    <tr><td>Las tallas salen todas mal</td><td>Escala μm/px incorrecta</td>
        <td>Revisar la sección de calibración del informe, o calibrar a mano en el Visor</td></tr>
    <tr><td>Confunde PP con LDPE</td><td>Es el límite conocido del modelo</td>
        <td>No tiene arreglo por parámetros. Ver la <a href="#polimeros">sección 12</a></td></tr>
    <tr><td>Una fibra aparece partida en dos</td><td>Segmentación</td>
        <td>Verificarla en el <a href="#visor">Visor</a>, que muestra la máscara real</td></tr>
    <tr><td><code>CUDA not available</code></td><td>No hay GPU NVIDIA utilizable</td>
        <td>No es un error: funciona en CPU, más lento</td></tr>
  </tbody>
</table>
</div>

<h3>Al entrenar</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Síntoma</th><th>Solución</th></tr></thead>
  <tbody>
    <tr><td><code>CUDA out of memory</code></td>
        <td>Bajar <b>batch</b> a 4 o 2, y la resolución a 640</td></tr>
    <tr><td><i>no kernel image is available for execution on the device</i></td>
        <td>PyTorch no trae kernels para tu tarjeta. Reejecutar <code>SETUP.bat</code>,
            que comprueba esto explícitamente</td></tr>
    <tr><td>El modelo va bien en validación y mal en tus fotos</td>
        <td>Dataset mixto: usar <code>best_real.pt</code></td></tr>
  </tbody>
</table>
</div>

<div class="aviso">
  <span class="et">Si necesitas ver el error completo</span>
  <p>Arranca con <code>iniciar_polyx.bat</code> en vez del acceso directo: muestra
  la consola, y ahí aparece el mensaje entero en vez de una ventana que se cierra.</p>
</div>
"""
},

# ── 18 ──────────────────────────────────────────────────────────────
{
"id": "estructura",
"titulo": "Estructura de carpetas",
"sub": "Qué es tuyo, qué es del programa, y qué se puede borrar.",
"html": r"""
<pre><code>Poly-X-Microplastics-main\
├── <span class="p">src\polyx\</span>               <span class="c">código fuente — del programa</span>
│   ├── launcher.py
│   ├── core\                 <span class="c">núcleo compartido (medida, calibración, informe, idioma)</span>
│   ├── gui\                  <span class="c">interfaz: detector\, trainer\, labeler\, viewer\…</span>
│   └── cli\                  <span class="c">comandos sin ventana (polyx-train, polyx-eval…)</span>
├── <span class="p">models\</span>                   <span class="c">TUYO — los pesos .pt entrenados</span>
├── <span class="p">runs\</span>                     <span class="c">TUYO — una carpeta por corrida, con fecha y hora</span>
├── <span class="p">data_microplastico\</span>       <span class="c">TUYO — dataset YOLO (images/ + labels/)</span>
├── <span class="p">.venv\</span>                    <span class="c">entorno de Python — se regenera con SETUP.bat</span>
├── docs\                     <span class="c">documentación</span>
│   └── manual\               <span class="c">Manual_PolyX.html — este manual</span>
├── tests\                    <span class="c">suite de pruebas de medida y calibración</span>
│
├── SETUP.bat                 <span class="c">[Win] instalador</span>
├── iniciar_polyx.bat         <span class="c">[Win] lanzador con consola</span>
├── Poly-X.vbs                <span class="c">[Win] lanzador sin consola</span>
├── actualizar.bat            <span class="c">[Win] actualizador</span>
├── Lanzar_macOS.command      <span class="c">[Mac] instalador + lanzador</span>
├── actualizar_macOS.command  <span class="c">[Mac] actualizador</span>
└── LEEME.txt                 <span class="c">versión corta de la instalación</span></code></pre>

<div class="aviso err">
  <span class="et">Las tres carpetas que no se borran</span>
  <p><code>models\</code>, <code>runs\</code> y <code>data_microplastico\</code>
  contienen <b>tu trabajo</b>, no el programa. Un modelo entrenado puede ser días
  de cómputo y ninguna de las tres se recupera de GitHub, porque nunca estuvieron
  ahí. <code>.venv\</code>, en cambio, se regenera cuando quieras volviendo a
  ejecutar <code>SETUP.bat</code>.</p>
</div>

<h3>Qué guarda cada corrida</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Archivo</th><th>Contenido</th></tr></thead>
  <tbody>
    <tr><td><code>images/*.png</code></td><td>las fotografías con las cajas dibujadas</td></tr>
    <tr><td><code>centroids.csv</code></td><td>una fila por partícula: clase, posición, confianza, talla</td></tr>
    <tr><td><code>metrics.json</code></td><td>aciertos, falsos positivos y falsos negativos por clase</td></tr>
    <tr><td><code>report.html</code></td><td>el informe autocontenido</td></tr>
    <tr><td><code>annotations/</code></td><td>los <code>.txt</code> YOLO, si había ground truth</td></tr>
  </tbody>
</table>
</div>

<div class="aviso">
  <span class="et">Las diferencias entre sistemas están en un solo archivo</span>
  <p>Abrir carpetas, lanzar el actualizador, elegir el dispositivo de cómputo:
  todo lo que difiere entre Windows y macOS vive en
  <code>polyx/core/plataforma.py</code>, y no repartido por el código. Es la razón
  de que el programa funcione igual en ambos sin dos versiones que mantener.</p>
</div>
"""
},

# ── 19 ──────────────────────────────────────────────────────────────
{
"id": "referencias",
"titulo": "Referencias y contacto",
"sub": "El método publicado sobre el que se apoya Poly-X.",
"html": r"""
<h3>Publicaciones</h3>
<ul>
  <li><b>Perez M, Parra S, Ferrada C, Bravo M, Perez PA, Quiroz W (2024).</b>
      Development of a new methodology for the determination of PET microplastics
      in sediment, based on microwave-assisted acid digestion.
      <i>PLoS ONE</i> <b>19</b>(12): e0314520.
      <a href="https://doi.org/10.1371/journal.pone.0314520">doi.org/10.1371/journal.pone.0314520</a></li>

  <li><b>Ferrada C, Perez M, Parra S, Salas E, Sepulveda F, Bravo MA, Quiroz W (2024).</b>
      Evaluation of microwave-assisted acid/oxidant digestion method for the
      detection of polyethylene microplastics in <i>Merluccius gayi</i> fish by
      Nile Red fluorescent staining and image analysis.
      <i>J. Chil. Chem. Soc.</i> <b>69</b>(1): 6082–6085.
      <a href="https://doi.org/10.4067/s0717-97072024000106082">doi.org/10.4067/s0717-97072024000106082</a></li>
</ul>

<h3>Alcance del repositorio</h3>
<p>El repositorio documenta <b>el programa</b>. Todo lo que forma parte de un
artículo en preparación —el pipeline de análisis del estudio, sus fotografías y
sus hallazgos— queda deliberadamente fuera, porque publicarlo adelantaría
resultados que aún no han salido.</p>

<h3>Pruebas</h3>
<p>La medida de forma y la calibración tienen suite propia, porque cada cifra que
producen acaba en una tabla y un cambio bienintencionado puede desplazarlas todas
sin que nada avise:</p>
<pre><code><span class="p">.venv\Scripts\python.exe -m pytest tests/ -q</span></code></pre>
<p>Son <b>43 pruebas sobre formas sintéticas de talla conocida</b>, sin depender
de ninguna anotación humana. Cada una fija además el <i>porque</i> de una decisión
de diseño, de modo que si alguien vuelve a intentar una variante ya descartada, la
suite se lo dice. <code>pytest</code> solo hace falta para desarrollar y no esta
en <code>requirements.txt</code>: una instalación de uso no lo necesita.</p>

<h3>Contacto</h3>
<p><b>Cristofher Ferrada</b><br>
Laboratorio de Química Ambiental · Pontificia Universidad Católica de Valparaíso<br>
<a href="mailto:cristofher.ferrada@pucv.cl">cristofher.ferrada@pucv.cl</a><br>
<a href="https://github.com/CrissFerrada/Poly-X-Microplastics">github.com/CrissFerrada/Poly-X-Microplastics</a></p>
"""
},

]
