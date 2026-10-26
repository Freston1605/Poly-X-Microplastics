"""English text of the Poly-X manual.

Content only: the engine that turns it into HTML lives in generar_manual.py.
Section ids match the Spanish file so both manuals share anchors and figures.
"""

TITULO_DOC = "Poly-X Manual — Microplastics detection suite"
DESCRIPCION = (
    "Complete Poly-X manual: step-by-step installation from GitHub on "
    "Windows and macOS, and a guide to all four modules."
)
SELLO = "User manual · v2.0.0"
TITULO_H1 = "Poly-X <em>analytics</em>"
BAJADA = (
    "Detection, sizing and classification of microplastics by Nile Red "
    "fluorescence under UV light, with YOLO v8/v11 models. From the GitHub "
    "download to a publication-ready report."
)
META = (
    "<b>Cristofher Ferrada</b> · PhD (c) in Chemistry<br>"
    "Environmental Chemistry Laboratory · Pontificia Universidad Católica de Valparaíso<br>"
    "Windows 10/11 and macOS · Python 3.11 · 2026"
)

ETIQUETA_INDICE = "Contents"
ETIQUETA_FIGURA = "Figure"
PALABRA_ESQUEMA = "Diagram"

PIE = (
    "<b>Poly-X v2.0.0</b> — Cristofher Ferrada, 2026. "
    "Repository: <a href='https://github.com/CrissFerrada/Poly-X-Microplastics'>"
    "github.com/CrissFerrada/Poly-X-Microplastics</a> · "
    "Contact: <a href='mailto:cristofher.ferrada@pucv.cl'>cristofher.ferrada@pucv.cl</a><br>"
    "Interface screenshots are regenerated against the installed program, so this "
    "manual shows the version in front of you and not an earlier one."
)


# ════════════════════════════════════════════════════════════════════
#  Diagrams: the dialogs that cannot honestly be photographed
# ════════════════════════════════════════════════════════════════════
_SVG_PY = """
<svg viewBox="0 0 760 470" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Opening screen of the Python installer for Windows">
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
        font-weight="700" fill="#9a6700">tick</text>
  <text x="716" y="308" font-family="Segoe UI,sans-serif" font-size="12"
        font-weight="700" fill="#9a6700">both</text>

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
     aria-label="macOS warning: unidentified developer">
  <rect width="760" height="400" fill="#f6f8fa"/>
  <rect x="215" y="40" width="330" height="320" rx="13" fill="#fff" stroke="#d0d7de"/>
  <circle cx="380" cy="104" r="34" fill="#fef3c7" stroke="#9a6700" stroke-width="2"/>
  <text x="380" y="118" text-anchor="middle" font-size="36" fill="#9a6700">⚠</text>
  <text x="380" y="176" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="15" font-weight="700" fill="#1f2328">“Lanzar_macOS.command”</text>
  <text x="380" y="197" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="15" font-weight="700" fill="#1f2328">cannot be opened because it is</text>
  <text x="380" y="222" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="12.5" fill="#424a53">from an unidentified developer.</text>
  <line x1="215" y1="268" x2="545" y2="268" stroke="#eaeef2"/>
  <rect x="243" y="288" width="126" height="34" rx="7" fill="#fff" stroke="#d0d7de"/>
  <text x="306" y="310" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="13" fill="#424a53">Move to Trash</text>
  <rect x="391" y="288" width="126" height="34" rx="7" fill="#0969da"/>
  <text x="454" y="310" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="13" font-weight="600" fill="#fff">Cancel</text>
  <text x="380" y="345" text-anchor="middle" font-family="-apple-system,Segoe UI,sans-serif"
        font-size="11.5" fill="#cf222e" font-weight="600">Neither: cancel, then open it with a right-click</text>
</svg>
"""

_SVG_MAC_ABRIR = """
<svg viewBox="0 0 760 400" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Finder context menu with the Open option">
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

  <rect x="286" y="150" width="238" height="196" rx="9" fill="#fff" stroke="#d0d7de"/>
  <rect x="292" y="176" width="226" height="28" rx="5" fill="#0969da"/>
  <text x="308" y="171" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Open With</text>
  <text x="308" y="195" font-family="-apple-system,Segoe UI,sans-serif" font-size="13.5"
        font-weight="700" fill="#fff">Open</text>
  <text x="308" y="226" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Move to Trash</text>
  <line x1="292" y1="240" x2="518" y2="240" stroke="#eaeef2"/>
  <text x="308" y="262" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Get Info</text>
  <text x="308" y="288" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Rename</text>
  <text x="308" y="314" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Compress</text>
  <text x="308" y="338" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" fill="#424a53">Duplicate</text>

  <path d="M560 190 L536 190" stroke="#1f6b5e" stroke-width="2.4" marker-end="url(#fl2)"/>
  <defs><marker id="fl2" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
    <path d="M0 0 L9 4.5 L0 9 z" fill="#1f6b5e"/></marker></defs>
  <text x="566" y="186" font-family="-apple-system,Segoe UI,sans-serif" font-size="12.5"
        font-weight="700" fill="#1f6b5e">RIGHT-click,</text>
  <text x="566" y="203" font-family="-apple-system,Segoe UI,sans-serif" font-size="12.5"
        font-weight="700" fill="#1f6b5e">then Open</text>
</svg>
"""

ESQUEMAS = {
    "python_installer": _SVG_PY,
    "gatekeeper": _SVG_GATEKEEPER,
    "mac_abrir": _SVG_MAC_ABRIR,
}


# ════════════════════════════════════════════════════════════════════
SECCIONES = [
    # ── 1 ───────────────────────────────────────────────────────────────
    {
        "id": "que-es",
        "titulo": "What Poly-X is",
        "sub": "The problem it solves, what it does and what it does not.",
        "html": r"""
<p>Poly-X is a desktop program that <b>counts, measures and classifies
microplastics</b> in microscope photographs. Particles are stained with
<b>Nile Red</b> and photographed under <b>254 nm UV light</b>: each polymer emits
with a distinct colour and brightness, and a <b>YOLO v8/v11</b> detection model
works on that emission, locating each particle, assigning it a class and
returning its size in micrometres.</p>

<p>It covers the whole cycle without leaving the program: you annotate images,
train a model on them, analyse a full batch and produce a report with the tables
and figures already assembled.</p>

<div class="modulos">
  <div class="mod"><span class="ic">🔬</span><b>Detector</b>
    <span>Analyses whole folders with a trained model and produces the report.</span></div>
  <div class="mod"><span class="ic">🎯</span><b>Trainer</b>
    <span>Trains YOLO v8 and v11 models with live curves and a dataset audit.</span></div>
  <div class="mod"><span class="ic">🏷</span><b>Labeller</b>
    <span>Annotates particles in YOLO format, with automatic pre-annotation.</span></div>
  <div class="mod"><span class="ic">📐</span><b>Viewer</b>
    <span>Inspects one image, calibrates micrometres per pixel and checks every measurement.</span></div>
</div>

<h3>The three polymers</h3>
<p>The model separates three classes, the ones used by the laboratory's published
method:</p>
<div class="tabla-env">
<table>
  <thead><tr><th>Class</th><th>Name</th><th>Emission observed under UV</th></tr></thead>
  <tbody>
    <tr><td><span class="tag pet">PET</span></td><td>Polyethylene terephthalate</td>
        <td>Red–salmon</td></tr>
    <tr><td><span class="tag pp">PP</span></td><td>Polypropylene</td>
        <td><b>Greenish, dull</b> yellow</td></tr>
    <tr><td><span class="tag ldpe">LDPE</span></td><td>Low-density polyethylene</td>
        <td>Plain yellow, <b>brighter</b></td></tr>
  </tbody>
</table>
</div>

<div class="aviso warn">
  <span class="et">What to know before trusting a class label</span>
  <p>PP and LDPE are <b>not separated by hue but by brightness</b>: both are
  yellowish. It is the most frequent confusion when annotating and the reason
  recall drops for those two classes. The figures are in
  <a href="#polimeros">section 12</a>.</p>
</div>

<h3>The complete workflow</h3>
<pre><code><span class="c">Microscope photographs (254 nm UV, Nile Red staining)</span>
        ↓
  🏷 <span class="p">Labeller</span>  → annotate PET / PP / LDPE in YOLO format
        ↓
  🎯 <span class="p">Trainer</span>   → train the model on those annotations
        ↓
  🔬 <span class="p">Detector</span>  → analyse the batch and produce the report
        ↓
  📐 <span class="p">Viewer</span>    → check particle by particle, and calibrate</code></pre>

<p>If you already have a trained model, the short path is <b>Detector only</b>:
the other three modules exist to build that model and to verify it.</p>

<div class="aviso">
  <span class="et">Scope of this manual</span>
  <p>Sections <b>2 to 6</b> are the installation: downloading from GitHub,
  installing and starting up for the first time, with a photograph of every step.
  Sections <b>7 to 15</b> walk through everything the program can do today,
  screen by screen. If you only came to install it, the first six are enough.</p>
</div>
""",
    },
    # ── 2 ───────────────────────────────────────────────────────────────
    {
        "id": "requisitos",
        "titulo": "Before you start",
        "sub": "What the machine needs, and how much space it will take.",
        "html": r"""
<div class="tabla-env">
<table>
  <thead><tr><th>Component</th><th>Windows</th><th>macOS</th></tr></thead>
  <tbody>
    <tr><td><b>System</b></td><td>Windows 10 or 11</td><td>11 Big Sur or later</td></tr>
    <tr><td><b>Python</b></td><td><b>3.11.x</b> — not 3.12 or later</td><td>3.9 or later</td></tr>
    <tr><td><b>RAM</b></td><td>8 GB minimum</td><td>8 GB minimum</td></tr>
    <tr><td><b>Disk</b></td><td colspan="2">About <b>6 GB</b> free: the environment with
        GPU-enabled PyTorch takes close to 5 GB</td></tr>
    <tr><td><b>Acceleration</b></td>
        <td>NVIDIA GPU optional. With one, training runs 20–30× faster; the
            installer picks the CUDA build that matches the card</td>
        <td><b>Apple Silicon:</b> integrated GPU via MPS.<br>
            <b>Intel:</b> CPU only, ~1 min per photo</td></tr>
    <tr><td><b>Internet</b></td><td colspan="2">Only during installation and updates.
        After that it works offline</td></tr>
  </tbody>
</table>
</div>

<div class="aviso err">
  <span class="et">Python 3.12 will not do</span>
  <p>On Windows you must install <b>Python 3.11.x</b>. Versions 3.12 and later have
  no wheels compatible with the PyTorch / NumPy 1.26 / Ultralytics 8.3 combination
  that Poly-X pins, and the installation fails halfway through with compilation
  errors that do not name the real cause.</p>
</div>

<h3>How long it takes</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Step</th><th>Typical time</th><th>How often</th></tr></thead>
  <tbody>
    <tr><td>Install Python 3.11</td><td>2–4 min</td><td>once per machine</td></tr>
    <tr><td>Download Poly-X from GitHub</td><td>under 1 min (≈ 5 MB)</td><td>once</td></tr>
    <tr><td>Run <code>SETUP.bat</code></td><td>5–15 min</td><td>once</td></tr>
    <tr><td>Start the program</td><td>2–4 s</td><td>every day</td></tr>
  </tbody>
</table>
</div>

<div class="aviso vio">
  <span class="et">One number that surprises people</span>
  <p>The GitHub download is <b>under 5 MB</b>, yet the finished installation takes
  several gigabytes. That is not a fault: the weight is <b>PyTorch with GPU
  support</b>, downloaded separately, around <b>2.8 GB</b> compressed. The program
  itself is small; the numerical library is not.</p>
</div>
""",
    },
    # ── 3 ───────────────────────────────────────────────────────────────
    {
        "id": "instalar-windows",
        "titulo": "Installing on Windows, step by step",
        "sub": "Five steps, with a photograph of each. Done once.",
        "anclas": [
            ("w-python", "Step 1 · Python 3.11"),
            ("w-descargar", "Step 2 · Download from GitHub"),
            ("w-extraer", "Step 3 · Unzip"),
            ("w-setup", "Step 4 · Run SETUP.bat"),
            ("w-arrancar", "Step 5 · First launch"),
        ],
        "html": r"""
<ol class="pasos">

<li id="w-python"><span class="t">Install Python 3.11.9</span>
<p>If the machine already has Python 3.11, skip to step 2. To check, open the
Start menu, type <code>cmd</code>, and in the black window type:</p>
<pre><code><span class="p">python --version</span>
<span class="o">Python 3.11.9</span>   <span class="c">← if you see this, you are set; go to step 2</span></code></pre>

<p>If you do not have it, go to
<a href="https://www.python.org/downloads/release/python-3119/">python.org/downloads/release/python-3119</a>.</p>

[[fig:web/w04_python_release|The 3.11.9 release page on python.org. Scroll to the bottom, where the
file table is.]]

<p>At the end of that page there is a table with every installer. The one you need
is <b>Windows installer (64-bit)</b>, marked <i>Recommended</i>.</p>

[[fig:web/w06_python_tabla_recorte|File table for release 3.11.9. On Windows download
<b>Windows installer (64-bit)</b>; on a Mac, <b>macOS 64-bit universal2 installer</b>.]]

<div class="aviso err">
  <span class="et">The checkbox that decides whether everything else works</span>
  <p>When the installer opens, <b>before</b> clicking <i>Install Now</i>, tick
  <b>Add python.exe to PATH</b> at the bottom. Without it, <code>SETUP.bat</code>
  cannot find Python and stops. This is by far the most common failure in this
  installation.</p>
</div>

[[esquema:python_installer|Opening screen of the Python installer on Windows. Tick both boxes at
the bottom and only then click <i>Install Now</i>. It is drawn rather than photographed because on
a machine that already has Python the installer shows the maintenance screen, not this one.]]

<p>Also tick <b>tcl/tk and IDLE</b> if the installer offers a component choice:
some of the program's dialogs use it.</p>
</li>

<li id="w-descargar"><span class="t">Download Poly-X from GitHub</span>
<p>Open <a href="https://github.com/CrissFerrada/Poly-X-Microplastics">github.com/CrissFerrada/Poly-X-Microplastics</a>.
No account or sign-in is needed: the repository is public.</p>

[[fig:web/w01_github_repo|The repository page. The green <b>Code</b> button sits at the top right
of the file listing.]]

<p>Click the green <b>Code</b> button and, in the menu that drops down, the last
entry: <b>Download ZIP</b>.</p>

[[fig:web/w03_github_download_zip|The <b>Code</b> menu open. The entry you want is
<b>Download ZIP</b>, at the bottom. The ones above are for people using Git or GitHub Desktop.]]

<div class="aviso">
  <span class="et">If you would rather use Git</span>
  <p>If you already have Git installed, this is equivalent and makes updating
  easier:</p>
  <pre><code><span class="p">git clone https://github.com/CrissFerrada/Poly-X-Microplastics.git</span></code></pre>
</div>
</li>

<li id="w-extraer"><span class="t">Unzip the archive</span>
<p>The file lands in your <b>Downloads</b> folder and weighs about <b>5 MB</b>.
Right-click it and choose <b>Extract all</b> (or select the file and use the
<b>Extract all</b> button in the Explorer toolbar).</p>

[[fig:win/p01_zip_descargado|The freshly downloaded ZIP, with the <b>Extract all</b> button visible
in the File Explorer toolbar. This machine runs Windows in Spanish, so the button reads
<i>Extraer todo</i>.]]

<div class="aviso warn">
  <span class="et">Do not run it from inside the ZIP</span>
  <p>Windows lets you open files inside a ZIP as if it were a folder, but it
  <b>extracts them to a temporary folder</b> that it later deletes. If you run
  <code>SETUP.bat</code> from there, the installation is lost. Extract for real
  first.</p>
</div>

<p>Choose a path <b>without accents or unusual characters</b> and with write
permission. <code>C:\PolyX\</code> or your Documents folder will do; a Desktop
synchronised with OneDrive can cause trouble if syncing locks files mid-install.</p>

<p>When it finishes you will have a folder called
<code>Poly-X-Microplastics-main</code> with everything inside.</p>

[[fig:win/p02_carpeta_extraida|The extracted folder. The file to run is <b>SETUP.bat</b>, selected
here near the bottom.]]
</li>

<li id="w-setup"><span class="t">Run SETUP.bat</span>
<p>Double-click <b>SETUP.bat</b>. A black window opens: that is normal, and it is
where everything happens. The first thing it does is ask <b>where to install</b>.</p>

[[fig:win/p03_setup_pregunta_carpeta|First screen of <code>SETUP.bat</code>. Press <kbd>Enter</kbd>
to install into the same folder you extracted, which is the recommended choice.]]

<p>Press <kbd>Enter</kbd> without typing anything and it installs right there. If
you prefer another location, paste the path and press <kbd>Enter</kbd>.</p>

<p>From then on the installer works on its own. It finds Python, creates the
<code>.venv</code> environment, checks which graphics card you have and downloads
<b>the PyTorch build that matches that card</b>.</p>

[[fig:win/p04_setup_python_gpu|The installer detecting the GPU and downloading PyTorch. Here it
recognised an NVIDIA card of <i>compute capability</i> 7.5 and chose CUDA 11.8: that is 2.8 GB, and
the part that takes longest.]]

<div class="aviso vio">
  <span class="et">Why it checks the card's generation and not just whether a GPU exists</span>
  <p>Knowing there is an NVIDIA card is not enough. <b>RTX 50xx</b> cards
  (Blackwell, <code>sm_120</code>) need CUDA 12.8: install the CUDA 11.8 wheels on
  one of those and <code>torch.cuda.is_available()</code> returns <code>True</code>,
  everything looks fine, and the failure only shows up at training time as
  <i>no kernel image is available for execution on the device</i> — a symptom that
  looks nothing like its cause. That is why the installer verifies, at the end,
  that PyTorch ships kernels for <b>your</b> specific card.</p>
</div>

[[fig:win/p05_setup_completado|The final check. A GPU being present is not enough: the installer
matches the card's architecture (<code>sm_75</code>) against the kernel list PyTorch ships. Below
it, the search for earlier installations — folders containing <code>.git</code> are always skipped.]]

<p>When it finishes it offers to create a Desktop shortcut and tells you how to
start the program.</p>

[[fig:win/p06_setup_final|Installation complete. It states where it installed, how to start it, and
what is missing: the <code>.pt</code> model, which does not come with the download.]]

<div class="aviso">
  <span class="et">If the machine already had an older Poly-X</span>
  <p>At the end, <code>SETUP.bat</code> looks for old installations. If it finds
  one, it shows what it holds — models, training runs, detections — and asks for
  confirmation before touching anything. The order is always
  <b>copy → verify → retire</b>: only if the copy succeeded does the old folder go
  to the <b>Recycle Bin</b>, never to a direct delete. Folders containing
  <code>.git</code> are skipped, because a development repository is not an old
  installation.</p>
</div>
</li>

<li id="w-arrancar"><span class="t">Start Poly-X</span>
<p>Double-click the <b>Poly-X</b> shortcut on the Desktop. If you chose not to
create one, the folder gives you three equivalent ways:</p>

<div class="tabla-env">
<table>
  <thead><tr><th>File</th><th>What it does</th><th>When to use it</th></tr></thead>
  <tbody>
    <tr><td><code>Poly-X.vbs</code></td><td>Opens the program <b>with no black window</b></td>
        <td>Normal, day-to-day use</td></tr>
    <tr><td><code>iniciar_polyx.bat</code></td><td>The same, but showing the console</td>
        <td>When something fails and you want to read the error</td></tr>
    <tr><td><code>.venv\Scripts\python.exe -m polyx.launcher</code></td>
        <td>Manual start from the terminal</td><td>For debugging</td></tr>
  </tbody>
</table>
</div>

[[fig:ui/01_launcher|The Launcher just opened. All four modules are reached from here, and the
selector at the top right switches the language of the whole program.]]

<div class="aviso ok">
  <span class="et">Installation finished</span>
  <p>If you see this screen, you are done. One thing remains before you can detect
  anything: copy a trained model into <code>models\</code>, because <b>models are
  not included in the download</b>. That is <a href="#modelo-pt">section 5</a>.</p>
</div>
</li>

</ol>

<h3>If something goes wrong during installation</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>What you see</th><th>What is happening</th><th>What to do</th></tr></thead>
  <tbody>
    <tr><td><code>Python no encontrado</code></td>
        <td>Python is missing, or was installed without <i>Add to PATH</i></td>
        <td>Reinstall Python 3.11.9 ticking <b>Add python.exe to PATH</b></td></tr>
    <tr><td>The black window closes instantly</td>
        <td>You ran it from inside the ZIP</td>
        <td>Extract properly and try again</td></tr>
    <tr><td><code>No module named tkinter</code></td>
        <td>Python was installed without tcl/tk</td>
        <td>Reinstall Python ticking <b>tcl/tk and IDLE</b></td></tr>
    <tr><td>It seems stuck at <i>Downloading torch</i></td>
        <td>It is 2.8 GB; it can take several minutes</td>
        <td>Wait. Only cancel if there is no progress after 15 minutes</td></tr>
    <tr><td><code>CUDA not available</code></td>
        <td>Not an error: no usable NVIDIA GPU</td>
        <td>It still works on CPU, more slowly. Safe to ignore</td></tr>
    <tr><td>Windows warns about a dangerous file</td>
        <td>SmartScreen distrusts downloaded <code>.bat</code> files</td>
        <td><i>More info</i> → <i>Run anyway</i></td></tr>
    <tr><td>Permission errors while writing</td>
        <td>The chosen folder is read-only or is syncing</td>
        <td>Install into <code>C:\PolyX\</code> or Documents</td></tr>
  </tbody>
</table>
</div>
""",
    },
    # ── 4 ───────────────────────────────────────────────────────────────
    {
        "id": "instalar-macos",
        "titulo": "Installing on macOS, step by step",
        "sub": "The same steps, with one important difference: Apple's security warning.",
        "html": r"""
<div class="aviso vio">
  <span class="et">About the images in this section</span>
  <p>The web pages are real photographs, identical on both systems. The
  <b>macOS-specific dialogs are drawn</b>, marked <i>Diagram</i>, because this
  manual was prepared on a Windows machine. They reproduce faithfully what you
  will see, but they are not screenshots and do not pretend to be.</p>
</div>

<ol class="pasos">

<li><span class="t">Check Python</span>
<p>macOS ships with Python, but it is worth confirming the version. Open
<b>Terminal</b> (Applications → Utilities) and type:</p>
<pre><code><span class="p">python3 --version</span>
<span class="o">Python 3.11.9</span></code></pre>
<p><b>3.9 or later</b> is enough. If you do not have it, download the
<b>macOS 64-bit universal2 installer</b> from the same python.org page shown in
<a href="#instalar-windows">section 3</a>.</p>
</li>

<li><span class="t">Download Poly-X from GitHub</span>
<p>Identical to Windows: open
<a href="https://github.com/CrissFerrada/Poly-X-Microplastics">the repository</a>,
green <b>Code</b> button → <b>Download ZIP</b>. Safari unzips the archive on its
own when the download finishes; if it does not, double-click the file.</p>
</li>

<li><span class="t">Open Lanzar_macOS.command with a right-click</span>
<p>On macOS there is <b>no separate installer</b>. A single file,
<code>Lanzar_macOS.command</code>, installs the first time and starts the program
every time after that.</p>

<div class="aviso vio">
  <span class="et">Why one file and not two</span>
  <p>Every downloaded <code>.command</code> needs its own security approval the
  first time it is opened. With a separate installer and launcher you would have
  to go through that warning twice, and the second time — with the program already
  installed — is exactly when it most looks like something broke.</p>
</div>

<p><b>Right-click</b> <code>Lanzar_macOS.command</code> → <b>Open</b> → confirm
<b>Open</b> in the dialog.</p>

[[esquema:mac_abrir|The Finder context menu. The first time you must open it with a
<b>right</b>-click and choose <b>Open</b>; an ordinary double-click will not work yet.]]
</li>

<li><span class="t">Get past the “unidentified developer” warning</span>
<p>If you double-click normally, macOS shows this warning:</p>

[[esquema:gatekeeper|The Gatekeeper warning. <b>Nothing is broken:</b> macOS blocks by default any
downloaded script not signed with an Apple developer account. Cancel, and open it with a
right-click.]]

<div class="aviso ok">
  <span class="et">Expected behaviour, not a fault</span>
  <p>macOS blocks every downloaded script that is not signed with an Apple
  developer account, which costs USD 99 a year. <b>Right-click → Open → Open</b>
  clears the warning, and you only need to do it <b>the first time</b>: after that
  an ordinary double-click works.</p>
  <p>If it still resists, in Terminal:</p>
  <pre><code><span class="p">xattr -d com.apple.quarantine Lanzar_macOS.command</span></code></pre>
</div>
</li>

<li><span class="t">Wait for the install, then it starts</span>
<p>That same file installs for <b>10–15 minutes</b> and then opens the program. It
detects the processor type on its own:</p>

<div class="tabla-env">
<table>
  <thead><tr><th>Mac</th><th>Acceleration</th><th>Performance</th></tr></thead>
  <tbody>
    <tr><td><b>Apple Silicon</b> (M1/M2/M3/M4)</td><td>Integrated GPU via <b>MPS</b></td>
        <td>Fast</td></tr>
    <tr><td><b>Intel</b></td><td>CPU only</td>
        <td>Around <b>1 minute per photo</b> on batches that get tiled</td></tr>
  </tbody>
</table>
</div>

<div class="aviso warn">
  <span class="et">Intel-based Macs</span>
  <p>PyTorch stopped publishing Intel builds from version 2.3 onwards, so the
  installer pins <b>2.2.2</b>, the last one with x86_64 support. It works
  correctly, but without GPU acceleration.</p>
</div>
</li>
</ol>

<h3>Problems specific to macOS</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>What you see</th><th>What to do</th></tr></thead>
  <tbody>
    <tr><td><i>cannot be opened because it is from an unidentified developer</i></td>
        <td><b>Right</b>-click the file → <b>Open</b>. Only the first time</td></tr>
    <tr><td><code>bad interpreter: /bin/bash^M</code></td>
        <td>The file arrived with Windows line endings. <b>Download it again from
            GitHub</b>; do not pass it around by email or WhatsApp</td></tr>
    <tr><td>Double-click does nothing</td>
        <td>The execute permission is missing: <code>chmod +x Lanzar_macOS.command</code></td></tr>
  </tbody>
</table>
</div>

<div class="aviso">
  <span class="et">Packaging it as an application</span>
  <p>Optionally, <code>construir_app_macOS.command</code> builds a normal
  <code>Poly-X.app</code> that can be dragged to the Dock. Full detail in
  <code>LEEME_macOS.md</code>.</p>
</div>
""",
    },
    # ── 5 ───────────────────────────────────────────────────────────────
    {
        "id": "modelo-pt",
        "titulo": "The trained model: the one thing not in the download",
        "sub": "Without a .pt file the Detector and the Viewer have nothing to detect with.",
        "html": r"""
<div class="aviso err">
  <span class="et">Important</span>
  <p>Trained models (<code>*.pt</code>) are <b>not included in the GitHub
  download</b>, because of their size. A medium model weighs tens of megabytes and
  GitHub is not the place to distribute it. The program installs fine, but the
  Detector and the Viewer cannot analyse anything until you copy one in.</p>
</div>

<h3>What to do</h3>
<p>Copy your <code>.pt</code> file into the <code>models\</code> folder of the
installation. The program creates it during setup, even if it stays empty:</p>

<pre><code>Poly-X-Microplastics-main\
├── polyx\
├── <span class="p">models\</span>              <span class="c">← the .pt file goes here</span>
│   └── bestdetectormedium.pt
├── runs\
└── SETUP.bat</code></pre>

<p>Nothing has to be renamed or configured: the Detector lists every
<code>.pt</code> it finds there.</p>

<h3>Where a model comes from</h3>
<div class="dl">
<dl>
  <dt>Someone who already trained it gives it to you</dt>
  <dd>This is the normal case in a laboratory: a model is trained once and the
      whole group uses it. A <code>.pt</code> is copied on a USB stick or a shared
      drive like any other file.</dd>
  <dt>You train it yourself</dt>
  <dd>With the <a href="#entrenador">Trainer</a>, from images annotated in the
      <a href="#etiquetador">Labeller</a>. This is the long road, and it is the one
      to take if your photographs come from a different microscope, stain or
      illumination.</dd>
</dl>
</div>

<div class="aviso warn">
  <span class="et">A model is not universal</span>
  <p>A model learns from the photographs it was trained on: their microscope,
  illumination, staining and size range. Applied to photographs taken differently
  it can fail silently — under-detecting, which is the error nobody notices — so
  it is worth verifying against a manual count before trusting the numbers.
  <a href="#flujos">Section 14</a> suggests how.</p>
</div>
""",
    },
    # ── 6 ───────────────────────────────────────────────────────────────
    {
        "id": "launcher",
        "titulo": "The Launcher: the entry screen",
        "sub": "Where each module is opened, and where the language is set.",
        "html": r"""
[[fig:ui/01_launcher|The Launcher. Each module opens in its own window, independent of the others.]]

<p>Each module starts as a <b>separate process</b>. That means you can have the
Labeller annotating and the Detector analysing at the same time, and that if one
fails it does not take the others down with it.</p>

<h3>The language selector</h3>
<p>Top right. It switches between <b>Spanish</b> and <b>English</b>, and the choice
is remembered between sessions; the first time it follows the system language.</p>

<div class="aviso">
  <span class="et">When it takes effect</span>
  <p>Because modules are separate processes, they read the language <b>as they
  open</b>. The change shows up as soon as you open the next module, not in
  windows that are already open.</p>
  <p>The <b>detection report also comes out in the chosen language</b>: titles,
  tables, figure captions, chart axes and the methods prose. Even the HTML
  <code>lang</code> attribute is set, so the browser's spell-checker and screen
  readers handle it properly.</p>
  <p>To force it without touching the interface, the environment variable
  <code>POLYX_IDIOMA=en</code>.</p>
</div>

<h3>The new-version notice</h3>
<p>When the Launcher opens it checks in the background whether GitHub is ahead. If
it is, a button appears with the new version's identifier. The check <b>does not
delay startup</b> and stays quiet on any failure: with no internet, or with GitHub
down, the notice simply does not appear.</p>
""",
    },
    # ── 7 ───────────────────────────────────────────────────────────────
    {
        "id": "detector",
        "titulo": "Detector module",
        "sub": "Batch analysis of a whole folder, and the final report. Nine screens.",
        "anclas": [
            ("d-modelos", "Models"),
            ("d-imagenes", "Images"),
            ("d-gt", "Manual GT"),
            ("d-parametros", "Parameters"),
            ("d-ejecutar", "Run"),
            ("d-resultados", "Results"),
            ("d-errores", "Errors"),
            ("d-comparar", "Compare"),
            ("d-reporte", "Report"),
        ],
        "html": r"""
<p>This is the main module: it takes a folder of photographs and a model, and
returns the counts, the sizes and a report. The nine screens on the left are
worked through from top to bottom, in that order.</p>

<h3 id="d-modelos">Models</h3>
[[fig:ui/02_det_modelos|The <b>Models</b> tab. Three slots, so up to three models can be compared
on exactly the same photographs.]]
<p>Up to <b>three <code>.pt</code> models</b> load at once. This is not a whim:
comparing two architectures on the same batch, with the same parameters, is the
only thing that lets a difference in metrics be attributed to the model rather
than to the conditions. It also accepts <b>drag and drop</b> of the file straight
onto the window.</p>

<h3 id="d-imagenes">Images</h3>
[[fig:ui/03_det_imagenes|The <b>Images</b> tab. The GT column shows, for each photograph, whether
a manual annotation exists to compare against.]]
<p>You pick a folder and every image is read, including those in subfolders. The
<b>GT</b> column flags which have ground truth: a <code>.txt</code> file with the
manual count. Without GT the program still detects, but it cannot compute hits or
errors, because there is nothing to compare against.</p>

<h3 id="d-gt">Manual GT</h3>
[[fig:ui/04_det_gt_manual|The <b>Manual GT</b> tab. A full annotator inside the Detector itself,
to mark the reference truth without switching modules.]]
<p>A built-in box annotator: draw by dragging, select, move, resize from the
corners, zoom with the wheel and pan with the middle button. It writes
<code>.txt</code> files in YOLO format next to the image.</p>
<p>It exists so you do not have to leave the Detector when a few annotations are
missing. For a campaign of hundreds of images, the
<a href="#etiquetador">Labeller</a> is the right tool, because it tracks progress
across sessions.</p>

<h3 id="d-parametros">Parameters</h3>
[[fig:ui/05_det_parametros|The <b>Parameters</b> tab. Confidence, IoU, micrometres per pixel and
inference resolution.]]
<div class="tabla-env">
<table>
  <thead><tr><th>Parameter</th><th>What it controls</th><th>Starting value</th></tr></thead>
  <tbody>
    <tr><td><b>Confidence</b></td><td>Threshold below which a detection is discarded</td>
        <td>0.25 — lower to 0.10 if nothing is detected</td></tr>
    <tr><td><b>IoU (NMS)</b></td><td>How much two boxes may overlap before merging</td>
        <td>0.45</td></tr>
    <tr><td><b>IoU (hits)</b></td><td>Minimum overlap for a detection to count as a hit</td>
        <td>0.50</td></tr>
    <tr><td><b>μm/pixel</b></td><td>The scale. Converts pixels into micrometres</td>
        <td>Automatic against the dish</td></tr>
    <tr><td><b>Resolution</b></td><td>The size at which the network sees the image</td>
        <td>Fast 1280 · Balanced 2560 · Maximum 4096</td></tr>
  </tbody>
</table>
</div>

<div class="aviso ok">
  <span class="et">Automatic calibration against the Petri dish</span>
  <p>The dish rim is found automatically and its known diameter fixes the
  <b>μm/px of each photograph</b>, with nothing to mark by hand. This matters
  because shooting distance varies between shots: in this study's material the
  real scale ranges from <b>31 to 50 μm/px</b>, a factor of 1.6. A single value for
  the whole batch would give sizes with up to <b>50 % error</b>.</p>
</div>

<p>There is also a button that <b>measures how much resolution your GPU can take</b>
before running out of memory, so you do not find out halfway through a batch.</p>

<div class="aviso vio">
  <span class="et">How far the polymer call holds</span>
  <p>Detecting a particle and saying which polymer it is are two different
  questions, and the second is far more fragile. Nile Red is
  <b>solvatochromic</b>: its emission responds to the <b>polarity</b> of its
  environment, not to chemical identity. That is why PET —a polar polyester—
  separates cleanly, while PP and LDPE, both non-polar polyolefins, share a hue
  and differ only in <i>brightness</i>; and brightness depends on exposure,
  focus, particle thickness and how much the stain took.</p>
  <p>The <b>Minimum confidence to assign</b> field lets the report say so rather
  than hide it: below that value the particle <b>still counts</b> as a detected
  particle, but is reported as <b>"unassignable"</b> instead of being given a
  polymer. At 0 it is off. The report states in Methods what percentage of the
  batch was left unassigned — exactly the figure a reviewer will ask for.</p>
  <p>Confirming the polymer itself requires spectroscopy —FTIR or Raman—. No
  fluorescent stain replaces that.</p>
</div>

<h3 id="d-ejecutar">Run</h3>
[[fig:ui/06_det_ejecutar|The <b>Run</b> tab. Before starting it warns whether the batch will be
tiled and how long it will take.]]

<div class="aviso vio">
  <span class="et">Tiling of large photographs</span>
  <p>Above a certain threshold, a photograph is analysed as <b>overlapping
  tiles</b> rather than whole. The reason is specific: at full resolution the
  particles fall below the network's <i>stride</i> and vanish — a 12 px particle in
  a 4096 px photo shrinks to 2 px when the network sees it at 640. The boxes are
  then mapped back to the original photo's coordinates and the overlaps merged with
  NMS, so <b>the results and the report always speak about the complete photograph,
  never about the tiles</b>.</p>
</div>

<h3 id="d-resultados">Results</h3>
[[fig:ui/07_det_resultados|The <b>Results</b> tab. Global metrics and the particle table, with sizes
already in micrometres.]]
<p>Global metrics (hits, false positives, false negatives, F1, precision and
recall) and the full table: one row per particle, with class, confidence, length,
width, area, aspect ratio and whether it is a <b>fibre</b> or a <b>fragment</b>. It
includes a <b>size histogram</b> by class and by size band, stacked by polymer, and
direct <b>CSV export</b>.</p>

<div class="aviso warn">
  <span class="et">Size is measured on the particle, not on its box</span>
  <p>The box of an elongated particle is nearly empty and depends on how it
  happened to fall: a fibre lying diagonally has a <b>square</b> box. Measured over
  7,129 annotated particles, the box <b>overestimates area by 1.87×</b>. That is
  why the particle is segmented inside the box and measured on the mask. The detail
  is in <a href="#medida">section 11</a>.</p>
</div>

<h3 id="d-errores">Errors</h3>
[[fig:ui/08_det_errores|The <b>Errors</b> tab. Confusion matrix and a gallery of the failures, with
a filter by type.]]
<p>It lists every wrong box and sorts it into <b>false positive</b> (detected
something that was not there), <b>false negative</b> (missed a real particle) or
<b>misclassified</b> (found it but gave it the wrong class). With the confusion
matrix alongside, it is immediately visible whether the problem is detection or
classification — which are fixed in different ways.</p>

<h3 id="d-comparar">Compare</h3>
[[fig:ui/09_det_comparar|The <b>Compare</b> tab. The loaded models, set against each other on the
same batch.]]
<p>A comparison table of the loaded models, photograph by photograph, with each
one's detections and — if ground truth exists — their hits, errors and overall F1.</p>

<h3 id="d-reporte">Report</h3>
[[fig:ui/10_det_reporte|The <b>Report</b> tab. Thirteen selectable sections and three presets. This
is where the analysis becomes a document.]]
<p>This is the module's output and deserves its own section: it is detailed in
<a href="#informe">section 13</a>.</p>
""",
    },
    # ── 8 ───────────────────────────────────────────────────────────────
    {
        "id": "entrenador",
        "titulo": "Trainer module",
        "sub": "Training your own YOLO model, with live curves. Nine screens.",
        "html": r"""
<p>Used when the available model does not suit your photographs: a different
microscope, stain or size range. It needs an already annotated dataset — the one
the <a href="#etiquetador">Labeller</a> produces — and a fair amount of patience.</p>

<h3>Model</h3>
[[fig:ui/11_ent_modelo|The <b>Model</b> tab. YOLO v8 or v11 family, sizes from nano to xlarge, or
your own starting weights.]]
<p>Supports <b>YOLO v8 and v11</b>, in every size. There is a checkbox to
<b>train v8 and v11 with the same configuration</b>: both runs reuse identical
<code>imgsz</code>, <code>batch</code>, epochs, seed and augmentation. That is what
allows a difference in metrics to be attributed to the architecture rather than to
the hyperparameters. They run in sequence because they share the GPU, and when
they finish the log summarises the comparison.</p>

<h3>Dataset</h3>
[[fig:ui/12_ent_dataset|The <b>Dataset</b> tab. Loading <code>data.yaml</code> with automatic split
detection and a box preview.]]
<p>It loads <code>data.yaml</code>, detects the train/val/test splits and validates
the dataset with <b>✓/✗</b> indicators: images present, matching labels, coherent
classes. It includes a <b>pre-training audit</b> with the class distribution and a
verdict on whether the dataset is fit to train on.</p>

<h3>Parameters</h3>
[[fig:ui/13_ent_parametros|The <b>Parameters</b> tab. Epochs, batch, resolution and optimiser, with
automatic recommendations.]]

<h3>Augmentation</h3>
[[fig:ui/14_ent_augmentacion|The <b>Augmentation</b> tab. Flips, rotation, mosaic, HSV, mixup and
copy-paste.]]

<h3>Train</h3>
[[fig:ui/15_ent_entrenar|The <b>Train</b> tab. Live curves of mAP, loss, precision and recall while
it runs.]]
<p>On starting it states explicitly whether it is using GPU or CPU, so that nobody
discovers three hours later that it was training on the processor.</p>

<div class="aviso ok">
  <span class="et"><code>best_real.pt</code>: the weights that work on your photos</span>
  <p>Ultralytics saves <code>best.pt</code> by the mAP of the full validation set,
  and in a mixed dataset that number is dominated by spiked laboratory dishes: the
  Loa dataset had <b>1,191 laboratory boxes against 47 of real sediment</b>. The
  model that wins that average is not necessarily the one that works best on
  sediment.</p>
  <p>When training ends, Poly-X evaluates every checkpoint against <b>only</b> the
  real sediment and saves the winner separately, as <code>best_real.pt</code>.
  <code>best.pt</code> is left untouched.</p>
</div>

<h3>Evaluate</h3>
[[fig:ui/16_ent_evaluar|The <b>Evaluate</b> tab. Validates any <code>.pt</code> against a YOLO
dataset, with per-class metrics.]]

<h3>Compare</h3>
[[fig:ui/17_ent_comparar|The <b>Compare</b> tab. Every earlier run with its metrics, so the thread
between experiments is not lost.]]

<h3>Export</h3>
[[fig:ui/18_ent_exportar|The <b>Export</b> tab. Converts the model to ONNX, TensorRT or CoreML.]]

<h3>Report</h3>
[[fig:ui/19_ent_informe|The <b>Report</b> tab. Generates the training HTML, with curves and the full
configuration.]]

<div class="aviso warn">
  <span class="et">If it runs out of memory</span>
  <p><code>CUDA out of memory</code> is the commonest training error. Lower the
  <b>batch</b> to 4 or 2, and the resolution to 640. That is preferable to shrinking
  the model, which costs more in metrics.</p>
</div>
""",
    },
    # ── 9 ───────────────────────────────────────────────────────────────
    {
        "id": "etiquetador",
        "titulo": "Labeller module",
        "sub": "Annotating hundreds of images without losing track between sessions.",
        "html": r"""
[[fig:ui/20_etiquetador|The Labeller. The list on the left separates what is reviewed from what is
pending, and that state is restored from disk when the folder is reopened.]]

<p>A box annotator in YOLO format. You draw by dragging, assign a class with a
right-click or the <kbd>1</kbd>–<kbd>9</kbd> keys, and there is per-image
<b>undo/redo</b>.</p>

<h3>Progress tracking</h3>
<p>It is built for campaigns of hundreds of images spread over several sessions.
The list separates three cases:</p>

<div class="tabla-env">
<table>
  <thead><tr><th>Mark</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>✓ name (n)</code></td><td>reviewed, with n particles</td></tr>
    <tr><td><code>· name (0)</code></td><td>reviewed, no particles — <b>that is data</b></td></tr>
    <tr><td><code>○ name</code></td><td>not reviewed yet</td></tr>
  </tbody>
</table>
</div>

<div class="aviso vio">
  <span class="et">Why passing over an image creates no file</span>
  <p>A <code>.txt</code> is written only when the image is <b>marked as
  reviewed</b> or when a <b>box is drawn</b>. Recording an image that was barely
  glanced at as “reviewed with zero” would falsify a census count: the zero of a
  dish that was looked at and the zero of a dish that was skipped mean different
  things, and the file cannot tell them apart afterwards.</p>
</div>

<h3>Automatic pre-annotation</h3>
<p>You can load a <code>.pt</code> model and let it propose the boxes, with
adjustable <code>conf</code> and <code>imgsz</code> and GPU if available. You then
correct them by hand. On this study's material it saves around <b>80 % of the
time</b> compared with annotating from scratch.</p>

<h3>Shortcuts</h3>
<div class="atajos">
<dl>
  <dt><kbd>Space</kbd></dt><dd>mark reviewed and advance</dd>
  <dt><kbd>Tab</kbd></dt><dd>jump to the next unreviewed image</dd>
  <dt><kbd>←</kbd> <kbd>→</kbd></dt><dd>move between images</dd>
  <dt><kbd>1</kbd>–<kbd>9</kbd></dt><dd>change the active class</dd>
</dl>
<dl>
  <dt><kbd>F</kbd></dt><dd>fit the image to the window</dd>
  <dt><kbd>Del</kbd></dt><dd>delete the selected box</dd>
  <dt><kbd>Ctrl</kbd>+<kbd>Z</kbd></dt><dd>undo</dd>
  <dt><kbd>Ctrl</kbd>+<kbd>Y</kbd></dt><dd>redo</dd>
</dl>
</div>

<h3>Details that prevent lost work</h3>
<ul>
  <li><b>Silent auto-save</b> every 60 seconds.</li>
  <li><b>Zoom is preserved</b> between images, via a checkbox in the right-hand
      panel. With a size tolerance, because the crops of a grid differ by 1 px
      through rounding and demanding exact equality lost the zoom on every
      change.</li>
  <li><b>Minimum box side: 2 px.</b> It used to be 5 px and <b>silently</b>
      discarded legitimate marks: the smallest particles in the study are about
      8 px across. Now, if a box is rejected, the status bar says so.</li>
  <li>It writes the <code>.txt</code> next to the image — or into
      <code>labels/</code> if the image lives in <code>images/</code> — and
      generates <code>classes.txt</code>.</li>
  <li>Thumbnails are generated in the background, so the window responds
      immediately.</li>
</ul>
""",
    },
    # ── 10 ──────────────────────────────────────────────────────────────
    {
        "id": "visor",
        "titulo": "Viewer module",
        "sub": "One image at a time: calibrating the scale and verifying every measurement.",
        "html": r"""
[[fig:ui/21_visor|The Viewer. The table lists every particle, and selecting a row shows <b>what was
measured</b>: the clean crop and, beside it, the mask with the measurement drawn on top.]]

<h3>Interactive μm/pixel calibration</h3>
<p>Two modes, and in both the program then asks for the real measurement:</p>
<div class="dl">
<dl>
  <dt>📏 Line — two clicks</dt>
  <dd>Mark a reference of known length and type how long it is.</dd>
  <dt>⭕ Circle — three clicks</dt>
  <dd>Mark three points on the rim and type the real diameter. This is the mode to
      use with Petri dishes, whose diameter is known; the dialog defaults to
      100,000 μm, the dish used in the study.</dd>
</dl>
</div>
<p>The bottom bar shows the scale in real time, in the form
<code>📐 0.4880 μm/px (line)</code>.</p>

<h3>Particle-by-particle review</h3>
<p>This is the function that makes the Viewer a verification tool and not just a
display. The table lists every particle with its number, class, type, length,
width and aspect ratio. Selecting a row shows <b>what was measured</b>:</p>
<ul>
  <li>on the left, the crop <b>with no marks</b>;</li>
  <li>on the right, the mask outline with the measurement drawn on it:
      <span style="color:#9a6700;font-weight:650">the Feret line in yellow</span>,
      <span style="color:#6639ba;font-weight:650">the geodesic path in magenta</span>,
      <span style="color:#1f6b5e;font-weight:650">the mask in green</span>;</li>
  <li>and the full arithmetic from pixels to micrometres.</li>
</ul>

<div class="aviso ok">
  <span class="et">The principle</span>
  <p>A size you cannot see being measured is a size you cannot verify. That is why
  the Viewer does not show only the number: it shows the particle, the mask it came
  from, and the line that was measured on it.</p>
</div>

<h3>What else it does</h3>
<ul>
  <li><b>Detect</b> with a loaded model, at a configurable resolution from 320 to
      8192 and on GPU if available. With tiny objects in large photographs
      <code>imgsz</code> is decisive: at low values the particles fall below the
      network's <i>stride</i> and nothing is detected.</li>
  <li><b>Load existing <code>.txt</code> labels</b> and see them over the image with
      sizes converted to μm. Useful for reviewing a manual count without going back
      to the Labeller.</li>
  <li><b>Load the predictions of a finished run</b> from
      <code>runs/detect_.../</code>, without running the model again. It always
      opens the original photograph and never the annotated PNG, which has the
      boxes painted on it.</li>
  <li><b>Drag and drop</b> an image or a model straight onto the canvas.</li>
  <li><b>Export</b> the annotated image, <code>detecciones.csv</code> and
      <code>resumen.json</code>.</li>
</ul>
""",
    },
    # ── 11 ──────────────────────────────────────────────────────────────
    {
        "id": "medida",
        "titulo": "How a particle's size is measured",
        "sub": "The criterion, its exceptions, its accuracy and its declared limits.",
        "html": r"""
<p>The general criterion is <b>the longest straight line that fits inside the
particle</b>: the greatest distance between two points on its outline, or
<i>maximum Feret diameter</i>. It does not depend on the orientation the particle
happened to fall in, and a jagged edge does not change it.</p>

<p>That line stops working when the particle is <b>contorted</b>: in a folded
fibre, the distance between the ends is the chord, and on a half-circle arc it
falls <b>35 % short</b>. For those cases the <i>geodesic diameter</i> is used: the
longest path that fits <b>inside</b> the particle, which, unable to leave the mask,
follows the curve.</p>

<div class="tabla-env">
<table>
  <thead><tr><th>Particle shape</th><th>What is reported as length</th></tr></thead>
  <tbody>
    <tr><td>Compact or irregular, but not folded</td>
        <td><b>Maximum Feret</b> — the longest straight line</td></tr>
    <tr><td>Elongated and contorted (fibre)</td>
        <td><b>Geodesic diameter</b> — follows the curve</td></tr>
  </tbody>
</table>
</div>

<p>The geodesic is applied only if the particle is <b>thin</b> (length ≥ 4 ×
thickness) and <b>non-convex</b> (solidity &lt; 0.90). Each condition comes from an
observed failure: without the first, any concavity makes the path go around the
particle instead of through it — a real 44 px clump was given 73; without the
second, length would come to depend on the rotation angle, which is precisely the
defect being eliminated.</p>

<div class="aviso ok">
  <span class="et">Accuracy</span>
  <p>Against synthetic shapes of known size — straight lines, rotated lines, arcs
  of 60°, 120° and 180°, a circle, a saw-toothed line and a notched clump — the
  length measured this way gives <b>0.6 % median error and 4.7 % in the worst
  case</b>. It is pinned in <code>tests/test_morfologia.py</code>, so a
  well-meaning change that makes it worse fails the suite.</p>
</div>

<div class="aviso warn">
  <span class="et">The equivalent rectangle is not a size</span>
  <p>The formula <i>L</i> = (<i>P</i> + √(<i>P</i>²−16<i>A</i>))/4 gives the length
  of a rectangle with the same area and perimeter, which is a different thing. It
  depends on the perimeter, so a jagged edge inflates it by <b>22.5 %</b>, and it
  is undefined for compact particles, where <i>P</i>² &lt; 16<i>A</i>. It is
  reported as a descriptor, because compared with the other two it betrays
  irregular edges, but <b>it is not used as a size</b>.</p>
  <p>A correction made in August 2026 came from exactly that: “22 % fibres and
  aspect ratios up to 21.1” had been reported, an artefact of using this formula as
  length. Measured with Feret and geodesic over 6,638 particles, the median aspect
  ratio is <b>1.58</b>, the maximum <b>8.7</b>, and fibres are <b>1.1 %</b>. The
  material is mostly compact fragments.</p>
</div>

<h3>From pixels to micrometres: the scale</h3>
<p>All of the above is measured in pixels. The conversion is <b>not a single factor
for the batch</b>: it depends on the shooting distance, and in this study's material
the real scale ranges from <b>31 to 50 μm/px</b>, a factor of 1.6.</p>

<p>So each photograph is calibrated with its own, against the Petri dish rim: the
approximate centre is found with Hough, the rim is sampled in <b>720
directions</b>, and a circle is fitted by least squares with outlier rejection. The
Hough radius is <b>not used</b>, because it can be off by 12 % and that error would
pass in full into every size.</p>

<div class="aviso vio">
  <span class="et">Which rim is the 100 mm, and which way it can be wrong</span>
  <p>The rim has a wall of about <b>2 mm</b> — measured on this study's
  photographs: the inner edge falls at 0.960 of the fitted radius and the outer at
  1.000. The nominal diameter of a Petri dish is ambiguous at that level: it may
  refer to the outer or to the usable inner edge. Here the <b>outer</b> is taken,
  which is the edge the circle fits. If the nominal figure referred to the inner
  edge, the correct scale would be <b>4.2 % larger</b> and every size would be
  <b>underestimated</b> by that amount. The bias can only go that way, because the
  outer is the larger of the two possible edges. It is declared in the report
  itself.</p>
</div>

<h3>Particles that touch</h3>
<p>Two particles in contact form a single blob, and measuring them together would
add their sizes. They are separated by <i>watershed</i> on the distance transform:
the centre of each is far from the background and the neck joining them is close,
so the cut falls at the neck. On circles of known size it separates them up to
<b>27 % overlap of the diameter</b>, without splitting any single-piece
particle.</p>

<h3>A fibre is measured whole</h3>
<p>The mask is <b>not</b> cropped to the detector's box when the particle is
elongated, nor passed through the touching-particle splitter: both cut real fibres.
In the case that exposed it, <b>53 % of the length</b> was lost — 369 px of
connected component became 174 — and an underestimated size does not show up in
the figures, only in the image. It is pinned in
<code>tests/test_fibra_no_se_trunca.py</code>.</p>

<div class="aviso err">
  <span class="et">Declared limitations</span>
  <p>Two particles overlapping <b>by more than 40 % of their diameter</b> are still
  measured as one: past that point there is no neck left to cut at.</p>
  <p>In a <b>tightly coiled fibre</b> the geodesic path cuts the corner inside each
  bend, underestimating by up to <b>19 %</b> in the tightest case tested.</p>
</div>
""",
    },
    # ── 12 ──────────────────────────────────────────────────────────────
    {
        "id": "polimeros",
        "titulo": "The three polymers, and why two get confused",
        "sub": "What you need to know to read the per-class recall.",
        "html": r"""
<div class="tabla-env">
<table>
  <thead><tr><th>ID</th><th>Class</th><th>Observed fluorescence</th><th>Box colour on screen</th></tr></thead>
  <tbody>
    <tr><td>0</td><td><span class="tag pet">PET</span></td><td>Red–salmon</td><td>🔴 red</td></tr>
    <tr><td>1</td><td><span class="tag pp">PP</span></td><td><b>Greenish, dull</b> yellow</td><td>🟠 orange</td></tr>
    <tr><td>2</td><td><span class="tag ldpe">LDPE</span></td><td>Plain yellow, <b>brighter</b></td><td>🟡 yellow</td></tr>
  </tbody>
</table>
</div>

<p>The colours on the right are only the box colours in the interface: <b>they do
not describe the real emission</b>. Measured inside the boxes of the training
dataset (mean RGB, n = 30 per class):</p>

<div class="tabla-env">
<table>
  <thead><tr><th>Class</th><th class="num-col">R</th><th class="num-col">G</th><th class="num-col">B</th></tr></thead>
  <tbody>
    <tr><td><b>PET</b></td><td class="num-col">116</td><td class="num-col">58</td><td class="num-col">65</td></tr>
    <tr><td><b>PP</b></td><td class="num-col">122</td><td class="num-col"><b>125</b></td><td class="num-col">32</td></tr>
    <tr><td><b>LDPE</b></td><td class="num-col">181</td><td class="num-col">162</td><td class="num-col">57</td></tr>
  </tbody>
</table>
</div>

<div class="aviso warn">
  <span class="et">PP and LDPE separate by brightness, not by hue</span>
  <p>Both are yellowish, but in PP green <b>matches or exceeds</b> red (122 against
  125) and the emission is considerably duller; in LDPE red rises to 181. It is the
  commonest confusion when annotating, and the reason per-class recall collapses
  there: <b>PET 0.98 · PP 0.70 · LDPE 0.54</b>.</p>
  <p>If your count separates PP from LDPE, those two figures are the limit of what
  the model can support today. The total particle count is far more reliable than
  the split between those two classes.</p>
</div>

<div class="aviso">
  <span class="et">Historical note</span>
  <p>Older documentation described PP as “orange”, which does not match the
  observed emission. It was corrected in August 2026 from the RGB measurement and
  direct observation.</p>
</div>
""",
    },
    # ── 13 ──────────────────────────────────────────────────────────────
    {
        "id": "informe",
        "titulo": "The detection report",
        "sub": "A self-contained HTML of publication quality, and its PDF.",
        "html": r"""
[[fig:ui/10_det_reporte|The <b>Report</b> tab. Thirteen section checkboxes and three presets: Full,
Short summary and Methodological.]]

<p>The report is the real output of the work: a <b>self-contained HTML</b> with
every image embedded in base64, which can be emailed without anything breaking, and
which exports to <b>PDF in one click</b>.</p>

<h3>What it can include</h3>
<div class="dl">
<dl>
  <dt>Prediction vs Ground Truth comparison gallery</dt>
  <dd>Side by side. Gallery images are re-encoded and their number capped so the
      file stays openable in a browser; <b>the metrics, by contrast, cover every
      image</b>.</dd>

  <dt>Calibration section</dt>
  <dd>Where each photograph's scale came from, its minimum, median and maximum, the
      <b>mean with a 95 % confidence interval</b>, and a figure over a real dish with
      the fitted circle and its diameter drawn on top.</dd>

  <dt>Size by folder and by photo</dt>
  <dd>Compares the size distribution between folders — each folder as a sampling
      site, station or condition — with box plots and a <b>Kruskal-Wallis</b>
      test.</dd>

  <dt>Core depth profile</dt>
  <dd>When photographs are named <code>section.core</code>, the section stops being
      just another folder and becomes an <b>ordered</b> variable. The report draws
      the profile with depth on the vertical axis — the convention for any sediment
      core — and answers whether particle count and median size rise or fall with
      depth, using <b>Spearman</b> and a <i>p</i> value by <b>permutation</b>.</dd>

  <dt>Measured-particle sheet</dt>
  <dd>The <b>6 fibres and the 6 largest particles</b>, each with its crop beside it
      and the measurement drawn on top. The split is deliberate: fibres are a
      minority and are exactly where the geodesic method acts.</dd>

  <dt>Model comparison</dt>
  <dd>A photo-by-photo table with each model's detections and, if ground truth
      exists, their hits, errors and overall F1.</dd>

  <dt>Size histogram</dt>
  <dd>By class and by size band, stacked by polymer.</dd>
</dl>
</div>

<div class="aviso vio">
  <span class="et">Why the correlation runs over sections and not over particles</span>
  <p>Two particles from the same dish are <b>not independent observations</b> of
  depth: they share dish, shot and conditions. Correlating particle by particle
  would inflate the sample size with repetitions of the same measurement and produce
  <i>p</i> values that mean nothing. The correlation is computed <b>over the
  sections</b>.</p>
</div>

<h3>Selectable sections</h3>
<p>Thirteen checkboxes and three presets: <b>Full</b>, <b>Short summary</b> and
<b>Methodological</b>. Unticking a section makes the rest <b>renumber
themselves</b> and the index adjust; a section that is ticked but has no data is
omitted anyway, rather than appearing empty.</p>

<h3>Selectable scope</h3>
<p>The report can cover the complete job, <b>only the photographs you tick</b>, or
both at once. The figures, the charts and the confusion matrix are
<b>recomputed over what you chose</b>, so the report always describes the
photographs it shows.</p>

<div class="aviso ok">
  <span class="et">And it comes out in your language</span>
  <p>Titles, tables, figure captions, chart axes and the methods prose follow the
  language chosen in the Launcher — not just the interface.</p>
</div>
""",
    },
    # ── 14 ──────────────────────────────────────────────────────────────
    {
        "id": "flujos",
        "titulo": "Recommended workflows",
        "sub": "Three scenarios, depending on where you start from.",
        "html": r"""
<h3>A · You already have a model and just want to analyse photographs</h3>
<p>The commonest case, and the shortest.</p>
<ol>
  <li>Copy the <code>.pt</code> into <code>models\</code>.</li>
  <li>Open the <b>Detector</b> from the Launcher.</li>
  <li><b>Models</b> → load the model.</li>
  <li><b>Images</b> → choose the folder.</li>
  <li><b>Parameters</b> → leave the automatic calibration if your photos include
      the whole dish; otherwise set μm/px by hand.</li>
  <li><b>Run</b> → ▶ start. Read the tiling notice before you begin.</li>
  <li><b>Report</b> → generate the HTML, and export to PDF if it needs sending.</li>
</ol>

<h3>B · You want to train a model for your own photographs</h3>
<ol>
  <li><b>Labeller</b> → annotate. Start with the densest station: it is the one
      that carries the main result and the one that teaches the model most.</li>
  <li>Recount a <b>random ~10 %</b> to estimate your intra-observer error.
      Reviewers ask for it, and without that figure there is nothing to compare the
      model's error against.</li>
  <li><b>Trainer</b> → <b>Dataset</b>: pass the audit before training.</li>
  <li><b>Train</b>. If you are comparing architectures, use the <b>v8 and v11 with
      the same configuration</b> checkbox.</li>
  <li>Keep <code>best_real.pt</code> if your dataset mixes laboratory and real
      material.</li>
  <li><b>Detector</b> → analyse and compare against your manual count.</li>
</ol>

<div class="aviso warn">
  <span class="et">One seed does not let you claim an architecture wins</span>
  <p>Two runs of the same model with different seeds can differ more than two
  architectures do from each other. If you are going to assert “v11 beats v8” in a
  document, repeat with <b>several seeds</b> and report the spread.</p>
</div>

<h3>C · You want to verify one particular measurement</h3>
<ol>
  <li><b>Viewer</b> → open the photograph.</li>
  <li><b>Load predictions</b> from the finished run, in
      <code>runs/detect_.../</code>. No need to run the model again.</li>
  <li>Select the particle's row in the table.</li>
  <li>Look at the crop and the mask with the measurement drawn on it: that is where
      you see whether the size came from a correct segmentation or a broken
      mask.</li>
</ol>

<div class="aviso">
  <span class="et">What the program cannot give you</span>
  <p>Poly-X counts and measures particles. To report <b>particles per kilogram</b>
  — the unit comparable with the literature — you need the <b>dry mass of each
  section</b>, which is measured in the laboratory and does not come out of the
  photographs. The program cannot invent it.</p>
</div>
""",
    },
    # ── 15 ──────────────────────────────────────────────────────────────
    {
        "id": "atajos",
        "titulo": "Keyboard shortcuts",
        "sub": "Those of the Labeller and the Manual GT save the most time.",
        "html": r"""
<h3>Labeller and Manual GT</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Key</th><th>What it does</th></tr></thead>
  <tbody>
    <tr><td><kbd>Space</kbd></td><td>mark the image reviewed and advance</td></tr>
    <tr><td><kbd>Tab</kbd></td><td>jump to the next unreviewed image</td></tr>
    <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>previous / next image (auto-saves)</td></tr>
    <tr><td><kbd>1</kbd> … <kbd>9</kbd></td><td>change the active class, or that of the selected box</td></tr>
    <tr><td><kbd>F</kbd></td><td>fit the image to the window</td></tr>
    <tr><td><kbd>Del</kbd></td><td>delete the selected box</td></tr>
    <tr><td><kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd></td><td>undo / redo</td></tr>
    <tr><td>Mouse wheel</td><td>zoom</td></tr>
    <tr><td>Middle button, or <kbd>Space</kbd>+drag</td><td>pan the image</td></tr>
    <tr><td>Left click + drag</td><td>draw a box</td></tr>
    <tr><td>Right click on a box</td><td>assign it a class</td></tr>
  </tbody>
</table>
</div>

<h3>Viewer</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Action</th><th>What it does</th></tr></thead>
  <tbody>
    <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>move through the folder</td></tr>
    <tr><td>Mouse wheel</td><td>zoom</td></tr>
    <tr><td>Drag and drop</td><td>open an image or load a <code>.pt</code> model</td></tr>
    <tr><td>2 clicks in line mode</td><td>calibrate by known length</td></tr>
    <tr><td>3 clicks in circle mode</td><td>calibrate by known diameter</td></tr>
  </tbody>
</table>
</div>
""",
    },
    # ── 16 ──────────────────────────────────────────────────────────────
    {
        "id": "actualizar",
        "titulo": "Updating and uninstalling",
        "sub": "Getting what is new without reinstalling, and removing the program cleanly.",
        "html": r"""
<h3>Updating</h3>
<p>Double-click the matching updater:</p>
<div class="tabla-env">
<table>
  <thead><tr><th>System</th><th>File</th></tr></thead>
  <tbody>
    <tr><td><span class="tag win">Windows</span></td><td><code>actualizar.bat</code></td></tr>
    <tr><td><span class="tag mac">macOS</span></td><td><code>actualizar_macOS.command</code></td></tr>
  </tbody>
</table>
</div>

<p>It checks whether there is a new commit on <code>main</code> and, if so,
downloads and replaces <b>only the program's files</b>. It <b>keeps</b> your
<code>.venv</code> environment, your <code>models/*.pt</code>, your
<code>runs/</code> and any local data. It does not need Git installed — it
downloads over HTTPS — only an internet connection.</p>

<div class="aviso">
  <span class="et">It does not matter which system you update from</span>
  <p>Each updater protects itself while it runs, but <b>it does update the other
  platform's files</b>. The project ends up complete for both, whichever you came
  from.</p>
</div>

<p>There is no need to remember to run it: the <a href="#launcher">Launcher</a>
gives notice on its own when GitHub is ahead.</p>

<h3>Uninstalling</h3>
<p>On Windows, <code>DESINSTALAR.bat</code>. It removes the environment and the
program's files. Check first that no models or results you want to keep are still
inside the folder: <code>models\</code> and <code>runs\</code> are yours, not the
program's.</p>

<div class="aviso warn">
  <span class="et">Before uninstalling</span>
  <p>Move your <code>.pt</code> files and your <code>runs/</code> out of the folder.
  A trained model can represent days of work and cannot be recovered from GitHub,
  because it was never there.</p>
</div>
""",
    },
    # ── 17 ──────────────────────────────────────────────────────────────
    {
        "id": "problemas",
        "titulo": "Troubleshooting",
        "sub": "What usually fails, why, and what to do.",
        "html": r"""
<h3>While installing</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Symptom</th><th>Cause</th><th>Fix</th></tr></thead>
  <tbody>
    <tr><td><code>Python no encontrado</code></td><td>Python missing, or installed without PATH</td>
        <td>Reinstall Python 3.11.9 ticking <b>Add python.exe to PATH</b></td></tr>
    <tr><td><code>No module named tkinter</code></td><td>Python without tcl/tk</td>
        <td>Reinstall ticking <b>tcl/tk and IDLE</b></td></tr>
    <tr><td>The console closes on its own</td><td>It was run from inside the ZIP</td>
        <td>Extract properly and retry</td></tr>
    <tr><td><code>bad interpreter: /bin/bash^M</code> <span class="tag mac">macOS</span></td>
        <td>The file arrived with Windows line endings</td>
        <td>Download it again from GitHub, without passing it through email or WhatsApp</td></tr>
    <tr><td><i>Unidentified developer</i> <span class="tag mac">macOS</span></td>
        <td>Gatekeeper blocks unsigned scripts</td>
        <td><b>Right</b>-click → <b>Open</b>, only the first time</td></tr>
  </tbody>
</table>
</div>

<h3>While detecting</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Symptom</th><th>Cause</th><th>Fix</th></tr></thead>
  <tbody>
    <tr><td><b>Nothing is detected</b></td><td>Confidence too high, or resolution too low</td>
        <td>Lower confidence to 0.10. And raise the resolution: with ~12 px particles
            in 4096 px photos, at 640 they collapse to ~2 px and disappear</td></tr>
    <tr><td>All the sizes come out wrong</td><td>Wrong μm/px scale</td>
        <td>Check the report's calibration section, or calibrate by hand in the Viewer</td></tr>
    <tr><td>PP confused with LDPE</td><td>A known limit of the model</td>
        <td>No parameter fixes it. See <a href="#polimeros">section 12</a></td></tr>
    <tr><td>A fibre appears split in two</td><td>Segmentation</td>
        <td>Check it in the <a href="#visor">Viewer</a>, which shows the real mask</td></tr>
    <tr><td><code>CUDA not available</code></td><td>No usable NVIDIA GPU</td>
        <td>Not an error: it works on CPU, more slowly</td></tr>
  </tbody>
</table>
</div>

<h3>While training</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>Symptom</th><th>Fix</th></tr></thead>
  <tbody>
    <tr><td><code>CUDA out of memory</code></td>
        <td>Lower <b>batch</b> to 4 or 2, and the resolution to 640</td></tr>
    <tr><td><i>no kernel image is available for execution on the device</i></td>
        <td>PyTorch has no kernels for your card. Re-run <code>SETUP.bat</code>,
            which checks this explicitly</td></tr>
    <tr><td>The model scores well in validation and badly on your photos</td>
        <td>Mixed dataset: use <code>best_real.pt</code></td></tr>
  </tbody>
</table>
</div>

<div class="aviso">
  <span class="et">If you need to read the full error</span>
  <p>Start with <code>iniciar_polyx.bat</code> instead of the shortcut: it shows the
  console, and the whole message appears there instead of a window that closes.</p>
</div>
""",
    },
    # ── 18 ──────────────────────────────────────────────────────────────
    {
        "id": "estructura",
        "titulo": "Folder structure",
        "sub": "What is yours, what belongs to the program, and what can be deleted.",
        "html": r"""
<pre><code>Poly-X-Microplastics-main\
├── <span class="p">src\polyx\</span>               <span class="c">source code — the program's</span>
│   ├── launcher.py
│   ├── core\                 <span class="c">shared core (measurement, calibration, report, language)</span>
│   ├── gui\                  <span class="c">interface: detector\, trainer\, labeler\, viewer\…</span>
│   └── cli\                  <span class="c">headless commands (polyx-train, polyx-eval…)</span>
├── <span class="p">models\</span>                   <span class="c">YOURS — the trained .pt weights</span>
├── <span class="p">runs\</span>                     <span class="c">YOURS — one folder per run, with date and time</span>
├── <span class="p">data_microplastico\</span>       <span class="c">YOURS — YOLO dataset (images/ + labels/)</span>
├── <span class="p">.venv\</span>                    <span class="c">Python environment — rebuilt by SETUP.bat</span>
├── docs\                     <span class="c">documentation</span>
│   └── manual\               <span class="c">Manual_PolyX.en.html — this manual</span>
├── tests\                    <span class="c">measurement and calibration test suite</span>
│
├── SETUP.bat                 <span class="c">[Win] installer</span>
├── iniciar_polyx.bat         <span class="c">[Win] launcher with console</span>
├── Poly-X.vbs                <span class="c">[Win] launcher without console</span>
├── actualizar.bat            <span class="c">[Win] updater</span>
├── Lanzar_macOS.command      <span class="c">[Mac] installer + launcher</span>
├── actualizar_macOS.command  <span class="c">[Mac] updater</span>
└── LEEME.txt                 <span class="c">short version of the installation</span></code></pre>

<div class="aviso err">
  <span class="et">The three folders never to delete</span>
  <p><code>models\</code>, <code>runs\</code> and <code>data_microplastico\</code>
  hold <b>your work</b>, not the program. A trained model can be days of compute and
  none of the three can be recovered from GitHub, because they were never there.
  <code>.venv\</code>, by contrast, can be rebuilt at any time by running
  <code>SETUP.bat</code> again.</p>
</div>

<h3>What each run stores</h3>
<div class="tabla-env">
<table>
  <thead><tr><th>File</th><th>Contents</th></tr></thead>
  <tbody>
    <tr><td><code>images/*.png</code></td><td>the photographs with the boxes drawn on</td></tr>
    <tr><td><code>centroids.csv</code></td><td>one row per particle: class, position, confidence, size</td></tr>
    <tr><td><code>metrics.json</code></td><td>hits, false positives and false negatives by class</td></tr>
    <tr><td><code>report.html</code></td><td>the self-contained report</td></tr>
    <tr><td><code>annotations/</code></td><td>the YOLO <code>.txt</code> files, if ground truth existed</td></tr>
  </tbody>
</table>
</div>

<div class="aviso">
  <span class="et">The differences between systems live in a single file</span>
  <p>Opening folders, launching the updater, choosing the compute device:
  everything that differs between Windows and macOS lives in
  <code>polyx/core/plataforma.py</code>, not scattered through the code. That is why
  the program behaves the same on both without two versions to maintain.</p>
</div>
""",
    },
    # ── 19 ──────────────────────────────────────────────────────────────
    {
        "id": "referencias",
        "titulo": "References and contact",
        "sub": "The published method Poly-X rests on.",
        "html": r"""
<h3>Publications</h3>
<ul>
  <li><b>Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).</b>
      Development of a new methodology for the determination of PET microplastics
      in sediment, based on microwave-assisted acid digestion.
      <i>PLoS ONE</i> <b>19</b>(12): e0314520.
      <a href="https://doi.org/10.1371/journal.pone.0314520">doi.org/10.1371/journal.pone.0314520</a></li>

  <li><b>Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).</b>
      Evaluation of microwave-assisted acid/oxidant digestion method for the
      detection of polyethylene microplastics in <i>Merluccius gayi</i> fish by
      Nile Red fluorescent staining and image analysis.
      <i>J. Chil. Chem. Soc.</i> <b>69</b>(1): 6082–6085.
      <a href="https://doi.org/10.4067/s0717-97072024000106082">doi.org/10.4067/s0717-97072024000106082</a></li>
</ul>

<h3>Scope of the repository</h3>
<p>The repository documents <b>the program</b>. Everything belonging to a paper in
preparation — the study's analysis pipeline, its photographs and its findings — is
deliberately left out, because publishing it here would pre-empt results that have
not been released yet.</p>

<h3>Tests</h3>
<p>Shape measurement and calibration have their own suite, because every figure
they produce ends up in a table and a well-meaning change can shift all of them
with nothing to warn you:</p>
<pre><code><span class="p">.venv\Scripts\python.exe -m pytest tests/ -q</span></code></pre>
<p>There are <b>43 tests on synthetic shapes of known size</b>, depending on no
human annotation. Each one also pins the <i>why</i> of a design decision, so that if
someone tries a discarded variant again, the suite says so. <code>pytest</code> is
needed only for development and is not in <code>requirements.txt</code>: a
use-only installation does not need it.</p>

<h3>Contact</h3>
<p><b>Cristofher Ferrada</b><br>
Environmental Chemistry Laboratory · Pontificia Universidad Católica de Valparaíso<br>
<a href="mailto:cristofher.ferrada@pucv.cl">cristofher.ferrada@pucv.cl</a><br>
<a href="https://github.com/CrissFerrada/Poly-X-Microplastics">github.com/CrissFerrada/Poly-X-Microplastics</a></p>
""",
    },
]
