# Poly-X — Microplastics Detection Suite

**Status:** Ongoing rebuild (May 2026)  
**Author:** Cristofher Ferrada · PhD (c) in Chemistry · PUCV  
**Version:** 2.0.0  
**GitHub:** https://github.com/CrissFerrada/Poly-X-Microplastics

## 🎯 Purpose

Platform for **automated detection and classification** of microplastics (PET, PP, LDPE) by Nile Red fluorescence under UV light (254 nm) using YOLO v8/v11 models (deep learning).

It integrates the whole pipeline: **training → labelling → detection → paper-quality HTML report**.

### Publications

- **Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).** Development of a new methodology for the determination of PET microplastics in sediment, based on microwave-assisted acid digestion. PLoS ONE 19(12): e0314520. https://doi.org/10.1371/journal.pone.0314520
- **Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).** Evaluation of microwave-assisted acid/oxidant digestion method for the detection of polyethylene microplastics in Merluccius gayi fish by Nile Red fluorescent staining and image analysis. J. Chil. Chem. Soc. 69(1): 6082-6085. https://doi.org/10.4067/s0717-97072024000106082

---

## 📦 Structure

### The root and its contract

The root deliberately mixes two surfaces with different audiences:

- **End-user surface** (the double-click surface seen right after the ZIP
  download): everything under `install/` — `SETUP.bat`, `iniciar_polyx.bat`,
  `Poly-X.vbs`, the updaters (`actualizar.{bat,ps1}`,
  `actualizar_macOS.command`, `actualizar_linux.sh`), the uninstaller and the
  packagers (`empaquetar_*.bat`, `migrar_instalacion.ps1`), plus `setup.sh`,
  `run.sh` and `polyx.desktop` for Linux. The assumption "they live in
  install/" is wired in: `polyx.core.platform.name_updater()` plus
  `polyx.gui.updates` build `ROOT/install/name_updater()`, `Poly-X.vbs`
  self-locates two levels up (project root), `SETUP.bat` resolves `SRC` as
  its parent, and `empaquetar_para_llevar.bat` builds the offline package
  from `src\`, `docs\` and `tools\manual\` and copies the install scripts
  flat into the bundle root. Moving them again means updating all of that and
  the cross-references between the scripts themselves.
- **Developer surface** (`src/`, `tests/`, `tools/`, `docs/`):
  `tools/audit` (UI-translation audit and report fixtures), `tools/manual`
  (the manual chain), `tools/packaging` (PyInstaller). The compiled manual
  lives in `docs/manual/` and IS versioned: a fresh ZIP download opens it
  offline. The PDFs stay ignored (regenerable from `tools/manual`).
- **Local research work**: `manuscript-workspace/` is not versioned — it is
  a draft-testing workspace, material of the paper that the README keeps
  out of the repository on purpose.

```
src/                      # Código fuente (paquete instalable)
└── polyx/                # El paquete `polyx` (importable)
    ├── __init__.py       # __version__ (fuente única, leída por hatchling)
    ├── launcher.py       # Shim Qt-free del entry point histórico
    ├── assets/           # Recursos empaquetados (polyx.ico)
    ├── cli/              # CLIs headless: train, eval, detect, report,
    │                     #   download-models
    ├── core/             # Motor puro, sin Qt
    │   ├── paths.py      # Rutas (ROOT, ASSETS_DIR, USER_DATA_DIR, ...)
    │   ├── platform.py   # Diferencias entre sistemas operativos
    │   ├── i18n.py       # tr(), idioma()
    │   ├── translations_en.py
    │   ├── observability.py  # Logs JSONL (logger_for, set_run_id)
    │   ├── yolo_wrap.py  # Wrapper YOLO (Detection, YoloModel)
    │   ├── metrics.py    # TP/FP/FN, IoU, matriz de confusión
    │   ├── assignment.py # Abstención: detectar ≠ identificar
    │   ├── dataset_audit.py
    │   ├── detection_run.py  # Motor headless de detección
    │   ├── training.py       # Motor headless de entrenamiento
    │   ├── evaluation.py     # Motor headless de evaluación
    │   ├── report_simple.py  # Reporte HTML sin Qt
    │   ├── detect/       # Inferencia YOLO
    │   ├── measure/      # Segmentación y tallas sobre la máscara
    │   ├── calibration/  # Calibración μm/píxel
    │   ├── data/         # Carga/validación de datasets
    │   ├── quality/      # Métricas, asignación, abstractness
    │   └── provenance/   # Trazabilidad de la corrida
    └── gui/              # Presentación: todo lo que toca Qt
        ├── __main__.py   # python -m polyx.gui
        ├── launcher.py   # Módulo 1: Menú principal (hero + 4 cards)
        ├── theme.py      # Paleta de colores (Segoe UI, Nile Red RGB)
        ├── iconos.py     # Iconos vectoriales (QPainterPath)
        ├── marca.py      # Identidad por ventana (logo + nombre)
        ├── widgets.py    # LogoBadge, HLine, custom widgets
        ├── preferencias.py  # Diálogo de preferencias (tema, idioma)
        ├── pdf_export.py # HTML → PDF
        ├── updates.py    # Buscador de actualizaciones
        ├── report_html.py  # Generador reporte (base64, paper-quality)
        ├── qt.py         # Guard de PySide6 (extra [gui])
        ├── detector/     # Módulo 2: Análisis en lote
        │   ├── window.py, state.py, runner.py
        │   └── pages/    # modelos, imagenes, parametros, gt_manual,
        │                 # ejecutar, resultados, errores, comparar, reporte
        ├── trainer/      # Módulo 3: Entrenamiento YOLO
        │   ├── window.py, state.py, runner.py, hw.py
        │   └── pages/    # modelo, dataset, parametros, augmentacion,
        │                 # entrenar, evaluar, comparar, exportar, informe
        ├── labeler/      # Módulo 4: Anotación YOLO manual
        └── viewer/       # Módulo 5: Inspección con calibración μm/píxel

tests/                    # Pytest (44 módulos + fixtures)
├── english_scan.py       # Scanner de remanentes en español (sin LibCST)
├── test_*_ingles.py      # Guardas de convención de idioma
├── fixtures/             # Fixtures compartidos y baselines del informe
└── formas/               # Casos de prueba geométricos

tools/                    # Herramientas de desarrollo (no se empaquetan)
├── audit/                # Auditoría de traducción + captura de fixtures
├── manual/               # Cadena del manual (capturar, generar, texto ES/EN)
└── packaging/            # PyInstaller (polyx.spec, build.py)

docs/                     # Documentación
├── manual/               # Manual compilado (versionado, offline)
├── assets/               # manual_screenshots/ (es, en, web, win)
├── development/          # diagnóstico, recetas, revisión de arquitectura
└── (LEEME_macOS.md en la raíz)

install/                  # Superficie del usuario final (doble clic)
├── SETUP.bat, iniciar_polyx.bat, Poly-X.vbs   # Windows
├── actualizar.{bat,ps1}, desinstalar.ps1, migrar_instalacion.ps1
├── empaquetar_*.bat
├── Lanzar_macOS.command, actualizar_macOS.command, ...  # macOS
├── setup.sh, run.sh, actualizar_linux.sh, polyx.desktop  # Linux
└── lanzar_polyx.py       # Shim Qt-free

legacy/                   # Versiones anteriores (referencia, no se toca)

models/                   # Pesos YOLO .pt (dev; en instalación van a user data)
runs/                     # Resultados de cada ejecución (gitignored)
data_microplastico/       # Dataset YOLO etiquetado (gitignored)
data/                     # Otros datasets

.github/workflows/        # ci.yml (tests + build smoke), publish.yml (PyPI)
```

---

## 🔧 Technology

| Component | Version | Purpose |
|---|---|---|
| **Python** | 3.11.x | Runtime |
| **PySide6** | 6.7.2 | GUI (Qt) |
| **Ultralytics** | 8.3.40 | YOLO inference + training |
| **OpenCV** | 4.10.0.84 | Image reading/annotation |
| **NumPy** | 1.26.4 | Computations |
| **Pillow** | 10.4.0 | PNG/JPEG processing |
| **Matplotlib** | 3.9.2 | Plots (reports) |
| **pyqtgraph** | 0.13.7 | Live plots (trainer) |

**Hardware:**
- Windows 10/11 · macOS 11 Big Sur or later
- 8 GB RAM minimum
- Acceleration: picked automatically in `src/polyx/core/platform.py` → **CUDA** (NVIDIA) →
  **MPS** (integrated GPU of Apple Silicon Macs) → **CPU**

**Portability.** Everything that differs between operating systems is concentrated in
`src/polyx/core/platform.py`, not scattered across the code:

| Before (Windows only) | Now |
|---|---|
| `os.startfile` (11 sites) | `abrir_en_el_sistema()` → `startfile` / `open` / `xdg-open` |
| `cmd /c start` | `lanzar_actualizador()` |
| `device = "0"` (fixed CUDA) | `dispositivo_disponible()` |
| hardcoded `actualizar.bat` | `nombre_actualizador()` |

> **Intel Macs:** PyTorch stopped publishing x86_64 wheels for macOS with version 2.3;
> the installer pins **2.2.2**, the last one that exists. No MPS (it requires Apple
> Silicon), so it runs on CPU.

> **`.gitattributes` is not cosmetic:** it forces LF on `*.command` and `*.sh`. With
> `core.autocrlf=true` on Windows and without it, the script reaches the Mac with CRLF
> and bash rejects it with `/bin/bash^M: bad interpreter`, a symptom that looks nothing
> like its cause. The `.command` files ship with mode `100755`: without the executable
> bit, double-clicking in Finder does nothing.

---

## 🎨 Themes and palette (src/polyx/gui/theme.py)

**Four themes**, selectable in *Preferences* (or via `POLYX_TEMA`):

| Theme | What it is for |
|---|---|
| `claro` | The usual one. Default, and the one shown in the manual |
| `oscuro` | Working at night and next to UV photos |
| `azul` | Deep blue, a colder accent |
| `contraste` | AAA. For projector and tired-eye viewing |

Each theme is a dictionary of **tokens** in `PALETTES`. On module import the active
one is resolved and its values are published as module attributes, so the rest of
the program keeps writing `T.INK` without knowing the themes exist.

> **That is why `from polyx.gui.theme import INK` is forbidden.** That import keeps
> whatever colour was active at startup and never changes again. Import the module
> (`from polyx.gui import theme as T`) and read `T.INK` at the point of use.

### Tokens

```python
INK / INK2 / INK3 / MUTED   # texto, de más a menos contraste
RULE / RULE_SOFT            # bordes: visible y apenas insinuado
BG                          # superficie elevada: tarjetas, barras, campos
BG_SOFT                     # fondo hundido: el lienzo de la ventana
ACCENT / ACCENT_D           # primario y su hover
ON_ACCENT                   # texto ENCIMA del acento; no siempre blanco
OK / WARN / ERR / VIO       # estado, como FONDO de botón relleno
OK_TX / WARN_TX / ...       # el mismo estado, como TEXTO sobre la tarjeta
FIELD_1 / FIELD_2 / RING    # campo del microscopio del launcher
TIP_BG / TIP_FG             # tooltip
```

**Why each state has two versions.** The tone that gives 5:1 with white text on top
is too dull to be read *as* text on a dark background. A single value cannot do both
jobs: choosing one left half the interface below 4.5:1. In the light themes both
coincide. All text/background pairs across the four themes pass **4.5:1**.

### What does NOT follow the theme

- **`CLASS_COLOR_HEX`** (PET/PP/LDPE) — these are domain data. If they changed with
  the theme, two screenshots of the same analysis would not be comparable.
- **`T.DOC`** — frozen palette (the light one) for the **HTML reports**. A report
  gets printed, attached and archived: if it came out dark because whoever generated
  it works at night, it would waste ink and two reports of the same analysis would
  not look alike. `report_html.py` and `informe.py` use `T.DOC.*`.

### How a theme change is applied

| Where | When |
|---|---|
| Launcher | **Instantly** — it rebuilds itself (`_construir_ui`) |
| Modules already open | They do not change: they are separate processes with the colour baked in |
| Modules opened afterwards | They already come up with the new theme |

It is the same contract as the language. Preferences are stored in
`~/.polyx_tema.json` (theme + animations) and `~/.polyx_idioma.json`.

### Animation

`theme.animaciones()` governs the launcher's microscope panel. Turning it off is not
just aesthetic: continuous movement bothers people with vestibular sensitivity, and
on a machine without a GPU it saves 30 frames per second. With animations off the
panel draws a fixed frame; it does not disappear.

### Icons

`src/polyx/gui/iconos.py` draws **40 icons** with `QPainterPath` on a 24×24 grid, with
the stroke proportional to the requested size.

**Emoji are not used as icons.** An emoji is drawn by the system font: it changes
shape between Windows 10, 11 and macOS, arrives in fixed colour —so it cannot be
tinted with the theme accent— and mixes styles, some flat and some with gradients,
something that shows as soon as four of them sit together in a grid.

```python
from polyx.gui import iconos
lbl.setPixmap(iconos.pixmap("detector", 26, T.ACCENT))
btn.setIcon(iconos.icon("guardar", 15, T.INK2))
```

For a `checkable` button —the sidebar ones— there is
`switchable_icon(nombre, tam, color_off, color_on)`: Qt stores a per-state map
inside the `QIcon` itself, so the active page's icon is tinted with the accent along
with its text, without listening to `toggled`.

Where they are used: Detector and Trainer sidebars
(`SIDEBAR_ITEMS`), page headers (`PAGE_ICON`), card
headers (second argument of `card()`) and the action buttons of the four modules.

> **Typographic marks that are part of a sentence are kept as text:** `✓ ✗ ⚠ → ←`.
> They are not icons, they are punctuation, and replacing them with pictograms
> inside a `QLabel` would force splitting the text into three widgets.

> **Be careful rewriting files with a script.** `Path.write_text()` translates
> `\n` to `\r\n` on Windows: over a file that already came with CRLF, a
> line-by-line sweep inserts a blank line for every line and doubles the file.
> You have to read and write with `newline=""`.

### Light theme values

```python
INK = "#1f2328"          # Texto principal
INK2 = "#424a53"         # Subtítulos
INK3 = "#656d76"         # Muted
ACCENT = "#0969da"       # Botones primarios (azul)
OK = "#1f6b5e"           # Verde (positivo)
WARN = "#9a6700"         # Naranjo (warning)
ERR = "#cf222e"          # Rojo (error)
VIO = "#6639ba"          # Púrpura

# Colores con que se dibujan las cajas de cada clase en pantalla.
# OJO: son colores de interfaz, NO describen la emisión real del Nile Red.
CLASS_COLOR_HEX = {
    "PET": "#e3342f",    # se dibuja rojo
    "PP": "#ff8c00",     # se dibuja naranjo
    "LDPE": "#ffd700",   # se dibuja amarillo
}
```

**Font:** Segoe UI (Windows-native)

### Real fluorescence of each polymer

Do not confuse with the interface colours above. Measured over the interior of
the boxes of the training dataset (mean RGB, n=30 per class):

| Class | R | G | B | Appearance |
|---|---|---|---|---|
| PET | 116 | 58 | 65 | red–salmon |
| PP | 122 | **125** | 32 | **dull yellowish green** |
| LDPE | 181 | 162 | 57 | plain yellow, bright |

**PP and LDPE share hue and are separated by brightness** (R 122 versus 181), plus the
greenish tinge of PP. It is the dominant confusion when annotating and it explains why
per-class recall sinks there: PET 0.98 · PP 0.70 · LDPE 0.54.

The documentation used to say PP was "orange", which does not match the observed
emission. Corrected in August 2026 from the measurement and the author's
observation.

---

## 🚀 Installation and usage

### 🪟 Windows

```batch
SETUP.bat            :: primera vez — crea .venv, detecta GPU, instala todo
iniciar_polyx.bat    :: uso diario
actualizar.bat       :: traer lo nuevo de GitHub
```

### 🍎 macOS

```bash
Lanzar_macOS.command       # instala la 1ª vez, arranca las siguientes
crear_icono_macOS.command  # crea Poly-X.app: abre sin ventana de Terminal
actualizar_macOS.command   # traer lo nuevo de GitHub
construir_app_macOS.command  # empaquetar Poly-X.app (opcional)
```

**Right** click → Open the first time (Gatekeeper blocks scripts
downloaded without Apple's signature).

The installer and launcher are in **a single file** on purpose: on macOS each
`.command` requires its own security approval the first time, and separating them
would mean going through that prompt twice — the second time, with the program
already installed, is when it most looks like something broke.

Open the **Launcher** → pick a module.

---

## 📊 Main modules

### 🏠 Launcher (Module 1)
Main menu: four cards (Detector, Trainer, Labeller, Viewer), theme picker,
update checker (`polyx.gui.updates`), manual/report openers, preferences.

---

### 🔬 Detector (Module 2)
**Purpose:** Batch analysis of images with a trained model.

**Typical flow:**
1. **Models** → Load up to 3 `.pt` models (compared in parallel)
2. **Images** → Pick a folder (recursive)
3. **Manual GT** (optional) → Annotate ground truth manually
4. **Parameters** → conf=0.25, IoU=0.45, um/px=2.5 (calibration)
5. **Run** → Batch inference with progress
6. **Results** → Table (class, conf, size)
7. **Errors** → Confusion matrix + FP/FN gallery
8. **Compare** → Compare runs (P/R/F1 curves)
9. **Report** → Self-contained HTML + browser

**Output:**
- `runs/YYYYMMDD_HHMMSS/images/*.png` — Annotated images
- `runs/.../centroids.csv` — class, x, y, conf, diam_um
- `runs/.../metrics.json` — TP/FP/FN per class
- `runs/.../report.html` — Paper-quality report (base64)

### 🎯 Trainer (Module 3)
**Purpose:** Train YOLO v8/v11 models from scratch or fine-tune.

**Flow:**
1. **Model** → YOLO v8/v11, size (nano, small, medium, large, xlarge)
2. **Dataset** → Dataset.yaml (train/val/test split)
3. **Parameters** → Epochs, batch, imgsz, optimizer (SGD, Adam)
4. **Augmentation** → Flip, rotation, mosaic, HSV, etc.
5. **Train** → Launch training + live loss curves
6. **Evaluate** → mAP, P, R, F1 per class (validation)
7. **Compare** → Compare multiple runs
8. **Export** → PT → ONNX/TensorRT/CoreML
9. **Report** → Training report

**Output:**
- `runs/detect/trainN/weights/best.pt` — Best model
- `runs/detect/trainN/results.csv` — Per-epoch metrics
- `runs/detect/trainN/confusion_matrix.png` — Confusion matrix

### 🏷 Labeller (Module 4, working)
Manual YOLO annotation with optional automatic pre-labelling.

**Verified with 552 crops** (August 2026). Startup 0.25 s, interface
responds in 0.06 s.

**Progress tracking.** The list distinguishes three states, and it is recovered from
disk when the folder is opened, so the count can be spread over several sessions:

| Mark | Meaning |
|---|---|
| `✓ name (n)` | reviewed, with n particles |
| `· name (0)` | reviewed, no particles — **that is data** |
| `○ name` | not yet reviewed |

A `.txt` file is written **only** when the image is marked as reviewed or when a box
is drawn. Skimming past does not create a file: recording an image that was barely
glanced at as "reviewed with zero" would falsify the census.

**Shortcuts:** `Espacio` marks reviewed and moves on · `Tab` jumps to the next
unreviewed · `F` reframes · `1-9` active class · `←/→` navigate · `Supr` deletes the
selected box · `Ctrl+Z/Y` undo/redo · wheel zoom · middle-button pan.

**Minimum box side: 2 px** (`BboxCanvas.SIDE_MINIMUM_PX`). It was 5 px and it silently
discarded legitimate marks: the study's smallest particles measure ~8 px per side.
Now, if a box is rejected, the status bar says so.

Zoom is kept between images (checkbox in the right panel), with size tolerance:
crops from a grid differ by 1 px due to rounding, and demanding exact equality
used to lose the zoom on every change.

### 📐 Viewer (Module 5, working)
Single-image inspection with μm/pixel calibration, detection and export.

**Interactive calibration.** Two modes: **line** (2 clicks + real length) and
**circle** (3 clicks over a circular edge + real diameter). The circle dialog
ships with 100000 μm as default, which is the study's Petri dish.

> It was unusable until August 2026: `QInputDialog.getDouble()` was called
> with `min=`/`max=`, which PySide6 does not accept as keywords, and it raised
> `AttributeError` when the points were completed. The maths was always right;
> the dialog blew up before reaching it. **The arguments must go
> positional:** `(parent, title, label, value, minValue, maxValue, decimals)`.

**Detection.** Configurable `imgsz` (320–8192, default 2080) and GPU when
available. It used to force `device="cpu"` and default `imgsz` 640: with
particles of ~12 px in 4096 px photos, at 640 they collapse to ~2 px and **nothing
is detected**. If the GPU runs out of memory, it explains how to lower `imgsz`
instead of showing the raw exception.

**Load labels (.txt).** Shows the YOLO annotations that accompany the
image, with sizes already converted to μm. Useful for reviewing the manual count
over the full plate without going back to the Labeller.

**Export** to `visor_<imagen>_<ts>/`: annotated PNG, `detecciones.csv`
(one row per particle, with diam_px, diam_um and area_um2) and `resumen.json`.

> Image reading and writing with `cv2.imdecode`/`imencode` over
> `np.fromfile`, not `imread`/`imwrite`: on Windows these fail with paths that
> carry accents and return `None` without warning.

---

## 🔑 Key classes

### `Detection` (src/polyx/core/detect/yolo.py)
```python
@dataclass
class Detection:
    class_id: int           # 0=PET, 1=PP, 2=LDPE
    class_name: str         # "PET", "PP", "LDPE"
    conf: float             # 0-1, confianza
    x1, y1, x2, y2: float   # Bbox píxeles (absolutos)
    diam_um: Optional[float]  # Diámetro del círculo de igual área REAL
    area_um2: Optional[float] # Área de la partícula, no de la caja
    largo_um: Optional[float] # Dimensión mayor, siguiendo la curva
    ancho_um: Optional[float]
    aspecto: Optional[float]    # largo/ancho reales
    curvatura: Optional[float]  # 1.0 = recta; ≥1.15 = curva
    morfotipo: Optional[str]    # "fibra" | "fragmento"
```

**The box does not describe the shape.** Measured over 7.129 annotated particles: the box
overestimates the area **1.87×**, and a fibre lying diagonally has a **square**
box — which is why it was reported as a fragment.

`src/polyx/core/measure/morphology.py` segments inside each box (Otsu on the highest-contrast
channel, the central connected component, clipped to the box) and measures over the
mask. **The length comes from two standard measures and the larger one is reported:**

| Measure | What it is | Fails on |
|---|---|---|
| **Maximum Feret** | greatest distance between two boundary points | curved fibres (it gives the chord: −34.6 % on a 180° arc) |
| **Geodesic diameter** | longest path *inside* the particle | thick particles (it rounds the concavities) |

The geodesic is used only if the particle meets **two** conditions, and each one
comes from an observed failure:

| Condition | Constant | Why |
|---|---|---|
| **Slender**, `largo/grosor ≥ 4` | `THINNESS_FOR_GEODESIC` | on a thick particle the path rounds the concavities: a real 44 px clump got 73 |
| **Non-convex**, `solidez < 0.90` | `SOLIDITY_CONVEX` | on a convex shape geodesic and Feret coincide by definition; a rotated bar measured 200 px at 0° and 45° but **208 at 15°, 30° and 60°** |

Against synthetic shapes of known size: **median error 0.6 %, worst case
4.7 %**. Pinned in `tests/test_morfologia.py`.

**The equivalent rectangle `L = (P + √(P²−16A))/4` is NOT a size** and must not
be used as one. It depends on the perimeter, so a jagged edge inflates it
(+22.5 % on a straight line with saw teeth), and it is only defined if P² ≥ 16A
—a circle gives P² = 4πA and the discriminant comes out negative—. It is kept in
`largo_rect_eq` as a descriptor: compared with the other two it betrays irregular
edges.

> **August 2026 correction.** The material was reported to have "22 % of fibres
> and aspect up to 21.1". It was an artefact of using `rect_eq` as length. Measured
> with Feret/geodesic over 6.638 particles: **median aspect 1.58, maximum 8.7,
> and 1.1 % of fibres**. The material is mostly compact fragments.

Cost 5.9 ms per particle. Only cv2 and numpy: `scipy` is **not** in
`requirements.txt` —it arrives as a transitive dependency of ultralytics— and
relying on it would break the install the day ultralytics stops shipping it.

### `YoloModel` (src/polyx/core/detect/yolo.py)
Lazy-loading wrapper for Ultralytics YOLO:
```python
model = YoloModel("bestdetectormedium.pt")
detections = model.predict("image.jpg", conf=0.25, device="0")
```

### `DetectorState` (src/polyx/gui/detector/state.py)
Global state of the Detector (Qt signals):
```python
state.model_slots: List[ModelSlot]      # Hasta 3 modelos
state.images: List[Path]                # Imágenes a analizar
state.params: InferenceParams           # conf, IoU, um_per_px
state.results: Dict[int, List[ImageResult]]  # Resultados por modelo
```

### `ImageResult` (src/polyx/gui/detector/state.py)
Result of one image:
```python
image_path: Path
predictions: List[Detection]
gt: List[Detection]             # Ground truth (si existe)
tp, fp, fn, miscls: int
annotated_png: Optional[bytes]  # PNG anotado en base64
```

### `MatchResult` (src/polyx/core/quality/metrics.py)
Prediction vs GT matching (IoU ≥ 0.5):
```python
tp: int          # True Positives (conf y clase correcta)
fp: int          # False Positives (sin GT)
fn: int          # False Negatives (GT sin pred)
miscls: int      # Bien localizado pero clase incorrecta
```

---

## 🔄 Data flow

### Detector
```
Imágenes (carpeta)
    ↓
YoloModel.predict() × N modelos
    ↓
Detection[] (bbox + clase + conf)
    ↓
read_yolo_txt() → Ground truth (si existe)
    ↓
match_image() → TP/FP/FN (IoU ≥ 0.5)
    ↓
compute_box_size_um() → diam_um (si um_per_px calibrado)
    ↓
ImageResult
    ↓
cv2.rectangle() + anotación → PNG
    ↓
runs/YYYYMMDD_HHMMSS/
```

### Trainer
```
Dataset (images/ + labels/ YOLO)
    ↓
YOLO(model_name).train(epochs=..., batch=...)
    ↓
Ultralytics training loop
    ↓
best.pt (checkpoint óptimo)
    ↓
results.csv + curves
    ↓
runs/detect/trainN/
```

---

## 📝 Conventions

- **English:** Comments, docstrings and identifiers in English. Spanish
  survives only inside the `tr()` layer and the UI strings it wraps.
- **Spanish:** Runtime UI strings and `tr()` dictionary keys remain Spanish;
  they are the translation layer, not developer-facing text.
- **Qt Signals:** `models_changed`, `run_started`, `run_finished`, etc.
- **Paths:** Always pathlib `Path`, never strings
- **Colors:** Use `T.ACCENT`, `T.OK`, etc., never hardcodes. As
  **text**, the `_TX` variant (`T.OK_TX`); as **background** of a filled
  button, the base one. In HTML reports, always `T.DOC.*`
- **Methods:** `_` private (widget, signal callback)

---

## 🔗 Key dependencies

- **Ultralytics YOLO** → Inference, training, export
- **OpenCV** → `cv2.rectangle()`, annotation, image reading
- **NumPy** → Bbox, IoU computations
- **Matplotlib** → Plots in reports (in Agg mode, non-interactive)
- **PySide6/Qt** → GUI (signals/slots, QObject)

---

## ⚙️ Common parameters

### Inference (DetectorState.params)
```python
conf: float = 0.25          # Umbral confianza
iou_nms: float = 0.45       # IoU para NMS
iou_tp: float = 0.50        # IoU para análisis de errores
imgsz: int = 640            # Tamaño entrada red
device: str = "0"           # "0" (GPU), "cpu"
um_per_px: float = 0.0      # Calibración (0 = sin calibrar)
size_min_um: float = 0.0    # Filtro inferior (0 = sin filtro)
size_max_um: float = 0.0    # Filtro superior
conf_asignacion: float = 0.0  # Confianza mínima para atribuir polímero (0 = off)
```

### Abstention: detecting ≠ identifying

`src/polyx/core/quality/assignment.py`. Below `conf_asignacion` the particle **still
counts** as a detected particle, but it is reported as **"unassignable"** instead
of attributing a polymer to it. The report declares in Methods what percentage of
the batch went unassigned.

The reason is chemical, not engineering: Nile Red is **solvatochromic** and its
emission responds to the **polarity** of the environment, not to the polymer's
identity. PET —a polyester— separates cleanly (recall 0.98); PP and LDPE are the
two apolar polyolefins, they share hue and differ only in brightness, which
depends on exposure, focus and thickness. That is why their recall sinks to
0.70 and 0.54.

**It is not a model class, and it must not become one.** A trained `unknown`
class is defined by the doubt of whoever annotates and not by what is seen: the
same particle would end up LDPE one day and unknown the next, and that noise
would land exactly on the two already weakest classes.

**It does not use the margin between the two most probable classes either**, which
would be a better criterion: after NMS, YOLO returns a single confidence and a
single class per box, and the scores of the rest are lost there. Recovering them
requires intercepting the raw head tensor, fragile across ultralytics versions.

### Training
```python
epochs: int = 100
batch: int = 16             # Menor si OOM
imgsz: int = 640            # Menor si OOM
device: str = "0"
optimizer: str = "SGD"      # o "Adam"
lr0: float = 0.01
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| "No module named tkinter" | Reinstall Python, tick tcl/tk |
| "CUDA not available" | Normal, CPU only works (slower) |
| Detector finds nothing | Lower conf to 0.10 in Parameters |
| OOM during training | Lower batch to 4-8, imgsz to 512-576 |
| Report does not open | Check docs/manual/Manual_PolyX.html manually |

---

## 📚 Automatically generated files

- **docs/manual/Manual_PolyX.html / Manual_PolyX.en.html** (+ their PDFs) —
  built from scratch by `tools/manual/generar_manual.py`: text, figures,
  index and stylesheet. Generated and not versioned.
- **tools/manual/manual_texto_es.py / manual_texto_en.py** — the manual's
  text. Content only; the engine is in `tools/manual/generar_manual.py`.
- **docs/assets/manual_screenshots/** — screenshots the manual embeds, one
  folder per source: `es/` and `en/` (interface, one language each), plus
  `web/` and `win/`.

To regenerate (screenshots first, manual after):
```bash
.venv\Scripts\python.exe tools\manual\capturar_manual.py --salida docs/assets/manual_screenshots/es
set POLYX_IDIOMA=en && .venv\Scripts\python.exe tools\manual\capturar_manual.py --salida docs/assets/manual_screenshots/en
.venv\Scripts\python.exe tools\manual\generar_manual.py --pdf
```

> The previous generator replaced figures *by position* inside an already written
> HTML. It was retired in September 2026 —it lives in the git history— because
> with the current manual it would misplace the images without warning.

---

## 🔮 Next improvements (under construction)

- ✅ Detector: Working, HTML report
- ✅ Trainer: Working, live curves
- ✅ Labeller: Working, verified with 552 crops (Aug 2026)
- ✅ Viewer: Working, calibration repaired and verified (Aug 2026)
- 📋 Dataset validation (duplicates, outliers)
- 🤖 Automatic parameter recommendations

---

## 📞 Contact

**Cristofher Ferrada**  
PhD (c) in Chemistry — Pontificia Universidad Católica de Valparaíso  
2026

**Repo:** https://github.com/CrissFerrada/Poly-X-Microplastics
