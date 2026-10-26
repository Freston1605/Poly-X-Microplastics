# Poly-X — Microplastics detection suite

> **Automated detection and classification of microplastics (PET, PP, LDPE)
> by Nile Red fluorescence under UV light (254 nm), using YOLO v8/v11 models.**

**Author:** Cristofher Ferrada · PhD (c) in Chemistry · PUCV · 2026
**Version:** 2.0.0 · **Windows 10/11 and macOS** · Python 3.11

---

## Installation

### From PyPI (headless engines + CLIs)

```bash
pip install polyx-microplastics            # core + CLIs: polyx-train, polyx-detect, polyx-eval, polyx-report
pip install polyx-microplastics[gui]       # + the Qt GUI (Launcher, Detector, Trainer, Labeller, Viewer)
polyx-download-models                      # fetch the default .pt weights into the user data dir
```

Requires **Python 3.11**. The headless engines and report generation work
without Qt; the GUI needs PySide6, which the `[gui]` extra pulls in. Model
weights are not bundled (~50 MB) and are fetched by `polyx-download-models`.

### From source (desktop app, Windows/macOS)

Download the repository, then run `install/SETUP.bat` (Windows) or
`install/Lanzar_macOS.command` (macOS). They create the virtual environment,
install the pinned stack and, on Windows, offer a Desktop shortcut. Full
step-by-step instructions (with photographs) are in
`docs/manual/Manual_PolyX.html`.

---

## Modules

| # | Module | Status | What it does |
|---|---|---|---|
| 1 | 🏠 **Launcher** | ✅ Working | Main menu — opens each module in its own window |
| 2 | 🔬 **Detector** | ✅ Working | Batch analysis with YOLO · paper-quality HTML report |
| 3 | 🎯 **Trainer** | ✅ Working | YOLO v8/v11 training with live curves |
| 4 | 🏷 **Labeller** | ✅ Working | Interactive YOLO annotation with automatic pre-labelling |
| 5 | 📐 **Viewer** | ✅ Working | Single-image inspection, calibration and particle-by-particle review |

---

## Features

### 🔬 Detector (batch analysis)
- Loads up to **three .pt models** at once for side-by-side comparison
- Runs over folders of images with a progress bar
- Error analysis: **true positives / false positives / false negatives /
  misclassified**, confusion matrix, gallery of the worst cases
- **Automatic calibration against the Petri dish**: the rim is found on its own
  and its known diameter sets the µm/px **of each photo**, with nothing to mark
  by hand. This matters because the shooting distance varies between takes: in
  this study's material the real scale ranges from 31 to 50 µm/px, a factor of
  1.6, so a single value for the whole batch would give sizes off by as much as
  50 %
- **Size and shape measured on the particle, not on its box**: length, width,
  area, aspect ratio and **fibre / particle** classification. The box of an
  elongated particle is nearly empty and depends on how it happened to land — a
  fibre lying diagonally has a square box — so across 7,129 annotated particles
  it overestimated area by **1.87×**
- **Size distribution histogram** per class (PET/PP/LDPE) and by size band,
  stacked by polymer
- **CSV export** straight from the results page
- **Drag and drop** for `.pt` models and images or folders
- **Adjustable inference resolution** with presets (Fast 1280 · Balanced 2560 ·
  Maximum detection 4096) and a button that measures what your GPU can take
- Self-contained HTML report (images embedded as **base64**, paper quality) with
  a **side-by-side prediction vs ground truth gallery**. Gallery images are
  re-encoded and capped in number so the file stays openable in a browser; the
  metrics still cover every image
- **PDF export** of the report in one click, ready to email
- **Selectable report sections**: thirteen tick boxes and three presets (Full ·
  Short summary · Methodological). Unticking renumbers the sections **on its
  own** and adjusts the table of contents; a ticked section with no data is left
  out anyway
- **Calibration section**: where each photograph's scale came from, its minimum /
  median / maximum, the **mean with a 95 % confidence interval** and a figure over
  a real dish with the fitted circle and its diameter drawn on it
- **Size per folder and per photograph** (optional): compares the size
  distribution across folders — each folder as a sampling site, a station or a
  condition — with box plots and a **Kruskal-Wallis** test
- **Depth profile of the core** (optional): when photographs are named
  `interval.core`, the interval stops being an arbitrary folder and becomes an
  **ordered** variable. The report draws the profile with depth on the vertical
  axis — the convention for any sediment core — and answers whether the **number
  of particles** and the **median size** rise or fall with depth, using
  **Spearman** with a **permutation** p value. The correlation runs over the
  intervals, not the particles: two particles from the same dish are not
  independent observations of depth
- **Measured particle cards**: the **6 fibres and the 6 largest particles**, each
  with its crop beside it and the measurement drawn on top (Feret in yellow,
  geodesic in magenta, mask in green). The split is deliberate: fibres are a
  minority and are exactly where the geodesic method comes into play
- **The report comes out in the application's language**, Spanish or English:
  headings, tables, figure captions and the methods prose
- **Selectable report scope**: the whole job, only the photos you tick, or both
  at once. Figures, charts and the confusion matrix are recomputed over what you
  chose, so the report always describes the photos it shows
- **Real model comparison** inside the report: a photo-by-photo table with each
  model's detections and, when ground truth exists, their TP/FP/FN and overall F1
- **Automatic tiling of large photographs**: above a threshold the photo is
  analysed in overlapping tiles, because at full resolution the particles fall
  below the network's stride and vanish. Boxes are mapped back to the original
  photo's coordinates and overlaps merged with NMS, so **results and report are
  always delivered on the whole photo, never on the tiles**
- The Run tab warns beforehand whether the batch will be tiled and how long it
  will take

### 🎯 Trainer
- Supports **YOLO v8 and v11**, sizes nano → xlarge
- Loads `data.yaml` with automatic train/val/test split detection
- **Dataset validation** with ✓/✗ indicators (images, labels, classes)
- Live loss curves (box loss, mAP, precision, recall)
- Configurable augmentation (flip, rotation, mosaic, HSV)
- Export to **ONNX / TensorRT / CoreML**
- Explicit GPU/CPU diagnosis when training starts
- **Train v8 and v11 with the same configuration** from a single tick box: both
  runs reuse identical `imgsz`, `batch`, epochs, seed and augmentation, which is
  what allows attributing the difference in metrics to the architecture and not
  to the hyperparameters. They run in sequence because they share the GPU, and
  when they finish the log summarises the comparison
- **Dataset audit** before training: class distribution, format and validation,
  with a verdict on whether the dataset is fit to train on
- **`best_real.pt`: the weight that works on your photographs.** Ultralytics
  keeps `best.pt` according to the mAP of the full validation set, and in a
  mixed dataset that number is dominated by the doped laboratory plates (in the
  Loa dataset, 1,191 laboratory boxes against 47 of real sediment). When
  training ends, Poly-X evaluates every checkpoint against **real sediment
  only** and keeps the winner apart. `best.pt` is preserved untouched

### 🏷 Labeller
- Interactive canvas: **drag** to draw normalised YOLO boxes
- Right-click to assign the class; keys `1`–`9` to switch the active class
- **Undo/Redo** (Ctrl+Z / Ctrl+Y) per image
- **Progress tracking**, designed for campaigns of hundreds of images spread
  over several sessions. The state is recovered from disk when the folder is
  opened and the list distinguishes three cases:

  | Mark | Meaning |
  |---|---|
  | `✓ name (n)` | reviewed, with n objects |
  | `· name (0)` | reviewed, zero objects — **that is a data point** |
  | `○ name` | not yet reviewed |

  A `.txt` is written **only** when the image is marked as reviewed or a box is
  drawn: skipping over an image creates no file, because recording a barely
  glanced image as "reviewed with zero" would falsify a census count.
- `Space` marks as reviewed and moves on · `Tab` jumps to the next unreviewed
  image · `F` reframes · `←/→` navigate · `Del` deletes the selected box
- **The zoom is preserved** between images (tick box in the right-hand panel)
- **Minimum box side: 2 px.** When a box is rejected, the status bar says so
  instead of discarding it in silence
- **Automatic pre-labelling** with a `.pt` model, with adjustable `conf` and
  `imgsz` and GPU when available
- **Silent auto-save** every 60 seconds; thumbnails are generated in the
  background so the window responds immediately
- Writes the YOLO `.txt` next to the image (or in `labels/` if it is in
  `images/`) and generates `classes.txt` in the root and in every subfolder

### 📐 Viewer
- Opens a single image, or steps through a folder with `← →`
- **Interactive µm/pixel calibration** in two modes:
  - 📏 **Line** (2 clicks): mark a known reference, type its real size
  - ⭕ **Circle** (3 clicks): mark 3 points on a circular edge, type the real
    diameter (handy with Petri dishes, whose diameter is known)
- The status bar shows `📐 0.4880 µm/px (line)` live
- Detection with a loaded `.pt` model, with **configurable inference resolution**
  (320–8192) and GPU when available. With tiny objects in high-resolution photos
  `imgsz` is decisive: at low values the particles fall below the network's
  stride and nothing is detected
- **Load `.txt` labels**: shows existing annotations over the image with their
  sizes converted to µm. Useful for reviewing a manual count without going back
  to the Labeller
- **Particle-by-particle review**: the table lists every particle with its
  **number**, class, type (fibre or fragment), length, width and aspect ratio.
  Selecting a row shows **what it was measured on**: the crop without marks on
  the left and, on the right, the mask outline with the measurement drawn over it
  — the Feret line in yellow, the geodesic path in magenta — plus the full
  pixels-to-micrometres arithmetic. A size you cannot see being measured cannot
  be verified
- **Load predictions from a closed run** in `runs/detect_.../`, without running
  the model again. It always opens the original photo and never the annotated
  PNG, which has the boxes painted onto it
- **Drag and drop** an image or a `.pt` model straight onto the canvas
- Exports: annotated image + `detecciones.csv` + `resumen.json`

---

## How a particle's size is measured

The general criterion is **the longest straight line that fits inside the
particle**, that is the greatest distance between two points of its outline: the
*maximum Feret diameter*. It does not depend on the orientation the particle
happened to land in, and a jagged edge does not disturb it.

That straight line stops working when the particle is **contorted**: in a bent
fibre the distance between the ends is the chord, and on a half-circle arc it
falls 35 % short. For those cases the *geodesic diameter* is measured instead —
the longest path that fits **inside** the particle, which cannot leave the mask
and therefore follows the curve.

| Particle shape | What is reported as length |
|---|---|
| Compact or irregular, but not bent | Maximum Feret — the longest straight line |
| Elongated and contorted (fibre) | Geodesic diameter — follows the curve |

The geodesic is only used when the particle is **thin** (length ≥ 4 × thickness)
and **non-convex** (solidity < 0.90). Without the first condition, any concavity
makes the path go around the particle instead of through it; without the second,
the length would start depending on the rotation angle, which is precisely the
defect this was meant to remove.

Against synthetic shapes of known size — straight bars, rotated bars, arcs of
60°, 120° and 180°, a circle, a bar with a jagged edge and a blob with a notch —
the length measured this way gives a **median error of 0.6 % and 4.7 % in the
worst case**. This is pinned down in `tests/test_morfologia.py`.

> **The equivalent rectangle is not a size.** The formula
> *L* = (*P* + √(*P*²−16*A*))/4 gives the length of a rectangle with the same
> area and the same perimeter, which is a different thing. It depends on the
> perimeter, so a jagged edge inflates it by 22.5 %, and it is undefined for
> compact particles, where *P*² < 16*A*. It is reported as a descriptor, because
> compared against the other two it exposes irregular outlines, but it is not
> used as a size.

### From pixels to micrometres: the scale

Everything above is measured in **pixels**. Converting to micrometres is not one
factor for the whole batch: it depends on the shooting distance, and in this
study's material the real scale ranges from 31 to 50 µm/px, a factor of 1.6. So
**each photograph is calibrated on its own**, against the Petri dish's rim: the
approximate centre is located with Hough, the edge is sampled in **720
directions** and a circle is fitted by least squares with outlier rejection.
Hough's radius is not used, because it can be off by 12 % and that error would go
straight into every size.

> **Which edge the 100 mm are, and which way it can be wrong.** The rim has a wall
> about **2 mm** thick — measured on this study's photographs: the inner edge falls
> at 0.960 of the fitted radius and the outer edge at 1.000. The nominal diameter
> of a Petri dish is ambiguous at that level: it may refer to the **outer** one or
> to the **usable inner** one. Here the **outer** is taken, since that is the edge
> the circle is fitted to. If the nominal figure referred to the inner edge, the
> correct scale would be **4.2 % larger** and every size would be
> **underestimated** by that amount. The bias can only run in that direction,
> because the outer edge is the larger of the two. This is stated in the report
> itself.

**Touching particles.** Two particles in contact form a single blob, and
measuring them together would add their sizes up. They are separated by
*watershed* on the distance transform: the centre of each one sits far from the
background while the neck joining them sits close, so the cut falls on the neck.
Against circles of known size it separates them up to **27 % overlap of the
diameter**, without splitting any particle that is a single piece.

**A fibre is measured whole.** The mask is not clipped to the detector's box
when the particle is elongated, nor passed through the touching-particle splitter:
both of those cut real fibres short. On the one that exposed it, 53 % of the
length was lost — 369 px of connected component came out as 174 — and an
underestimated size does not show in the figures, only in the image. It is pinned
down in `tests/test_fibra_no_se_trunca.py`.

**Stated limitations.** Two particles overlapping by more than 40 % of their
diameter are still measured as one: past that point there is no neck left to cut
along. And in a tightly coiled fibre the geodesic path cuts the corner at each
bend, underestimating by up to 19 % in the tightest case tested.

---

## Polymer classes

| ID | Class | Observed fluorescence (Nile Red, UV) | Interface colour |
|---|---|---|---|
| 0 | **PET** | Red–salmon | 🔴 `#e3342f` |
| 1 | **PP** | **Dull yellow-green** | 🟠 `#ff8c00` |
| 2 | **LDPE** | Bright, clear yellow | 🟡 `#ffd700` |

The interface colours are there to tell the boxes apart on screen and **do not
describe the real emission**. Measured inside the training-set boxes (mean RGB,
n = 30 per class): PET 116/58/65 · PP 122/125/32 · LDPE 181/162/57.

**PP and LDPE share a hue and separate by brightness** (R 122 against 181), plus
PP's greenish cast. That is the dominant confusion when annotating, and it
explains why per-class recall collapses there: PET 0.98 · PP 0.70 · LDPE 0.54.

---

## Language

The interface is available in **Spanish and English**. The selector sits in the
Launcher's top bar and the choice is remembered between sessions. The first time,
the system language is used; `POLYX_IDIOMA=en` forces it without touching the
interface.

Modules are separate processes and read the language when they open, so the
change takes effect as soon as you open the next module.

**The detection report comes out in the chosen language**, not just the
interface: headings, tables, figure captions, chart axes and the methods prose.
The HTML `lang` attribute follows too, so the browser's spell checker and screen
readers treat it correctly.

To see what is still untranslated:

```bat
.venv\Scripts\python.exe tools\audit\audit_translations.py
```

It walks the syntax tree of every module looking for `tr()` calls and checks them
against the dictionary. It must report **0 untranslated**. With `--list` it
also prints the strings that are missing.

## Themes

Four themes, under **Preferences** (top-bar button, or `Ctrl+,`):

| Theme | What for |
|---|---|
| **Light** | The original one. Default |
| **Dark** | Working at night, and next to UV photographs |
| **Midnight blue** | Deep blue, with a cooler accent |
| **High contrast** | Meets AAA. For projectors and for tired eyes |

The theme applies **immediately in the launcher**, which rebuilds itself.
Modules already open keep theirs — they are separate processes, with the colours
baked into their stylesheet — and any you open afterwards come up with the new
one. Same contract as the language.

`POLYX_TEMA=oscuro` forces it without touching the interface.

> **HTML reports deliberately do not follow the theme.** A report gets printed,
> emailed and filed: turning it dark because whoever produced it prefers working
> at night would waste ink and make two reports of the same analysis look
> unalike.

Every text/background pair in all four themes clears the **4.5:1** WCAG asks for
normal text.

### Motion

The launcher's microscope panel is animated: particles drift and a sweep marks
them with detection boxes — which is what the program does. Under
**Preferences → Motion** it can be turned off, leaving the panel on a still
frame, if the motion bothers you or the machine is struggling.
`POLYX_SIN_ANIMACION=1` does the same from the environment.

---

## Requirements

| Component | Windows | macOS |
|---|---|---|
| System | 10 / 11 | 11 Big Sur or later |
| Python | **3.11.x** (not 3.12+) | 3.9 or later |
| RAM | 8 GB minimum | 8 GB minimum |
| Acceleration | NVIDIA GPU optional (20–30× faster for training). The installer picks the CUDA build that matches the card | **Apple Silicon:** built-in GPU via MPS. **Intel:** CPU only |

Main dependencies: `PySide6 6.7`, `Ultralytics 8.3`, `OpenCV 4.10`, `NumPy 1.26`,
`Matplotlib 3.9`

> **Intel Macs:** PyTorch stopped shipping builds for Intel processors from 2.3
> onwards, so the installer pins **2.2.2**, the last release with x86_64 support.
> It works correctly, but without GPU acceleration: expect about 1 minute per photo
> on tiled batches.

---

## Installation

Clone or download the repository, then follow the instructions for your system.

### 🪟 Windows

Run once:

```bat
SETUP.bat
```

It creates `.venv`, detects the GPU and installs PyTorch, Ultralytics and
PySide6. Nothing else has to be installed by hand.

In full, the installer:

1. **Asks where to install Poly-X** (press ENTER to install into the same
   folder, or type another path).
2. Detects Python 3.11 and creates the `.venv` environment.
3. Detects the GPU and downloads PyTorch with **the CUDA version matching that
   card**. RTX 50xx cards (Blackwell, `sm_120`) need CUDA 12.8: with the CUDA
   11.8 wheels `torch.cuda.is_available()` returns `True` and the failure only
   shows up when training starts, as *"no kernel image is available for
   execution on the device"*.
4. Installs the remaining dependencies, verifies that everything imports and
   **checks that PyTorch ships kernels for your specific GPU**.
5. Registers the installed version, so the launcher can warn about new ones.
6. **Looks for earlier Poly-X installations** on the machine. If it finds any,
   it shows what data they hold, migrates it to the new one and sends the old
   folder to the **Recycle Bin** (see below, under Updating).
7. Offers to **create a "Poly-X" Desktop shortcut** and **tells you how to
   start it**.

### 🍎 macOS

**Right-click** `Lanzar_macOS.command` → **Open** → **Open**.

That single file does both jobs: the first run installs everything (10–15 min),
later runs just start the app. There is deliberately no separate installer — on
macOS every `.command` file needs its own security approval the first time, and
keeping it to one file means that only happens once.

The installer detects whether the Mac is Apple Silicon or Intel, picks the matching
PyTorch build, verifies that everything imports, and offers to create a Desktop
shortcut.

> #### ⚠️ "cannot be opened because it is from an unidentified developer"
>
> **This is expected and does not mean anything is broken.** macOS blocks any
> downloaded script that is not signed with an Apple developer account (USD 99/year).
>
> **Fix:** **right-click** → **Open** → confirm **Open**. Only needed the first time;
> a normal double-click works afterwards. If it still refuses, in Terminal:
> `xattr -d com.apple.quarantine Lanzar_macOS.command`

Full details in **[LEEME_macOS.md](LEEME_macOS.md)**.

---

### Opening without a console window (Windows)

Double-click **`Poly-X.vbs`** (project root): it opens the Launcher directly,
with no black window. It resolves its own location, so it works from any folder —
just download the repository and double-click, no path editing needed.

To keep it at hand, right-click `Poly-X.vbs` → *Send to* → *Desktop (create
shortcut)*, and give it the `assets\polyx.ico` icon.

`iniciar_polyx.bat` does the same but showing the console, useful for seeing
error messages if something fails at startup.

> **Important:** the trained models (`*.pt`) are **not** included in the GitHub
> download because of their size. Copy your `.pt` file into the `models\` folder
> to use the Detector and the Viewer.

> Requires **Python 3.11.x** (not 3.12+). If you do not have it, download it from
> [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/)
> ticking *"Add Python to PATH"*.

## Usage

Double-click the **Poly-X** Desktop shortcut, or `iniciar_polyx.bat` (Windows) /
`Lanzar_macOS.command` (macOS).

That opens the **Launcher** → pick a module. Or straight from a terminal:

```bat
REM Windows
.venv\Scripts\python.exe -m polyx.launcher
.venv\Scripts\python.exe -m polyx.gui.detector
.venv\Scripts\python.exe -m polyx.gui.trainer
.venv\Scripts\python.exe -m polyx.gui.etiquetador
.venv\Scripts\python.exe -m polyx.gui.visor
```

```bash
# macOS
.venv/bin/python -m polyx.launcher
.venv/bin/python -m polyx.gui.detector
.venv/bin/python -m polyx.gui.trainer
.venv/bin/python -m polyx.gui.etiquetador
.venv/bin/python -m polyx.gui.visor
```

---

## Updating

To pull the latest version published on GitHub **without reinstalling anything**,
double-click:

| System | File |
|---|---|
| 🪟 Windows | `actualizar.bat` |
| 🍎 macOS | `actualizar_macOS.command` |

It checks whether there is a new commit on `main`; if there is, it downloads and
replaces only the program files. It **keeps** your `.venv` environment, your
`models/*.pt`, your `runs/` and any local data. Git does not need to be installed
(it downloads over HTTPS).

Each updater protects itself while running, but **does update the other platform's
files**: whichever one you run, the project stays complete for both.

### Automatic update notice

There is no need to remember to run it: when the launcher opens, it checks in the
background whether GitHub is ahead and, if so, shows a button with the new
version's identifier. The check does not slow the startup down and stays silent
on any failure — without internet or with GitHub down, the notice simply does not
appear.

### If the machine already had Poly-X

`SETUP.bat` looks for other installations at the end of the process. If it finds
any, it shows what they contain — models, trainings, detections, datasets, with
their sizes — and asks for confirmation before touching anything.

The order is **copy → verify → remove**, never the other way around:

- The data the new installation does not have are copied. **No existing file is
  overwritten.**
- If anything fails while copying, the old folder is left intact and a warning
  is shown.
- Only if everything went well does the old folder go to the **Recycle Bin**. It
  is not deleted: if something is missing, it can be restored.

Folders containing `.git` are always skipped: a development repository is not an
old installation, and removing it would take unversioned work with it.

---

## Project layout

```
polyx/
├── launcher.py          # Main menu
├── core/                # Shared modules (theme, yolo_wrap, metrics, report_html,
│                        #   calibracion, morfologia, procedencia, i18n, plataforma)
├── detector/            # Module 2: batch analysis (9 pages)
├── trainer/             # Module 3: YOLO training (9 pages)
├── etiquetador/         # Module 4: interactive annotation
└── visor/               # Module 5: inspection + µm/px calibration
models/                  # Trained .pt weights
runs/                    # Results of each run
data_microplastico/      # YOLO dataset (images/ + labels/)
tests/                   # Test suite for measurement, calibration and portability

SETUP.bat                # 🪟 Installer
iniciar_polyx.bat        # 🪟 Launcher
actualizar.bat           # 🪟 Updater
Lanzar_macOS.command     # 🍎 Installer + launcher (both in one)
crear_icono_macOS.command    # 🍎 Icon that opens with no Terminal window
actualizar_macOS.command # 🍎 Updater
construir_app_macOS.command  # 🍎 Build Poly-X.app (optional)
```

Everything that differs between systems — opening folders, launching the updater,
picking the compute device — lives in `polyx/model/plataforma.py` rather than being
scattered across the code.

---

## The full workflow

```
Microscope images (UV 254 nm, Nile Red staining)
        ↓
  🏷 Labeller  → annotate PET/PP/LDPE in YOLO format
        ↓
  🎯 Trainer   → train a YOLO v8/v11 model
        ↓
  🔬 Detector  → batch analysis + HTML report
        ↓
  📐 Viewer    → detailed inspection + µm/px calibration
```

---

## Tests

Shape measurement and calibration have their own suite, because every figure they
produce ends up in a table of the paper and a well-meant change can shift them
all without anything warning you.

```bash
.venv\Scripts\python.exe -m pytest tests/ -q
```

43 tests against **synthetic shapes of known size**, depending on no human
annotation whatsoever. Each one also pins down the *why* of a design decision, so
that if someone tries a variant that was already discarded, the suite says so.

`pytest` is only needed for development and is **not in `requirements.txt`**: an
installation meant for use does not need it.

---

## Related publications

- **Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).** Development
  of a new methodology for the determination of PET microplastics in sediment,
  based on microwave-assisted acid digestion. *PLoS ONE* 19(12): e0314520.
  https://doi.org/10.1371/journal.pone.0314520
- **Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).**
  Evaluation of microwave-assisted acid/oxidant digestion method for the detection
  of polyethylene microplastics in *Merluccius gayi* fish by Nile Red fluorescent
  staining and image analysis. *J. Chil. Chem. Soc.* 69(1): 6082-6085.
  https://doi.org/10.4067/s0717-97072024000106082

---

## User manual

`docs/manual/Manual_PolyX.en.html` (English) and
`docs/manual/Manual_PolyX.html` (Spanish) are the full
manual: step-by-step installation from GitHub with real photographs of every
step — on Windows and macOS — all four modules screen by screen, how size is
measured, the report, the shortcuts and troubleshooting. They are versioned
and open on a double-click, so a repo download carries the manual offline:
every image is embedded. Alongside them,
`Manual_PolyX.en.pdf` and `Manual_PolyX.pdf` for printing or emailing;
those are generated and not versioned.

To regenerate it against the current interface:

```bash
# 1. interface screenshots, one pass per language
.venv\Scripts\python.exe tools\manual\capturar_manual.py --salida docs/assets/manual_screenshots/es
POLYX_IDIOMA=en .venv\Scripts\python.exe tools\manual\capturar_manual.py --salida docs/assets/manual_screenshots/en

# 2. the manual, both languages and their PDFs
.venv\Scripts\python.exe tools\manual\generar_manual.py --pdf
```

The text lives in `tools/manual/manual_texto_es.py` and
`tools/manual/manual_texto_en.py`; the engine — template, stylesheet and
figure numbering — in `tools/manual/generar_manual.py`.

---

## Repository scope

This repository documents **the program**. Everything belonging to a paper in
preparation — the study's analysis pipeline, its photographs and its findings —
is deliberately left out, because publishing it here would pre-empt results that
are not out yet.

---

## Licence and contact

**Cristofher Ferrada** — PhD (c) in Chemistry, Pontificia Universidad Católica de
Valparaíso, 2026.

Environmental Chemistry Laboratory · Pontificia Universidad Católica de
Valparaíso
[cristofher.ferrada@pucv.cl](mailto:cristofher.ferrada@pucv.cl)

Repository: https://github.com/CrissFerrada/Poly-X-Microplastics
