# MICRA — Microplastic Identification, Counting and Recognition Application

> **Automated detection and classification of microplastics (PET, PP, LDPE)
> by Nile Red fluorescence under UV light (254 nm), using YOLO v8/v11 models.**

**Status: pre-alpha** · Linux supported · Windows best effort · macOS unsupported
**Python 3.11–3.12** · formerly **Poly-X**

MICRA is a research instrument, not a product: it is used by the maintainer and
named collaborators, it carries no general-public support promise, and the
supported surface is the one CI exercises. See [Support](#support).

---

## Installation

### Support

| Tier | Platform | What is tested |
|---|---|---|
| 1 | Linux x86_64 | the full suite, headless and offscreen Qt, on Python 3.11 and 3.12 |
| 2 | Windows 10/11 | nothing in CI; the code is portable and `uv sync` is expected to work |
| 3 | macOS | nothing; no launcher, bundle or platform claim ships |

Support means the triple in the row is exercised. A Tier 2 setup that works on
your machine is welcome, but it is not a promise.

### On a cluster (the paper path)

Training runs in the Apptainer image built from the offline wheelhouse. It
carries the CUDA 12.6 build that still compiles the V100 kernels and installs
nothing from the network on a compute node:

```bash
python tools/hpc/build_wheelhouse.py --wheelhouse wheelhouse
git rev-parse HEAD > source_sha.txt
apptainer build micra.sif container/hpc.def
apptainer exec --nv micra.sif python -m polyx.hpc.check_env
```

See `docs/hpc-environment.md` and `docs/hpc.md`. The image installs the same
wheel, so it exposes both `micra` and the legacy `polyx` commands.

### On a lab machine

```bash
uv sync --locked --extra gui      # creates/updates .venv from uv.lock
uv run micra --help               # command list
uv run micra gui                  # launcher; needs the gui extra
```

`uv sync` resolves the **CPU** torch wheel (2.7.1) from the default index, which
is what CI uses. For a local GPU, install the matching wheel on top:

```bash
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Model weights are not bundled. Nothing is published yet, so
`micra download-models` has no source to fetch from: copy a `.pt` file into the
`models/` folder of the user data directory (`~/.local/share/polyx/models` on
Linux; `POLYX_HOME` overrides).

---

## Modules

| # | Module | Status | What it does |
|---|---|---|---|
| 1 | 🏠 **Launcher** | ✅ Working | Main menu — opens each module in its own window |
| 2 | 🔬 **Detector** | ✅ Working | Batch analysis with YOLO · paper-quality HTML report |
| 3 | 🎯 **Trainer** | ✅ Working | YOLO v8/v11 training with live curves |
| 4 | 🏷 **Labeller** | ✅ Working | Interactive YOLO annotation with automatic pre-labelling |
| 5 | 📐 **Viewer** | ✅ Working | Single-image inspection, calibration and particle-by-particle review |
| 6 | 🧪 **Pipeline** | ✅ Working | Headless family-agnostic training: `micra pipeline` (one execution, COCO import) and the Hydra multiruns `sweep_app`, `sweep_best`, `campaign_app` |

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
  training ends, MICRA evaluates every checkpoint against **real sediment
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
- **Load predictions from a closed canonical run** in `runs/detect_.../`
  (the folder has `run.json`), without running the model again. It always
  opens the original photo and never the annotated PNG, which has the boxes
  painted onto it. Runs from older versions are left untouched but need an
  older Poly-X version to open
- **Drag and drop** an image or a `.pt` model straight onto the canvas
- Exports: annotated image + `detections.csv` + `viewer_summary.json` in a
  `viewer_<image>_<timestamp>/` folder (never into a run folder)

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
worst case**. This is pinned down in `tests/test_morphology.py`.

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

For crops where the dish is not visible, the plate cannot be measured. Supply
an English per-image index instead: `calibration_index.csv` with the headers
`image,um_per_px`, keyed by image basename. Run with
`micra detect --calibrate index --calibration-index calibration_index.csv` (the
flag alone also selects index mode); each photo then uses **its own** scale. A
photo with no row is reported with `image_not_in_index` and stays uncalibrated
rather than silently inheriting a batch-wide value. Image basenames must be
unique within an index-calibrated batch: two photos with the same name in
different folders are refused before any inference, since a single row would
otherwise apply to both. Legacy Spanish index files are rejected by name or
header with an error naming the canonical format; they are never searched for,
rewritten or deleted.

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
down in `tests/test_fiber_is_not_truncated.py`.

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

The interface is available in **Spanish and English**. The selector lives in
**Preferences** (the Launcher's top-bar button, or `Ctrl+,`) and the choice is
remembered between sessions. The first time, the system language is used;
`POLYX_LOCALE=en` forces it without touching the interface.

Changing the language applies immediately, including to modules that are
already open: they share the preference file and watch it for changes.

**The detection report comes out in the chosen language**, not just the
interface: headings, tables, figure captions, chart axes and the methods prose.
Numbers, dates and CSV column headers do not change with the language: they are
the report's machine contract, so the same run reads the same digits everywhere.
A report already saved keeps the language it was generated in.

To see which strings are still untranslated:

```bat
.venv\Scripts\python.exe tools\i18n\cli.py audit
```

It compares the committed template with the source tree and fails on any empty
or fuzzy translation, a placeholder mismatch or a user-visible string that
bypasses the translation call. To see by eye which strings were never
extracted, generate the pseudo-locale and select it:

```bat
.venv\Scripts\python.exe tools\i18n\cli.py pseudo
set POLYX_LOCALE=qps_ploc
```

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
one. A language change is the exception: it reaches modules that are already
open, because every process watches the shared preference file.

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

| Component | Minimum |
|---|---|
| Python | 3.11 or 3.12 (3.13 and later are not supported yet) |
| RAM | 8 GB |
| GPU | optional; an NVIDIA card makes training 20–30× faster |
| Dependencies | `PySide6 6.7`, `Ultralytics 8.3`, `OpenCV 4.10`, `NumPy 1.26`, `Matplotlib 3.9`, pinned in `uv.lock` |

The GUI needs Qt and therefore the `gui` extra; the headless commands do not.

---

## Usage

```bash
uv run micra                # command list
uv run micra gui            # launcher; pick a module
uv run micra train --help   # headless training
```

The GUI modules can also be opened directly with `python -m polyx.gui.launcher`,
`polyx.gui.detector`, `polyx.gui.trainer`, `polyx.gui.labeler` and
`polyx.gui.viewer`.

---

## Project layout

```
src/polyx/
├── cli/                 # micra / polyx command line: train, detect, eval, report, pipeline
├── core/                # Qt-free shared code: paths, platform, labels, detection run, editing
├── gui/                 # Qt shell: launcher, detector, trainer, labeler, viewer, report
├── hpc/                 # cluster entry points: check_env, weights, job staging
├── pipeline/            # Hydra/Lightning training pipeline and its configs
├── locales/             # gettext catalogs (es)
└── assets/
tests/                   # the suite; measurement and calibration are pinned numerically
tools/                   # check runner, i18n workflow, manual generator, HPC wheelhouse
container/               # Containerfile and Apptainer definition
cluster/                 # SLURM submission scripts
docs/                    # ADRs, HPC guides, manual
```

Everything that differs between systems — opening folders, picking the compute
device, the user data location — lives in `src/polyx/core/platform.py` rather
than being scattered across the code.

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
uv run pytest -q
```

43 tests against **synthetic shapes of known size**, depending on no human
annotation whatsoever. Each one also pins down the *why* of a design decision, so
that if someone tries a variant that was already discarded, the suite says so.

`pytest` is a development dependency and is not installed with the runtime
stack.

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

The generated manual in `docs/manual/` predates the pre-alpha rewrite and still
describes the retired installers. It is out of scope until after the paper: it is
not regenerated in the meantime, and this README is authoritative wherever the
two disagree. The interface walkthroughs remain useful as a screen-by-screen
reference.

---

## Repository scope

This repository documents **the program**. Everything belonging to a paper in
preparation — the study's analysis pipeline, its photographs and its findings —
is deliberately left out, because publishing it here would pre-empt results that
are not out yet.

---

## Origin and credit

MICRA is a derivative of **Poly-X**, created by **Cristofher Ferrada** at the
Pontificia Universidad Católica de Valparaíso. The original project is at
[CrissFerrada/Poly-X-Microplastics](https://github.com/CrissFerrada/Poly-X-Microplastics);
its authorship and the MIT licence are preserved.

This repository is maintained by **Franco Ignacio Valdés Navarro** in the
**Laboratorio de Química Ambiental, Instituto de Química, Pontificia Universidad
Católica de Valparaíso**, as the laboratory's continuation of the work.

## Licence

MIT for the code; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights are
published under CC BY 4.0 once a release exists.

Repository: https://github.com/Freston1605/micra
