# Glossary

The project's domain language. Terms here are the ones code, docs and
conversation should use; when a term is missing or fuzzy, sharpen it here
first instead of letting a second meaning into the code.

## Run

One execution of a pipeline. The word is generic and is always qualified by
the pipeline that ran: **detection run** (the canonical engine over a set of
images), **training run** (one config-driven execution of the headless
pipeline), **evaluation run** or **validation run**. Some shipped identifiers
(`RunResult`, `run.json`, `load_run`) keep the unqualified name for historical
reasons; they are documented legacy, not a second meaning of the term. A
**training execution** is the concrete folder of artifacts one run produced.

## Protocol

The framework-neutral part of a training configuration: split, seed, epochs,
batch, image size, optimizer, learning rate, momentum, weight decay, schedule,
warmup, augmentation subset, AMP and early stopping. Every family adapter
consumes the resolved protocol unchanged, so the model is the only variable in
a benchmark. The resolved protocol is written verbatim into each training
execution's artifacts. _Avoid_: hyperparameters (too narrow — it also fixes
the split and the evaluator).

## The triptych

The three saved views of a detection run over one plate: the
**predictions** view (class + confidence labels), the **GT** view
(ground truth in the class colours) and the **combined** view
(predictions under a GT overlay in the GT colour with GT-prefixed
labels). One folder per model per run. Owned by the plate annotation
module; the boxes' thickness scales with the plate and an unknown class
gets one fallback colour everywhere.

## User data dir

The per-user writable home of everything Poly-X generates: the models
cache, detector runs, training runs and correction datasets
(`%LOCALAPPDATA%\polyx` on Windows, `~/Library/Application Support/polyx`
on macOS, `~/.local/share/polyx` elsewhere; `POLYX_HOME` overrides).
Avoid: the repo-root `runs/` folders — those are legacy locations of
older versions, not the location. Decided in ADR-0001.

## YOLO-label codec

The module (`polyx/core/labels.py`) that owns the YOLO `.txt` format and
its folder conventions for the whole program: parse, serialize, locate
(the `images/` ↔ `labels/` mapping, both directions), `classes.txt`, and
the EXIF-safe image size. One contract, one owner: it is the only import
path for label I/O, and no module — including `polyx.core.data` — carries
a second codec or re-exports these names from the package roots.

The conventional layout pairs a dataset root's `images/` with a sibling
`labels/`, mirroring the relative subpath (`images/train/a.jpg` ↔
`labels/train/a.txt`); no split name is hard-coded. Reads prefer the
conventional file and fall back to a legacy same-directory or nested
`labels/` file with a warning; writes go to the conventional location, or
next to the image in a plain folder. An explicit GT folder is a flat
override (flat read, then mirrored read, flat write). `classes.txt` lives
at the dataset root for a split layout and in the folder itself for a
plain one. `data.yaml` is not the codec's and the codec does not read it
today. Class-name precedence today is explicit `class_names` →
`classes.txt` → `STUDY_CLASSES`; the planned dataset module
(`polyx/core/data/`, still empty) will own the YAML names and pass them in
as `class_names`, adding the `data.yaml` step once it lands. The codec never
parses YAML. This fallback is a **dataset-label** convention, not a run reader:
pre-canonical run folders are not opened by it (see *Canonical run folder*
below).

## Detection run

One execution of the canonical engine (`polyx/core/detection_run`) over a set
of images and one or more loaded models. It decodes, calibrates, reads GT,
predicts, measures, matches, draws and persists; its output is a `RunResult`
and a canonical run folder. There is no second "headless" or "GUI" detection
engine: the GUI runs this one on a thread and the CLI runs it synchronously.

## RunResult

The Qt-free record of a detection run (`polyx/core/detection_run/result.py`):
`run_dir`, the selected images, the active models in dense engine order, the
`RunParams`, per-image calibrations, the per-model `ImageResult` lists,
aggregate metrics, status (`completed` / `aborted` / `failed`), error,
skipped images and timestamps. Reports, the CLI and `load_run` consume it;
the GUI adds signals and review verdicts on top but does not extend it.

## Operating point

The confidence threshold a run is scored at, together with the criterion used
to count its errors. Two criteria are legitimate and disagree by construction:
**localisation** asks only whether the particle was found, while **with class**
also requires the polymer, so a well-placed box with the wrong class is a false
positive of the predicted class and a false negative of the true one. Every
figure names the criterion it uses. A *recommendation* sweeps the confidences
the run observed and returns the best one under a named criterion. The module
that owns both the arithmetic and the criteria is
`core/detection_run/operating_point.py`.
_Avoid_: F1 (unqualified), score

## Tiling mode and plan

The decision about splitting a large photo before inference. The **mode** is
`auto` (tile only above a threshold), `always` or `never`; these are the only
values accepted at any boundary — legacy Spanish spellings are no longer read.
The **plan** is the
geometry of one tiled photo — tile side, overlap, tile count and the photo's
longer side — recorded in `run.json` instead of prose, so a reader can phrase
it in their own language. A photo with no plan goes through in a single pass.
The decision, its geometry and its sentence belong to `core/detect/tiling.py`;
neither a page nor the engine formats the numbers again.
_Avoid_: mosaic (in code), `siempre`/`nunca` (legacy spellings, not values)

## Calibration refusal

How the Viewer answers a drawing that cannot support a µm/px — a zero-length
line, a non-positive real size, two coincident points or three collinear ones.
It is **refused by name**: a stable English token whose sentence is chosen
where it is shown. Nothing is applied — the µm/px in force stays — and the
drawing stays on screen, so the same points can be finished with another real
size. It is never a zero scale: zero is a scale, and every size computed with
it would silently come out zero.
_Avoid_: "calibration failed" (does not say what happened to the scale),
0 µm/px as a refusal

## Per-image calibration index

The canonical English CSV `calibration_index.csv`, with the headers
`image,um_per_px`, that supplies one µm/px scale per image basename. It is the
safe source for crops where the Petri dish is not visible: a single batch-wide
manual scale would mis-size photos shot at different distances. The file is
named explicitly, never searched for; a missing row is a warning
(`image_not_in_index`) and never a silent fallback. The legacy Spanish files
and headers are refused by name. Because the basename is the lookup key, an
index-calibrated batch must give every photo a unique basename: two images with
the same name in different folders are refused before inference rather than one
row silently scaling both.
_Avoid_: calibration index CSV (ambiguous: could mean a legacy file),
batch-wide fallback for crops

## Warning code

A stable English token recorded in an artifact when a measurement cannot be completed as requested. It names what happened; the sentence a person reads is chosen at display time, and numeric details travel as separate structured values.
_Avoid_: warning message, error text

## Model alias and display name

Two values for the same model. The **alias** is the stable, filesystem-safe
folder key inside a run (`model_1`, or the weights stem / a name somebody
typed); it is an artifact and is never translated. The **display name** is the
prose the interface and the report show: `_("Model %(n)s")` when the alias was
generated, and the alias verbatim when it was chosen. The `alias_generated`
flag on `ModelInfo`/`ModelSlot` says which, and
`core/detection_run/identity.py` owns the resolution.
_Avoid_: model name (ambiguous), translated alias

## Canonical run folder

The folder a detection run owns, written by the canonical engine:
`run.json` (schema version, status, params, models, calibrations, metrics and
image records) plus `detections.csv` (predictions only) and one folder per
model alias with the triptych PNGs, `labels/*.txt` and `classes.txt`. Read it
with `load_run`; do not add a second schema. Pre-canonical folders
(`summary.json` + `centroids.csv`, old GUI per-model labels and legacy
calibration index CSVs) are legacy inputs: new code does not read them, and the
CLI and Viewer report that an old run needs an older version of Poly-X. The
canonical per-image index is a distinct, English-only file; see
[Per-image calibration index](#per-image-calibration-index).

## Machine-readable artifact

A file whose content carries a token that a program reads or writes across a disk boundary. This is the unit the vocabulary spec covers: neither pixels nor translated prose are artifacts in this sense, and neither is state that never reaches a file. Every machine-readable artifact belongs to one of four classes.
_Avoid_: output, file, report (as umbrella terms)

## Canonical run artifact

A machine-readable artifact written by a detection run into the folder it owns: its manifest, prediction table, per-model labels and report identifiers.
_Avoid_: run output

## Cross-cutting log

A structured record of events written by several modules to one shared contract, not owned by a single detection run.
_Avoid_: debug log, console output

## Tool output

A machine-readable artifact written by a supporting tool — training, evaluation, export or preference storage — outside a detection run's folder.
_Avoid_: run artifact

## Legacy input

A machine-readable artifact from a pre-canonical format. New code neither reads nor writes it.
_Avoid_: legacy run (when the file, not the run, is meant)

## Report section

One numbered, self-contained part of the HTML report — summary, methods,
gallery, and so on — chosen before the report is rendered. A section has a
stable id that doubles as its HTML anchor, so a link into an archived report
keeps working; the id is a key, not a translated word. Its visible title is a
translated English message.
_Avoid_: tab, chapter, page

## Batch (report)

The set of detections a report describes, after any narrowing to a chosen
subset of images. Every total, figure and table in the document describes
this same set; the report never mixes one batch's numbers with another
batch's images.
_Avoid_: run (the run is the recorded execution, which may be wider),
selection

## Shape bodies (report)

The two section bodies the shape pass produces in one go: the particle-shape
section and its size-by-folder companion. They share the per-class size
grouping, so they are computed once and read from `ReportContext`.
_Avoid_: form results, shape tuple

## GT (ground truth)

The manual annotation of an image, stored as a YOLO `.txt`. An existing
`.txt` IS the GT even with zero boxes: a reviewed empty plate is data
(those particles the detector finds there are false positives), not an
absence of annotation. Predictions are never written next to the photo —
that would destroy the GT irrecoverably.

## Pre-annotation

A run's prediction labels (`.txt` with the confidence as a sixth column)
reopened in the Labeller as a starting point: counting 500 particles from
the model's boxes is another task than doing it from zero. The confidence
column is what keeps the record while the boxes are corrected.

## The study's classes

`["PET", "PP", "LDPE"]` — the fallback class list when no `classes.txt`
governs a label file. Defined once, in the codec. Not the same thing as
the per-class interface colours in `gui/theme.py`: those are domain data
that must not change with the theme; this is the class order.

## Census honesty

The Labeller's rule that a `.txt` is written only for images that were
marked reviewed or drawn on. "Reviewed, zero particles" is a datum; an
image barely glanced at must not be recorded as reviewed. The policy
lives in the GUI flows — the codec writes exactly what it is told.

## Editing core

The one Qt-free implementation of box editing (`polyx/core/editing.py`):
the box under edit in absolute image pixels, its normalized YOLO currency
through the label codec, and a per-image editor that owns the boxes, the
selection, the bounded undo history and every gesture — draw, move, resize
by handle, delete, clear, class assignment. Both annotation canvases are
thin Qt shells over it, so the 2 px census minimum, the handle geometry,
one undo entry per gesture and the active class are defined once and cannot
drift between pages. Hit-test distances arrive in image pixels: the shell
turns its roughly constant widget distance into that through the current
zoom, which is why the core never needs Qt.
_Avoid_: canvas logic, editor widget (the widget is the shell).

## Source language

English. It is the language every user-facing string is written in, and the
English sentence is the message id itself: there is no separate key. The
source language is also the fallback — a missing translation renders the
English msgid, which is correct text rather than an error. Owned by
`polyx.core.i18n`; decided in ADR-0003.

## Target locale

A language tag that has a catalog directory, such as `es`. The tag is the
directory name under `src/polyx/locales/`, so adding a locale is adding a
directory. A regional tag resolves through its language first (`es_AR` →
`es`) and then to English; it is not a separate catalog unless one exists.
Not to be confused with the source language: the source is always English,
the target locale is what the reader asked for.

## Catalog (catalogue)

The translation file for one target locale:
`src/polyx/locales/<tag>/LC_MESSAGES/messages.po`. The `.po` is the
translator handoff — plain data, openable by any CAT tool or text editor,
with the English msgid, the translated `msgstr` and the source locations.
`messages.pot` is the committed template every catalog is merged from; the
`.mo` next to each `.po` is the compiled build artifact the runtime reads,
never hand-edited and never committed. The catalog is the only home for
translated text; Python holds English only.

## Extraction

The step that walks the source tree for translation calls (`_`, `ngettext`,
`pgettext`, `npgettext`, `N_`, `lazy`) and writes the committed template
`src/polyx/locales/messages.pot`. Run it with
`python tools/i18n/cli.py extract` whenever a call is added, removed or
moved: editing source shifts line references, and the audit fails while the
template disagrees with the tree. Babel performs the walk; the project
extractor also sees calls inside f-strings.

## Pseudo-locale

A generated, development-only catalog (`qps_ploc`) whose translations are
bracketed and accented so every translated string is visibly mangled. It is
there to make an unextracted string obvious by eye — an English sentence on
screen is a string that bypassed the translation call — and to expose layout
truncation. Generated with `python tools/i18n/cli.py pseudo`, selected with
`POLYX_LOCALE=qps_ploc`, gitignored and excluded from the wheel and the
frozen application. It is not a shipped locale and not part of any gate.

## Retired: `tr()`

The old translation function that looked up a Spanish sentence in
`polyx/core/translations_en.py` and returned English. It is no longer
current: the module is deleted, Spanish is no longer a key, and the name
collided with Qt's own `tr()`. Say `_()` (or `ngettext`/`pgettext`) from
`polyx.core.i18n`; an old document that still calls `tr()` the translation
layer is describing the scheme ADR-0003 replaced.

