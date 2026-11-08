# Glossary

The project's domain language. Terms here are the ones code, docs and
conversation should use; when a term is missing or fuzzy, sharpen it here
first instead of letting a second meaning into the code.

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
the EXIF-safe image size. One contract, one owner — no module
re-implements the format.

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

## Canonical run folder

The folder a detection run owns, written by the canonical engine:
`run.json` (schema version, status, params, models, calibrations, metrics and
image records) plus `detections.csv` (predictions only) and one folder per
model alias with the triptych PNGs, `labels/*.txt` and `classes.txt`. Read it
with `load_run`; do not add a second schema. Pre-canonical folders
(`summary.json` + `centroids.csv`, or the old GUI per-model labels) are legacy
inputs, readable without migration.

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
grouping, so they are computed once and read from `ReportContext`. They live
next to their producer in `polyx/gui/report/sections.py`.
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

