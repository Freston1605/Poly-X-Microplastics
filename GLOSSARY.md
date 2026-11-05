# Glossary

The project's domain language. Terms here are the ones code, docs and
conversation should use; when a term is missing or fuzzy, sharpen it here
first instead of letting a second meaning into the code.

## YOLO-label codec

The module (`polyx/core/labels.py`) that owns the YOLO `.txt` format and
its folder conventions for the whole program: parse, serialize, locate
(the `images/` ↔ `labels/` mapping, both directions), `classes.txt`, and
the EXIF-safe image size. One contract, one owner — no module
re-implements the format.

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
