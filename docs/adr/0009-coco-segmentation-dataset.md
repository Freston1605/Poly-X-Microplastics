# ADR-0009: COCO segmentation training dataset from CVAT masks

- **Status:** Accepted
- **Date:** 2026-11-18
- **Landed in:** #196

## Context

The benchmark needs a segmentation ground-truth dataset, but the shipped
human-in-the-loop review tool corrects boxes only: SAM2, Otsu and YOLO-seg
masks exist in memory for measurement and are deliberately discarded. There is
no mask editor in the project and building one would spend the campaign's
budget on tooling instead of models. A second problem is fairness: if the
segmentation dataset lived inside detection run folders, ADR-0002's box-only
contract would break and a run would no longer mean the same thing on two
machines.

CVAT already produces high-quality polygon masks and exports them as COCO
instance JSON. Any other tool (Label Studio among them) is acceptable while it
emits the same format, so the workflow is not tied to one vendor. COCO is
lossless, has one id per instance and is consumable by every family after a
per-job conversion.

## Decision

**An external CVAT COCO export is the canonical segmentation training dataset.
It is imported by the same headless command, audited by the existing dataset
audit and stored as a versioned tool output under the user data dir. Detection
run folders stay box-only; no machine-generated mask becomes ground truth.**

The command is extended, not duplicated: `polyx-pipeline coco_import=...`
reads a CVAT instance JSON, validates its structure, checks that referenced
images are inside the export, and hands the polygons to `audit_coco_masks` in
the existing `polyx.core.quality.dataset_audit` module. That is the same
`DatasetAudit` type and the same check suite the YOLO box path uses — class
distribution per split, class coverage, imbalance, dominance, size similarity
and validation size — plus polygon geometry checks. A polygon needs at least
three finite, in-bounds points and a non-zero area; RLE is refused because the
project has no RLE codec. An invalid or degenerate polygon, an unknown
category or a mask without an image makes the audit fail and stops the import
before any version is written.

The split of each plate comes from an image-level `split` attribute (`train`,
`val` or `test`; `validation` and similar spellings are normalised). An export
without the marker imports every image as train and the audit warns that no
validation split exists: class coverage and metrics cannot be audited until the
annotator marks one. This is a warning, not a rejection, because splitting an
exported batch is a separate deterministic step.

The validated dataset is stored under `USER_DATA_DIR/segmentation_datasets/
<name>/vN/`:

```
dataset.json   # canonical COCO instances, file_name resolved inside the version
images/...     # the referenced images, at their export-relative paths
audit.json     # the shared dataset audit that accepted the import
manifest.json  # version, class distribution, hashes, GT policy, batch target
```

`dataset.json` rewrites every `file_name` to `images/<original name>`, so a
later conversion does not depend on the export folder or on the machine that
produced it. The manifest records the version, the per-class distribution per
split, the SHA-256 of the JSON and every stored image, the source tool, the
ground-truth policy and the first-batch target. Each re-import increments `vN`;
nothing is overwritten. This path is a **tool output** in the glossary sense
and deliberately does not sit under `RUNS_DIR` or `RUNS_TRAIN_DIR`. ADR-0002 is
unchanged and detection runs never gain mask files. Images are read before the
version folder is created, so a missing or unsafe image path leaves nothing
half-copied behind.

An annotation or image whose provenance names a machine generator (SAM2, Otsu,
YOLO-seg and friends) rejects the whole export. The rejection is all-or-nothing
because a partially reviewed export cannot be trusted as ground truth without a
review step this tool does not have. Machine masks remain diagnostics and
pluggable proposal generators only.

The first batch target is **roughly ten plates, stratified across PET, PP and
LDPE and the imaging domains represented in the study**, so every class lands
in both training and validation before a larger annotation effort. The target
is written into each import manifest and report so it travels with the data.

## Considered options

- **Build a mask editor inside Poly-X.** Rejected: out of scope, slow to build
  and no better than the external annotators the researchers already use.
- **Write masks into each detection run folder.** Rejected: it changes the
  canonical run folder and contradicts ADR-0002; the same run would carry
  different data on different machines.
- **A second, mask-only validator.** Rejected: it would drift from the dataset
  audit's class and split checks. The existing auditor is extended with a new
  input surface instead.
- **Accept machine masks and flag them as low-confidence GT.** Rejected: a
  pseudo-mask would become the benchmark reference, which the spec forbids.
- **Store one mutable folder instead of versions.** Rejected: an import could
  silently replace a dataset a published result was trained on.
- **Write a YOLO `data.yaml` beside the COCO dataset.** Rejected: the stored
  dataset has no YOLO label files, so a YOLO declaration would pretend to be a
  dataset it is not; per-family conversion in #197 owns that shape.

## Consequences

- The command's one seam now covers both a training execution and a dataset
  import; no second CLI or test surface is introduced.
- Per-family conversion (Ultralytics YOLO-seg, Mask R-CNN) remains a job-time
  step into scratch. COCO stays the only canonical mask format.
- The first batch is small on purpose; its audit is expected to warn about the
  validation size, which is a warning, not a rejection. A larger batch is a
  data-collection task, not a code change.
- A machine-generated marker anywhere in an export blocks the import until it
  is removed by the annotator; that is the intended safety property.
