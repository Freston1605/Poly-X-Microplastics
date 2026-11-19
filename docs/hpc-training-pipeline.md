# Headless training pipeline (local backend)

The pipeline is a headless path parallel to the shipped GUI Trainer. It does
not change the Trainer or its native training wrapper. This page describes the
first slice: one YOLO detection training execution on the local backend.

## Running one execution

The command takes Hydra-style `key=value` overrides. `output_dir` and either
`data_yaml` (a training execution) or `coco_import` (a dataset import) are
required; every other value comes from the composed configuration.

```bash
polyx-pipeline \
  data_yaml=/path/to/dataset/data.yaml \
  output_dir=/path/to/executions/tracer \
  protocol.epochs=2 \
  protocol.early_stopping.patience=1
```

The command prints the output directory, the selected checkpoint, the
selection metric and the criterion it was computed under. A non-zero exit
status means the execution failed; the printed error names the reason.

## Configuration sections

- `protocol` — framework-neutral and consumed unchanged by every adapter.
  Defaults: 150 epochs, image size 1280, batch 8, SGD with learning rate 0.01,
  momentum 0.937 and weight decay 0.0005, cosine schedule, five warmup epochs,
  FP16 AMP enabled, and early stopping on `with_class_score` (mode `max`,
  patience 30, minimum delta 0.001, best weights restored).
- `backend` — where the execution runs. The local backend is one process with
  no scheduler and no cluster-specific values.
- `family` — which adapter executes the protocol. `yolo_detection` wraps the
  Ultralytics YOLO detection model in one Lightning module.

## Artifact layout

One output directory holds everything needed to review or reproduce the run:

```
config.yaml            # fully resolved configuration
resolved_protocol.yaml # the protocol exactly as the adapter consumed it
provenance.json        # git commit, dataset hashes, dataset audit, version data
training_state.json    # adapter state, AMP precision, early-stopping decisions
metrics.json           # named figures, per-class figures, selection criterion
report.md              # human-readable summary, criteria included
logs/training.log      # plain-text execution log
checkpoints/best.pt    # best weights, selected by the with-class score
checkpoints/epoch_NNN.pt # one checkpoint per epoch
```

## Scoring

The shared evaluator consumes COCO-format predictions and COCO ground truth
built from the existing label codec. Every figure names its criterion:

- `with_class_score` — COCO mAP@[0.50:0.95], class-aware: a detection counts
  only when the polymer identity is correct. This is the selection metric.
- `localisation_score` — COCO AP@0.50, class-agnostic: whether the particle was
  found, reported beside the selection score.
- `mAP50` and `mAP50-95` — class-aware averages with their thresholds named.

The criterion is always on the real-domain validation split. Localisation is
never used to select a checkpoint.

## Importing the segmentation dataset

Masks are not produced inside Poly-X. An external CVAT COCO export is imported
through the same command with `coco_import` instead of `data_yaml`:

```bash
polyx-pipeline \
  coco_import=/path/to/cvat_export \
  dataset_name=first_batch \
  output_dir=/path/to/executions/import
```

The import validates the instance JSON and every referenced image, audits the
polygons with the existing dataset audit, and stores a versioned tool output in
`USER_DATA_DIR/segmentation_datasets/<name>/vN/`: `dataset.json` (canonical
COCO instances with paths resolved inside the version), the images, `audit.json`
and `manifest.json` with class distribution, hashes, ground-truth policy and the
first-batch target. The execution folder receives the resolved configuration,
the manifest, the audit and a `report.md`; it never receives masks. Re-importing
creates the next `vN` instead of overwriting an earlier dataset.

Each image may carry a `split` attribute (`train`, `val`, `test`); without one
it is imported as train and the audit warns that validation coverage cannot be
measured. A polygon needs at least three finite, in-bounds points and a non-zero
area, and RLE is refused because there is no RLE codec. Invalid or degenerate
polygons, unknown categories or annotations without an image reject the import.
So does any annotation or image whose provenance names a machine generator
(SAM2, Otsu, YOLO-seg): machine masks are diagnostics and proposal generators,
never ground truth.

The first batch target is roughly ten plates stratified across PET, PP and LDPE
and the imaging domains represented in the study, so every class appears in
both training and validation. The target is recorded in every import manifest
and report. Detection run folders are never touched: they stay box-only.

## Decisions and vocabulary

See [ADR-0007](adr/0007-one-lightning-loop.md) for the single-loop decision,
[ADR-0009](adr/0009-coco-segmentation-dataset.md) for the COCO segmentation
dataset, and the [glossary](../GLOSSARY.md) for *run*, *protocol*,
*segmentation training dataset* and *COCO mask dataset*.
