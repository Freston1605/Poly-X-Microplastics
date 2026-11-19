# Headless training pipeline (local backend)

The pipeline is a headless path parallel to the shipped GUI Trainer. It does
not change the Trainer or its native training wrapper. This page describes the
local backend and the per-family executions: one YOLO detection training
execution and two segmentation executions — YOLO-seg and Mask R-CNN — sharing
one protocol, one Lightning loop, one evaluator and one artifact layout. The
three families can also run as one multirun campaign; the cluster procedure is
in [hpc.md](hpc.md).

## Running one execution

The command takes Hydra-style `key=value` overrides. `output_dir` and exactly
one dataset source are required: `data_yaml` (a YOLO box dataset),
`coco_dataset` (the imported COCO mask dataset) or `coco_import` (an import).
Every other value comes from the composed configuration.

```bash
polyx-pipeline \
  data_yaml=/path/to/dataset/data.yaml \
  output_dir=/path/to/executions/tracer \
  protocol.epochs=2 \
  protocol.early_stopping.patience=1
```

The command prints the output directory, the selected checkpoint, the
selection metric and the criterion it was computed under, then the
localisation figure and its criterion beside it. A non-zero exit status means
the execution failed; the printed error names the reason.

## Running a campaign

A campaign is one Hydra multirun whose jobs are the model families. Each job
composes the same protocol and writes its own execution under
`<campaign_root>/<family.run_name>/` (the family name by default); the model is
the only variable. Both training source overrides are supplied because the
detection family consumes the YOLO box dataset and the segmentation families
consume the canonical COCO dataset:

```bash
python -m polyx.pipeline.campaign_app --multirun \
  campaign_root=/path/to/campaign \
  data_yaml=/path/to/dataset/data.yaml \
  coco_dataset=/path/to/segmentation_datasets/first_batch/v1 \
  family=yolo_detection,yolo_segmentation,mask_rcnn \
  protocol.epochs=150
```

The Hydra basic launcher runs the three jobs sequentially inside one process
and one allocation. Add `backend=slurm` on the cluster; the wrapper, queue
tiers and sync-back procedure are on [hpc.md](hpc.md). The local campaign is
proven by `tests/test_pipeline_campaign.py`.

## Configuration sections

- `protocol` — framework-neutral and consumed unchanged by every adapter.
  Defaults: 150 epochs, image size 1280, batch 8, SGD with learning rate 0.01,
  momentum 0.937 and weight decay 0.0005, cosine schedule, five warmup epochs,
  FP16 AMP enabled, and early stopping on `with_class_score` (mode `max`,
  patience 30, minimum delta 0.001, best weights restored).
- `backend` — where the execution runs. The local backend is one process with
  no scheduler and no cluster-specific values.
- `family` — which adapter executes the protocol. `yolo_detection` wraps the
  Ultralytics YOLO detection model in one Lightning module; `yolo_segmentation`
  wraps the segmentation model the same way; `mask_rcnn` is the sibling
  non-Ultralytics adapter built on torchvision. The task is a config value,
  never a code edit.
- `scratch_dir` — ephemeral cache root for per-family conversion. It defaults
  to `$POLYX_HPC_SCRATCH/polyx/pipeline`, or the system temporary directory
  when the cluster variable is absent. Nothing written there is an artifact.
- `weights_dir` / `weights_manifest` — the staged weights directory and its
  `sha256sum` manifest. Empty resolves to `POLYX_WEIGHTS_DIR` /
  `POLYX_WEIGHTS_MANIFEST`. When either is configured, `family.weights` is
  resolved to a local file, verified, and handed to the framework; an absent,
  unpinned or mismatching file stops the execution. `training_state.json`
  records the resolved path and `weights_verified`. With neither configured the
  framework's own default is used and reported unverified — local debugging
  only; cluster jobs always stage and pin.

A detection family consumes only `data_yaml`; a segmentation family consumes
only `coco_dataset`. A campaign job may declare both, so one multirun can serve
both kinds of family; a single execution still refuses a missing source, and a
segmenter can never train on boxes or a detector be scored against masks. The
mismatch would silently produce a meaningless benchmark, so it is refused
before any training starts.

`protocol.augmentation_subset` is the benchmark campaign's common subset, and
it is exactly the intersection the first-wave families can express identically:
`hsv` and `flip_lr`. Their magnitudes are protocol values (`hsv_h`, `hsv_s`,
`hsv_v`, `flip_lr_probability`) and both families apply them with the same
arithmetic, so the transform is not a second variable. A benchmark execution
refuses any transform outside that intersection, so `mosaic` cannot enter a
campaign by accident. Family-specific transforms belong to an exploratory run:
`protocol.benchmark=false` with `protocol.exploratory_augmentation=[mosaic]`.

Every adapter records in `training_state.json` what it applied and what it
skipped, the magnitudes, and `family_specific: false`: a transform a family
cannot express is reported as skipped, never replaced by a framework-specific
one. Mask R-CNN implements `hsv` and `flip_lr` and skips `mosaic`.

## Artifact layout

One output directory holds everything needed to review or reproduce the run,
for every family:

```
config.yaml            # fully resolved configuration
resolved_protocol.yaml # the protocol exactly as the adapter consumed it
provenance.json        # git commit, dataset hashes, audit, dataset source, conversion
training_state.json    # adapter state, AMP precision, early-stopping decisions
metrics.json           # named figures, per-class figures, selection criterion
report.md              # human-readable summary, criteria included
logs/training.log      # plain-text execution log
checkpoints/best.pt    # best weights, selected by the with-class score
checkpoints/epoch_NNN.pt # one family-native export per epoch
checkpoints/lightning/last.ckpt # real Lightning checkpoint: weights, optimizer, scheduler, epoch
```

`provenance.json` always names the dataset source (`yolo_boxes` or
`coco_instances`). A YOLO-seg execution also records its job-time conversion:
format, cache key, whether the cache was reused, and the scratch path that
holds the intermediate. Mask R-CNN reads the canonical JSON directly and
records no conversion. The intermediate is never an artifact; the canonical
dataset stays COCO.

## Scoring

The shared evaluator consumes COCO-format predictions and COCO ground truth.
Every figure names its criterion, and the criterion names the overlap it used:

- detection — box IoU on the YOLO label codec's ground truth;
- segmentation — mask IoU between rasterized COCO polygons.

The figures are the same shape for both families:

- `with_class_score` — class-aware mAP@[0.50:0.95]: an instance counts only
  when the polymer identity is correct. This is the selection metric, and the
  checkpoint with the best value is the one kept as `best.pt`.
- `localisation_score` — class-agnostic AP@0.50: whether the particle was
  found, reported beside the selection score.
- `mAP50` and `mAP50-95` — class-aware averages with their thresholds named.
- `mask_iou`, `mask_mAP50` and `mask_mAP50-95` — segmentation only: the mean
  best mask IoU per ground-truth instance and the mask mAP figures.

Per-class figures repeat the same criteria for every category present in the
validation split, and `report.md` lists them. Overlap is only computed between
a prediction and a ground-truth instance of the same image, so identical
geometry in another image is a false positive, not a match. The criterion is
always on the real-domain validation split. Localisation is never used to
select a checkpoint.

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

## Training a segmentation family

`yolo_segmentation` trains from the versioned COCO mask dataset. The COCO
polygons are converted to the family's folder contract (`images/<split>`,
`labels/<split>` with YOLO-seg polygon lines, `data.yaml`) at job time, and
cached under `scratch_dir/seg_datasets/<cache key>`. The key is the SHA-256 of
the canonical `dataset.json`, so a second execution over the same dataset
reuses the cache and a different dataset cannot pick up stale labels.

```bash
polyx-pipeline \
  coco_dataset="$POLYX_HOME/segmentation_datasets/first_batch/v1" \
  family=yolo_segmentation \
  output_dir=/path/to/executions/seg-tracer \
  protocol.epochs=1
```

The command prints the checkpoint, the with-class selection score with its
criterion and the localisation figure next to it. The canonical dataset is
read-only here: the run folder holds no YOLO labels, no `data.yaml` and no
masks, and deleting the scratch cache costs only a rebuild. A one-epoch run on
the mask fixture is exercised end to end by
`tests/test_pipeline_yolo_seg.py::test_yolo_seg_trains_from_coco_masks_for_one_epoch`,
which imports the fixture through the command and then trains it; the test is
the reproducible demo until a real CVAT batch exists.

## Training Mask R-CNN from the same masks

`mask_rcnn` is the non-Ultralytics sibling: torchvision reads the canonical
`dataset.json` directly, so no framework-specific intermediate is written for
it. Polygons are rasterized into instance masks at load time, torchvision's
reserved background label 0 is added to the targets and removed from the
predictions, and the pretrained detection heads are rebuilt for the dataset's
class count. The same protocol, early stopping, shared evaluator and artifact
layout apply unchanged:

```bash
polyx-pipeline \
  coco_dataset="$POLYX_HOME/segmentation_datasets/first_batch/v1" \
  family=mask_rcnn \
  output_dir=/path/to/executions/mask-rcnn-tracer \
  protocol.epochs=1
```

`family.weights` names the torchvision detection factory
(`maskrcnn_resnet50_fpn_v2` by default). Setting it to `none` builds an
untrained model, which is useful for offline debugging when the pretrained
weights cannot be downloaded. The one-epoch demo is
`tests/test_pipeline_mask_rcnn.py::test_mask_rcnn_trains_from_coco_masks_for_one_epoch`,
and the real Lightning loop is exercised by
`test_one_epoch_completes_under_the_real_lightning_loop`.

The two segmentation families are proven interchangeable, and the model is the
only variable:

- `test_both_segmentation_families_resolve_the_same_protocol` asserts that
  `resolved_protocol.yaml` parses to the same mapping across a YOLO-seg run and
  a Mask R-CNN run over the same dataset (YAML equality, not byte equality);
- `test_the_shared_evaluator_scores_both_segmentation_families_identically`
  asserts the same figure names, criteria and operating-point scores from
  identical prediction geometry;
- `test_benchmark_augmentation_is_the_true_common_subset` asserts that both
  families declare and apply the same subset and the same HSV/flip magnitudes,
  and that `mosaic` is not part of the benchmark;
- `test_the_two_families_apply_the_same_effective_hsv_and_flip` compares the
  torchvision-side transform against Ultralytics' own `RandomHSV`/`RandomFlip`
  for the same seed, so parity is asserted on the pixels and boxes, not on the
  transform names.

## How one epoch is trained

The Ultralytics families share one wrapper:

- the pretrained network is rebuilt from its own yaml with the dataset's class
  count, and the pretrained weights are transferred into the matching layers;
- `train_dataloader` and `val_dataloader` reuse the Ultralytics dataset and
  batch collation over the job-time labels;
- `training_step` hands the batch to the Ultralytics network's own loss;
- `configure_optimizers` applies the protocol's optimizer, weight decay,
  momentum, warmup and cosine schedule;
- at the end of the epoch, Ultralytics validation scores the current weights
  and the checkpoint is written; its output directory is pointed at scratch so
  no global runs directory is touched and nothing lands in the artifact layout.

Mask R-CNN follows the same shape with torchvision's own pieces: a
`DataLoader` over the canonical COCO instances and their rasterized masks, the
detector's own loss dictionary summed into one Lightning loss, and the same
optimizer and schedule. Its per-epoch checkpoints hold the model state dict and
the validation hook polygonizes the predicted masks into the COCO currency the
shared evaluator already consumes.

At the end of every epoch each adapter writes two kinds of checkpoint. The
family-native export (`epoch_NNN.pt`, and `best.pt` restored from it) keeps its
historical shape for compatibility. `checkpoints/lightning/last.ckpt` is a real
Lightning checkpoint written by `Trainer.save_checkpoint`: it carries the
weights, optimizer, scheduler and epoch progress, and it is the file a resumed
execution hands to `Trainer.fit(ckpt_path=...)`. An Ultralytics `.pt` is not a
Lightning checkpoint and is never used for resume. If a progress snapshot has
no Lightning checkpoint (one written before this behaviour existed), the
execution refuses to resume instead of silently restarting the epoch loop.

On a machine without a GPU, set `protocol.amp.enabled=false`: FP16 AMP is a GPU
request recorded in the protocol, and Lightning cannot run FP16 mixed precision
on CPU.


## Decisions and vocabulary

See [ADR-0007](adr/0007-one-lightning-loop.md) for the single-loop decision,
[ADR-0009](adr/0009-coco-segmentation-dataset.md) for the COCO segmentation
dataset, [ADR-0010](adr/0010-optuna-objective.md) for the shared optimization
objective, and the [glossary](../GLOSSARY.md) for *run*, *protocol*, *campaign*,
*segmentation training dataset* and *COCO mask dataset*.
