# Headless training pipeline (local backend)

The pipeline is a headless path parallel to the shipped GUI Trainer. It does
not change the Trainer or its native training wrapper. This page describes the
local backend and the per-family executions: YOLO detection and YOLO-seg in
both v8 and v11 variants, plus Mask R-CNN — all sharing one protocol, one
Lightning loop, one evaluator and one artifact layout. The five variants can
also run as one multirun campaign; the cluster procedure is in
[hpc.md](hpc.md).

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

Those terminal lines are **machine tokens** (`OUTPUT:`, `CHECKPOINT:`,
`SELECTION:`, `CRITERION:`, `STUDY:`, `BEST_VALUE:` and the like), parsed by
job scripts and logs exactly as they are printed. They are deliberately not
wrapped in `_()`: they are not interface copy and a translated key could not be
consumed by a script. The artifact text (`report.md`, figure criteria, state
keys) is English by the project's source-language rule, and human-facing GUI
copy keeps going through the translation catalog.

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
  non-Ultralytics adapter built on torchvision. The `_v11` configs share the
  same adapters and differ only in staged weights and `run_name`. The task is a
  config value, never a code edit.
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
progress.json          # atomic per-epoch resume snapshot
report.md              # human-readable summary, criteria included
logs/training.log      # plain-text execution log
checkpoints/best.pt    # best live weights, selected by with-class F1
checkpoints/epoch_NNN.pt # live-weight export per epoch (no staged EMA)
checkpoints/lightning/checkpoint_NNN.ckpt # immutable per-epoch Lightning state
completion.json        # written last, only after every required artifact exists
```

`completion.json` is the completion authority. A campaign skips a family only
when it exists under the same protocol; a family with a progress snapshot but
no completion resumes, and a final epoch already on disk is published rather
than retrained. The log line is written before completion, and every JSON
artifact is published atomically. A fresh execution refuses an output directory
that already holds an execution — pass `resume=true` to continue it or choose a
new directory — so a new run can never inherit a stale completion marker.

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

- `with_class_score` — class-aware operating-point F1 at the protocol's
  `confidence` (default 0.25) and 0.50 box/mask IoU. It reuses the project's
  `Criterion`/`Score` definition: a well-placed box of the wrong polymer is a
  false positive of the predicted class and a false negative of the true one.
  This is the selection metric, and the checkpoint with the best value is the
  one kept as `best.pt`. The threshold and IoU rule are recorded beside it.
- `localisation_score` — the same operating-point F1 with the class ignored,
  reported beside the selection score.
- `mAP50` and `mAP50-95` — COCO interpolated average precision, class-aware,
  with the reference per-image/per-category 100-detection cap. The cap is
  applied per image **and category**, so a plate crowded with one polymer
  cannot push another class's confident detections out of the ranking.
  Predictions are collected at a low floor; each family may emit up to 1000 per
  image, far above the reference per-category cap, so a busy polymer cannot
  consume another class's share before the evaluator caps each category.
  Only the operating point applies `protocol.confidence`. Masks use
  a port of pycocotools' boundary-crossing rasterizer, so a polygon COCO counts
  as a hit is not emptied by integer rounding; the port is checked against
  pycocotools on the accepted input formats. A family that returns an instance
  mask at the original resolution (Mask R-CNN) has that mask scored directly,
  because recovering a polygon from it loses half a pixel per edge; YOLO-seg's
  own boundary polygons are scored as polygons, and asking it for full-resolution
  retina masks instead would upsample every candidate on every plate. AP is
  reported separately and never selects a checkpoint or drives Optuna.
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
cached under `scratch_dir/seg_datasets/<cache key>`. The key hashes the
canonical `dataset.json` and every image's actual bytes, so two stores with the
same JSON but different pixels cannot share a cache. Each COCO annotation must
have exactly one polygon ring: YOLO-seg would otherwise turn a multi-part
particle into several training instances while Mask R-CNN keeps it as one. Such
an export is refused before training, with an actionable error asking for one
annotation per part or a single exterior polygon. Two annotations collapse when
the pinned loader's own parsing gives them the same class and derived box:
`verify_image_label` reads each line, converts the normalized polygon with
`segments2boxes` and removes duplicate rows **after** that conversion, so a
triangle and a square with the same extrema are one target even though their
label lines differ. The converter builds its key from the line the shared label
codec is about to write and from that same box, and reports the image and
annotation ids. Label destinations come from the project's label codec, not
from a re-derived path.

The Mask R-CNN sibling walks the pinned loader's own path: the `ceil` resize of
the long side, the 1000-point segment resampling, the `LetterBox` pad, the
segment-derived boxes and `box_candidates`. Segmentation uses the loader's 0.01
area threshold and box-only detection uses 0.10. Boxes and segments are derived
from the serialized six-decimal polygon, which is what the loader reads back,
so a target that survives YOLO's rounding is kept and one that does not is
dropped.

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
- `train_dataloader` reuses the Ultralytics dataset and batch collation over
  the job-time labels;
- `training_step` hands the batch to the Ultralytics network's own loss;
- `configure_optimizers` applies the protocol's optimizer, weight decay,
  momentum, warmup and cosine schedule;
- at the end of the epoch, one Ultralytics prediction pass over the validation
  split produces COCO predictions; the shared evaluator scores them, and that
  score is the only one recorded. Ultralytics' own validator is not run a second
  time and no discarded figures are presented as a cross-check.

Mask R-CNN follows the same shape with torchvision's own pieces: a
`DataLoader` over the canonical COCO instances and their rasterized masks, the
detector's own loss dictionary summed into one Lightning loss, and the same
optimizer and schedule. Its per-epoch checkpoints hold the model state dict and
the validation hook polygonizes the predicted masks into the COCO currency the
shared evaluator already consumes.

At the end of every epoch each adapter writes two kinds of checkpoint. The
family-native export (`epoch_NNN.pt`, and `best.pt` restored from it) keeps its
historical shape for compatibility. Each epoch also writes its own
`checkpoints/lightning/checkpoint_NNN.ckpt` with `Trainer.save_checkpoint`: it
carries the weights, optimizer, scheduler and epoch progress, and the progress
snapshot is published atomically only after that checkpoint exists. A
preemption therefore restores a matched pair, never new weights with older
selection bookkeeping. The snapshot also carries the fit's own record — the
precision Lightning actually ran, the device, the verified weights and the
augmentation — so a publication recovered after the last epoch states the same
facts instead of inventing them. An Ultralytics `.pt` is not a Lightning checkpoint and
is never used for resume. If a progress snapshot has no Lightning checkpoint
(one written before this behaviour existed), the execution refuses to resume
instead of silently restarting the epoch loop.

The snapshot also names the dataset it trained on, as one digest over every
image and label it hashed. A resume compares that digest against the dataset the
new invocation resolved and refuses a mismatch: relabelled images or an edited
COCO export are a different dataset, and the kept weights and best figures
belong to the bytes that produced them. The protocol comparison alone could not
see that, so a changed dataset used to resume silently against the old
checkpoint.

Both the native export and `best.pt` are written from the live Lightning
network, without the staged checkpoint's `ema` entry: Ultralytics prefers an
`ema` key over `model` when loading, so re-saving the staged file would validate
and keep weights that never trained in this run. Validation loads a fresh
inference wrapper from that live-weight snapshot. Ultralytics' predictor fuses
and freezes the network it is handed, so scoring with the live training module
would break the next backward pass; the two-epoch real-framework test guards
this.

On a machine without a GPU, Lightning substitutes `bf16-mixed` for the requested
`16-mixed`. `training_state.json` records `amp_requested` (`16-mixed`) next to
`precision` (`bf16-mixed`) and derives `amp_active` from what actually ran, so a
CPU execution never claims FP16. FP16 itself remains a GPU acceptance check.

The wrapper is not merely import-compatible: `tests/test_pipeline_real_framework.py`
builds `YOLO("yolov8n.yaml")` offline, rebuilds its head for the fixture's three
classes and runs one real training epoch and prediction pass under real
Lightning.


## Sweeps

A hyperparameter search is a Hydra Optuna multirun, not a campaign. It runs
inside one allocation, one trial at a time, and reuses its SQLite study when the
same study name and storage are supplied:

```sh
python -m polyx.pipeline.sweep_app --multirun \
  output_dir="$POLYX_HPC_WORK/results/sweep-<timestamp>" \
  data_yaml="$POLYX_HOME/data/data.yaml" \
  hydra.sweeper.n_trials=20
python -m polyx.pipeline.sweep_best --output-dir "$POLYX_HPC_WORK/results/sweep-<timestamp>"
```

- `output_dir` is the sweep root; it is required and is not a trial directory.
- The study file derives from `hydra.sweeper.study_name`, so overriding only the
  name also moves the SQLite file; `sweep_best` resolves the same path. It is
  never on scratch, and a resumed multirun extends the same study.
- `PolyxOptunaSweeper` injects the actual study trial number into each job as
  `sweep_trial_id`; identity never comes from parameters or Hydra's resettable
  job index. Before asking for new trials it scans the persisted records and
  relaunches every trial left `RUNNING`: one with a progress snapshot continues
  its checkpoint, one without a snapshot retries the same parameters in a clean
  directory, and one whose execution finished but never reached Optuna is told
  from its own metrics. A recovered trial keeps its study number; two leases
  under `trials/.locks/` keep runners apart — a job holds `trial_NNN.lock`
  while it trains and a runner holds `trial_NNN.recovery.lock` while it
  relaunches one, so the recovery and the job it starts never block each
  other. The lease carries its owner's host and pid, so a lease whose local
  owner is gone is taken over at once; an owner that cannot be probed (another
  node) is taken over once it has shown no activity — no `progress.json` write
  — for `POLYX_LEASE_TTL` (an hour by default), so a killed job cannot block
  resumption. A directory belongs to a study only when its record names that
  study **and** that database; both sweep
  drivers publish the same `record.json` through one writer, and a trial whose
  number Optuna never sampled has no parameters to retry and is left untouched.
- The resolved study name and storage are written once to
  `<output_dir>/sweep_config.yaml` and reused by `sweep_best`. The export
  resolves the winner strictly by `study.best_trial.number` and requires the
  winner's record (same trial and study), `completion.json`, readable
  `metrics.json` and `checkpoints/best.pt`; a partial directory is reported as
  missing instead of being exported.
- The search space lives in `hydra.sweeper.params` and may only touch protocol
  hyperparameters; split, seed, augmentation subset and the evaluator are fixed.
  The sampler is a TPE seeded from the protocol, direction `maximize` and no
  pruner: trials are separate processes, so Lightning early stopping stops each
  one.
- The objective is the shared evaluator's with-class score, the same number a
  single execution reports.
- `python -m polyx.pipeline.sweep_best` reads the plugin's
  `optimization_results.yaml` and the per-trial artifacts and writes
  `<output_dir>/sweep_best.json` with the winning parameters, value, criterion
  and trial directory.

`python -m polyx.pipeline.sweep_app --multirun` is the only sweep entry point,
locally and on the cluster: the same command with `backend=slurm` is the job the
submission script runs. `sweep_space.py` refuses any declared name outside
`protocol.<searchable field>` and any direction other than `maximize`, so split,
seed, augmentation subset and the evaluator cannot enter the search space. The
plugin path is proven by `tests/test_pipeline_optuna_multirun.py`, including
study resume, the fixed-field guard and the best-artifact step.

## Decisions and vocabulary

See [ADR-0007](adr/0007-one-lightning-loop.md) for the single-loop decision,
[ADR-0009](adr/0009-coco-segmentation-dataset.md) for the COCO segmentation
dataset, [ADR-0010](adr/0010-optuna-objective.md) for the shared optimization
objective, and the [glossary](../GLOSSARY.md) for *run*, *protocol*, *campaign*,
*segmentation training dataset* and *COCO mask dataset*.
