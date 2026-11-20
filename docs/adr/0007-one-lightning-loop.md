# ADR-0007: One Lightning loop for every model family

- **Status:** Accepted (amended 2026-11-20 by #202: callbacks and validation)
- **Date:** 2026-11-18
- **Landed in:** #194

## Context

The shipped Trainer is a GUI flow wrapped around one native Ultralytics
training call, and the headless CLI exposes only a few flags. It has no seed,
no shared early stopping and no way to train the segmentation families the
benchmark needs. Adding a second native loop for each new family would make the
comparison unfair: every difference in methodology would be confounded with
the difference between models.

The illustrative Lightning YOLO wrapper from the pipeline discussion is not a
drop-in solution. Its loss shape, model construction and Lightning API calls
have to be checked against the installed versions, and a wrapper that merely
imports is not evidence that it trains. The wrapper is therefore kept behind
one command seam whose local execution could be driven end to end before any
cluster time was requested.

## Decision

**There is one training loop: a single PyTorch Lightning `Trainer`. Ultralytics
YOLO detection is its first adapter; every other family is a sibling adapter
under the same contract.**

The headless pipeline is a new path parallel to the GUI Trainer. The GUI and
its native training wrapper are left unchanged. Hydra composes a
framework-neutral **protocol** (split, seed, epochs, batch, image size,
optimizer, learning rate, momentum, weight decay, schedule, warmup,
augmentation subset, AMP and early stopping). The resolved protocol is written
to the artifacts and handed to the adapter unchanged; the adapter translates
it into framework calls instead of redefining any value.

The YOLO detection adapter wraps the Ultralytics neural module in a Lightning
module, reuses the Ultralytics dataset and batch collation, and returns
COCO-format predictions per validation epoch. Lightning, Ultralytics and torch are imported lazily, so
importing the command does not load a GPU stack. One shared evaluator scores
those predictions against COCO ground truth built from the existing label
codec; it always names the criterion, selects on the with-class operating-point
F1 at one confidence and IoU threshold, and reports localisation beside it.
COCO AP is reported separately and never selects a checkpoint.

The module logs that score as `with_class_score`, and two stock Lightning
callbacks own the rest: `ModelCheckpoint` keeps the strictly best score seen,
and `EarlyStopping` measures patience against the protocol's `min_delta`. A
marginal gain smaller than the delta therefore still replaces the kept
checkpoint while patience keeps expiring — the two thresholds are two
behaviours, not one flag. Checkpoints are written through an atomic
`CheckpointIO` (write beside, rename into place) and `save_last=True` keeps
the resume point complete under a SLURM time limit. Callback state, including
the wait count and the kept best, lives in the checkpoint, so a resumed run
continues the stopping decision instead of restarting it. Before publication
the kept checkpoint is validated once more, so the metrics and `best.pt`
describe the same weights.

Every execution emits the same artifact layout: resolved configuration,
protocol, provenance (git commit, dataset hashes and audit), checkpoints,
metrics, a report and logs. The local backend runs the same command as later
backends; only the backend composition changes.

## Considered options

- **One native training loop per family.** Rejected: it is exactly the
  confound the benchmark exists to remove. Each family would inherit a
  different schedule, seed handling and stopping rule.
- **A hand-rolled epoch loop.** Rejected: checkpointing, AMP and callback
  ordering would have to be reimplemented and would become a second
  methodology to keep in step with Lightning.
- **Extending the GUI Trainer's native wrapper.** Rejected: it couples the
  headless campaign to Qt and changes shipped detection-run behaviour, which
  ADR-0002 keeps box-only.
- **Trusting the discussion snippet as-is.** Rejected: it was written against
  an unverified API shape. The adapter is exercised through the command seam
  with a tiny fixture before it is used on a GPU.

## Consequences

- Adding a family means adding one adapter under the same contract; no change
  to the loop, evaluator or artifact layout is needed.
- Protocol parity is a file comparison: the resolved protocol in the artifacts
  is the same mapping every adapter received.
- FP16 AMP is requested through the protocol and recorded in the training
  state as `amp_requested`; the precision Lightning actually ran is recorded
  separately and `amp_active` follows it (on CPU, Lightning substitutes
  `bf16-mixed` for `16-mixed`). The GPU smoke job is the acceptance check for
  the real FP16 pass.
- The first adapter was detection-only. The COCO mask dataset and the YOLO-seg
  and Mask R-CNN adapters arrived in later tickets without changing this
  decision; they use the same loop, evaluator and artifact layout. Mask R-CNN
  keeps torchvision's own mask paste and only bounds it: the transform's
  `postprocess` is called in chunks of `MASK_PASTE_CHUNK` masks, each turned
  into a bit-packed mask before the next chunk is built, and validation runs one
  image per step. A 4K plate otherwise asks for hundreds of original-resolution
  masks at once, which the V100 cannot hold. The evaluator unpacks a packed
  mask only while computing a figure, so the exact raster is scored without a
  polygon round-trip and without keeping the arrays live.
- An interrupted execution resumes from a real Lightning checkpoint
  (`checkpoints/lightning/last.ckpt`, with the immutable
  `checkpoint_NNN.ckpt` beside it), which restores weights, optimizer,
  scheduler, epoch progress and the callbacks' state. A family-native `.pt`
  export is never passed to `Trainer.fit` as a checkpoint. Every family
  validates inside Lightning's validation loop; YOLO exports the epoch's
  network first and predicts from that file, because Ultralytics fuses and
  freezes the network it is handed. The export carries no staged `ema` entry,
  so the metric and `best.pt` describe the weights that actually trained.
- The training loop, its callbacks and the shared score live in
  `lightning_loop.py`; an adapter supplies only a `LightningModule` and a
  `LightningDataModule`. Adding a family cannot change the stopping rule,
  the schedule, the selection metric or the artifact record.
- Completion is a separate atomic record written after metrics, checkpoint,
  provenance and report. A final epoch left without that record is published,
  not retrained or skipped.
