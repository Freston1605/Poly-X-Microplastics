# ADR-0007: One Lightning loop for every model family

- **Status:** Accepted
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
module, reuses the Ultralytics data path and returns COCO-format predictions
per validation epoch. Lightning, Ultralytics and torch are imported lazily, so
importing the command does not load a GPU stack. One shared evaluator scores
those predictions against COCO ground truth built from the existing label
codec; it always names the criterion, selects on the with-class score and
reports localisation beside it. Early stopping watches that one score with the
protocol's patience and minimum delta, and the best epoch's weights are copied
to one stable checkpoint path.

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
  state. On a machine without a supported accelerator the setting remains a
  request; the GPU smoke job is the acceptance check for the real AMP pass.
- The first adapter was detection-only. The COCO mask dataset and the YOLO-seg
  and Mask R-CNN adapters arrived in later tickets without changing this
  decision; they use the same loop, evaluator and artifact layout.
- An interrupted execution resumes from a real Lightning checkpoint
  (`checkpoints/lightning/last.ckpt`), which restores weights, optimizer,
  scheduler and epoch progress. A family-native `.pt` export is kept for
  compatibility and best-weight restoration but is never passed to
  `Trainer.fit` as a checkpoint.
