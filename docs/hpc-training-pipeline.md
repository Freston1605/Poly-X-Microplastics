# Headless training pipeline (local backend)

The pipeline is a headless path parallel to the shipped GUI Trainer. It does
not change the Trainer or its native training wrapper. This page describes the
first slice: one YOLO detection training execution on the local backend.

## Running one execution

The command takes Hydra-style `key=value` overrides. Both `data_yaml` and
`output_dir` are required; every other value comes from the composed
configuration.

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

## Decisions and vocabulary

See [ADR-0007](adr/0007-one-lightning-loop.md) for the single-loop decision and
the [glossary](../GLOSSARY.md) for the generic terms *run* and *protocol*.
