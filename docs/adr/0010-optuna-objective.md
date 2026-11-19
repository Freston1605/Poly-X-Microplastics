# ADR-0010: Optuna optimizes the shared with-class score

- **Status:** Accepted
- **Date:** 2026-11-18
- **Landed in:** #200

## Context

A benchmark campaign tries several protocol configurations per model family.
If optimization introduced its own scoring routine, the tuned models would be
selected by a different methodology from the benchmark's reported figures, and
the comparison would no longer be fair. The pipeline already has one shared
evaluator (ADR-0007): it consumes COCO predictions, names its criterion and
selects on the with-class score on the real-domain validation split while
reporting localisation alongside. Hyperparameter search must reuse that number.

Hydra composes configuration and its Optuna sweeper plugin owns the trial loop;
hand-rolling a search would duplicate code the plugin already provides. Cluster
jobs are preempted and hit time limits, so study state has to live on the work
filesystem and survive a restart. Hydra launches each trial as a separate
process, which makes Optuna-level pruning across processes an unnecessary
source of complexity.

## Decision

**Optuna, through the Hydra sweeper, maximizes the shared evaluator's
with-class score. The search space is protocol-only, the study is a resumable
SQLite file on work, and no cross-process pruning is attempted.**

`conf/sweep.yaml` sets `direction: maximize`, a `TPESampler` seeded from
`protocol.seed`, `n_jobs: 1`, a small verification trial count and a declared
`hydra.sweeper.params` space containing only protocol hyperparameters (learning
rate, momentum, weight decay). Split, seed, augmentation subset and evaluator
are absent from that space and cannot vary between trials. Each trial writes a
normal execution under
`<output_dir>/trials/trial_<study trial number>`, so a resumed multirun cannot
overwrite an earlier trial's directory; `python -m polyx.pipeline.sweep_best`
folds the plugin's result into `sweep_best.json`. Both the real search multirun
(`python -m polyx.pipeline.sweep_app --multirun`) and the in-process smoke
driver (`polyx-pipeline --sweep`) read the same sweeper node and call the same
`run_execution`; the objective function returns `selection["value"]` from the
shared evaluator, never an adapter's own metric.

The study is `sqlite:///<work>/studies/<study-name>.db`, supplied by the work
layout from `POLYX_HPC_WORK` or by an explicit `hydra.sweeper.storage` override.
Optuna creates it with `load_if_exists=True`, so a resumed campaign reuses the
same study and its completed trials. The best trial's configuration, value,
metric name and criterion are written as `sweep_best.json` next to the
experiment's other artifacts; `sweep_config.yaml` records the resolved space
and the fields held fixed.

No pruner is configured and the driver passes `NopPruner`: there is no
cross-process pruning. Lightning's early stopping (ADR-0007) handles stopping
inside each trial on the same with-class score. This is a deliberate omission,
not a missing setting.

## Considered options

- **Optimize mAP50 or an adapter's native Ultralytics metric.** Rejected: it
  would be a second methodology and would reward localisation without polymer
  identity. Localisation stays a reported figure.
- **Hand-rolled random or grid search.** Rejected: Optuna owns optimization;
  a custom loop would duplicate the sweeper and lose SQLite resume.
- **Searching split, seed or augmentation subset.** Rejected: those define the
  experiment; changing them confounds every comparison.
- **Cross-process Optuna pruning.** Rejected: Hydra trials are separate
  processes and a single GPU is used sequentially. Per-trial Lightning early
  stopping gives the same protection without the coordination cost.
- **Storing the study on scratch or in the repository.** Rejected: scratch is
  temporary and the repository cannot be written from a compute node.

## Consequences

- Optimization cannot drift from reporting: both read `metrics.json` from the
  same evaluator and the criterion is saved with the best configuration.
- A resumed run appends trials to the existing study; the best artifact always
  reflects the whole study, not just the latest invocation.
- The offline wheelhouse must include `hydra-optuna-sweeper` and its pinned
  `optuna` release; all three Hydra pins are locked together.
- Tuning remains sequential by default. Distributed parallelism and Optuna
  pruning are future refinements, not part of this decision.
