# HPC workflow

This page is the map and the operating procedure for the headless training
pipeline on the OCÉANO cluster. It covers the three first-wave model families as
one benchmark campaign, the queue tiers used for each kind of job, checkpoint
resume and the small-artifact sync-back. The companion pages go deeper:

- [hpc-training-pipeline.md](hpc-training-pipeline.md) — the command, the
  protocol, the evaluator and the artifact layout;
- [hpc-environment.md](hpc-environment.md) — the pinned V100 environment, offline
  wheelhouse and pre-flight check;
- [hpc-submission.md](hpc-submission.md) — the work/scratch layout, the
  administrator checklist, the smoke-job procedure and the acknowledgement.

The cluster campaign and the smoke job are **Pending** cluster verification:
there is no OCÉANO access or GPU in the development environment, so no cluster
result is claimed here. The local multirun is implemented and exercised by the
project gate. See [README](../README.md) and [GLOSSARY](../GLOSSARY.md) for the
vocabulary (*run*, *protocol*, *campaign*).

## What a campaign is

A **campaign** is one Hydra multirun whose jobs are the model families, all
under the identical protocol and the same datasets. Each job is a normal
training execution: it writes its own resolved configuration, provenance,
checkpoints, metrics and report under `<campaign_root>/<family>/`. The model
family is the only variable.

`python -m polyx.pipeline.campaign_app --multirun` is the campaign command. The
Hydra basic launcher runs the jobs sequentially **inside one allocation**, so a
campaign never queues one SLURM job per family or per configuration. A
hyperparameter search is a different command:
`python -m polyx.pipeline.sweep_app --multirun` runs Optuna trials, also inside
one allocation (see the pipeline page and
[ADR-0010](adr/0010-optuna-objective.md)).

The three first-wave families and the source each consumes:

| Family | Framework | Dataset source |
| --- | --- | --- |
| `yolo_detection` | Ultralytics YOLO detection | `data_yaml` (YOLO box labels) |
| `yolo_segmentation` | Ultralytics YOLO-seg | `coco_dataset` (canonical COCO masks) |
| `mask_rcnn` | torchvision | `coco_dataset` (canonical COCO masks) |

A campaign job carries both sources; the family's task decides which one it
uses. A single-execution invocation still refuses the wrong source for the
family, so a segmenter cannot silently train on boxes.

## Weights

Jobs never download a model: `cluster/smoke_job.sh` and
`cluster/campaign_job.sh` require `POLYX_WEIGHTS_DIR` and
`POLYX_WEIGHTS_MANIFEST`, run `python -m polyx.hpc.stage_weights`, and pass both
values to every execution. `stage_weights` verifies the staged files against
the manifest and, when `POLYX_WEIGHTS_URL` is set on a networked node, fetches
the missing ones first. Training then resolves `family.weights` to a local,
verified file and fails closed if it is absent, unpinned or does not match its
digest. See the [weights section](hpc-environment.md#verified-model-weights) for
the manifest and staging details.

## Queue usage

| Job | Quality of service | GPUs | Allocation |
| --- | --- | --- | --- |
| Smoke / environment check | short test tier (`short`) | one (`--gres=gpu:1`) | one job |
| Campaign (three families, 150 epochs) | normal GPU tier (`normal`) | one (`--gres=gpu:1`) | one job, jobs sequential |
| Long single execution | long tier, only on request and after approval | one | one job |

Request one GPU per job and stay within the job-count and quality-of-service
limits. The long tier is used only when a specific job genuinely needs it.
Every job writes checkpoints to the work filesystem so a preemption or the
24-hour cap is recoverable by resubmitting with `resume=true`.

## Running the campaign on OCÉANO

The submission scripts read every cluster-specific value from the environment;
see the [work and scratch layout](hpc-submission.md#work-and-scratch-layout).
They never contain a username, key or personal path.

```sh
cp docs/hpc-environment.example.env .env.hpc   # fill in on the cluster
set -a; . ./.env.hpc; set +a
bash cluster/submit_campaign.sh
```

`cluster/submit_campaign.sh` submits one job to the normal GPU tier with
`--gres=gpu:1` and a 24-hour time limit (override with `POLYX_HPC_QOS` and
`POLYX_HPC_TIME`). `cluster/campaign_job.sh` runs the multirun and then syncs
each family back:

```sh
python -m polyx.pipeline.campaign_app --multirun \
  campaign_root="$POLYX_HPC_WORK/results/campaign-<timestamp>" \
  data_yaml="$POLYX_HOME/data/data.yaml" \
  coco_dataset="$POLYX_HOME/segmentation_datasets/first_batch/v1" \
  family=yolo_detection,yolo_segmentation,mask_rcnn \
  backend=slurm \
  protocol.epochs=150

python -m polyx.hpc.sync_back --campaign "$CAMPAIGN_ROOT" "$SYNC_ROOT"
```

Each family writes under `<campaign_root>/<family.run_name>`, which defaults to
the family name. To put two detection variants in one campaign, add family
configs that share `name: yolo_detection` but declare different `run_name` and
`weights`, and list them in the multirun; each variant gets its own directory
and its own metrics. The same command runs locally with the local backend for
debugging; drop `backend=slurm` and the environment variables. The project gate
proves the local multirun with real Hydra and real Lightning in
`tests/test_pipeline_campaign.py`.

### Resuming a preempted campaign

Set `POLYX_CAMPAIGN_ROOT` to the previous campaign root and `POLYX_RESUME=true`,
then submit again. The job passes `resume=true` to every family and each family
decides for itself:

- a family with a `progress.json` resumes from its Lightning checkpoint
  (`checkpoints/lightning/last.ckpt`); completed epochs are not rerun;
- a family whose `training_state.json` already exists for the same protocol is
  **skipped**, so a finished execution's metrics are never recomputed or
  overwritten;
- a family that never started runs normally, because there is no snapshot to
  resume;
- a root that holds an execution for a different protocol is refused; start a
  new campaign root instead.

The engine still refuses a resume without a snapshot, or with a snapshot that
has no Lightning checkpoint.

```sh
POLYX_CAMPAIGN_ROOT="$POLYX_HPC_WORK/results/campaign-20261118-101500" \
POLYX_RESUME=true bash cluster/submit_campaign.sh
```

## What is synced back

Only the agreed artifact manifest is copied, one directory per family:

```
<family>/config.yaml            resolved configuration
<family>/resolved_protocol.yaml the protocol every family shared
<family>/provenance.json        git commit, dataset hashes and audit
<family>/training_state.json    precision, AMP, early stopping, resume state
<family>/progress.json          last completed epoch and Lightning checkpoint
<family>/metrics.json           named figures, per-class figures, criterion
<family>/report.md              human-readable summary
<family>/logs/training.log      execution log
<family>/checkpoints/best.pt    kept weights (the only checkpoint copied)
```

Datasets, environments, intermediate conversions, the per-epoch checkpoint
series and the Lightning checkpoint stay on the cluster. The small artifacts
have a 1 MB cap; the kept checkpoint has its own larger cap. An oversized
artifact is an error, never a truncated copy. `polyx.hpc.sync_back` implements
this allow-list; `--campaign` iterates the family directories.

## Verification status

| Check | Status |
| --- | --- |
| Local three-family multirun under one protocol | Verified by `tests/test_pipeline_campaign.py` |
| Local real-Lightning resume for YOLO and Mask R-CNN | Verified by `tests/test_pipeline_resume_lightning.py` |
| Local campaign sync-back of every family | Verified by `tests/test_hpc_slurm_backend.py` |
| OCÉANO smoke job (GPU visibility, `sm_70`, AMP, one epoch) | **Pending** cluster verification |
| OCÉANO campaign (three families, 150 epochs) | **Pending** cluster verification |

The exact smoke procedure is in
[hpc-submission.md](hpc-submission.md#manual-smoke-run--pending-cluster-verification).
