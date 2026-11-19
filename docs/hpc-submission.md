# SLURM submission and OCÉANO workflow

This page covers how the headless training pipeline is submitted to the OCÉANO
cluster: the work and scratch layout, the short-test smoke job, checkpoint
resume, sync-back, the administrator checklist and the publication
acknowledgement. The campaign as a whole — the three-family multirun, its queue
tier and its per-family sync-back — is described in [hpc.md](hpc.md). The
environment itself — the pinned container, the offline wheelhouse, hash
verification and the pre-flight check — is described in
[hpc-environment.md](hpc-environment.md). No username, key, credential or
personal path is stored in this repository.

## Work and scratch layout

Every location is supplied through the environment. The helper
`polyx.hpc.layout.ClusterLayout.from_env` resolves them with `pathlib.Path` and
refuses a user data location outside the work filesystem.

| Purpose | Location | Variable |
| --- | --- | --- |
| Work root (datasets, results, checkpoints) | work filesystem | `POLYX_HPC_WORK` |
| Temporary files | scratch filesystem | `POLYX_HPC_SCRATCH` |
| User data location, pointing at work | `work/userdata` | `POLYX_HOME` |
| Account and partition for submission | — | `POLYX_HPC_ACCOUNT`, `POLYX_HPC_PARTITION` |
| Short test quality of service | `short` by default | `POLYX_HPC_QOS` |

Datasets and results live on work; temporary files go to scratch through
`TMPDIR`; checkpoints live on work so a later job can resume them. The
gitignored local configuration is `.env.hpc`, created from the committed
placeholder:

```sh
cp docs/hpc-environment.example.env .env.hpc
# Fill in .env.hpc on the cluster; it is ignored by the existing .env.* rule.
set -a; . ./.env.hpc; set +a
bash cluster/submit_smoke.sh
```

`cluster/submit_smoke.sh` builds one `sbatch` submission from those variables.
It requests exactly one GPU with `--gres=gpu:1` and selects the short test
quality of service with `--qos` (default `short`). Set `POLYX_HPC_QOS` to the
normal GPU tier for the sequential campaign; use the long tier only for a
specific job and only after approval. A multirun executes as trials inside one
allocation, never as an array of queued jobs. The campaign has its own
submission, `cluster/submit_campaign.sh`, which defaults to the normal GPU tier;
see [hpc.md](hpc.md).

## Campaign submission

`bash cluster/submit_campaign.sh` submits the three-family campaign as one job
on the normal GPU tier with one GPU and a 24-hour limit. The job body runs
`python -m polyx.pipeline.campaign_app --multirun` and then syncs every family
back with `python -m polyx.hpc.sync_back --campaign`. Reuse a campaign root with
`POLYX_CAMPAIGN_ROOT` and `POLYX_RESUME=true` to continue after a preemption or
a time limit. The exact commands and the sync-back artifact list are in
[hpc.md](hpc.md).

## Administrator checklist — confirm before the first submission

Confirm with the administrators before the first submission: container runtime
availability, GPU request syntax, login-node internet access, account category
and quotas, data-origin declaration requirements, sponsor letter, and the long quality-of-service approval.

- [ ] **Container runtime:** confirm which runtime is available on the compute
      node, whether the pinned Apptainer definition builds on a login node and
      which work/scratch paths must be bound into the container.
- [ ] **GPU request syntax:** confirm that `--gres=gpu:1` is the accepted way
      to request one V100 and that the GPU partition name is correct.
- [ ] **Login-node internet access:** confirm whether the login node can reach
      the package index and the weight host, so the wheelhouse and weights can
      be prepared there; compute nodes need no internet.
- [ ] **Account category and quotas:** confirm the account category, the work,
      home and scratch quotas, and the job-count and quality-of-service limits.
- [ ] **Data-origin declaration:** confirm what must be declared about the
      origin of the datasets and where the declaration is recorded.
- [ ] **Sponsor letter:** confirm whether a sponsor letter is required for the
      account request and who must provide it.
- [ ] **Long quality-of-service approval:** confirm whether the long
      quality-of-service tier needs separate approval and how to request it.

## Manual smoke run — Pending cluster verification

There is no OCÉANO access and no GPU in the local development environment, so
this acceptance criterion cannot be executed here. It is **Pending** cluster
verification; the procedure below is exact and no smoke result is claimed.

1. Prepare the wheelhouse and the hash-pinned weights on a node with internet
   access, then transfer them to work (see
   [hpc-environment.md](hpc-environment.md)).
2. Export the variables from `.env.hpc` and submit: `bash cluster/submit_smoke.sh`.
3. In the job, `python -m polyx.hpc.check_env` proves GPU visibility, prints the
   reported architecture list and the compiled architecture list, and runs a
   real FP16 AMP pass. It exits zero only when CUDA is available, `sm_70` (the
   V100 architecture) is reported, the runtime is CUDA 12.6 and the AMP pass
   produced a finite FP16 result.
4. The same job runs one short training execution:
   `python -m polyx.pipeline.command data_yaml=... output_dir=... backend=slurm protocol.epochs=1`.
   Confirm `training_state.json` reports `amp_active: true`, `device: cuda` and
   `epochs_completed: 1`, and that `metrics.json` names the selection criterion.
5. `python -m polyx.hpc.sync_back RUN_DIR LOCAL_DIR` copies only the small
   artifacts back to work. Inspect the job log for a clean exit.

The runnable check at step 3 works locally too: on a machine without a GPU it
prints `CUDA available: no` and `AMP pass: not run`, exits non-zero and never
invents a pass. That is the honest local result, not the cluster smoke test.

## Checkpoint resume

Each epoch writes a per-epoch checkpoint under `checkpoints/` on work and
refreshes `progress.json` with the completed epoch count, the last checkpoint,
the early-stopping bookkeeping and the resolved protocol. The progress snapshot
also names the real Lightning checkpoint at
`checkpoints/lightning/last.ckpt`, which is what makes a resumed execution
continue the optimizer, scheduler and epoch loop rather than restart it. If a
job is preempted or hits its time limit, resubmit with `resume=true`:

```sh
python -m polyx.pipeline.command data_yaml=... output_dir=WORK_RUN \
  backend=slurm protocol.epochs=150 resume=true
```

The engine refuses to resume without a progress snapshot, refuses to resume a
different protocol, and refuses a snapshot with no Lightning checkpoint, so a
changed methodology (or a restart) can never be mixed into an interrupted
execution. `tests/test_pipeline_local_tracer.py` proves, through the command
seam, that an interrupted execution continues from its last checkpoint and
keeps the earlier best weights. The real Lightning loop is proven by
`tests/test_pipeline_resume_lightning.py`, which interrupts a YOLO and a Mask
R-CNN run after one epoch and resumes both from a real Lightning checkpoint.

## Small-artifact sync-back

Only the agreed manifest is copied: resolved configuration, protocol,
provenance, training state, progress, metrics, report and logs, plus the kept
`best.pt` checkpoint. The per-epoch checkpoint series, the Lightning resume
checkpoint, datasets and intermediate predictions stay on the cluster. The small
artifacts share a 1 MB cap and the kept checkpoint has its own larger cap; an
allow-listed artifact above its cap is an error, not a partial copy. This is
implemented by `polyx.hpc.sync_back` and invoked by the smoke and campaign jobs;
`python -m polyx.hpc.sync_back --campaign CAMPAIGN_ROOT LOCAL_ROOT` syncs every
family into its own directory.

## Publication acknowledgement

Any paper that used the cluster carries this wording, the English form
published in the cluster access and use policy:

> This research/project/thesis/course was partially supported by HPC OCÉANO (FONDEQUIP NºEQM170214).

Only the resulting publication is emailed to the cluster contacts. Run
artifacts, logs and failed runs are never emailed. The contact addresses are
public and are deliberately not committed to the repository.
