---
id: T007
title: Add src/polyx/training module and HPCC SLURM/Apptainer templates
wave: 2
deps: [T001]
status: pending
agent: null
base: null
worktree: null
task_branch: null
files:
  - src/polyx/training/dataset.py
  - src/polyx/training/train.py
  - src/polyx/training/evaluate.py
  - src/polyx/training/export.py
  - src/polyx/training/__init__.py
  - src/polyx/training/config.yaml
  - hpcc/sbatch/train.sbatch
  - hpcc/apptainer/polyx.def
  - hpcc/scripts/bootstrap_env.sh
  - hpcc/README.md
  - tests/test_hpcc_templates.py
---

# T007 — Add src/polyx/training module and HPCC SLURM/Apptainer templates

## Context

Success criterion SC7 requires a documented, reproducible remote-training workflow: SSH into the HPCC, submit a SLURM job running an Apptainer (Singularity) image that trains a YOLO model on GPU nodes and returns a versioned `.pt` + metrics, with no manual steps beyond editing a config. Datasets/outputs live in shared/project storage or scratch (never `$HOME`); scratch is purged so outputs must be staged back. The job must fail fast if the Singularity image or dataset path is missing (E5).

## Approach

- Create `src/polyx/training/` grouping dataset prep, train entrypoint, evaluate, export, driven by a portable YAML config (runs on laptop or cluster).
- Model artifact convention: `models/<name>-<version>.pt` plus sidecar metadata (dataset, params, metrics).
- Add `hpcc/sbatch/train.sbatch` (SLURM template with `--nv` GPU, explicit `-B` bind mounts, `APPTAINER_CACHEDIR`/`TMPDIR` on scratch, `PYTHONNOUSERSITE=1`, fail-fast check that the bound dataset path exists inside the container).
- Add `hpcc/apptainer/polyx.def` building a SIF from a Docker base (`pytorch/pytorch` or `nvidia/cuda`) + requirements.txt.
- Add `hpcc/scripts/bootstrap_env.sh` and `hpcc/README.md` documenting SSH/rsync sync and data placement.
- Add `tests/test_hpcc_templates.py` checking templates contain the fail-fast dataset-path check and no secrets/credentials.
- Moved from Interface contract: training config schema (YAML) `model`, `dataset_yaml`, `epochs`, `batch`, `imgsz`, `device`, `output_dir`, `artifact_name`; artifact path `models/<name>-<version>.pt` with sidecar `<name>-<version>.json` containing dataset, params, metrics.

## Interface contract

- None

## Intent coverage
- SC7

## Acceptance criteria

1. A maintainer can edit a config and run training locally or via the sbatch template.
2. The sbatch template fails fast with a clear error if the Singularity image or dataset path is missing (E5).
3. No SSH credentials or secrets are stored in the repo (N4).
4. `hpcc/README.md` documents SSH/rsync sync and data placement.

## Verify

```bash
python -m pytest tests/test_hpcc_templates.py -q
```

Heavy: no
## Log

- 2026-10-25 — created by planner
