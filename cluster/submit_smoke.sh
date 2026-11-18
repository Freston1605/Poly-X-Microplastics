#!/usr/bin/env bash
# Submit one smoke job to the short test quality of service with one GPU.
# Every cluster-specific value comes from the environment (see
# docs/hpc-submission.md); nothing here is tied to a person or a login name.
set -euo pipefail

: "${POLYX_HPC_ACCOUNT:?set POLYX_HPC_ACCOUNT in the gitignored local config}"
: "${POLYX_HPC_PARTITION:?set POLYX_HPC_PARTITION in the gitignored local config}"
: "${POLYX_HPC_WORK:?set POLYX_HPC_WORK in the gitignored local config}"
: "${POLYX_HPC_SCRATCH:?set POLYX_HPC_SCRATCH in the gitignored local config}"

QOS="${POLYX_HPC_QOS:-short}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${POLYX_HPC_SCRATCH}/logs"

sbatch \
  --job-name=polyx-smoke \
  --account="${POLYX_HPC_ACCOUNT}" \
  --partition="${POLYX_HPC_PARTITION}" \
  --qos="${QOS}" \
  --gres=gpu:1 \
  --nodes=1 \
  --time=00:30:00 \
  --output="${POLYX_HPC_SCRATCH}/logs/polyx-smoke-%j.out" \
  --error="${POLYX_HPC_SCRATCH}/logs/polyx-smoke-%j.err" \
  "${HERE}/smoke_job.sh"
