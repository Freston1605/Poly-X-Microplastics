#!/usr/bin/env bash
# Submit one benchmark campaign to the normal GPU tier with one GPU.
# Every cluster-specific value comes from the environment (see docs/hpc.md);
# nothing here is tied to a person or a login name.
set -euo pipefail

: "${POLYX_HPC_ACCOUNT:?set POLYX_HPC_ACCOUNT in the gitignored local config}"
: "${POLYX_HPC_PARTITION:?set POLYX_HPC_PARTITION in the gitignored local config}"
: "${POLYX_HPC_WORK:?set POLYX_HPC_WORK in the gitignored local config}"
: "${POLYX_HPC_SCRATCH:?set POLYX_HPC_SCRATCH in the gitignored local config}"

# The campaign runs sequentially on the normal GPU tier. The long tier is only
# for a specific job that needs it, and only after administrator approval.
QOS="${POLYX_HPC_QOS:-normal}"
TIME="${POLYX_HPC_TIME:-24:00:00}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${POLYX_HPC_SCRATCH}/logs"

sbatch \
  --job-name=polyx-campaign \
  --account="${POLYX_HPC_ACCOUNT}" \
  --partition="${POLYX_HPC_PARTITION}" \
  --qos="${QOS}" \
  --gres=gpu:1 \
  --nodes=1 \
  --time="${TIME}" \
  --export=ALL \
  --output="${POLYX_HPC_SCRATCH}/logs/polyx-campaign-%j.out" \
  --error="${POLYX_HPC_SCRATCH}/logs/polyx-campaign-%j.err" \
  "${HERE}/campaign_job.sh"
