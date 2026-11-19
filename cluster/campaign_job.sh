#!/usr/bin/env bash
# Body of the campaign job: three families in one Hydra multirun inside this
# allocation, then a small-artifact sync-back per family.
set -euo pipefail

: "${POLYX_HPC_WORK:?set POLYX_HPC_WORK in the gitignored local config}"
: "${POLYX_HPC_SCRATCH:?set POLYX_HPC_SCRATCH in the gitignored local config}"

export TMPDIR="${POLYX_HPC_SCRATCH}/polyx"
export POLYX_HOME="${POLYX_HPC_WORK}/userdata"
mkdir -p "${TMPDIR}" "${POLYX_HPC_WORK}/results"

# Reuse a root only when resuming the campaign; a new campaign gets a new one.
CAMPAIGN_ROOT="${POLYX_CAMPAIGN_ROOT:-${POLYX_HPC_WORK}/results/campaign-$(date +%Y%m%d-%H%M%S)}"
RESUME="${POLYX_RESUME:-false}"
SYNC_ROOT="${POLYX_HPC_WORK}/results/local/$(basename "${CAMPAIGN_ROOT}")"
mkdir -p "${CAMPAIGN_ROOT}" "${SYNC_ROOT}"

python -m polyx.pipeline.campaign_app --multirun \
  campaign_root="${CAMPAIGN_ROOT}" \
  data_yaml="${POLYX_HOME}/data/data.yaml" \
  coco_dataset="${POLYX_HOME}/segmentation_datasets/first_batch/v1" \
  family=yolo_detection,yolo_segmentation,mask_rcnn \
  backend=slurm \
  protocol.epochs=150 \
  resume="${RESUME}"

python -m polyx.hpc.sync_back --campaign "${CAMPAIGN_ROOT}" "${SYNC_ROOT}"
