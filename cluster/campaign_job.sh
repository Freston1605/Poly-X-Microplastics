#!/usr/bin/env bash
# Body of the campaign job: three families in one Hydra multirun inside this
# allocation, then a small-artifact sync-back per family.
set -euo pipefail

: "${POLYX_HPC_WORK:?set POLYX_HPC_WORK in the gitignored local config}"
: "${POLYX_HPC_SCRATCH:?set POLYX_HPC_SCRATCH in the gitignored local config}"
: "${POLYX_WEIGHTS_DIR:?set POLYX_WEIGHTS_DIR to the staged weights directory}"
: "${POLYX_WEIGHTS_MANIFEST:?set POLYX_WEIGHTS_MANIFEST to the sha256 manifest}"

export TMPDIR="${POLYX_HPC_SCRATCH}/polyx"
export POLYX_HOME="${POLYX_HPC_WORK}/userdata"
mkdir -p "${TMPDIR}" "${POLYX_HPC_WORK}/results"

# Reuse a root only when resuming the campaign; a new campaign gets a new one.
CAMPAIGN_ROOT="${POLYX_CAMPAIGN_ROOT:-${POLYX_HPC_WORK}/results/campaign-$(date +%Y%m%d-%H%M%S)}"
RESUME="${POLYX_RESUME:-false}"
SYNC_ROOT="${POLYX_HPC_WORK}/results/local/$(basename "${CAMPAIGN_ROOT}")"
mkdir -p "${CAMPAIGN_ROOT}" "${SYNC_ROOT}"

# Container-first: POLYX_CONTAINER names the pinned image/definition. Without
# it the job assumes a pre-activated conda+uv environment on the node.
run_polyx() {
  if [ -n "${POLYX_CONTAINER:-}" ]; then
    if ! command -v apptainer >/dev/null 2>&1; then
      echo "ERROR: POLYX_CONTAINER is set but apptainer is not available" >&2
      exit 1
    fi
    apptainer exec --nv "${POLYX_CONTAINER}" python "$@"
  else
    python "$@"
  fi
}

run_polyx -m polyx.hpc.stage_weights
run_polyx -m polyx.pipeline.campaign_app --multirun \
  campaign_root="${CAMPAIGN_ROOT}" \
  data_yaml="${POLYX_HOME}/data/data.yaml" \
  coco_dataset="${POLYX_HOME}/segmentation_datasets/first_batch/v1" \
  family=yolo_detection,yolo_detection_v11,yolo_segmentation,yolo_segmentation_v11,mask_rcnn \
  backend=slurm \
  weights_dir="${POLYX_WEIGHTS_DIR}" \
  weights_manifest="${POLYX_WEIGHTS_MANIFEST}" \
  protocol.epochs=150 \
  resume="${RESUME}"

run_polyx -m polyx.hpc.sync_back --campaign "${CAMPAIGN_ROOT}" "${SYNC_ROOT}"
