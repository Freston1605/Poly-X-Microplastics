#!/usr/bin/env bash
# Body of the smoke job: prove the environment, run one short training
# execution and sync only the small artifacts back to the work filesystem.
set -euo pipefail

: "${POLYX_HPC_WORK:?set POLYX_HPC_WORK in the gitignored local config}"
: "${POLYX_HPC_SCRATCH:?set POLYX_HPC_SCRATCH in the gitignored local config}"
: "${POLYX_WEIGHTS_DIR:?set POLYX_WEIGHTS_DIR to the staged weights directory}"
: "${POLYX_WEIGHTS_MANIFEST:?set POLYX_WEIGHTS_MANIFEST to the sha256 manifest}"

export TMPDIR="${POLYX_HPC_SCRATCH}/polyx"
export POLYX_HOME="${POLYX_HPC_WORK}/userdata"
mkdir -p "${TMPDIR}" "${POLYX_HPC_WORK}/results"

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

RUN_DIR="${POLYX_HPC_WORK}/results/smoke-$(date +%Y%m%d-%H%M%S)"

run_polyx -m polyx.hpc.check_env
run_polyx -m polyx.hpc.stage_weights
run_polyx -m polyx.pipeline.command \
  data_yaml="${POLYX_HOME}/data/data.yaml" \
  output_dir="${RUN_DIR}" \
  backend=slurm \
  weights_dir="${POLYX_WEIGHTS_DIR}" \
  weights_manifest="${POLYX_WEIGHTS_MANIFEST}" \
  protocol.epochs=1
run_polyx -m polyx.hpc.sync_back "${RUN_DIR}" "${POLYX_HPC_WORK}/results/local"
