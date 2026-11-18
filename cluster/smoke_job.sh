#!/usr/bin/env bash
# Body of the smoke job: prove the environment, run one short training
# execution and sync only the small artifacts back to the work filesystem.
set -euo pipefail

: "${POLYX_HPC_WORK:?set POLYX_HPC_WORK in the gitignored local config}"
: "${POLYX_HPC_SCRATCH:?set POLYX_HPC_SCRATCH in the gitignored local config}"

export TMPDIR="${POLYX_HPC_SCRATCH}/polyx"
export POLYX_HOME="${POLYX_HPC_WORK}/userdata"
mkdir -p "${TMPDIR}" "${POLYX_HPC_WORK}/results"

RUN_DIR="${POLYX_HPC_WORK}/results/smoke-$(date +%Y%m%d-%H%M%S)"

python -m polyx.hpc.check_env
python -m polyx.pipeline.command \
  data_yaml="${POLYX_HOME}/data/data.yaml" \
  output_dir="${RUN_DIR}" \
  backend=slurm \
  protocol.epochs=1
python -m polyx.hpc.sync_back "${RUN_DIR}" "${POLYX_HPC_WORK}/results/local"
