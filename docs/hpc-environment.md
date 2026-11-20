# Portable V100 environment

This document describes the environment for the headless training pipeline on
the OCÉANO cluster and on any other machine with an NVIDIA V100. It covers the
pinned container, the offline wheelhouse, hash verification, the pre-flight
check and the fallback for clusters without a container runtime. The workflow
that uses this environment — partition requests, quality of service,
checkpoints and sync-back — is described in
[hpc-submission.md](hpc-submission.md); this page is only about making the
stack runnable and trustworthy. The decision and its rationale are recorded in
[ADR-0008](adr/0008-v100-environment.md).

## Why the framework is pinned

The GPU partition has NVIDIA V100 PCIe 32 GB cards with driver 560.35.03 and
CUDA 12.6. A V100 has compute capability 7.0, which PyTorch names `sm_70`. The
deep-learning build must be compiled for that architecture; the default public
wheel is not the contract and may resolve to a newer CUDA build that dropped
`sm_70`, so a job would fail at kernel launch rather than at import time.

The environment therefore pins:

- base image by manifest digest, not by tag;
- `uv` version `0.13.7`, the same version that writes `uv.lock`;
- `torch==2.7.1+cu126` and `torchvision==0.22.1+cu126` from
  `https://download.pytorch.org/whl/cu126`, which report `sm_70` in their
  compiled architecture list and match driver 560 / CUDA 12.6;
- every other package from `uv.lock`, with no floating versions.

`torch` and `torchvision` are transitive dependencies of Ultralytics and are
constrained, not declared directly, in `pyproject.toml` (ticket #194 owns
dependency declarations). The wheelhouse builder removes their default-index
pins and substitutes the CUDA wheels explicitly; it never relies on what the
default index happens to serve.

## Container-first

`container/Containerfile` is the OCI definition. It builds the wheelhouse from
the lockfile in one stage and installs from it with `--no-index` in the runtime
stage. Build and run it locally with Podman:

```sh
podman build -f container/Containerfile \
  --build-arg POLYX_SOURCE_SHA="$(git rev-parse HEAD)" \
  -t polyx-v100:local .
podman run --rm --device nvidia.com/gpu=all polyx-v100:local
```

The check is the container's default command. On a machine without a GPU it
prints that CUDA is unavailable and exits non-zero; that is the correct, honest
result, not a broken build.

The build stamps the exact commit into the image. Verify it before trusting a
training run: `podman run --rm --entrypoint cat polyx-v100:local /opt/polyx/SOURCE_SHA`
must print the same value as `git rev-parse HEAD`. Training provenance uses
`POLYX_SOURCE_SHA` when the image has no `.git` directory. The Apptainer build
is a separate cluster-side check; it consumes `source_sha.txt` through
`%files` and is not claimed as verified locally.

`container/hpc.def` is the Apptainer definition for the cluster. It starts from
the same pinned digest and installs from the same wheelhouse, which carries the
project's wheel as well as the dependencies. Build it on a login node, where the
build context contains a generated `wheelhouse/`:

```sh
python tools/hpc/build_wheelhouse.py --wheelhouse wheelhouse
git rev-parse HEAD > source_sha.txt
apptainer build polyx-v100.sif container/hpc.def
apptainer exec --nv polyx-v100.sif python -m polyx.hpc.check_env
```

Podman is available in the local development environment; Docker, Apptainer
and Singularity are not. The OCI build is the one verified locally. **No
Apptainer build has been run locally**, and the cluster build is a smoke job
for ticket #199, not a claim made here.

## Offline wheelhouse

`tools/hpc/build_wheelhouse.py` exports the pinned requirements from `uv.lock`,
downloads every wheel once on a networked machine, builds the project's own
wheel here, and writes a `SHA256SUMS` manifest plus a ready-to-use
`requirements.txt`. Building the project wheel on this side is what keeps the
node offline: the runtime image has no source tree, no build backend and no
index, and installs the package from the wheel like any other dependency. The
wheel build also needs `NOTICE`, which `pyproject.toml` force-includes for the
pycocotools licence; omitting it fails the build with
`Forced include not found`.

```sh
python tools/hpc/build_wheelhouse.py --wheelhouse wheelhouse
python tools/hpc/build_wheelhouse.py --dry-run   # plan only; no downloads
```

The generated requirements file contains the locked pins with the two CUDA
wheels substituted. On a compute node with no internet:

```sh
uv venv /work/polyx/.venv
VIRTUAL_ENV=/work/polyx/.venv uv pip install \
  --no-index --find-links wheelhouse -r wheelhouse/requirements.txt
VIRTUAL_ENV=/work/polyx/.venv uv pip install \
  --no-index --no-deps wheelhouse/polyx_microplastics-*.whl
```

Verify a copied wheelhouse before using it:

```sh
python -c 'from pathlib import Path; from polyx.hpc.hashing import verify_manifest; \
print(verify_manifest(Path("wheelhouse"), Path("wheelhouse/SHA256SUMS")))'
```

An empty list means every wheel is present and matches its digest. The build
and the verification are two separate steps so a partial copy is caught before
a job starts.

## Verified model weights

Model weights are PyTorch pickles: loading one executes code. The HPC path is
hash-first. Maintain the `sha256sum`-style manifest next to the weights:

```text
<64 hex digits>  bestdetectormedium.pt
```

Obtain the digest from the release that publishes the file, never from the file
you just downloaded. Then fetch with the manifest; a missing entry or a wrong
digest is a hard failure, and a rejected file is deleted so it cannot be picked
up by a later run:

```sh
export POLYX_WEIGHTS_URL=https://release-host.invalid/weights
python -c 'from pathlib import Path; from polyx.hpc.weights import fetch_weights; \
fetch_weights(Path("weights.sha256"), Path("weights"), base_url=__import__("os").environ["POLYX_WEIGHTS_URL"])'
```

`polyx-download-models --sha256 <digest>` remains available for interactive
use.

### How the training path uses the manifest

The job scripts require `POLYX_WEIGHTS_DIR` and `POLYX_WEIGHTS_MANIFEST` and run
`python -m polyx.hpc.stage_weights` before any training. On a login node with
`POLYX_WEIGHTS_URL` set, the staging step fetches the missing files and verifies
every digest; on a compute node the URL is unset and a missing file fails the
job with its name.

The pipeline then passes the same directory and manifest to every family
(`weights_dir`, `weights_manifest`). `family.weights` is resolved to a local
file and checked against the manifest; a file that is absent, unpinned or does
not match its digest stops the execution before the framework is called. The
Ultralytics adapter hands the absolute path to `YOLO`, and the torchvision
adapter loads the verified state dict into an untrained model instead of asking
for `weights=DEFAULT`. Only when neither the directory nor the manifest is
configured — a local debugging machine — does the framework use its own default
weights, and `training_state.json` records `weights_verified: false`.

An architecture declaration (`family.weights=yolov8n.yaml`) and an explicitly
untrained detector (`family.weights=none`) carry no weights to verify.

## Pre-flight environment check

```sh
python -m polyx.hpc.check_env
```

The command prints what it observed: CUDA availability, the CUDA version, the
architecture list reported by PyTorch, the compiled architecture list of the
wheel (visible even without a GPU), the device name, and the result of a real
FP16 matmul under `torch.autocast`. It exits zero only when all of these hold:

- CUDA is available;
- `sm_70` is in the reported architecture list;
- the runtime is CUDA 12.6;
- the AMP pass produced a finite FP16 result.

On a machine with no GPU it reports `CUDA available: no`, `AMP pass: not run`
and `Result: FAIL`, and exits non-zero. It never reports a pass it did not
observe and never substitutes the compiled architecture list for a real device
check. The compiled list is diagnostic only: `torch.cuda.get_arch_list()`
returns an empty list when no device is visible, which is exactly why the
hardware part of this criterion is verified as a smoke job on OCEANO by ticket
**#199**, not on the local development machine.

The offline test suite exercises both paths with a torch-shaped test double;
`tests/test_hpc_environment.py` is the named proof.

## Fallback: conda plus uv

On a cluster with no container runtime, create the same environment directly.
Run the first two commands on a node with internet access; the rest are
offline from the wheelhouse.

```sh
conda create -y -n polyx-v100 python=3.11
conda activate polyx-v100
python -m pip install uv==0.13.7
python tools/hpc/build_wheelhouse.py --wheelhouse /work/polyx/wheelhouse
uv venv /work/polyx/.venv
VIRTUAL_ENV=/work/polyx/.venv uv pip install \
  --no-index --find-links /work/polyx/wheelhouse \
  -r /work/polyx/wheelhouse/requirements.txt
VIRTUAL_ENV=/work/polyx/.venv uv pip install \
  --no-index --no-deps /work/polyx/wheelhouse/polyx_microplastics-*.whl
/work/polyx/.venv/bin/python -m polyx.hpc.check_env
```

Point `POLYX_HOME` and the job's work directory at the work filesystem, keep
temporary files in scratch and back up final artifacts to work. Never install
into the home quota.

## Cluster-specific configuration

No key, token, username or personal path is committed. Copy the committed
placeholder into a gitignored local file and fill it in on the cluster:

```sh
cp docs/hpc-environment.example.env .env.hpc
# .env.hpc is gitignored; edit it locally only
```

The HPC tools read ordinary environment variables; the pipeline reads
`POLYX_HOME` through the existing path module. Everything cluster-specific —
account, partition, work/scratch locations and the weights URL — stays in that
local file or in the job submission environment.

Before the first submission, confirm with the administrators: container
runtime availability, GPU request syntax, login-node internet access, account
category and quotas, data-origin declaration requirements, sponsor letter, and
whether the long quality-of-service tier needs approval. The full checklist,
the exact acknowledgement wording and the rule that only the resulting paper
is emailed to the cluster contacts are in
[hpc-submission.md](hpc-submission.md).
