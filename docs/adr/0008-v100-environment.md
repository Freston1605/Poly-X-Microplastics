# ADR-0008: V100/cu126 offline environment

- **Status:** Accepted
- **Date:** 2026-11-18
- **Landed in:** #195

## Context

The GPU partition of OCÉANO has NVIDIA V100 PCIe 32 GB cards with driver
560.35.03 and CUDA 12.6. A V100 has compute capability 7.0 (`sm_70`). The
headless training pipeline must use automatic mixed precision with FP16 to use
the Tensor Cores, and every package in the environment must be reproducible on
another cluster or locally.

The environment cannot be assembled from defaults for three reasons:

1. The default public deep-learning wheel is not the contract. Its CUDA build
   may drop `sm_70`, so a job imports cleanly and fails at kernel launch. The
   wheel that still compiles `sm_70` is the CUDA 12.6 one from the pinned
   PyTorch index.
2. Compute nodes have no internet access. Everything they install must already
   be on the work filesystem, and its bytes must be checkable after a copy.
3. Model weights are PyTorch pickles and therefore executable. Fetching one
   from a URL without a pinned digest is untrusted code execution.

A cluster without a container runtime must still be able to run the pipeline,
and no cluster-specific value (account, partition, key, username or personal
path) may be committed.

## Decision

1. The OCI definition (`container/Containerfile`) and the Apptainer definition
   (`container/hpc.def`) start from the same Python 3.11 slim base pinned by an
   amd64 manifest digest, plus a pinned `uv` version. The runtime stage
   installs exclusively from a downloaded wheelhouse with `--no-index`.
2. `torch==2.7.1+cu126` and `torchvision==0.22.1+cu126` come explicitly from
   `https://download.pytorch.org/whl/cu126`. They replace the default-index
   pins that `uv.lock` carries for those transitive dependencies. The default
   public wheel is not used.
3. `tools/hpc/build_wheelhouse.py` exports the lockfile, downloads every other
   wheel as pinned there, downloads the CUDA wheels with their runtime
   dependencies, builds the project's own wheel, writes an offline
   `requirements.txt` and a `SHA256SUMS` manifest. Installation on a compute
   node is `uv pip install --no-index --find-links <wheelhouse> -r
   <wheelhouse>/requirements.txt` plus `uv pip install --no-index --no-deps
   <wheelhouse>/polyx_microplastics-*.whl`: nothing is built from source on the
   node, and `NOTICE` travels inside the wheel.
4. Weights are downloaded only when a `sha256sum`-style manifest pins their
   digest. `polyx.hpc.weights.fetch_weights` re-verifies cached files, deletes
   rejected files and refuses an empty manifest. Digests come from the release,
   never from the downloaded bytes.
5. `python -m polyx.hpc.check_env` reports CUDA availability, the CUDA version,
   the architecture list from `torch.cuda.get_arch_list()`, the wheel's
   compiled architecture flags, and the result of a real FP16 autocast matmul.
   It exits non-zero when any check fails. With no GPU it reports the AMP pass
   as *not run*; the compiled flags are diagnostic only and never stand in for
   a device check. The hardware portion is a smoke job on OCEANO and is
   verified by ticket #199.
6. A conda environment plus the same pinned `uv` and wheelhouse is the
   documented fallback for clusters without a container runtime. Cluster
   values live in environment variables or a gitignored file backed by the
   committed `docs/hpc-environment.example.env` placeholder.

The rationale for the CUDA pin is therefore kernel compatibility with the V100
under driver 560, not a preference for a newer toolkit: `sm_70` support is what
the cu126 build guarantees and what the default build may not.

## Considered options

- **Use the default public wheel and check the architecture in the smoke
  job.** Rejected: it turns a resolvable build-time fact into a failed GPU
  allocation, and the architecture check would only detect the problem after
  cluster time was consumed.
- **Pin the framework in `pyproject.toml`.** Rejected for this change:
  dependency declarations belong to ticket #194. Constraining torch in the
  wheelhouse builder keeps the CUDA override explicit and in one place without
  fighting the lockfile.
- **Install from the network on each node.** Rejected: compute nodes have no
  internet, and repeated resolution is not reproducible.
- **Trust a weight because its URL is official.** Rejected: a URL can change
  or be intercepted; only a digest published with the release is evidence.
- **Report a pass when the wheel's compiled flags contain `sm_70`.** Rejected:
  that proves the binary could support the architecture, not that this machine
  has a working CUDA device or a successful AMP pass. It is the failure mode
  the local check exists to prevent.
- **Commit a filled-in cluster configuration for convenience.** Rejected: it
  would put accounts, paths or credentials in version control.

## Consequences

- Container and Apptainer builds are bit-for-bit pinned to the same base and
  framework build; the OCI one is verified locally with Podman. Apptainer is
  not installed locally, so its build is only executed on the cluster.
- Wheelhouses are self-describing and verifiable offline; a partial copy fails
  before a job starts.
- Weight downloads fail closed on a missing or wrong digest; a rejected file
  is removed so it cannot be reused accidentally.
- The local environment check has two honest outcomes on a GPU-less machine:
  CUDA absent and AMP not run. A PASS can only come from a real GPU, which is
  why the hardware criterion is delegated to #199.
- No cluster-specific secret, username or absolute personal path is tracked;
  contributors copy the committed placeholder into a gitignored local file.
- Documentation, the example configuration and this decision ship with the
  tooling they describe, in `docs/hpc-environment.md`.
