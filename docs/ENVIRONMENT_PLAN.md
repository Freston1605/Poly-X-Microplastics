# Runtime and environment plan — Poly-X takeover

> **SUPERSEDED IN PART — read `docs/GSD_DIRECTIVES.md` §"READ FIRST" first.**
> Sections 0.2, 2, 3 and 4 below were written without reading `.gsd/DECISIONS.md` and `.gsd/KNOWLEDGE.md`. Three claims in them are wrong or superseded:
> - **§4's ROCm 10.1 / torch 2.14 recommendation is void.** D031 and L044 already probed that exact pairing on this RX 9070 XT and found `torchvision::nms` broken. The supported pair is ROCm 7.2.4 + torch 2.10.0+rocm7.2.4 + torchvision 0.25.0 + ultralytics 8.4.182, in a Docker GPU worker (D031).
> - **§0.2's "add a torch pin to requirements.txt" is void.** D039 forbids `requirements.txt` gaining a torch or sam2 pin; the fix belongs inside the isolated `.sam2env/` venv.
> - **§2's "no venv for the project at all" is wrong.** The M007 worktree runs the suite with the system `/usr/bin/python3` (3.14.8: PySide6, cv2, matplotlib, Pillow, psutil, PyYAML, pytest, numpy 2.5.3 — but no torch, no ultralytics). L029 records that arrangement. §3's Python-3.12 recommendation remains sound as a *reproducibility* measure, not as a blocker.
> The resource findings in §1, §5 and §6 stand.

**Snapshot:** 2026-10-20T19:01:22Z · schema 1.1 · `valid: true` · completeness `complete_with_informational_notes`
**Snapshot file:** `/home/Franco/.hermes/cache/scratch/resource-snapshot.json` (mode 0600)
**Scope:** the machine that will run the M007 SAM-teacher pipeline and the CPU measurement sweep, as it actually is today.

---

## 1. What this machine actually is

| Resource | Value | Source | Notes |
|---|---|---|---|
| CPU | 16 logical / **8 physical** | `psutil.cpu_count` | |
| CPU effective capacity | 16.0 cores | affinity 16, no cgroup quota, no scheduler | `limiting_sources: host_logical, process_affinity, python_process_count` |
| CPU worker ceiling | 16 | | |
| RAM total | **16.67 GB** | host total | cgroup v2 `memory.max` = null (non-root scope, no limit set) |
| RAM available | **8.68 GB** | host available | cgroup current usage 1.42 GB |
| Swap | 8.34 GB total, 8.32 GB free | | |
| Accelerator | **AMD Radeon RX 9070 XT**, 17.10 GB device-reported pool, `gfx1201` (RDNA4) | `amd-smi`, `management_query: visible` | `runtime_usable_devices: null` — `runtime_compatibility: not_tested` |
| Disk (working fs) | 929 GB total, **147 GB free** (85% used) | `shutil.disk_usage` + `statvfs` | user-available 146.6 GB; writable; single filesystem, no separate scratch/output fs |
| Scheduler | none | no Slurm vars | `enforcement: not_applicable` |
| Container | not detected | cgroup v2 `scope: non_root` | |
| Python | 3.14.7 system | | only 3.14 is installed |

**The single most important fact on this page:** RAM is the binding constraint, not the GPU. 8.68 GB available against a 17 GB VRAM card. And the *planner already told us what that means*:

```
GPU/accelerator-bound, 1 worker @ 8 GiB   -> status: insufficient_memory, 0 workers
CPU measurement sweep, 1 GiB/worker        -> 7 workers x 2 threads   (binding: memory)
I/O bench over phantom corpus, 512 MiB    -> 14 workers x 1 thread  (binding: memory)
```

A single 8 GiB worker does not fit. That is not a rounding error — it means **you cannot run one SAM-teacher process with an 8 GiB budget on this box while keeping a safe reserve.** The teacher is `hiera-tiny` (the smallest SAM 2.1 variant) at `sam2.1_hiera_tiny.pt`, 156 MB, so the *weights* are not the problem; the problem is the activation memory of image-encoder inference over full-resolution plates.

---

## 2. The environment as it stands: it does not exist yet

This is the finding that actually blocks the takeover, and it is not about hardware.

```
$ for v in /home/Franco/.venv /home/Franco/pytorch ...; do $v/bin/python -c "import torch"
```

**Every venv on the box is empty of the science stack.**

| venv | Python | torch | ultralytics | cv2 | numpy | scipy | PySide6 | pytest |
|---|---|---|---|---|---|---|---|---|
| `/home/Franco/.venv` | 3.14.8 | – | – | – | ✓ | – | – | ✓ |
| `/home/Franco/pytorch` | 3.14.8 | – | – | – | – | – | – | – |
| `/var/home/Franco/.venv` | 3.14.8 | – | – | – | ✓ | – | – | ✓ |
| `/var/home/Franco/pytorch` | 3.14.8 | – | – | – | – | – | – | – |

There is **no venv for the Poly-X project at all** — no `.venv` in the repo, no `venv/`, nothing. The `requirements.txt` pins (`ultralytics==8.3.40`, `numpy==1.26.4`, `opencv-python==4.10.0.84`, `PySide6==6.7.2`, `Pillow==12.3.0`, `matplotlib==3.9.2`, `pyqtgraph==0.13.7`, `psutil==6.0.0`) have never been installed on this machine. The 210-passing-tests claim in the sidecar evidence was produced somewhere else.

**Consequence for the takeover plan:** Block A's first step is not "merge M007" — it is **"build a reproducible environment,"** because you cannot run the suite, the bench, or the teacher without one. And the environment has a hard compatibility wall.

---

## 3. The compatibility wall: Python 3.14 vs `numpy==1.26.4`

`requirements.txt` pins `numpy==1.26.4` with a correct and well-documented reason (ultralytics 8.3.40 calls `np.trapz`, removed in numpy 2.0). But:

```
$ python3 -c "import json,urllib.request; ... numpy/1.26.4 ..."
 cp tags: ['cp310', 'cp311', 'cp312', 'cp39', 'pp39']
 has cp314: False
 classifiers: 3.9, 3.10, 3.11, 3.12   <- no 3.13, no 3.14
```

**numpy 1.26.4 has no Python 3.14 wheel, and 3.14 is the only interpreter installed.** `uv pip compile --python-version 3.14` "resolves" it by falling back to the sdist — which means a from-source C build of numpy on every fresh setup, on a machine with 8.7 GB of available RAM and a 3.14 C-API that numpy 1.26 predates by two years. That build will fail or produce a subtly broken numpy.

The repo's own CI sidesteps this: `.github/workflows/ci.yml` pins `python-version: '3.11'`, where numpy 1.26.4 has a proper `cp311` wheel. **So CI is green and the local machine cannot run the same code.** That is a reproducibility gap of exactly the kind your own literature review and the FONDECYT review both flag.

### The fix (and it is a one-liner, not a rewrite)

`uv` 0.12.24 is installed and can provision a matching interpreter on demand:

```
$ uv python list --all-versions | grep 3.12
 cpython-3.12.15-linux-x86_64-gnu   <download available>
```

**Decision: standardize the project on Python 3.12.** It is the newest version for which `numpy==1.26.4` ships a manylinux wheel, it matches what CI already uses (3.11 → 3.12 is a small, safe step, and 3.12 has a `cp312` numpy 1.26.4 wheel), and it is the version for which **AMD's own ROCm index publishes a `linux_x86_64` cp312 wheel** (see §4). Do this *before* anything else in the takeover plan.

```
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install -r requirements.txt          # resolves to cp312 wheels, no source builds
```

Then pin it in the repo so nobody has to rediscover it: add `.python-version` (3.12) and a `requires-python = ">=3.12,<3.13"` line, and change CI to `python-version: '3.12'`.

---

## 4. The GPU: visible, permissioned, and genuinely usable — but deliberately unused

Management visibility is not usability, so here is each gate separately.

| Gate | Result | Evidence |
|---|---|---|
| **Visibility** | ✅ PASS | `amd-smi` returned the RX 9070 XT; `rocm-smi` present; `HIP version: 7.1.52802-9999`; `gfx_target_version 120001` = gfx1201 |
| **Device node permission** | ✅ PASS | `/dev/kfd` opens `O_RDWR` as your user (not just present — actually opened). `/dev/dri/renderD128` present. `amdgpu` kernel module loaded. You are in `wheel` and `libvirt`. |
| **Driver/runtime** | ⚠️ PARTIAL | `amdgpu` module loaded, but **no `/opt/rocm` installation** on disk; `hipcc` is present but `rocm_agent_enumerator` is missing from `/usr/bin` |
| **Framework package** | ✅ AVAILABLE | `torch-2.14.0+rocm10.1.0-cp314-cp314-linux_x86_64.whl` and `torch-2.13.0+rocm10.1.0-cp314-...` exist in `stable.repo.amd.com/rocm/whl-next/`, plus `amd-torch-device-gfx1201-2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64.whl` and `cp313`/`cp314` variants |
| **Runtime usability** | ❌ NOT TESTED | No torch ROCm wheel installed anywhere; `runtime_usable_devices: null`; `runtime_compatibility: not_tested` |

**This is an unusually good position for an AMD box.** AMD publishes ROCm PyTorch wheels for `gfx1201` (RX 9070 XT) as *stable* (not nightly) for cp310–cp314 on Linux. The `amd-torch-device-gfx1201` package is the piece that carries the RDNA4 kernels, and it is published for cp312 and cp314 on linux_x86_64.

**But M007 deliberately does not use any of it.** From `tools/provision_sam2_runtime.py`:

```python
# CPU-only wheels: the teacher runs on the desktop CPU, and the CUDA wheels
# would add gigabytes of nvidia-* dependencies for a device this pipeline
# deliberately does not require.
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_SPEC = "torch==2.14.1+cpu"
```

That comment is wrong for *this* machine in one specific way: it says "CUDA wheels" but the threat on an AMD box is the **default PyPI torch wheel**, which is CUDA-bundled — `torch-2.14.1-cp314-cp314-manylinux_2_28_x86_64.whl` is **554.6 MB** and drags in `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`, `nvidia-cusparselt-cu13` and `triton==3.8.0`. On an AMD GPU that entire stack is dead weight: it consumes disk, it consumes install time, and it can never execute a kernel.

**So: the CPU-only pin is right in intent, wrong in reasoning, and it should be rewritten to say why.** Concretely:

- **Keep the CPU-only decision for the teacher** (M007's own bench is designed around it, and reproducibility of published numbers should not depend on a consumer GPU's ROCm stack).
- **Fix the comment and the pin mechanism.** Replace the hand-rolled `CPU_INDEX` constant with an explicit extra-index, and add a guard that fails loudly if a CUDA-flavoured torch is ever resolved. Right now `requirements.txt` has **no torch pin at all** — torch arrives transitively via `ultralytics==8.3.40`, which means the *default PyPI* (CUDA-bundled, 554 MB) wheel is what you get unless you intervene. **Add an explicit `torch` pin with the CPU extra index.**

**The genuinely valuable option, which is a decision for you and not for me:**

| | CPU-only torch | ROCm torch on the 9070 XT |
|---|---|---|
| Install | 196 MB, no extras | ~2–3 GB with `amd-torch-device-gfx1201` |
| Teacher throughput | Slow (hours for a corpus) | Potentially 10–30× faster |
| Reproducibility of published numbers | Highest (no GPU in the loop) | Depends on driver/ROCm version being pinned |
| Risk | None | gfx1201 kernel availability; ROCm version vs. your kernel driver; AMD notes training on RX 9070 can hit GPU resets and suggests `TORCH_BLAS_PREFER_HIPBLASLT=0` |
| Python | 3.12 or 3.14 | 3.12 (cp312 wheel confirmed) |

My recommendation: **publish results from the CPU-only path** (it is what M007 is already built and tested against, and it is the defensible choice for a Methods section), but **run your own iteration loops on the GPU** if it proves out. Keep the two paths explicitly separate in the provenance record, and never mix CPU-produced and GPU-produced numbers in one table without saying which produced which. If you do adopt ROCm, pin the ROCm version, the driver, and the exact wheel URLs — the `amd-torch-device-gfx1201` split is a variant wheel, and unpinned variant wheels are a reproducibility hazard.

Note also the ROCm pitfall that bites RDNA4 specifically: **nightly ROCm wheels lack gfx1201 kernels** and fail with `device kernel image is invalid`. Use the stable set.

---

## 5. Revised takeover plan, re-sequenced by what the hardware allows

The original plan (docs/TAKEOVER_PLAN.md) was written without this resource data. Three changes, one of them significant.

### Block 0 (NEW, week 0) — Make the machine able to run the code

This precedes everything, because nothing else is verifiable until it exists.

- **0.1** `uv python install 3.12`; `uv venv --python 3.12 .venv`. Add `.python-version` and `requires-python = ">=3.12,<3.13"` to the repo. Change CI from 3.11 → 3.12.
- **0.2** Install with an explicit CPU index and an explicit torch pin, so the 554 MB CUDA wheel can never be resolved:
  ```
  # requirements.txt — add explicitly, do not let ultralytics pull torch transitively
  --extra-index-url https://download.pytorch.org/whl/cpu
  torch==2.14.1+cpu
  ```
  Verify no `nvidia-*` package lands in the venv (`pip list | grep nvidia` → empty). This is checkable in CI and should be a test.
- **0.3** Install the repo requirements and **run the full suite**. Record the real passing count. The sidecar says 210; AGENTS.md says 128; the tree has since gained the translation tests and M007's 11 new files. **Three different numbers is exactly the kind of thing a reviewer notices.** Fix AGENTS.md to the measured value.
- **0.4** Install the M007 SAM runtime via `tools/provision_sam2_runtime.py --check` and confirm `torch`, `torchvision`, `sam2` import and the checkpoint SHA256 matches. Confirm no second torch appears in the venv.
- **0.5** Write down the reference environment in the repo (`docs/environment.md`): Python 3.12.x, the exact pins, the CPU-only decision and its reason, the hardware, and the SAM checkpoint hash. This is the artifact that makes a future number checkable.

### Block A (revised) — consolidate, now that the suite can run

Unchanged in content, but now executable:
- **A1** Merge `milestone/M007` (15 commits, 198 files, +49,327) into `main`. It is 15 ahead of the merge base with zero conflicts.
- **A2** Run the suite on the merged tree; fix the AGENTS.md test count. *Now possible because of 0.3.*
- **A3** Add the **ultralytics-present CI job**. Your CI installs exact pins — confirm torch is among them, then add a one-line smoke test that imports `polyx.model.yolo_wrap` and runs a single inference. A green suite that never touches the ML path is not evidence.
- **A4** Wire the **M007 fidelity bench** into CI on the pinned phantom corpus (the branch's `--check` recomputes the baseline from committed pixels with no torch — exactly CI-shaped). This makes the segmentation claim machine-checkable by a reviewer instead of asserted in a Markdown table.
- **A5** Commit the literature review into the repo (`docs/literature/`, currently untracked). Commit the PRISMA artifacts; leave the 4.1 MB `records_dedup.jsonl` ignored.
- **A6** Delete the empty `polyx/core/`, `polyx/detector/`, `polyx/trainer/` directories (0 `.py` files each, only stale `__pycache__`), and confirm `__pycache__/`/`*.pyc` are gitignored — 228 `.pyc` files are currently untracked noise.

### Block C (revised) — the memory budget is the real constraint

The CLI work is unchanged, but the *runtime shape* has to fit 8.7 GB of RAM:

- **C1** `polyx/cli/{detect,evaluate,audit}`. Unchanged — and now more valuable, because a headless path is what lets CI exercise the real pipeline within a memory budget.
- **C2** JSON run record. Unchanged.
- **C3** Detection probability vs. particle size (Tanoiri 2026 curve). Unchanged.
- **C4** Blanks, recovery, detection limit. Unchanged.
- **C5 (NEW)** **Make the CLI streaming and bounded.** A plate is ~2,000–4,800 px; the teacher must not hold a whole corpus in RAM. Process image-by-image, bound the worker pool at **7 workers × 2 threads** for CPU measurement work (the planner's figure) and **1 worker** for teacher inference, with an explicit per-image memory ceiling. Record the measured peak RSS in the run record so the numbers in the paper are accompanied by the resource envelope that produced them.
- **C6 (NEW)** **Choose one teacher concurrency and record it.** With 8.68 GB available and 8.32 GB of swap, the safe envelope is 1–2 teacher processes. Batch size 1, `num_workers=0` for the dataloader (workers multiply memory), and pin `OMP_NUM_THREADS` so the 16 logical cores do not thrash. A GPU path would change this by 10–30×, which is the argument for testing the ROCm install — but the published numbers should be reproducible on the CPU path.

### Blocks B, D, E, F — unchanged

The architecture (B), the science (D), the provenance/licensing (E) and the proposal reconciliation (F) are unaffected by the hardware. **But D now has a hard scheduling implication:** the nine-shape taxonomy (D1) and the detection-probability curves (D3, C3) are re-measurements over the *whole* corpus. On a CPU-only teacher that is a multi-hour-to-multi-day job, and it must be scheduled **after** the pipeline is frozen, which is what the original plan already said. The resource data just makes the constraint concrete: **one corpus re-measurement per pipeline change, not three.** Get the pipeline right first.

---

## 6. The disk number that matters more than it looks

147 GB free, 85% used, **one filesystem** for source, scratch, models and outputs. The skill's guidance applies directly: free space on the input filesystem does not establish space on the output filesystem, because there is only one.

Budget what actually lands there:

| Item | Size | Note |
|---|---|---|
| Repo (incl. `.git` 59 MB) | 148 MB | |
| `torch==2.14.1+cpu` | 196 MB | |
| `torchvision` CPU | ~7 MB | |
| `sam2.1_hiera_tiny.pt` | 156 MB | |
| ROCm path instead | **~2–3 GB** | `amd-torch-device-gfx1201` + `rocm-sdk` variants |
| **Avoided**: default CUDA torch | 554 MB + ~2.5 GB of `nvidia-*`/`triton` | never installable on an AMD card |
| Teacher mask corpus (one corpus) | hundreds of MB to GB | scales with image count × resolution |
| Training runs, `runs/` | GB-scale per experiment | `runs/` is gitignored |
| pip + uv caches | GB-scale | purgeable |

**147 GB is enough for the CPU path with room to spare. It is tight but workable for the ROCm path plus multiple training runs.** Two actions: (a) add the ROCm-vs-CPU disk delta to your decision if you go the GPU route; (b) **purge the pip/uv cache and set a retention policy for `runs/` before starting the corpus work**, because with 85% already used, a 3-epoch YOLO-seg training sweep that writes checkpoints per run is the single most likely thing to fill the disk mid-milestone.

Note also that `runs/` and `runs_train/` are gitignored, and `models/` + `data_microplastico/` are gitignored. That is correct for size but means **the published numbers are not reproducible from the repo alone** — which is Block E5, and it is now the item most likely to be forgotten, because everything else looks self-contained.

---

## 7. Verification notes

- **The snapshot is a real artifact**, taken with the skill's detector, validated (`valid: true`, zero errors, schema 1.1), stored at 0600 in the scratch dir.
- **No stress test, benchmark, large allocation, or write probe was run.** No driver was installed or mutated. No device was reset. The accelerator diagnostic plan was generated but **not executed** — the skill forbids it without a manual runtime check, and the runtime check requires installing a torch ROCm wheel, which is your decision, not mine.
- **The AMD index query was read-only HTTPS**, listing published wheel filenames. It establishes *availability*, not usability.
- **The compatibility findings are from PyPI/AMD metadata**, not from an install: numpy 1.26.4's cp-tag set and classifiers, torch 2.14.1's wheel sizes and `nvidia-*` dependencies, and the `gfx1201` wheel list.
- **The ROCm runtime usability is explicitly untested.** `runtime_usable_devices: null` and `runtime_compatibility: not_tested` in the snapshot are the accurate state, and they should stay that way in any document you publish until you have actually run `torch.cuda.is_available()` (which for ROCm reports through the same API) in a ROCm-enabled venv and seen it return a device.
- **The `hipcc` ROCm 7.1 toolchain is present but `/opt/rocm` is not installed**, and `rocm_agent_enumerator` is missing from `/usr/bin`. That combination is consistent with a partial ROCm install, and it is the thing to resolve first if you pursue the GPU path.
- **One thing I did not verify:** whether the RX 9070 XT's 17.10 GB "device-reported pool" is backed by the full 16 GB of VRAM available to a torch process, or is a ROCm-management view that includes some reserved memory. Treat 16 GB as the practical ceiling and less for training with activations.
