# Synthesis

<!-- Written by the decider role. The planner treats Decisions as settled. -->

## Settled

- Stack baseline — Python 3.11, PySide6 6.7.2, Ultralytics 8.3.40, NumPy <2, Pillow 12.3.0 (INTENT.md Constraints; evidence-codebase.md § Finding: Engine lives in polyx/core... no, requirements.txt pins).
- Distribution channel — GitHub Releases only, no PyPI this milestone (INTENT.md Scope out / Constraints).
- Qt-free import contract — `polyx/core` must not import Qt or ultralytics (INTENT.md Prohibition N1).
- OS-specific code location — only `polyx/core/platform.py` (AGENTS.md; INTENT.md Prohibition N3).
- `DEFAULT_MODEL` — correct to `models/bestdetectormedium.pt` with graceful fallback (INTENT.md Scope in; evidence-codebase.md § Finding: DEFAULT_MODEL points to repo root).
- Empty `polyx/model/` + orphan pycache — remove after grep gate for `polyx.model`/`polyx.ml` (INTENT.md Prohibition N2; evidence-pitfalls.md § Finding: No hpcc/ or training/ module...; evidence-codebase.md).
- Docs layout/count drift — CLAUDE.md/README/AGENTS.md updated to `polyx/core` + `polyx/gui` and 876 tests (INTENT.md Success criteria 5; DOCS-AUDIT.md remediation queue).
- User-facing GUI strings — left in Spanish; only source identifiers/comments/docstrings translated (INTENT.md Prohibition N5).
- Legacy `legacy/` folder — not touched this milestone (INTENT.md Scope out).
- HPCC credentials — env/SSH agent only, never in repo (INTENT.md Prohibition N4; evidence-pitfalls.md § Finding: SSH credentials and datasets are already kept out of the repo).

## Decisions

### Stack (packaging toolchain)

- **Decision**: Create a PEP 621 `pyproject.toml` as the single build manifest (src layout, console scripts `polyx`/`polyx-gui`), keeping `requirements.txt` as the validated source of truth until the wheel install is proven on all three OSes; build wheels with the standard `build` frontend and publish via GitHub Actions. PyInstaller builds rely on `pyinstaller-hooks-contrib`'s built-in `hook-torch` + `hook-ultralytics` (>= 2024.9) plus an explicit `collect_data_files('ultralytics')` for `cfg/default.yaml`; require PyInstaller >= 6.11 on Linux.
- **Runner-up**: Writing custom PyInstaller hooks for torch/ultralytics — unnecessary because hooks-contrib already provides and tests them; only data-file collection and OS-specific binary quirks need handling (evidence-stack.md § Finding: pyinstaller-hooks-contrib ships first-party hooks).
- **Evidence**: evidence-stack.md § "pyinstaller-hooks-contrib ships first-party hooks for both torch and ultralytics (2024.9+)"; evidence-stack.md § "ultralytics data files (cfg/default.yaml) are not auto-collected by the hook in older setups"; evidence-stack.md § "torch hook has OS-specific binary handling"; evidence-pitfalls.md § "PyInstaller one-file builds require sys._MEIPASS-aware resource resolution".
- **Confidence**: high

### Layout (src/ migration)

- **Decision**: Move `polyx/` → `src/polyx/` and make `paths.ROOT` point at the repository/project root (not `src/`) by changing the parent count from `parents[2]` to `parents[3]`, so `MODELS_DIR`, `RUNS_DIR`, `DATA_DIR`, `ASSETS_DIR`, `DEFAULT_MODEL` and the `.venv` lookup in `platform.environment_python()` keep resolving to the project root. Update `launcher._spawn_module`'s `cwd` the same way. Update the i18n audit scripts (`tests/test_comments_ingles.py`, `test_identifiers_ingles.py`, `test_docstrings_ingles.py`) to use `REPO_ROOT / "src" / "polyx"`.
- **Runner-up**: Letting `ROOT` follow `src/` (relocating models/runs/assets under `src/`) — rejected because it breaks the documented data layout and the `.venv` discovery contract (evidence-pitfalls.md § Finding: paths.ROOT drives models/, runs/, data/, assets/ and the .venv lookup).
- **Evidence**: evidence-pitfalls.md § "src/ layout breaks all three current entry surfaces unless paths are re-derived"; evidence-pitfalls.md § "paths.ROOT drives models/, runs/, data/, assets/ and the .venv lookup"; evidence-stack.md § "src/ layout does not inherently break PyInstaller if the package is installed or pathex is set".
- **Confidence**: high

### Entry-point strategy (PyInstaller + installed CLI)

- **Decision**: Keep a top-level stub script (relocate `lanzar_polyx.py` to repo root, `from polyx.gui.launcher import main` guarded so it works when `src/` is on `sys.path` via `pathex=["src"]` in the spec or an editable install) and make `python -m polyx` / `python -m polyx.gui` the canonical installed entry points; replace `Poly-X.vbs`'s `raiz\polyx` folder probe with a probe for `src\polyx` (or drop it in favor of the installed console script) and point it at the installed entry point.
- **Runner-up**: Pointing PyInstaller directly at `src/polyx` as a directory — not supported by PyInstaller (issue #8074); a stub/`__main__.py` is required (evidence-stack.md § Finding: PyInstaller does not support module:entry_point syntax).
- **Evidence**: evidence-stack.md § "PyInstaller does not support module:entry_point syntax; a stub script is required"; evidence-stack.md § "src/ layout does not inherently break PyInstaller if the package is installed or pathex is set"; evidence-stack.md § "Poly-X.vbs assumes polyx/ sits at repo root — breaks under src/ layout"; evidence-pitfalls.md § "src/ layout breaks all three current entry surfaces unless paths are re-derived".
- **Confidence**: high

### Packaging & distribution (wheel + one-file exe per OS)

- **Decision**: Ship a pip wheel (built in CI) plus a PyInstaller one-file executable for Windows, Linux and macOS, attached to a GitHub Release; CI matrix runs the full 876-test suite on ubuntu-24.04/windows-latest/macos-latest with `QT_QPA_PLATFORM=offscreen`, then builds and smoke-tests each artifact before release. Asset/model resolution in frozen mode must prefer `sys._MEIPASS` when present.
- **Runner-up**: Conda-pack or a single cross-OS artifact — rejected: conda-pack is not a container runtime and couples to host glibc/CUDA; one-file-per-OS is required by SC2/N6 (evidence-pitfalls.md § "PyInstaller one-file builds require sys._MEIPASS-aware resource resolution"; INTENT.md Success criteria 2/6).
- **Evidence**: evidence-pitfalls.md § "PyInstaller one-file builds require sys._MEIPASS-aware resource resolution"; evidence-pitfalls.md § "Ultralytics/PyTorch bundling is large and needs hidden imports/hooks"; INTENT.md Success criteria 2, 4, 6.
- **Confidence**: medium — bundling size/hidden imports on Linux/macOS untested; verify in wave 1

### HPCC container image

- **Decision**: Use Apptainer (Linux Foundation fork of Singularity) as the runtime, building a SIF from a Docker base image (`pytorch/pytorch` or `nvidia/cuda`) plus `requirements.txt` via a `.def` file, run with `--nv` for GPU. Bind only the needed host paths explicitly, set `PYTHONNOUSERSITE=1`, and put `APPTAINER_CACHEDIR`/`TMPDIR` on scratch. Provide conda-pack only as a fallback if a cluster lacks Apptainer.
- **Runner-up**: Installing conda directly on the HPCC shared filesystem — explicitly discouraged by multiple centers (inode quotas, I/O load, system-library mismatches) (evidence-stack.md § Finding: conda environments directly on HPC filesystems are discouraged).
- **Evidence**: evidence-stack.md § "Apptainer (renamed Singularity) is the HPC-standard container runtime; conda-pack is a different, lighter artifact"; evidence-stack.md § "conda environments directly on HPC filesystems are discouraged; containerize them"; evidence-pitfalls.md § "Apptainer auto-binds $HOME/$PWD/tmp but not arbitrary host paths".
- **Confidence**: high

### HPCC data placement & sync

- **Decision**: Datasets and large outputs live in shared/project storage or scratch (never `$HOME`, which has small quotas); scratch is temporary and purged, so `.pt` + metrics must be copied back to a permanent location after training. Sync via `rsync` deltas over SSH; the sbatch template fails fast if the bound dataset path does not exist inside the container.
- **Runner-up**: Storing datasets in `$HOME` or only in scratch — hits quota or loses data on purge (evidence-pitfalls.md § Finding: HPCC datasets belong in shared/project storage, not home).
- **Evidence**: evidence-pitfalls.md § "HPCC datasets belong in shared/project storage, not home; scratch is temporary"; evidence-pitfalls.md § "rsync is the standard sync mechanism between laptop and HPCC"; evidence-pitfalls.md § "Apptainer auto-binds $HOME/$PWD/tmp but not arbitrary host paths".
- **Confidence**: high

### i18n audit approach

- **Decision**: Keep the existing audit scripts (`auditar_traduccion.py` + `tests/test_*_ingles.py`) as the CI gate but retarget their root from `REPO_ROOT / "polyx"` to `REPO_ROOT / "src" / "polyx"` after the move; scan all `.py` outside `tr()` for non-ASCII identifiers and non-English comments/docstrings; fail CI on any untranslated source string. Do not touch user-facing Spanish GUI strings (N5).
- **Runner-up**: Rewriting the audit as a new tool — unnecessary; the existing scripts already encode the rule and only need their path root updated (evidence-pitfalls.md § "Existing failure modes already present in the codebase").
- **Evidence**: evidence-pitfalls.md § "Existing failure modes already present in the codebase"; INTENT.md Success criteria 8; INTENT.md Prohibition N5.
- **Confidence**: high

### Version bump

- **Decision**: NEEDS-USER — choose between `2.1.0` (minor: additive packaging/release work on an already-shipped 2.x line) and `3.0.0` (major: first public release). No evidence leans either way; this is a values/branding call.
- **Runner-up**: n/a (pure values call, no lean).
- **Evidence**: INTENT.md Open questions ("Should the version be bumped to 2.1.0 or 3.0.0 for the first release?").
- **Confidence**: low

## For the planner

- **Wave-1 blockers**: PyInstaller bundling of torch + ultralytics on Linux/macOS (hidden imports, `sys._MEIPASS` resource resolution, MKL DLL collection on Windows) — smoke-test a minimal frozen app importing the detector before any release work; HPCC cluster specifics (shared group path, scratch purge policy, service account) are undocumented and must be confirmed with the maintainer; `paths.ROOT` parent-count change must land together with the `src/` move or all asset/model discovery breaks.
- **Walking skeleton**: `pip install -e .` on a clean checkout → `python -m polyx` launches the GUI with the detector page reporting "no model found" + download hint when weights are absent (E1/E2), then with `models/bestdetectormedium.pt` present auto-loads (SC3); one PyInstaller one-file build on one OS importing ultralytics end-to-end without crash (N6).
- **Pitfalls → tasks**: `paths.ROOT` parent count (`parents[2]`→`parents[3]`) and `launcher._spawn_module` cwd → layout task; `Poly-X.vbs` folder probe → entry-point task; i18n audit `REPO_ROOT / "polyx"` roots → audit task; orphan `segmentacion_aprendida` pycache + empty `polyx/model` → hygiene task after grep gate; CLAUDE.md/README stale layout & test counts → docs task; `APPTAINER_CACHEDIR`/`TMPDIR` on scratch + `PYTHONNOUSERSITE=1` + explicit `-B` binds → HPCC template task; fail-fast on missing dataset path → HPCC task (E5).

## User rulings

- (none yet — version bump and HPCC access are NEEDS-USER and carried to Still unknown)

## Still unknown

- Version bump (`2.1.0` vs `3.0.0`) — NEEDS-USER (values call, no lean).
- Who has SSH/SLURM access and whether a shared service account exists — NEEDS-USER (INTENT.md Open questions; no public source for PUCV cluster layout).
- PUCV HPCC specifics (shared group storage path, scratch purge policy, installed Apptainer version/CUDA) — plan: wave-1 spike with maintainer; accept risk of documenting as placeholders otherwise.
- PyInstaller hidden-import/hook behavior on macOS and Linux at build time — plan: wave-1 spike (smoke build) before release.