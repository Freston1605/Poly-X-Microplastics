# Evidence — pitfalls

Dimension: pitfalls
Questions assigned: Does `src/` layout break the existing `lanzar_polyx.py` / `Poly-X.vbs` launchers?; Where do datasets live on the HPCC (shared scratch vs. user home) and how are they synced?

## Finding: src/ layout breaks all three current entry surfaces unless paths are re-derived

- **Claim**: The repo-root layout is hard-coded in at least five places that assume `polyx/` sits directly under the project root: `Poly-X.vbs` checks `raiz & "\polyx"` and runs `python -m polyx.launcher` with `CurrentDirectory = raiz`; `lanzar_polyx.py` does `from polyx.gui.launcher import main` (only works if repo root is on sys.path); `polyx/gui/launcher.py::_spawn_module` sets `cwd=Path(__file__).resolve().parents[2]` (repo root today, would become `src/` after the move); `polyx/core/paths.py` sets `ROOT = Path(__file__).resolve().parents[2]` (repo root today, would become `src/`); and tests `test_arranque_instrumentado.py`, `test_launcher_trainer.py`, `test_trainer_runner.py` assert those exact parent counts. None of these auto-adjust to `src/polyx/`.
- **Source**: `Poly-X.vbs` lines 22-44; `lanzar_polyx.py`; `polyx/gui/launcher.py` `_spawn_module`; `polyx/core/paths.py` line 4; `tests/test_arranque_instrumentado.py:91`; `tests/test_launcher_trainer.py:74`; `tests/test_trainer_runner.py:47-48`
- **Confidence**: high
- **Why it matters here**: INTENT SC1 requires `python -m polyx` / `python -m polyx.gui` to launch after the move; the VBS/PyInstaller shim and subprocess cwd must be re-pointed or every secondary process launches in the wrong directory.

## Finding: VBS launcher's "polyx folder exists" gate becomes permanently false after the move

- **Claim**: `Poly-X.vbs` refuses to run unless `raiz\polyx` exists next to it; under `src/` layout that folder is `src\polyx`, so the launcher would always show "No se encuentra la carpeta 'polyx'" and quit. The same applies to `iniciar_polyx.bat`/`SETUP.bat` conventions that probe `.venv` and `polyx` at root.
- **Source**: `Poly-X.vbs` lines 24-30; INTENT.md scope in (move `polyx/` → `src/polyx/`)
- **Confidence**: high
- **Why it matters here**: The Windows "double-click to run" UX (a stated distribution path) breaks unless the launcher probes `src\polyx` or is replaced by the installed console script.

## Finding: paths.ROOT drives models/, runs/, data/, assets/ and the .venv lookup

- **Claim**: `paths.ROOT` is the single anchor for `MODELS_DIR`, `RUNS_DIR`, `DATA_DIR`, `ASSETS_DIR`, and `DEFAULT_MODEL`; `platform.environment_python()` imports `ROOT` from `polyx.core.paths` to find `.venv`. Moving to `src/` without changing the parent count relocates all data directories under `src/` and breaks venv discovery; `ASSETS_DIR = ROOT / "polyx" / "assets"` would become `src/polyx/assets` only if `ROOT` stays repo root.
- **Source**: `polyx/core/paths.py`; `polyx/core/platform.py` `environment_python()`
- **Confidence**: high
- **Why it matters here**: INTENT SC3 (auto-discover `models/bestdetectormedium.pt`) and the runs/assets layout depend on `ROOT` pointing at the repo/project root, not `src/`.

## Finding: PyInstaller one-file builds require sys._MEIPASS-aware resource resolution

- **Claim**: PyInstaller extracts `--add-data` files into a temp dir exposed as `sys._MEIPASS` in one-file mode; code that resolves assets via `__file__`/relative paths or `sys.executable` breaks in the bundle. The existing `lanzar_polyx.py` entry shim and `paths.ROOT` parent-count logic are both vulnerable because they assume a source tree.
- **Source**: PyInstaller docs "Using Spec Files" / "Advanced Topics" (pyinstaller.org/en/stable); issue #7951 discussion on referencing resources via `sys._MEIPASS` or `__file__`
- **Confidence**: medium
- **Why it matters here**: INTENT SC2 requires one-file executables on three OSes that do not crash on launch; asset/model discovery must use a helper that prefers `sys._MEIPASS` when frozen.

## Finding: Ultralytics/PyTorch bundling is large and needs hidden imports/hooks

- **Claim**: Bundling torch + ultralytics into a one-file executable produces hundreds of MB and commonly fails on hidden imports (torch, ultralytics, cv2) unless hooks/`--hidden-import`/`--collect-all` are used; the risk is explicitly listed in INTENT.md Risks.
- **Source**: INTENT.md Risks ("PyInstaller + PyTorch/Ultralytics bundling size and hidden imports on Linux/macOS (untested)"); PyInstaller usage docs on `--hidden-import`/`--collect-all`
- **Confidence**: medium
- **Why it matters here**: SC2/N6 require smoke-testing each built artifact; without hooks the exe crashes on first import of the detector.

## Finding: HPCC datasets belong in shared/project storage, not home; scratch is temporary

- **Claim**: On HPC clusters, home directories carry small quotas and are for code/config, while datasets and large outputs live in shared project/group storage or scratch; scratch is fast but purged after inactivity and discarded when a job ends, so results must be copied back to a permanent location.
- **Source**: HCC-DOCS "Using scratch space" (hcc.unl.edu); CHPC/Utah storage services; Stanford RC filesystems doc; RIT storage tutorial
- **Confidence**: high
- **Why it matters here**: INTENT SC7 requires a reproducible remote-training workflow; putting datasets in `$HOME` hits quota and putting them only in scratch loses them. Outputs (`.pt` + metrics) must be staged back.

## Finding: rsync is the standard sync mechanism between laptop and HPCC

- **Claim**: `rsync` is the workhorse for HPC data transfer because it sends only deltas; typical workflow is `rsync -avz` of the dataset into the cluster project/scratch dir and back of `runs/`/weights after training. Apptainer/Singularity containers need explicit bind mounts (`-B`) for host paths, and `APPTAINER_CACHEDIR`/`TMPDIR` should point at scratch to avoid home quota.
- **Source**: NCSU HPC file transfer docs; Dartmouth HPC cookbook; Berkeley Savio Apptainer docs; mssm minerva Apptainer docs
- **Confidence**: high
- **Why it matters here**: E5 requires the job to fail fast if the dataset path is missing; bind-mounting the dataset path into the container and checking it exists before training is the concrete check.

## Finding: Apptainer auto-binds $HOME/$PWD/tmp but not arbitrary host paths

- **Claim**: Apptainer/Singularity automatically mounts `$HOME`, `$PWD`, `/tmp`, and in some sites `/sc/...`; any other host directory (dataset store, model output) must be bound explicitly with `-B host:container`. Python packages in `$HOME/.local` can leak into the container and shadow the image.
- **Source**: mssm minerva Apptainer docs; Berkeley Savio Apptainer docs
- **Confidence**: high
- **Why it matters here**: A training image that silently imports a stale `~/.local` package breaks reproducibility; the sbatch template must set `PYTHONNOUSERSITE=1` and bind only the needed paths.

## Finding: SSH credentials and datasets are already kept out of the repo by .gitignore

- **Claim**: `.gitignore` already excludes `.env`, datasets/, data/, runs/, *.pt, and the project folder; N4 (no secrets in repo) is satisfied structurally, but there is no CI check for leaked keys, and large datasets are transferred outside git.
- **Source**: `.gitignore` (datasets/, data/, runs/, *.pt, .env); INTENT N4
- **Confidence**: high
- **Why it matters here**: The HPCC workflow must not introduce secrets into the repo; any new hpcc/ templates should use env/SSH agent and never embed keys.

## Finding: No hpcc/ or training/ module exists yet

- **Claim**: There is no `hpcc/` directory and no `polyx/training/` grouping in the repo; training code lives under `polyx/gui/trainer/` and `runs_train/`. The HPCC track (SLURM sbatch, Singularity def, env bootstrap) is entirely greenfield.
- **Source**: repo listing (`ls`), INTENT scope in
- **Confidence**: high
- **Why it matters here**: SC7 work is additive; nothing existing must be broken, but the trainer's path assumptions (`RAIZ_PROYECTO = parents[3]`) will need revisiting under `src/`.

## Finding: Existing failure modes already present in the codebase

- **Claim**: Several INTENT risks already manifest in code: `DEFAULT_MODEL` points at repo root (`bestdetectormedium.pt`) instead of `models/` (evidence-codebase); `polyx/model/` is an empty dead dir with stale pycache; `segmentacion_aprendida` pycache has no source; CLAUDE.md lists Pillow 10.4.0 vs actual 12.3.0; tests/audit scripts hard-code `REPO_ROOT / "polyx"` paths that will move. These are the exact hygiene items INTENT SC5/E6 target.
- **Source**: `polyx/core/paths.py:10`; evidence-codebase.md; `tests/test_comments_ingles.py`, `test_identifiers_ingles.py`, `test_docstrings_ingles.py` (all use `REPO_ROOT / "polyx"`)
- **Confidence**: high
- **Why it matters here**: The migration must update these audits' root paths or the i18n gate fails on a moved tree; the grep gate for `polyx.model` must run before deletion.

## Assigned questions — answers

- Does `src/` layout break the existing `lanzar_polyx.py` / `Poly-X.vbs` launchers? → **Yes.** `Poly-X.vbs` probes `raiz\polyx` (would be `src\polyx`) and quits if absent; `lanzar_polyx.py` imports `polyx` from root; `launcher._spawn_module` and `paths.ROOT` use `parents[2]` which shifts from repo root to `src/`. All must be re-pointed (VBS to `src\polyx` or replaced by installed entry point; parent counts to `parents[3]`; asset/model paths verified). Sources: `Poly-X.vbs`, `lanzar_polyx.py`, `polyx/gui/launcher.py`, `polyx/core/paths.py`.

- Where do datasets live on the HPCC (shared scratch vs. user home) and how are they synced? → **Datasets and large outputs live in shared/project storage or scratch (not home, which has small quotas); scratch is temporary and purged, so outputs must be copied back. Sync via `rsync` deltas over SSH; Apptainer needs explicit `-B` bind mounts and scratch-based cache dirs.** The exact PUCV cluster layout (shared group path, scratch purge policy, service account) is not documented in the repo and needs the user/maintainer's confirmation (INTENT open questions). Sources: HCC-DOCS, CHPC/Utah, Stanford RC, Berkeley Savio Apptainer docs, NCSU/Dartmouth transfer docs.

## Dead ends

- `polyx/model/` directory — empty except stale `__pycache__`; not a real package, so no runtime dependency to preserve (but grep gate still required before deletion per N2).
- Searching the repo for `hpcc/`, `*.sbatch`, `*.def` — none exist; the HPCC track is greenfield, not a fix-up of existing scripts.
- Searching for `pyproject.toml`/`setup.cfg`/`setup.py` — none; packaging is entirely requirements.txt + ad-hoc scripts, so there is no existing build config to migrate, only to create.
- `environment_python()` relying on `paths.ROOT` — under src/ it would look for `.venv` under `src/`; dead end unless `ROOT` is corrected to repo root (or the venv lookup is decoupled from `ROOT`).
- Web search for PUCV-specific cluster paths — no reliable public source; institutional details must come from the maintainer (INTENT open question).