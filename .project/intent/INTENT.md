# Intent — Poly-X (industry-standard packaging & release readiness)

Lane: standard
Review panel: off
Finding skeptics: off
Surfaces: CLI, desktop-app, installer

## Summary
Poly-X is a Nile-Red microplastics detection suite (Python/PySide6/PyTorch) whose engine already lives in `polyx/core` and whose GUI/tests are green (876 tests), but whose packaging, layout, and release story are pre-1.0: code sits at repo root, docs describe a superseded `polyx/model` layout, weights are not discoverable on a clean install, and only Windows is targeted. This milestone moves the package to an industry-standard `src/` layout with a real `pyproject.toml`, makes the app OS-agnostic (Windows + Linux + macOS), adds a Linux build, and produces both a PyInstaller one-file executable and a pip wheel published via GitHub Releases — preparing Poly-X for its first public release. A parallel MLOps track lays the groundwork for iterating on new models and training on HPCC (SLURM + Singularity/Apptainer over SSH). Source code is standardized to English; only GUI-facing strings may be localized.

## Problem
- No official release exists; distribution is ad-hoc (`.bat`/`.command`/`.ps1` scripts + PyInstaller shim).
- `polyx/` at repo root and no `pyproject.toml` → not installable as a package; `DEFAULT_MODEL` points at repo root instead of `models/`, so auto-discovery fails on a clean checkout.
- Docs (CLAUDE.md/README) describe `polyx/model/` and `polyx/gui/etiquetador` that no longer exist; claim "43 tests" vs 876.
- Only Windows is targeted; no Linux/macOS build or CI matrix.
- OS-specific assumptions leak outside `polyx/core/platform.py`.
- Source code carries Spanish identifiers/comments from the original fork; not yet fully internationalized.

## Users
Researchers and lab technicians (PUCV chemistry group and collaborators), moderate Python literacy, running on Windows (primary) and, after this milestone, Linux/macOS. They install via a downloaded executable or `pip install` and run the GUI or CLI. Maintainers (HPCC users) train new YOLO models on GPU nodes via SLURM.

## Success criteria
1. `pip install .` (and `pip install polyx-*.whl`) succeeds on a clean checkout on Windows, Linux, and macOS, and `python -m polyx` / `python -m polyx.gui` launch the app.
2. PyInstaller produces a one-file executable for Windows, Linux, and macOS in CI, attached to a GitHub Release.
3. On a clean install, the detector auto-discovers `models/bestdetectormedium.pt` (or reports a clear "no model found" with a download hint) instead of silently failing.
4. CI matrix runs the full test suite (876 tests) on ubuntu-24.04, windows-latest, and macos-latest with `QT_QPA_PLATFORM=offscreen`.
5. Docs (CLAUDE.md, README, AGENTS.md) describe the actual `polyx/core` + `polyx/gui` layout and current test counts; `polyx/model/` and orphan pycache removed.
6. A release checklist (version bump, changelog, tag, artifact upload) is documented and runnable by a maintainer.
7. A documented, reproducible remote-training workflow: a maintainer can SSH into the HPCC, submit a SLURM job (Singularity/Apptainer image) that trains a YOLO model on GPU nodes and returns a versioned `.pt` + metrics, with no manual steps beyond editing a config.
8. Source code (identifiers, comments, docstrings, commit messages, tests) is 100% English; Spanish (or any other language) appears only inside the `tr()` UI-string layer and user-facing GUI text. Verified by an automated audit reporting 0 untranslated identifiers.

## Edge coverage

| Edge | Criterion | Category | Disposition | Detail |
|------|-----------|----------|-------------|--------|
| E1 | SC1 | empty | held-out | Empty install (no weights) → app starts, detector page shows "no model found" + download hint. |
| E2 | SC3 | boundary | held-out | Weight present at `models/` → auto-loaded; absent → hint; never crashes. |
| E3 | SC4 | concurrency | dismissed | CI jobs are independent per OS; no shared state. |
| E4 | SC2 | boundary | held-out | PyInstaller build must succeed with no console window on Windows (GUI mode). |
| E5 | SC7 | boundary | held-out | HPCC job fails fast with a clear error if Singularity image or dataset path is missing. |
| E6 | SC8 | encoding | held-out | Audit scans all `.py` outside `tr()`; any non-ASCII identifier or Spanish comment fails. |
| E7 | SC5 | none | dismissed | Docs are static Markdown; no data-shape edge applies. |
| E8 | SC6 | none | dismissed | Checklist is procedural documentation; no data-shape edge applies. |

## Prohibitions

| Prohibition | Criterion | Must not | Disposition | Detail |
|-------------|-----------|----------|-------------|--------|
| N1 | SC1 | Must not break the existing 876-test suite or Qt-free import contract of `polyx/core`. | criterion SC1 | Any import of Qt/ultralytics in core fails CI. |
| N2 | SC5 | Must not delete `polyx/model` without confirming nothing imports `polyx.model` (grep gate). | criterion SC5 | CI grep for `polyx.model`/`polyx.ml` must be empty. |
| N3 | SC1 | Must not introduce OS-specific code outside `polyx/core/platform.py`. | dismissed | Already enforced by AGENTS.md; this milestone only adds platform helpers there. |
| N6 | SC2 | Must not ship an executable that crashes on launch on any target OS. | criterion SC2 | Smoke-test each built artifact in CI before release. |
| N7 | SC3 | Must not silently swallow a missing-model error; must surface a hint. | criterion SC3 | Detector page shows explicit "no model found" + download hint. |
| N8 | SC4 | Must not skip any OS in the CI matrix. | criterion SC4 | All three OS jobs must be green before merge. |
| N9 | SC6 | Must not publish a release without the checklist being completed. | criterion SC6 | Maintainer signs off each checklist item. |
| N4 | SC7 | Must not store SSH credentials or secrets in the repo. | criterion SC7 | Credentials via env/SSH agent only; CI checks for leaked keys. |
| N5 | SC8 | Must not translate or alter user-facing Spanish GUI strings in this milestone. | criterion SC8 | Only source-code identifiers/comments; `tr()` strings untouched. |

## Scope: in
- Move `polyx/` → `src/polyx/`; add `pyproject.toml` (PEP 621, src layout, console scripts).
- Fix `DEFAULT_MODEL` → `models/bestdetectormedium.pt` with graceful fallback.
- Remove empty `polyx/model/` and orphan pycache (after grep gate).
- Update CLAUDE.md, README.md, AGENTS.md to current layout/counts.
- CI matrix: ubuntu/windows/macos, offscreen Qt, build + test.
- PyInstaller specs + GitHub Actions release workflow (exe/wheel per OS).
- Release checklist doc (versioning, changelog, tag, upload).
- I18n audit pass: scan all `.py` for non-English identifiers/comments outside `tr()` and fix; remove any leftover Spanish module/file names.
- CI gate: run the translation audit in CI; fail on untranslated source strings.
- `src/polyx/training/` module grouping: dataset prep, train entrypoint, evaluate, export — reusable locally and on HPCC.
- HPCC templates under `hpcc/` (SLURM sbatch scripts, Singularity/Apptainer definition, env bootstrap) + docs.
- Model artifact convention: `models/<name>-<version>.pt` + sidecar metadata (dataset, params, metrics) for traceability.
- Config-driven training (YAML) so the same config runs on laptop or cluster.

## Scope: out (vetoes)
- No new detection features or model training changes — packaging/release only.
- No PyPI publication (GitHub Releases only) — can revisit later.
- No change to the Qt-free `polyx/core` import contract.
- No removal of legacy `legacy/` folder (out of scope for this milestone).
- No translation of user-facing GUI strings.

## Constraints
- Stack: Python 3.11 (CI), PySide6 6.7.2, Ultralytics 8.3.40, NumPy <2. Keep requirements.txt as source of truth until pyproject is validated.
- No deadline; iterate until green on all 3 OSes.
- Distribution: GitHub Releases only.
- Existing tests must stay green; new packaging tests (install smoke, import check) added.
- HPCC uses SLURM + Singularity/Apptainer; training configs must be portable between local (CPU/GPU) and cluster.
- Source language: English only. User-facing strings may be localized via `tr()` (es/en at minimum).

## Current state (brownfield)

- **What exists**: Engine in `polyx/core` (detect/measure/quality/provenance/data/i18n, Qt-free); GUI in `polyx/gui`; 876 tests green; refactor waves complete.
- **Must not break**: `polyx/core` Qt-free import contract; existing GUI entry points; 876 tests.
- **Doc-vs-code rulings**: CLAUDE.md layout → fix-doc; README "43 tests" → fix-doc; `polyx/model/` leftover → fix-code (remove after grep); DEFAULT_MODEL path → fix-code; Pillow version claim → fix-doc.
- **Ground truth**: `.project/research/evidence-codebase.md`, `.project/research/DOCS-AUDIT.md`

## Risks
- PyInstaller + PyTorch/Ultralytics bundling size and hidden imports on Linux/macOS (untested).
- `src/` migration may break relative asset paths (models/, runs/, assets/) and tests that assume repo-root layout.
- NumPy <2 and Pillow 12.3 pins may conflict with platform wheels.
- No real field data to validate the packaged app end-to-end.
- Singularity image / CUDA version drift between laptop and cluster.
- SLURM queueing makes iteration slow; need smoke tests locally first.
- SSH credentials and data transfer (datasets) security.
- Hidden Spanish strings in tests/fixtures or legacy scripts could break the audit gate; need a baseline pass first.

## Open questions
- [RESEARCH] Which PyInstaller hooks are needed for ultralytics/torch on each OS?
- [RESEARCH] Does `src/` layout break the existing `lanzar_polyx.py` / `Poly-X.vbs` launchers?
- [RESEARCH] Singularity vs Apptainer vs conda-pack for the cluster image?
- [RESEARCH] Where do datasets live on the HPCC (shared scratch vs. user home) and how are they synced?
- [NEEDS-USER] Should the version be bumped to 2.1.0 or 3.0.0 for the first release?
- [NEEDS-USER] Who has SSH/SLURM access, and is there a shared service account?

## Corrections
- 2026-10-25: user requested MLOps/HPCC training workflow (SLURM + Singularity, SSH) added to scope.
- 2026-10-25: user requested full English standardization of source code; Spanish only for GUI-facing strings.
