---
id: T001
title: Migrate polyx/ to src/polyx/ layout and re-derive project paths
wave: 1
deps: []
status: pending
agent: null
base: null
worktree: null
task_branch: null
files:
  - src/polyx/core/paths.py
  - src/polyx/gui/launcher.py
  - src/polyx/__init__.py
  - src/polyx/launcher.py
  - lanzar_polyx.py
  - Poly-X.vbs
  - tests/test_arranque_instrumentado.py
  - tests/test_launcher_trainer.py
  - tests/test_trainer_runner.py
  - tests/test_comments_ingles.py
  - tests/test_docstrings_ingles.py
  - tests/test_identifiers_ingles.py
  - tests/test_report_headings_ingles.py
  - tests/test_translate_identifiers.py
  - tests/test_model_discovery.py
---

# T001 — Migrate polyx/ to src/polyx/ layout and re-derive project paths

## Context

Poly-X has no installable package: `polyx/` sits at repo root and `paths.ROOT` uses `parents[2]`, which after a move to `src/polyx/` would silently relocate `models/`, `runs/`, `data/`, `assets/` and the `.venv` lookup under `src/`. This task moves the package to `src/polyx/`, re-derives every parent count so `ROOT` still points at the project root, fixes `DEFAULT_MODEL` to models/bestdetectormedium.pt (expected-new) with a graceful fallback, and updates the three entry surfaces (stub, VBS, subprocess cwd) plus the i18n audit roots. The detector must auto-discover the weight or report a clear "no model found" + download hint instead of crashing.

## Approach

- Move the entire `polyx/` tree to `src/polyx/` (git mv preferred) without changing file contents except the path derivations below.
- In `src/polyx/core/paths.py`: change `ROOT` to `parents[3]`; set `DEFAULT_MODEL = MODELS_DIR / "bestdetectormedium.pt"`; add `discover_default_model() -> Optional[Path]` returning `DEFAULT_MODEL` if it exists else `None`. Keep `MODELS_DIR/RUNS_DIR/DATA_DIR/ASSETS_DIR` anchored at `ROOT`.
- In `src/polyx/gui/launcher.py`: `_spawn_module` cwd becomes `parents[3]`; `open_manual`/`open_readme` root lookups become `parents[3]`.
- Update `lanzar_polyx.py` to `from polyx.gui.launcher import main` guarded by a `sys.path` insert of `src/` when run from repo root; or relocate it to repo root with `pathex` support.
- Update `Poly-X.vbs` folder probe from `raiz\polyx` to `raiz\src\polyx` (or replace with the installed console script invocation).
- Retarget i18n audit scripts/tests from `REPO_ROOT / "polyx"` to `REPO_ROOT / "src" / "polyx"` (known modules become `src/polyx/...`).
- Update tests asserting `parents[2]` cwd to `parents[3]`.
- Add `tests/test_model_discovery.py` covering present/absent weight cases (E1/E2).
- Model-discovery seam (moved from Interface contract): `polyx.core.paths.ROOT` resolves to the repository/project root (not `src/`); `polyx.core.paths.DEFAULT_MODEL == ROOT / "models" / "bestdetectormedium.pt"`; `polyx.core.paths.discover_default_model() -> Optional[pathlib.Path]` returns the path when it exists, else `None`; `DEFAULT_MODEL` and `discover_default_model()` are the only model-discovery seam consumed by the detector page.

## Interface contract

- None

## Intent coverage
- SC1
- SC3

## Acceptance criteria

1. `pip install -e .` succeeds on a clean checkout and `python -m polyx` / `python -m polyx.gui` launch.
2. With `models/bestdetectormedium.pt` present, the detector auto-loads it (E2: present → auto-loaded); with it absent, the app starts and the detector page shows "no model found" + download hint (E1: empty install → hint; never crashes).
3. `paths.ROOT` equals the project root; `MODELS_DIR/RUNS_DIR/DATA_DIR/ASSETS_DIR/DEFAULT_MODEL` all resolve under it.
4. The full test suite (876 tests) still passes after the move, with audit roots retargeted.

## Verify

```bash
python -m pytest tests/test_model_discovery.py tests/test_arranque_instrumentado.py tests/test_launcher_trainer.py -q
```

Heavy: no
## Log

- 2026-10-25 — created by planner
