---
id: T002
title: Add pyproject.toml and PyInstaller one-file build with smoke test
wave: 1
deps: [T001]
status: pending
agent: null
base: null
worktree: null
task_branch: null
files:
  - pyproject.toml
  - polyx.spec
  - lanzar_polyx.py
  - .github/workflows/release.yml
  - tests/test_pyinstaller_entry.py
---

# T002 — Add pyproject.toml and PyInstaller one-file build with smoke test

## Context

Poly-X has no build manifest and no reproducible frozen build. This task creates a PEP 621 `pyproject.toml` (src layout, console scripts `polyx`/`polyx-gui`), a PyInstaller spec that relies on hooks-contrib's `hook-torch`/`hook-ultralytics` plus explicit `collect_data_files('ultralytics')`, and a release workflow producing one-file executables for Windows/Linux/macOS. The frozen app must not crash on launch (N6) and must show the launcher with the model hint when weights are absent (E4).

## Approach

- Create `pyproject.toml` with `[build-system]` (setuptools or hatchling), `[project]` metadata, `src` layout, console scripts, and dependencies mirrored from `requirements.txt` (keep requirements.txt as source of truth until validated).
- Create `polyx.spec` with `pathex=["src"]`, `collect_data_files('ultralytics')`, `hookspath` relying on hooks-contrib, and `console=False` on Windows (GUI mode, E4).
- Add a `sys._MEIPASS`-aware resource helper in `src/polyx/core/paths.py` (or a new src/polyx/core/resources.py) so frozen asset/model resolution prefers `sys._MEIPASS` when present.
- Add `.github/workflows/release.yml` building wheel + one-file exe per OS and attaching to a GitHub Release.
- Add `tests/test_pyinstaller_entry.py` asserting the spec declares windowed mode and the resource helper resolves `sys._MEIPASS`.
- Moved from Interface contract: console scripts `polyx = "polyx.gui.launcher:main"` (or `polyx.__main__:main`), `polyx-gui = "polyx.gui.launcher:main"`; frozen resource resolution `resource_root() -> Path` returns `Path(sys._MEIPASS)` when `getattr(sys, "_MEIPASS", None)` is set, else the project root (`paths.ROOT`); spec entry stub repo-root `lanzar_polyx.py` (`from polyx.gui.launcher import main; main()`).

## Interface contract

- None

## Intent coverage
- SC2

## Acceptance criteria

1. `python -m build` (or `pip install .`) produces a wheel and `pip install polyx-*.whl` works.
2. `pyinstaller polyx.spec` succeeds on at least one OS and the resulting binary launches without crash (N6).
3. The built binary runs in GUI mode on Windows (no console window, E4).
4. The release workflow builds and smoke-tests artifacts for ubuntu-24.04, windows-latest, and macos-latest.

## Verify

```bash
python -m pytest tests/test_pyinstaller_entry.py -q
```

Heavy: no
## Log

- 2026-10-25 — created by planner
