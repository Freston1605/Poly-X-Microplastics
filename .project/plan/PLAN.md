# Plan — Poly-X (industry-standard packaging & release readiness)

Project verify: `.venv\Scripts\python.exe -m pytest -q` (run with `QT_QPA_PLATFORM=offscreen` on CI)

## Config

- max_review_cycles: 3
- wave_budget: none
- review_panel: off
- finding_skeptics: off

## Wave 1 — risk burn-down

Goal: Prove the `src/` migration keeps asset/model discovery and entry points working, and that a PyInstaller one-file build of the detector imports ultralytics end-to-end without crashing. If these fail, the rest of the milestone is blocked.
Review depth: full

## Wave 2 — walking skeleton + features

Goal: `pip install -e .` on a clean checkout → `python -m polyx` launches the GUI and the detector page reports "no model found" + download hint when weights are absent, auto-loads models/bestdetectormedium.pt (expected-new) when present; plus docs, release checklist, training/HPCC templates, and the i18n CI gate land.
Review depth: full

## Wave 3 — CI + release

Goal: CI matrix green on ubuntu-24.04/windows-latest/macos-latest running the full 876-test suite and building/smoke-testing each artifact; release workflow attaches wheel + one-file exe per OS to a GitHub Release.
Review depth: full

## Surface contract

### CLI — T001

Criteria: SC1
Entry: `python -m polyx` (or the installed `polyx` console script)
States: missing weights → app starts, detector page shows "no model found" + download hint; weights present → auto-loaded model.
Walkthrough:
1. Install editable (`pip install -e .`), run `python -m polyx`, confirm launcher opens.
2. Open Detector → Models page; confirm hint or auto-load per weights.

### desktop-app — T001

Criteria: SC3
Entry: Launcher window (`python -m polyx.gui`)
States: launcher opens with four module cards; modules spawn in separate processes with cwd at repo root.
Walkthrough:
1. Launch `python -m polyx.gui`; click each module card; confirm subprocess starts.

### installer — T002

Criteria: SC2
Entry: PyInstaller one-file executable (`dist/polyx` / `dist/polyx.exe`)
States: on launch, no console window on Windows (GUI mode); detector imports ultralytics without crash; model hint shown when weights absent.
Walkthrough:
1. Build with `pyinstaller polyx.spec`; run the produced binary; confirm it does not crash and shows the launcher.

## Intent coverage

| Criterion | Task | Acceptance |
|-----------|------|------------|
| SC1 | T001 | AC1 |
| SC2 | T002 | AC1 |
| SC3 | T001 | AC2 |
| SC4 | T005 | AC1 |
| SC5 | T003 | AC1 |
| SC5 | T004 | AC1 |
| SC6 | T006 | AC1 |
| SC7 | T007 | AC1 |
| SC8 | T008 | AC1 |

## Held-out checks

| Edge | Task | Test |
|------|------|------|
| E1 | T001 | tests/test_model_discovery.py |
| E2 | T001 | tests/test_model_discovery.py |
| E4 | T002 | tests/test_pyinstaller_entry.py |
| E5 | T007 | tests/test_hpcc_templates.py |
| E6 | T008 | tests/test_identifiers_ingles.py |

## Dependency notes

- T002 depends on T001 because the PyInstaller spec and stub import `polyx` from `src/` and rely on `paths.ROOT` pointing at the repo root.
- T003 depends on T001 because the grep gate for `polyx.model`/`polyx.ml` must run against the moved tree and deletion happens after the move.
- T004 depends on T001 because docs describe the new `src/polyx` layout and parent counts.
- T005 depends on T001, T002, T003, T008 because CI runs the full suite, the i18n audit, and builds/smoke-tests artifacts.
- T007 depends on T001 because training/HPCC templates reference project root paths and the `src/` layout.
- T008 depends on T001 because the audit scripts' `REPO_ROOT / "polyx"` roots must be retargeted to `REPO_ROOT / "src" / "polyx"`.