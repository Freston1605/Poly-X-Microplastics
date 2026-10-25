# Evidence — codebase

Repo root: /var/home/Franco/Projects/Poly-X-Microplastics
Scanned: 2026-10-25
Checks run: `pytest -q` in verify sidecar at d69db801eaf637a4492bc3c84fb72fcac9c90f55 → 876 passed, 0 failures, 0 skips (123.37s); Python 3.11.17 venv; QT_QPA_PLATFORM not set locally but suite passed.

## Map

- **Stack**: Python 3.11 (CI pins 3.11; venv 3.11.17; local also has 3.14.8). PySide6 6.7.2, Ultralytics 8.3.40, OpenCV 4.10.0.84, NumPy 1.26.4 (pinned <2), Pillow 12.3.0, Matplotlib 3.9.2, pyqtgraph 0.13.7, PyYAML 6.0.2, psutil 6.0.0, libcst 1.9.0. No pyproject/setup.cfg; requirements.txt is the manifest; CI (.github/workflows/ci.yml) installs requirements + pytest on ubuntu-24.04 with offscreen Qt.
- **Entry points**: `python -m polyx.gui` (launcher), `python -m polyx.launcher` (historic shim), `python -m polyx.gui.detector|trainer|viewer|labeler`; `lanzar_polyx.py` for PyInstaller. `polyx/launcher.py` re-exports `polyx.gui.launcher`.
- **Architecture**: Pure engine in `polyx/core` (detect/measure/quality/provenance/data/i18n, Qt- and ultralytics-free import contract) consumed by `polyx/gui` (Qt presentation: launcher, detector, trainer, viewer, labeler, report_html, theme). `polyx/core/__init__.py` re-exports the public API. Legacy `polyx/model` directory is empty (only stale __pycache__); docs still refer to it.
- **Conventions**: English identifiers/comments/docstrings; Spanish only inside `tr()` UI strings; pathlib.Path always; colors via `polyx.gui.theme` module attribute (never `from ... import INK`); private methods prefixed `_`; Qt signals named `models_changed` etc. Tests pin Spanish locale via conftest (`POLYX_IDIOMA=es`) and isolate logs to tmp_path.
- **Maturity**: Full suite green (876 tests). Detector/Trainer/Labeller/Viewer all shipped. `polyx/core` is the restructured engine (wave1/wave2 refactor complete; M005 legacy shim removed). Orphan `segmentacion_aprendida` pycache with no source. `polyx/model` is a dead empty dir.
- **Recent activity**: Last commits are the refactor waves (model→ml→core rename, 161 import paths), translation to English (M006), security audit (Pillow bump, HTML escaping), CI bootstrap, byte-identity report gate. Work clusters in polyx/core, polyx/gui/*, tests.

## Finding: Engine lives in polyx/core, but CLAUDE.md and paths still say polyx/model

- **Claim**: The documentation and `polyx/core/paths.py` reference a layout that no longer exists: `polyx/model/` is empty (only `__pycache__`), and the engine now lives under `polyx/core/`. CLAUDE.md's structure tree still lists `polyx/model/` and `polyx/core/` as a "temporary shim".
- **Source**: `polyx/model/` (empty dir), `polyx/core/__init__.py`, `CLAUDE.md` lines 23-40
- **Confidence**: high
- **Why it matters here**: Any planner or reviewer following CLAUDE.md will look for files in the wrong package; docs are stale relative to code.

## Finding: DEFAULT_MODEL points to repo root, not models/

- **Claim**: `DEFAULT_MODEL = ROOT / "bestdetectormedium.pt"` resolves to the repository root, while docs and the models page assume weights live under `models/` (`models/bestdetectormedium.pt`). No `.pt` exists at root in a clean checkout (weights are gitignored), so auto-discovery never finds the default model.
- **Source**: `polyx/core/paths.py:10`, `CLAUDE.md` "models/bestdetectormedium.pt", `polyx/gui/detector/pages/models.py:180-216`
- **Confidence**: high
- **Why it matters here**: Detector's "use default model" button reports the model missing on a normal install.

## Finding: Stale pycache for a removed module

- **Claim**: `polyx/core/__pycache__/segmentacion_aprendida.cpython-*.pyc` exists with no corresponding `segmentacion_aprendida.py` source, left over from the refactor.
- **Source**: `polyx/core/__pycache__/segmentacion_aprendida.cpython-311.pyc`
- **Confidence**: high
- **Why it matters here**: Signals incomplete cleanup; not a runtime issue but a hygiene/traceability note.

## Finding: CLAUDE.md technology table is out of date vs requirements.txt

- **Claim**: CLAUDE.md lists Pillow 10.4.0, but requirements.txt pins Pillow 12.3.0 (security hardening) and the CI installs 12.3.0. NumPy is correctly pinned to 1.26.4.
- **Source**: `CLAUDE.md` technology table, `requirements.txt`
- **Confidence**: high
- **Why it matters here**: Docs-derived version claims are unreliable; trust manifests.

## Finding: Test suite fully green at audited HEAD

- **Claim**: Full pytest run in the verify sidecar at d69db80 passed 876 tests with 0 failures and 0 skips (2508 warnings, mostly deprecation noise).
- **Source**: command `pytest -q` → "876 passed, 2508 warnings in 123.37s"
- **Confidence**: high
- **Why it matters here**: Baseline is healthy; no half-built test areas observed.

## Apparent intent

- The project is mid-way through a large refactor: package renamed `model`→`ml`→`core`, detectors moved under `polyx/gui`, and a new protocol/factory-based detection layer (`polyx/core/detect/`) added alongside YOLO. Apparent goal: a Qt-free, testable science engine with pluggable detectors (YOLO, native, SAM) and better provenance/observability. — based on `polyx/core/detect/` and commit messages.

## Open questions for define

- Should `DEFAULT_MODEL` be corrected to `models/bestdetectormedium.pt`, or should weights be expected at repo root?
- Is the empty `polyx/model/` directory and orphan pycache safe to delete, or is something still expected to import from `polyx.model`?
- Are the docs (CLAUDE.md) slated to be updated to the new `polyx/core` + `polyx/gui` layout, or is the old layout still the "intended" public surface?

## Blocked areas

- none