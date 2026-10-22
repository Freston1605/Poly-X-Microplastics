# Evidence — codebase

<!-- Written by the codebase mapper during gsd-path-inspect. Read by define
     (brownfield mode), researchers (fifth standard dimension), the planner
     (conventions are binding), and reviewers. -->

Repo root: /var/home/Franco/Projects/Poly-X-Microplastics
Scanned: 2026-10-18
Checks run: `python -m pytest -q` (sidecar, QT_QPA_PLATFORM=offscreen, MPLCONFIGDIR=/tmp/mplconfig) → 210 passed in 24.11s; `python auditar_traduccion.py` → 0 untranslated of 702 distinct strings; `python -m compileall -q polyx tests` → clean.

## Map

- **Stack**: Python desktop app (PySide6 6.7.2 pinned; runs on 3.11 per docs, 3.14 in this sidecar); Ultralytics 8.3.40, OpenCV 4.10.0.84, NumPy 1.26.4, Pillow 12.3.0 (upgraded from 10.4.0), Matplotlib 3.9.2, pyqtgraph 0.13.7, PyYAML 6.0.2, psutil 6.0.0. Four modules launched as subprocesses from a launcher: Detector, Trainer, Etiquetador, Visor. ~21.2k LOC across 84 Python files in `polyx/` + 15 test files.
- **Entry points**: `polyx/launcher.py` (main menu, `python -m polyx.launcher`); `polyx/detector/__main__.py`, `polyx/trainer/__main__.py`, `polyx/etiquetador/__main__.py`, `polyx/visor/__main__.py`. Launcher spawns each module via `sys.executable -m polyx.<module>`. OS-specific behavior isolated in `polyx/core/plataforma.py`.
- **Architecture**: Shared `polyx/core/` (theme, i18n, paths, metrics, yolo_wrap, report_html, widgets, iconos, plataforma, preferencias, updates) feeds four independent Qt processes. Detector: sidebar + 9 pages (modelos, imagenes, parametros, gt_manual, ejecutar, resultados, errores, comparar, reporte). Trainer: 9 pages. State objects (`DetectorState`, `TrainerState`) hold runtime config. Batch inference in `detector/runner.py`; training in `trainer/runner.py`. Reports generated as self-contained HTML (`core/report_html.py`, 2768 LOC).
- **Conventions**: Spanish comments/variables, English docstrings; paths via `pathlib.Path`; colors via `polyx/core/theme.py` tokens (`T.ACCENT`, etc.), never literals; theme imported as module (`from .core import theme as T`), never `from .core.theme import INK`; images via `cv2.imdecode/imencode` over `np.fromfile`; commit messages `type(scope): description` in Spanish; tests run with `QT_QPA_PLATFORM=offscreen`.
- **Maturity**: 210 tests, all pass in sidecar (core logic + morphology/calibration/assignment/report generation). No tests touch GUI windows or YOLO inference/training paths (ultralytics not installed in this sidecar). i18n fully translated (702/702). Four themes, 40 vector icons. Legacy scripts retained at repo root under `legacy/`.
- **Recent activity**: Last commits (Oct 2026) cluster on security (Pillow CVE bump, HTML escaping in reports), CI bootstrap (GitHub Actions, pinned installs), and UI polish (per-window icons/taskbar names, themes, vector icons, microscope animation). Detector morphology and depth-profile work in Aug–Sep 2026.

## Finding: Test count in docs is stale (128 vs 210)

- **Claim**: AGENTS.md states "pasan los 128" tests, but the suite at HEAD contains 210 tests and all pass.
- **Source**: `AGENTS.md:25`; `tests/` (15 files); run `python -m pytest -q` → "210 passed".
- **Confidence**: high
- **Why it matters here**: Any inspection or CI gate that expects 128 will misread the suite as broken or over-coverage; docs auditors should flag.

## Finding: Environment in this sidecar does not match pinned requirements

- **Claim**: The verify sidecar Python is 3.14 with NumPy 2.5.3, PySide6 6.11.2, OpenCV 4.14, and no ultralytics; requirements.txt pins Python 3.11, NumPy 1.26.4, PySide6 6.7.2, OpenCV 4.10.0.84, Ultralytics 8.3.40. Tests pass because they never import ultralytics and `yolo_wrap` imports it lazily, so inference/training code paths are unexercised.
- **Source**: `requirements.txt`; `pip list` in sidecar; `polyx/core/yolo_wrap.py` (lazy import); run `python -c "import ultralytics"` → ModuleNotFoundError.
- **Confidence**: high
- **Why it matters here**: A green test run here does not validate the YOLO pipeline; regressions in `yolo_wrap`, runner, or training could ship. Treat test evidence as core-logic only.

## Finding: Pillow version in CLAUDE.md lags requirements.txt

- **Claim**: CLAUDE.md technology table lists Pillow 10.4.0, but requirements.txt pins Pillow 12.3.0 (security bump, 33 advisories) and `.gitignore`-adjacent history shows the bump in commit `5a6578d`.
- **Source**: `CLAUDE.md` (Tecnología table); `requirements.txt:12`; `git log --oneline` → `5a6578d fix(security): Pillow 10.4.0 -> 12.3.0`.
- **Confidence**: high
- **Why it matters here**: Docs describing the stack are out of date; planners relying on CLAUDE.md may assume the older Pillow API/behavior.

## Finding: CLAUDE.md omits two shipped modules from its structure tree

- **Claim**: The `📦 Estructura` tree in CLAUDE.md lists `launcher`, `core`, `detector`, `trainer`, `legacy` but omits `polyx/etiquetador/` and `polyx/visor/`, although both are first-class modules wired into the launcher (cards 3 and 4) and shipped with `__main__.py`, windows, state, and canvas.
- **Source**: `CLAUDE.md` (Estructura); `polyx/launcher.py:728,733,825,826`; `polyx/etiquetador/`, `polyx/visor/` directories.
- **Confidence**: high
- **Why it matters here**: A planner reading only the structure tree would think the app has two modules; the Etiquetador and Visor are substantial (1253 and 1392 LOC).

## Finding: `polyx/core/pdf_export.py` is used but only by the report page

- **Claim**: `pdf_export` (69 LOC) is imported solely by `detector/pages/reporte.py` and guarded by `is_available()`; if QtWebEngine is absent, PDF export silently no-ops.
- **Source**: `polyx/detector/pages/reporte.py:16,519,557`; `polyx/core/pdf_export.py`.
- **Confidence**: medium
- **Why it matters here**: PDF export is an optional capability; report generation still succeeds as HTML. Worth confirming QtWebEngine is bundled in the installer.

## Finding: No automated tests for GUI modules or end-to-end detection

- **Claim**: All 210 tests exercise `core/` helpers and morphology/calibration/assignment/report generation; none instantiate Qt windows, run inference, or train a model.
- **Source**: `tests/` file list; grep for `DetectorWindow|TrainerWindow|VisorWindow|LabelerWindow` in `tests/` → only `test_plataforma.py`.
- **Confidence**: high
- **Why it matters here**: Coverage is deep on numeric/report logic but zero on UI and ML paths; regressions in windows or YOLO wrapping would not be caught by CI.

## Apparent intent

- The project is becoming a full 4-module desktop suite (Detector → Trainer → Etiquetador → Visor) for Nile-Red fluorescence microplastic analysis, with paper-quality bilingual HTML reports. — based on `polyx/launcher.py`, `CLAUDE.md`, report_html.py.
- Recent work (Sep–Oct 2026) prioritizes UX polish (themes, vector icons, animation, per-window taskbar identity) and security/CI hardening rather than new detection features. — based on git log subjects.
- The `report_html.py` module (2768 LOC) is the heaviest single file and likely the main output surface; bilingual reporting is fully wired. — based on `polyx/core/report_html.py`, `auditar_traduccion.py`.

## Open questions for define

- Is the test-count drift (128 → 210) intentional growth or did a bulk of tests land without updating AGENTS.md?
- Should CI gain an ultralytics-present environment (or a smoke test importing `yolo_wrap` and running one inference) to cover the ML path?
- Is PDF export (`pdf_export`) expected to work in the shipped app, or is HTML-only acceptable?
- Are the Etiquetador and Visor considered production-ready (docs say yes) despite having no automated tests?

## Blocked areas

- Inference/training paths could not be exercised: ultralytics is not installed in this sidecar and NumPy/PySide6/OpenCV versions diverge from pins. Static inspection only for those modules.
- `.project/archive` and installed GSD bundles excluded per brief; not inspected.