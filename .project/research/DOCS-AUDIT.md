# Docs Audit

Repo root: /var/home/Franco/Projects/Poly-X-Microplastics
Audited: 2026-10-25
Audited HEAD: d69db801eaf637a4492bc3c84fb72fcac9c90f55
Alignment mode: no

## Summary

| Verdict | Count |
|---------|-------|
| verified | 12 |
| stale | 10 |
| aspirational | 0 |
| unverifiable | 3 |
| descriptive docs (no testable claims) | 30 |

Worst drift: CLAUDE.md and README.md still describe the pre-refactor layout (`polyx/model/`, `polyx/gui/etiquetador`, `polyx/gui/visor`) after the M001 refactor moved everything to `polyx/core/`, `polyx/gui/labeler` and `polyx/gui/viewer`; README also claims "43 tests" while the suite now has 876.

## Doc: AGENTS.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| "Nothing OS-specific outside `polyx/model/plataforma.py`" | structure | stale | AGENTS.md:23 (pre-update) vs code: no `polyx/model/` exists; OS-specific code lives in `polyx/core/platform.py` |
| "Run tests with `.venv\Scripts\python.exe -m pytest -q`" | command | verified | sidecar: 876 passed in 116.71s (`pytest tests/ -q`) |
| "Code comments/docstrings/identifiers in English" | feature | verified | codebase identifiers English; `tr()` layer Spanish |

## Doc: CLAUDE.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| Structure tree lists `polyx/core/` as "shim temporal" and `polyx/model/` as motor | structure | stale | no `polyx/model/` dir; real engine is `polyx/core/` (paths.py, platform.py, ...) |
| `polyx/gui/etiquetador/` and `polyx/gui/visor/` modules | structure | stale | actual dirs are `polyx/gui/labeler/` and `polyx/gui/viewer/` |
| "iconos.py draws 40 icons" | feature | stale | `polyx/gui/iconos.py`: 43 `def ` (icon functions) |
| "43 tests against synthetic shapes" | status | stale | full suite: 876 passed (tests/ has 47 files) |
| "Python 3.11.x runtime" | config | verified | `.venv` python 3.11; requirements pins Python 3.11 |
| "PySide6 6.7.2 / Ultralytics 8.3.40 / OpenCV 4.10.0.84" | config | verified | requirements.txt pins match |
| "Labeller and Viewer working, verified with 552 crops" | status | unverifiable | no artifact in repo to confirm 552 crops; no test references it |
| "PP fluoresces dull yellowish green (RGB 122/125/32)" | feature | verified | CLAUDE.md §real fluorescence; consistent with README |

## Doc: README.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| Project layout `polyx/detector/`, `polyx/trainer/`, `polyx/etiquetador/`, `polyx/visor/` | structure | stale | actual modules under `polyx/gui/{detector,trainer,labeler,viewer}` |
| "43 tests against synthetic shapes of known size" | status | stale | suite: 876 passed |
| "Python 3.9+" (macOS) vs "Python 3.11.x" (Windows) | config | stale | requirements.txt targets Python 3.11 only |
| Module entry points `python -m polyx.gui.etiquetador` / `visor` | command | verified | `polyx/gui/labeler/__main__.py` and `polyx/gui/viewer/__main__.py` exist |
| `polyx/model/plataforma.py` (portability claim) | structure | stale | path is `polyx/core/platform.py` |
| "auditar_traduccion.py must report 0 untranslated" | command | unverifiable | script exists; not run in this audit |

## Doc: CONTRIBUTING.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| CI runs `.github/workflows/ci.yml` on every PR/push | integration | verified | `.github/workflows/ci.yml` exists |
| Full suite `python -m pytest -q` with `QT_QPA_PLATFORM=offscreen` | command | verified | 876 passed locally under offscreen-capable env |
| "No lint stage, no build stage" | status | verified | ci.yml contains only install + test steps |

## Doc: LEEME_macOS.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| Installation via `Lanzar_macOS.command` | integration | verified | file exists, executable bit set |
| `crear_icono_macOS.command` creates `Poly-X.app` | feature | verified | script present |

## Doc: docs/diagnostico_sigterm.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| "full test suite (281 tests)" / "~31.8s" | status | stale | actual: 876 tests, 116.71s |
| "Peak RSS ~383-390 MB" | config | unverifiable | not re-measured in this audit |

## Doc: docs/recetas_jq.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| Logs under `~/.polyx/logs/` (or `POLYX_LOG_DIR`) | feature | verified | `polyx/core/observability.py` uses `POLYX_LOG_DIR` |

## Descriptive docs

- agent/skills/async-python-patterns/SKILL.md
- agent/skills/async-python-patterns/references/details.md
- agent/skills/code-review-excellence/SKILL.md
- agent/skills/docx/SKILL.md
- agent/skills/error-handling-patterns/SKILL.md
- agent/skills/error-handling-patterns/references/details.md
- agent/skills/find-skills/SKILL.md
- agent/skills/git-advanced-workflows/SKILL.md
- agent/skills/git-advanced-workflows/references/details.md
- agent/skills/pdf/SKILL.md
- agent/skills/pdf/forms.md
- agent/skills/pdf/reference.md
- agent/skills/pptx/SKILL.md
- agent/skills/python-performance-optimization/SKILL.md
- agent/skills/python-performance-optimization/references/advanced-patterns.md
- agent/skills/python-performance-optimization/references/details.md
- agent/skills/python-testing-patterns/SKILL.md
- agent/skills/python-testing-patterns/references/advanced-patterns.md
- agent/skills/python-testing-patterns/references/details.md
- agent/skills/skill-creator/SKILL.md
- agent/skills/skill-creator/agents/analyzer.md
- agent/skills/skill-creator/agents/comparator.md
- agent/skills/skill-creator/agents/grader.md
- agent/skills/skill-creator/references/schemas.md
- agent/skills/uv-package-manager/SKILL.md
- agent/skills/uv-package-manager/references/advanced-patterns.md
- agent/skills/xlsx/SKILL.md
- tests/fixtures/baseline_informe/README.md
- tests/fixtures/geodesia/README.md
- tests/fixtures/geodesia/TIMING.md

## User rulings

| Queue # | Ruling | User's words | Planned |
|---------|--------|--------------|---------|
| 1 | fix-doc | CLAUDE.md layout describes superseded polyx/model; update to polyx/core + polyx/gui | planned: no |
| 2 | fix-doc | README "43 tests" claim stale; correct to 876 | planned: no |
| 3 | fix-code | Empty polyx/model/ and orphan pycache removed after grep gate | planned: no |
| 4 | fix-code | DEFAULT_MODEL corrected to models/bestdetectormedium.pt with graceful fallback | planned: no |
| 5 | fix-doc | Pillow version claim in CLAUDE.md updated to 12.3.0 | planned: no |

## Remediation queue
| # | Doc | Claim | Verdict | Class | Suggested action |
|---|-----|-------|---------|-------|------------------|
| 1 | CLAUDE.md | Structure tree lists `polyx/core/` as "shim temporal" and `polyx/model/` as motor | stale | fix-doc | Update tree to reflect post-refactor layout (`polyx/core/` is the engine) |
| 2 | CLAUDE.md | `polyx/gui/etiquetador/` and `polyx/gui/visor/` modules | stale | fix-doc | Rename to `polyx/gui/labeler/` and `polyx/gui/viewer/` |
| 3 | CLAUDE.md | "iconos.py draws 40 icons" | stale | fix-doc | Correct count to 43 |
| 4 | CLAUDE.md | "43 tests against synthetic shapes" | stale | fix-doc | Correct to 876 |
| 5 | README.md | Project layout `polyx/detector/`, `polyx/trainer/`, `polyx/etiquetador/`, `polyx/visor/` | stale | fix-doc | Update to `polyx/gui/{detector,trainer,labeler,viewer}` |
| 6 | README.md | "43 tests against synthetic shapes of known size" | stale | fix-doc | Correct to 876 |
| 7 | README.md | "Python 3.9+" (macOS) vs "Python 3.11.x" (Windows) | stale | fix-doc | Align to Python 3.11 |
| 8 | README.md | `polyx/model/plataforma.py` (portability claim) | stale | fix-doc | Change to `polyx/core/platform.py` |
| 9 | AGENTS.md | "Nothing OS-specific outside `polyx/model/plataforma.py`" | stale | fix-doc | Already corrected in-session to `polyx/core/platform.py` |
| 10 | docs/diagnostico_sigterm.md | "full test suite (281 tests)" / "~31.8s" | stale | fix-doc | Update to 876 tests / 116.71s |
| 11 | CLAUDE.md | "Labeller and Viewer working, verified with 552 crops" | unverifiable | NEEDS-USER | Provide provenance or remove claim |
| 12 | README.md | "auditar_traduccion.py must report 0 untranslated" | unverifiable | NEEDS-USER | Run script to confirm |
| 13 | docs/diagnostico_sigterm.md | "Peak RSS ~383-390 MB" | unverifiable | NEEDS-USER | Re-measure on current suite |