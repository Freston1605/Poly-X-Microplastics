# Docs Audit

Repo root: /var/home/Franco/Projects/Poly-X-Microplastics
Audited: 2026-10-18
Audited HEAD: 33f99eb5b544ea85c68ad206313b611be5f1303c
Alignment mode: no

## Summary

| Verdict | Count |
|---------|-------|
| verified | 26 |
| stale | 4 |
| aspirational | 0 |
| unverifiable | 8 |
| descriptive docs (no testable claims) | 27 |

Worst drift: CLAUDE.md and AGENTS.md still quote Pillow 10.4.0 / 128 tests while the repo now ships Pillow 12.3.0 and 210 tests.

## Doc: AGENTS.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| "pasan los 128" tests | status | stale | tests collected: 210 (pytest -q → 210 passed) |
| ".venv\Scripts\python.exe -m pytest -q" | command | verified | tests/ present; pytest collects 210 |
| "Comentarios y variables en español" | feature | unverifiable | style convention; not mechanically checkable |
| "Rutas siempre con pathlib.Path" | structure | verified | polyx/ uses Path throughout (paths.py) |
| "Colores desde polyx/core/theme.py" | feature | verified | polyx/core/theme.py defines T tokens |
| "Nada específico de un SO fuera de plataforma.py" | structure | verified | polyx/core/plataforma.py exists; OS calls centralized |
| "Imágenes con cv2.imdecode/imencode" | feature | verified | polyx/ uses cv2.imdecode (report_html.py) |
| Etiquetador/Visor "en construcción" (histórico) | status | unverifiable | no date evidence in repo |

## Doc: CLAUDE.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| "Versión: 2.0.0" | status | verified | polyx/__init__.py: __version__ = "2.0.0" |
| "Python 3.11.x" | config | unverifiable | no python-version pin in repo; system 3.14 used |
| "PySide6 6.7.2" | config | verified | requirements.txt: PySide6==6.7.2 |
| "Ultralytics 8.3.40" | config | verified | requirements.txt: ultralytics==8.3.40 |
| "OpenCV 4.10.0.84" | config | verified | requirements.txt: opencv-python==4.10.0.84 |
| "NumPy 1.26.4" | config | verified | requirements.txt: numpy==1.26.4 |
| "Pillow 10.4.0" | config | stale | requirements.txt: Pillow==12.3.0 |
| "Matplotlib 3.9.2" | config | verified | requirements.txt: matplotlib==3.9.2 |
| "pyqtgraph 0.13.7" | config | verified | requirements.txt: pyqtgraph==0.13.7 |
| "core/iconos.py dibuja 40 iconos" | feature | verified | polyx/core/iconos.py: 40 quoted icon names |
| "LADO_MINIMO_PX 2 px" | feature | verified | polyx/etiquetador/canvas.py:52 LADO_MINIMO_PX = 2.0 |
| "scipy no está en requirements.txt" | integration | verified | requirements.txt has no scipy |
| "Etiquetador verificado con 552 recortes (ago 2026)" | status | unverifiable | no artifact/count in repo |
| "Visor calibración reparada (ago 2026)" | status | unverifiable | no test evidence in repo |
| "Estructura polyx/..." | structure | verified | ls polyx/ matches documented tree |

## Doc: CONTRIBUTING.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| "Un único job en .github/workflows/ci.yml" | structure | verified | .github/workflows/ci.yml exists |
| "Instala requirements.txt con pins exactos" | config | verified | requirements.txt pins versions |
| "python -m pytest -q con QT_QPA_PLATFORM=offscreen" | command | verified | tests pass locally (210 passed) |
| "No hay linter ni build" | feature | verified | no lint config in repo |
| "No hace falta modelo, GPU ni dataset" | feature | verified | suite runs without .pt/models |
| "pytest no está en requirements.txt" | config | verified | requirements.txt has no pytest |

## Doc: LEEME_macOS.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| "los 49 tests automáticos" | status | stale | tests collected: 210 |
| "Python 3.9 o superior" | config | unverifiable | no python-version constraint in repo |
| "PyTorch 2.2.2 para Mac Intel" | config | unverifiable | network/installer not run |
| "Poly-X.log en ~/Library/Logs" | feature | unverifiable | static claim; not executed |

## Doc: README.en.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| "43 tests against synthetic shapes" | status | stale | shape-related test files sum 65 (morfologia 21 + calibracion 20 + fibra 12 + anillo 9 + mascara 3) |
| ".venv\Scripts\python.exe -m pytest tests/ -q" | command | verified | tests/ present; 210 collected |
| "pytest no está en requirements.txt" | config | verified | requirements.txt has no pytest |

## Doc: README.md

| Claim | Type | Verdict | Evidence |
|-------|------|---------|----------|
| "tests/ suite de pruebas" | structure | verified | tests/ directory present with 14 test files |
| ".venv\Scripts\python.exe -m pytest tests/ -q" | command | verified | tests/ present; 210 collected |

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

## User rulings

<!-- Appended after the ruling walk (standalone runs). Verbatim, append-only.
     accept-drift rulings suppress the item in future audits.
     `planned: no` rows are the alignment queue: gsd-path and gsd-path-plan offer
     to absorb them until drained. When a ruling becomes a task, set
     planned to the task id — never delete the row. -->

| Queue # | Ruling | User's words | Planned |
|---------|--------|--------------|---------|

None.

## Remediation queue

| # | Doc | Claim | Verdict | Class | Suggested action |
|---|-----|-------|---------|-------|------------------|
| 1 | AGENTS.md | "pasan los 128" tests | stale | fix-doc | Update test count to 210 |
| 2 | CLAUDE.md | "Pillow 10.4.0" | stale | fix-doc | Update to 12.3.0 to match requirements.txt |
| 3 | LEEME_macOS.md | "los 49 tests automáticos" | stale | fix-doc | Update to 210 |
| 4 | README.en.md | "43 tests against synthetic shapes" | stale | fix-doc | Recount shape-related tests (65) or redefine subset |
| 5 | CLAUDE.md | "Python 3.11.x" | unverifiable | NEEDS-USER | Confirm supported Python range; no pin present |
| 6 | CLAUDE.md | "Etiquetador verificado con 552 recortes (ago 2026)" | unverifiable | NEEDS-USER | Provide measurement artifact or remove claim |
| 7 | CLAUDE.md | "Visor calibración reparada (ago 2026)" | unverifiable | NEEDS-USER | Provide verification evidence |
| 8 | LEEME_macOS.md | "Python 3.9 o superior" | unverifiable | NEEDS-USER | Confirm min supported Python |
| 9 | LEEME_macOS.md | "PyTorch 2.2.2 para Mac Intel" | unverifiable | NEEDS-USER | Verify installer pin |
| 10 | AGENTS.md | "Comentarios y variables en español" | unverifiable | NEEDS-USER | Style policy; not mechanically checkable |
| 11 | AGENTS.md | Etiquetador/Visor "en construcción" (histórico) | unverifiable | NEEDS-USER | Clarify or remove historical note |
| 12 | LEEME_macOS.md | "Poly-X.log en ~/Library/Logs" | unverifiable | NEEDS-USER | Verify path in installer |
