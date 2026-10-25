---
id: T008
title: Retarget i18n audit roots and add CI gate for untranslated source
wave: 2
deps: [T001]
status: pending
agent: null
base: null
worktree: null
task_branch: null
files:
  - tests/test_comments_ingles.py
  - tests/test_docstrings_ingles.py
  - tests/test_identifiers_ingles.py
  - tests/test_report_headings_ingles.py
  - tests/test_translate_identifiers.py
  - auditar_traduccion.py
---

# T008 — Retarget i18n audit roots and add CI gate for untranslated source

## Context

The i18n audit scripts and tests hard-code `REPO_ROOT / "polyx"` roots that move to `REPO_ROOT / "src" / "polyx"` after T001. Success criterion SC8 requires scanning all `.py` outside `tr()` for non-English identifiers/comments and failing CI on any untranslated source string (E6), while leaving user-facing Spanish GUI strings untouched (N5).

## Approach

- Retarget every audit root (`REPO_ROOT / "polyx"` → `REPO_ROOT / "src" / "polyx"`) in the tests and `auditar_traduccion.py`.
- Update known-module paths and floors if the module count changes.
- Ensure the audit scans all `.py` under `src/polyx` and fails on non-ASCII identifiers or non-English comments/docstrings outside `tr()`.
- Do not alter user-facing Spanish GUI strings.

## Interface contract

- None

## Intent coverage

- SC8

## Acceptance criteria

1. Audit tests pass on the moved tree with roots retargeted (E6: audit scans all `.py` outside `tr()`; any non-ASCII identifier or Spanish comment fails).
2. `auditar_traduccion.py` reports 0 untranslated source strings.
3. User-facing Spanish GUI strings remain unchanged (N5).

## Verify

```bash
python -m pytest tests/test_comments_ingles.py tests/test_identifiers_ingles.py tests/test_docstrings_ingles.py -q
```

Heavy: no
## Log

- 2026-10-25 — created by planner
