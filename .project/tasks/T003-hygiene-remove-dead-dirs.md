---
id: T003
title: Remove empty polyx/model and orphan pycache after grep gate
wave: 1
deps: [T001]
status: pending
agent: null
base: null
worktree: null
task_branch: null
files:
  - src/polyx/core/__pycache__/segmentacion_aprendida.cpython-311.pyc
  - tests/test_no_polyx_model_imports.py
---

# T003 — Remove empty polyx/model and orphan pycache after grep gate

## Context

`polyx/model/` is an empty dead directory left over from the refactor and `segmentacion_aprendida` has an orphan pycache with no source. Prohibition N2 requires confirming nothing imports `polyx.model`/`polyx.ml` before deletion. This task runs that grep gate and removes the dead artifacts, adding a CI-facing test that the import is gone.

## Approach

- Run `grep -R "polyx\.model\|polyx\.ml" --include=*.py .` (excluding `.venv`); fail the task if any hit remains.
- Delete `src/polyx/model/` (and its pycache) and `src/polyx/core/__pycache__/segmentacion_aprendida.*.pyc`.
- Add `tests/test_no_polyx_model_imports.py` that greps the repo for `polyx.model`/`polyx.ml` imports and asserts none.
- Update AGENTS.md wording already points at `polyx/core/platform.py` (done in-session).

## Interface contract

- None

## Intent coverage

- SC5

## Acceptance criteria

1. `polyx/model/` directory and the orphan `segmentacion_aprendida` pycache no longer exist.
2. No `.py` file imports `polyx.model` or `polyx.ml`.
3. `tests/test_no_polyx_model_imports.py` passes.

## Verify

```bash
python -m pytest tests/test_no_polyx_model_imports.py -q
```

Heavy: no
## Log

- 2026-10-25 — created by planner
