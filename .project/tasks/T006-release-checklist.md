---
id: T006
title: Document release checklist (version bump, changelog, tag, upload)
wave: 2
deps: []
status: pending
agent: null
base: null
worktree: null
task_branch: null
files:
  - docs/RELEASE_CHECKLIST.md
---

# T006 — Document release checklist (version bump, changelog, tag, upload)

## Context

Success criterion SC6 requires a documented, runnable release checklist covering version bump, changelog, tag, and artifact upload, with maintainer sign-off per item (N9). The version bump itself (2.1.0 vs 3.0.0) remains a NEEDS-USER decision carried in SYNTHESIS.

## Approach

- Create `docs/RELEASE_CHECKLIST.md` with ordered steps: choose version, bump `__version__` and pyproject, update changelog, run full suite, tag, push, verify CI build, download artifacts, smoke-test, publish GitHub Release, sign off.
- Reference the release workflow and the PyInstaller spec.
- Note the NEEDS-USER version decision as a gating step.

## Interface contract

- None

## Intent coverage

- SC6

## Acceptance criteria

1. `docs/RELEASE_CHECKLIST.md` exists and covers version bump, changelog, tag, and artifact upload.
2. Each item is checkable and references the commands/workflows that run it.

## Verify

```bash
test -f docs/RELEASE_CHECKLIST.md && echo ok
```

Heavy: no
## Log

- 2026-10-25 — created by planner
