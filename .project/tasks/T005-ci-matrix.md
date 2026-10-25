---
id: T005
title: CI matrix across ubuntu/windows/macos with offscreen Qt and artifact builds
wave: 3
deps: [T001, T002, T003, T008]
status: pending
agent: null
base: null
worktree: null
task_branch: null
files:
  - .github/workflows/ci.yml
  - .github/workflows/release.yml
---

# T005 — CI matrix across ubuntu/windows/macos with offscreen Qt and artifact builds

## Context

The existing CI only runs on ubuntu-24.04 with install + test. Success criterion SC4 requires the full 876-test suite on ubuntu-24.04, windows-latest, and macos-latest with `QT_QPA_PLATFORM=offscreen`, and SC2 requires building and smoke-testing each artifact before release.

## Approach

- Rewrite `.github/workflows/ci.yml` as a matrix over ubuntu-24.04/windows-latest/macos-latest: checkout, setup Python 3.11, install requirements, run `pytest -q` with `QT_QPA_PLATFORM=offscreen`.
- Extend `.github/workflows/release.yml` (from T002) to build the wheel and one-file exe per OS, smoke-test each binary (launch + exit), and attach to a GitHub Release on tag.
- Do not skip any OS in the matrix (N8).

## Interface contract

- None

## Intent coverage

- SC4

## Acceptance criteria

1. CI matrix has all three OS jobs green.
2. Full suite (876 tests) runs on each OS with `QT_QPA_PLATFORM=offscreen`.
3. Each built artifact is smoke-tested before release.

## Verify

```bash
test -f .github/workflows/ci.yml && test -f .github/workflows/release.yml && echo ok
```

Heavy: no
## Log

- 2026-10-25 — created by planner
