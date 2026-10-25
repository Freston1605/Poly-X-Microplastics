---
id: T004
title: Update CLAUDE.md, README.md, AGENTS.md to current layout and counts
wave: 2
deps: [T001]
status: pending
agent: null
base: null
worktree: null
task_branch: null
files:
  - CLAUDE.md
  - README.md
  - AGENTS.md
  - docs/diagnostico_sigterm.md
---

# T004 — Update CLAUDE.md, README.md, AGENTS.md to current layout and counts

## Context

Docs describe the superseded `polyx/model` layout, `polyx/gui/etiquetador`/`visor` (now `labeler`/`viewer`), "43 tests" (now 876), Pillow 10.4.0 (now 12.3.0), and the removed polyx/model/plataforma.py (now `polyx/core/platform.py`). This task reconciles them with the post-refactor `polyx/core` + `polyx/gui` layout and the `src/` migration.

## Approach

- Rewrite CLAUDE.md's structure tree to show `src/polyx/core/` as the engine and `src/polyx/gui/{detector,trainer,labeler,viewer}`; remove `polyx/model` references; correct icon count (43) and test count (876); update Pillow to 12.3.0.
- Update README.md layout/entry points/test count/Python version/portability path.
- Update AGENTS.md OS-specific-code path to `polyx/core/platform.py` (already corrected) and test command.
- Update docs/diagnostico_sigterm.md test count (876) and timing.
- Do not translate user-facing GUI strings (N5).

## Interface contract

- None

## Intent coverage

- SC5

## Acceptance criteria

1. CLAUDE.md, README.md, AGENTS.md describe `polyx/core` + `polyx/gui` and the `src/` layout.
2. Test count stated as 876 everywhere it appears.
3. No stale references to `polyx/model/`, `etiquetador`, or `visor` remain in these docs.

## Verify

```bash
grep -R "polyx/model\|etiquetador\|43 tests" CLAUDE.md README.md AGENTS.md || true
```

Heavy: no
## Log

- 2026-10-25 — created by planner
