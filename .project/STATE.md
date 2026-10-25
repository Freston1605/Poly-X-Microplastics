---
pipeline: gsd-path/v2
project: poly-x-microplastics
milestone: null
phase: build
                    # inspect only for brownfield; greenfield starts at define
                    # roadmap only in program flow (CHARTER.md exists)
status: active
branch: gsd-path/M001
                    # the router rebinds before any next-milestone file change
                    # the bound branch is never main; ship integrates it there
archive: null       # persisted archive transaction path; never recomputed
integration_default: direct # direct | pull-request | external-landing; project setting
integration: direct # current milestone; may override the default before build
integration_source: default # default | milestone; preserves override provenance
default_branch: main # origin default branch; initialization records it once, nothing changes it
---

# Project State

> **Migrated to GSD Pi on 2026-10-25.** This corpus was imported as milestone
> M011 (slices S01-S03, eight tasks) and the GSD Path v2 run was stopped while
> T001 was in progress. The branch `gsd-path/M001` and `stash@{0}` keep the
> abandoned run's history; the live plan is in the GSD database, not here.

One file, always current. The router reads this first; every phase updates
it on completion. If this file and the artifacts disagree, the artifacts win
— fix this file.

## Log

<!-- append one line per transition: date, phase, event -->
- 2026-10-25 — inspect — project initialized
- 2026-10-25 — inspect — router bound initial milestone
- 2026-10-25 — inspect — docs audit gate failed: remediation queue binding
- 2026-10-25 — inspect — docs audit gate passed after remediation queue fix
- 2026-10-25 — inspect — inspection artifacts passed
- 2026-10-25 — define — definition started
- 2026-10-25 — define — milestone intent approved
- 2026-10-25 — research — research started
- 2026-10-25 — research — research manifest at .project/research/RESEARCH.md; dispatched: stack, pitfalls; skipped: domain, similar; 4 questions
- 2026-10-25 — decide — decision synthesis started
- 2026-10-25 — decide — decision synthesis at .project/research/SYNTHESIS.md; 8 decisions, 2 NEEDS-USER
- 2026-10-25 — plan — planning started
- 2026-10-25 — plan — plan approved
- 2026-10-25 — build — build started
- 2026-10-25 — build — build blocked: blocked recovery
- 2026-10-25 — build — build recovery
