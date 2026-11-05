# ADR-0001: Runs live in the per-user data directory, not the repo

- **Status:** Accepted
- **Date:** 2026-11-05
- **Landed in:** #26

## Context

Poly-X is a Windows-first desktop application for analysing microplastics in
Nile Red fluorescence images. Historically every run (detector and trainer)
wrote its outputs — annotated images, `centroids.csv`, `metrics.json`,
`report.html`, training weights — into `runs/` and `runs_train/` **inside the
project directory**. That had three problems:

1. **The install/update flow fights the user's data.** `SETUP.bat` and
   `actualizar.bat` copy, replace and (on uninstall) delete files in the
   project folder. Anything the user generated had to be carefully excluded
   from every destructive operation, all over the codebase.
2. **Portability.** A run produced on one machine was not meaningfully
   portable to another, yet it sat inside the repository tree alongside
   versioned sources, inviting accidents (`git clean -fdx`, a stale `.venv`
   rebuild, an updater that "helps").
3. **Two platforms, two conventions.** Windows users expect per-user state
   under `%LOCALAPPDATA%`; macOS and Linux have their own per-user data dirs.

## Decision

**All user-generated state — runs, training outputs, logs — lives in the
per-user data directory, resolved by `polyx/core/platform.py`:**

| Platform | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\polyx` |
| macOS | `~/Library/Application Support/polyx` |
| Linux | `$XDG_DATA_HOME/polyx` (default `~/.local/share/polyx`) |
| override | `POLYX_HOME` (tests, portable installs) |

The repository keeps only sources, tests and versioned assets. The updater,
the installer and the uninstaller never touch the data dir. Paths are always
derived through `platform.user_data_dir()` — never hard-coded — and nothing
OS-specific lives outside `polyx/core/platform.py`.

## Consequences

- **The install surface is simpler and safer**: SETUP/actualizar/DESINSTALAR
  can copy, replace or delete the program folder without endangering runs.
- **Existing installs tolerate both layouts**: tooling that reads runs looks
  in the data dir first and falls back to the legacy `runs/` and `runs_train/`
  folders in the project root, which are treated as read-only legacy.
- **Documentation must say where runs live** — CLAUDE.md, the manual and the
  install scripts were updated to match (the manual documents all three
  platforms and `POLYX_HOME`).
- **Tests use `POLYX_HOME`** pointing at a tmp dir; no test writes into the
  developer's real data dir.
- **This reverses nothing about the repo layout**: `models/` and `data/`
  remain the user's and stay in the project folder (they are inputs, not
  outputs).

## Alternatives considered

- **Keep runs in the repo, exclude them everywhere.** Already the bug farm
  that produced this decision; every new destructive path would need its own
  exclusion.
- **Split per module** (detector runs in the repo, trainer in the data dir).
  Two conventions for the same artifact kind; rejected for consistency.
