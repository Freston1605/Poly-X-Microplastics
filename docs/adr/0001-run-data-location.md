# All user-generated run data lives in the per-user data directory

- **Status:** Superseded by [English-only run artifact vocabulary](../specs/run-artifact-vocabulary.md) (spec #182, ticket #190). This is the original short-form record of ADR-0001; the canonical numbered record is [0001-runs-in-user-data-dir.md](0001-runs-in-user-data-dir.md). The per-user data **location** rule still holds; the "remains openable by hand" consequence below does not: pre-canonical run folders are left untouched but are not read by new code, and opening them needs an older Poly-X version. ADR-0004 stays in force.

Detector runs, training runs and the review dialog's correction dataset
are written to the per-user data dir (`%LOCALAPPDATA%\polyx`,
`~/Library/Application Support/polyx`, `~/.local/share/polyx`;
`POLYX_HOME` overrides), never next to the code. Eleven GUI sites used
to derive `<repo>/runs` by counting `Path(__file__).parents` —
arithmetic that scatters a wheel install's results into
`.venv/lib/python3.11/runs` or crashes outright on read-only system
Pythons.

## Considered options

- **Context rule** (checkout → project root, installed → user data):
  rejected — it makes "where are my runs" conditional on how the package
  was obtained, for no functional gain.
- **Everything next to the code**: rejected — it forfeits read-only
  installs, the reason `core/paths.py` exists.

## Consequences

- The repo-root `runs/`, `runs_train/` and `dataset_correcciones/`
  folders of older versions are **legacy**: no automatic migration; old
  data stays where it is and is not rewritten. **Superseded by spec #182:**
  new code does not open those runs; they need an older Poly-X version.
- `POLYX_HOME` is the portable-install escape hatch: pointing it next to
  the app puts the data "next to the app" again.
- The rule is pinned by `tests/test_paths.py`. Relocating the data later
  orphans what users have accumulated and needs its own ADR.
