# All user-generated run data lives in the per-user data directory

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
  data stays where it is and remains openable by hand.
- `POLYX_HOME` is the portable-install escape hatch: pointing it next to
  the app puts the data "next to the app" again.
- The rule is pinned by `tests/test_paths.py`. Relocating the data later
  orphans what users have accumulated and needs its own ADR.
