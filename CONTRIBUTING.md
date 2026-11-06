# Contributing to Poly-X

## What the CI does

Four workflows run on **every PR** and on every push to `main`:

- **`ruff.yml`** — `ruff check` and `ruff format --check` over `src/`, `tests/`,
  `tools/` and `install/`, using the ruff version locked in `uv.lock`. It
  installs no project dependencies, so it finishes in seconds.
- **`pre-commit.yml`** — runs every hook in `.pre-commit-config.yaml` with
  `pre-commit run --all-files`, so a hook you never installed locally still
  fails the PR.
- **`ci.yml`** — quality, tests and build:
  1. Checks `uv.lock` is current and smoke-tests the installer scripts before
     any dependency sync.
  2. Installs the Qt libraries needed without a display (`libegl1`, `libgl1`,
     `libxkbcommon0`, `libdbus-1-3`, `libfontconfig1`) and syncs the locked
     environment (`uv sync --locked --extra gui`) on Python 3.11.
  3. Runs `ty` from `uv.lock`: blocking on `src/polyx/core` + `src/polyx/cli`,
     advisory on the GUI. Diagnostics appear as GitHub annotations.
  4. Runs the full suite on Python 3.11 and 3.12 with
     `QT_QPA_PLATFORM=offscreen` (the suite opens no windows; PySide6 is the
     only thing that asks for a platform).
  5. The `build` job builds the wheel and smoke-imports it.
- **`publish.yml`** — publishes to PyPI on `v*` tags via Trusted Publishing.

Before pushing, run the same checks locally:

```bash
uv run ruff check src/ tests/ tools/ install/
uv run ruff format --check src/ tests/ tools/ install/
uv run ty check src/polyx/core src/polyx/cli
```

After editing dependencies, run `uv lock` and commit the lock: CI runs
`uv lock --check`, and the suite asserts every declared package is present in
`uv.lock`.

The whole gate — suite, ruff and ty — in one command, from any checkout or
git worktree:

```bash
python tools/check.py                              # full suite, serial
python tools/check.py tests/test_annotate.py -q    # one file, serial
python tools/check.py --parallel                   # full suite, one worker per core
python tools/check.py tests/test_annotate.py --parallel=4 -q
```

In a worktree, `tools/check.py` uses that worktree's `.venv` only when it can
import PySide6. A partial environment created by `uv run` without the GUI
extra is skipped in favour of the main checkout's interpreter, with a warning.
If you want the worktree to own its environment, run `uv sync --extra gui`
there first.

Tests run **serially by default**. CI always runs serially and remains the
reproducibility reference. `--parallel` (bare: `-n auto`) or `--parallel=N`
(`-n N`) is an explicit local convenience; the gate never enables it on its
own. A failure under `--parallel` must be re-run without the flag before it is
treated as real: xdist can surface ordering or shared-state problems that a
serial run does not. Every other argument (`tests/...`, `-k`, `-q`, …) is
forwarded to pytest untouched.

## Running the same tests locally

### Windows (the development machine)

```batch
.venv\Scripts\python.exe -m pytest -q
```

### macOS / Linux

```bash
python -m pytest -q          # inside the project .venv
```

- The suite resolves the language at startup and uses **Spanish** unless you set
  `POLYX_IDIOMA` (see `tests/conftest.py`). If the report comes out in English
  and a test that reads report text fails, do not touch the test: it is pinned on
  purpose.
- No model, GPU or dataset is needed: the `.pt` weights and images are not part
  of the suite (they are in .gitignore).

## Debugging a red build

1. Open the failed PR on GitHub and look at the red workflow:
   - `Ruff` → run `uv run ruff check src/ tests/ tools/ install/` and
     `uv run ruff format --check src/ tests/ tools/ install/`. Fix with
     `ruff check --fix` / `ruff format`; do not reformat unrelated files.
   - `pre-commit` → the hook name and its diff are in the log. Run
     `uv run pre-commit run --all-files` locally; `pre-commit clean` resets a
     stale hook environment.
   - `Sync dependencies` → someone changed a pin or the lock is stale. Run
     `uv lock` and commit `uv.lock`. **Do not** relax a pin: `numpy>=2`
     breaks ultralytics 8.3.40 validation (they removed `np.trapz`), and
     bumping `PySide6` can change Qt dialog behaviour.
   - `Type check (engine)` → run
     `uv run ty check src/polyx/core src/polyx/cli`. `Type check (GUI)` is
     advisory and does not fail the build.
   - `Run test suite` → the failing test is printed with its assertion. The
     suite is deterministic: if it passes locally it should pass in CI; if it
     fails only in CI, check the language (`POLYX_IDIOMA`) and the locale.
2. Reproduce locally with the same command as the step (Linux above).
3. Push the fix to the same branch: the job re-runs on its own, and the new push
   cancels the previous run.

## Local pre-commit guardrails

The hooks in `.pre-commit-config.yaml` run locally and in the `pre-commit`
workflow, so a hook that fails on your machine fails the PR too. Install them
once in the dev environment, then run them manually at any time:

```bash
uv sync --extra gui
uv run pre-commit install
uv run pre-commit run --all-files
```

Two local hooks enforce the conventions:

- `english-added-lines` — the lines a commit adds to `src/` and `tests/` carry
  no Spanish comments, docstrings or identifiers. On a whole-tree run, a
  `src/` file with no staged changes is scanned in full; `tests/` keeps the
  added-lines rule so grandfathered Spanish names still pass.
- `no-hardcoded-fallback` — no literal fallback in a
  `CLASS_COLOR_HEX.get(...)` lookup; use `T.FALLBACK_COLOR_HEX`.

The same checks are listed in [`CODING_STANDARDS.md`](CODING_STANDARDS.md).

## Before opening a PR

- The full suite passes locally (`pytest -q`, zero failures), and
  `ruff check` / `ruff format --check` are clean (see above).
- The local hooks pass: `uv run pre-commit run --all-files` (or the commit
  hook ran on your last commit).
- The PR body follows `.github/PULL_REQUEST_TEMPLATE.md`; that file is the
  single owner of the required fields.
- Comments, docstrings and identifiers are in English; Spanish survives only as
  the runtime UI source inside the `tr()` layer (see `AGENTS.md` and
  `CLAUDE.md`).
- The PR body names its originating spec on its own line: `Spec: #N` (or
  `Spec: none` when there is no spec). Review tooling resolves the spec
  source from that line before falling back to commit-message references.

---

## The 2 AM diagnostic (observability)

- Every module keeps a JSONL journal in `~/.polyx/logs/<module>.jsonl`
  (`POLYX_LOG_DIR` redirects it). Rotation: 5 MB × 3 copies.
- Read it with jq:
  ```bash
  jq -r '"\(.ts) \(.level) \(.event) \(.data)"' ~/.polyx/logs/detector.jsonl | tail -40
  ```
- Event names are English `verb.object` pairs (run.finish, imgsz.fallback,
  report.error, train.error, update.available). Failures go at level
  ERROR and carry the traceback in `exc`.
- More detail: `POLYX_LOG=debug python -m polyx.gui.detector`
- The signals that should raise an alarm: `imgsz.fallback` (the batch ran
  at less resolution than requested), `run.sin_forma_alta` (many particles
  left unmeasured), and any `*.error` event.
