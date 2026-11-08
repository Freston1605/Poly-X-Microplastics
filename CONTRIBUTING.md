# Contributing to Poly-X

## What the CI does

Three workflows run on **every PR** and on every push to `main`:

- **`ruff.yml`** — `ruff check` and `ruff format --check` over `src/`, `tests/`,
  `tools/` and `install/`, using the ruff version locked in `uv.lock`. It
  installs no project dependencies, so it reports in seconds instead of after
  the gate's suite. It runs the same two commands the gate runs, no more.
- **`pre-commit.yml`** — runs every hook in `.pre-commit-config.yaml` with
  `pre-commit run --all-files`, so a hook you never installed locally still
  fails the PR.
- **`ci.yml`** — the gate, plus the 3.12 suite and the wheel build:
  1. Checks `uv.lock` is current and smoke-tests the installer scripts before
     any dependency sync.
  2. Installs the Qt libraries needed without a display (`libegl1`, `libgl1`,
     `libxkbcommon0`, `libdbus-1-3`, `libfontconfig1`) and syncs the locked
     environment (`uv sync --locked --extra gui`) on Python 3.11.
  3. Runs `python tools/check.py`: translation audit, palette rule, the full
     suite (serial, `QT_QPA_PLATFORM=offscreen`), ruff and ty. The gate is the
     one owner of the checks, so a check added to `tools/check.py` cannot
     silently miss CI. ty blocks on `src/polyx/core` + `src/polyx/cli`.
  4. Types the GUI on the same 3.11 environment as an advisory step: it
     reports and never fails the build.
  5. A separate `test` job runs the suite on Python 3.12, and `build` builds
     the wheel, loads the packaged catalogs and smoke-imports it.

`publish.yml` is separate and does not run per PR: it publishes to PyPI on `v*`
tags via Trusted Publishing.

Before pushing, run the same checks locally:

```bash
uv run ruff check src/ tests/ tools/ install/
uv run ruff format --check src/ tests/ tools/ install/
uv run ty check src/polyx/core src/polyx/cli
```

After editing dependencies, run `uv lock` and commit the lock: CI runs
`uv lock --check`, and the suite asserts every declared package is present in
`uv.lock`.

The whole gate — translation audit, suite, ruff and ty — in one command,
from any checkout or git worktree:

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

- The suite pins the interface language with `POLYX_LOCALE=es`. If the report
  comes out in English and a test that reads report text fails, do not touch
  the test: it is pinned on purpose.
- No model, GPU or dataset is needed: the `.pt` weights and images are not part
  of the suite (they are in .gitignore).

## Debugging a red build

1. Open the failed PR on GitHub and look at the red workflow or job:
   - `Ruff` → run `uv run ruff check src/ tests/ tools/ install/` and
     `uv run ruff format --check src/ tests/ tools/ install/`. Fix with
     `ruff check --fix` / `ruff format`; do not reformat unrelated files.
   - `pre-commit` → the hook name and its diff are in the log. Run
     `uv run pre-commit run --all-files` locally; `pre-commit clean` resets a
     stale hook environment.
   - `CI` → `quality` → read the failing step. `Lockfile is current` means the
     lock is stale: run `uv lock` and commit `uv.lock`. **Do not** relax a pin:
     `numpy>=2` breaks ultralytics 8.3.40 validation (they removed `np.trapz`),
     and bumping `PySide6` can change Qt dialog behaviour. `Run the gate` prints
     the command that failed; reproduce it with `python tools/check.py`. The
     `Type check (GUI)` step is advisory and never fails the build.
   - `CI` → `test` → the failing test is printed with its assertion. The suite
     is deterministic: if it passes locally it should pass in CI; if it fails
     only in CI, check the language (`POLYX_LOCALE`) and the locale.
   - `CI` → `build` → the wheel, the packaged catalogs or a smoke import
     failed; the log names the command.
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

Three local hooks enforce the conventions:

- `english-added-lines` — the lines a commit adds to `src/` and `tests/` carry
  no Spanish comments, docstrings or identifiers. On a whole-tree run, a
  `src/` file with no staged changes is scanned in full; `tests/` keeps the
  added-lines rule so grandfathered Spanish names still pass.
- `i18n-audit` — the committed `messages.pot` matches the source tree, every
  shipped catalog is fully translated with matching `%(name)s` placeholders,
  and no user-visible string bypasses the translation call.
- `no-hardcoded-fallback` — no literal fallback in a
  `CLASS_COLOR_HEX.get(...)` lookup; use `T.FALLBACK_COLOR_HEX`.

The same checks are listed in [`CODING_STANDARDS.md`](CODING_STANDARDS.md).

## Translating the interface

English is the source language: every user-facing string is an English
sentence wrapped in a gettext call, and the English sentence is the message id.
Spanish is a catalog, not code. The workflow is extract → translate → compile →
test, and every step has a command:

```bash
python tools/i18n/cli.py extract   # regenerate src/polyx/locales/messages.pot
python tools/i18n/cli.py update    # merge the template into every catalog
python tools/i18n/cli.py compile   # write the .mo files the runtime reads
python tools/i18n/cli.py audit     # the gate: fail on any drift
python tools/i18n/cli.py pseudo    # generate the development pseudo-locale
```

1. **Extract.** `extract` walks `src/polyx` for translation calls and rewrites
   the committed template `src/polyx/locales/messages.pot`. Run it after
   adding, removing or moving a wrapped string: editing source shifts the line
   references, so a template that was fine yesterday can be stale today.
2. **Update.** `update` merges the template into every shipped catalog under
   `src/polyx/locales/<tag>/LC_MESSAGES/messages.po`. Existing translations are
   kept; new entries arrive empty or fuzzy.
3. **Translate.** Edit the `.po` directly. It is plain data any CAT tool or
   editor can open. Fill every `msgstr`; keep every `%(name)s` placeholder and
   use the catalog's plural forms. A translator does not need Python.
4. **Compile.** `compile` writes the `.mo` files the runtime reads. They are
   gitignored build artifacts: the wheel compiles them in a build hook and the
   frozen application adds them as data files. A development checkout also
   compiles a missing catalog on demand into a temporary cache, so a fresh
   clone is not silently English-only.
5. **Test.** `python tools/check.py` runs the translation audit, the suite,
   ruff and ty. The audit is also the `i18n-audit` pre-commit hook, so the
   same failure appears before the commit leaves your machine.

The template is committed on purpose: the audit compares it with a fresh walk
of the source and fails when they disagree, which is how a forgotten wrapper
is caught in the pull request that introduced it. Never edit the `.pot` by
hand — regenerate it with `extract`.

To see which strings were never extracted, generate the pseudo-locale and run
the application with it:

```bash
python tools/i18n/cli.py pseudo
POLYX_LOCALE=qps_ploc python -m polyx.gui
```

Every translated string appears bracketed and accented; an English sentence on
screen is a string that bypassed the translation call. The pseudo-locale is
development-only: it is gitignored, excluded from the wheel and the frozen
application, and it is not part of any gate.

Adding a language is adding a directory: create
`src/polyx/locales/<tag>/LC_MESSAGES/messages.po` with an empty catalog, run
`update`, and translate it. No Python change and no entry in a language list:
the directory is the registration and Babel supplies the display name.

## Before opening a PR

- The full suite passes locally (`pytest -q`, zero failures), and
  `ruff check` / `ruff format --check` are clean (see above).
- The local hooks pass: `uv run pre-commit run --all-files` (or the commit
  hook ran on your last commit).
- The PR body follows `.github/PULL_REQUEST_TEMPLATE.md`; that file is the
  single owner of the required fields.
- Comments, docstrings and identifiers are in English. User-facing strings are
  written in English and wrapped in a gettext call; Spanish lives only in the
  catalogs under `src/polyx/locales/` (see `CODING_STANDARDS.md`, ADR-0003 and
  ADR-0004).
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
