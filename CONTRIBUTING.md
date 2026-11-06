# Contributing to Poly-X

## What the CI does

A single GitHub Actions job (`.github/workflows/ci.yml`) runs on **every PR** and
on every push to `main`:

1. Installs the Qt libraries needed without a display (`libegl1`, `libgl1`,
   `libxkbcommon0`, `libdbus-1-3`, `libfontconfig1`).
2. Installs `requirements.txt` **with the exact pins** (`numpy==1.26.4`,
   `ultralytics==8.3.40`, `PySide6==6.7.2`, …) on Python 3.11. If someone bumps a
   dependency, CI flags it here, not in the lab.
3. Runs the full suite: `python -m pytest -q` with `QT_QPA_PLATFORM=offscreen`
   (the suite opens no windows; PySide6 is the only thing that asks for a
   platform).

Beyond the suite, the job runs `ruff check` (lint), `ruff format --check`
(formatting; line length 100 in `pyproject.toml`) and `ty` — type checking is
blocking on `src/polyx/core` + `src/polyx/cli` and advisory on the GUI. A
follow-up `build` job builds the wheel and smoke-imports it. Before pushing,
run the same checks locally:

```bash
uv run ruff check src/ tests/ tools/ install/
uv run ruff format --check src/ tests/ tools/ install/
uv run ty check src/polyx/core src/polyx/cli
```

The whole gate — suite, ruff and ty — in one command, from any checkout or
git worktree:

```bash
python tools/check.py                              # full suite, serial
python tools/check.py tests/test_annotate.py -q    # one file, serial
python tools/check.py --parallel                   # full suite, one worker per core
python tools/check.py tests/test_annotate.py --parallel=4 -q
```

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

1. Open the failed PR on GitHub and look at the red step:
   - `Install pinned requirements` → someone changed a pin. pip's error says
     which one. **Do not** relax the pin: `numpy>=2` breaks ultralytics 8.3.40
     validation (they removed `np.trapz`), and bumping `PySide6` can change Qt
     dialog behaviour.
   - `Run test suite` → the failing test is printed with its assertion. The
     suite is deterministic: if it passes locally it should pass in CI; if it
     fails only in CI, check the language (`POLYX_IDIOMA`) and the locale.
2. Reproduce locally with the same command as the step (Linux above).
3. Push the fix to the same branch: the job re-runs on its own, and the new push
   cancels the previous run.

## Before opening a PR

- The full suite passes locally (`pytest -q`, zero failures), and
  `ruff check` / `ruff format --check` are clean (see above).
- The PR body follows `.github/PULL_REQUEST_TEMPLATE.md`; that file is the
  single owner of the required fields.
- Comments, docstrings and identifiers are in English; Spanish survives only as
  the runtime UI source inside the `tr()` layer (see `AGENTS.md` and
  `CLAUDE.md`).

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
