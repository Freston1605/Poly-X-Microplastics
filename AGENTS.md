# Poly-X — instructions for agents

**All project documentation is in [CLAUDE.md](CLAUDE.md).** Read it before
touching anything: architecture, modules, data flow, key classes,
palette, parameters, and the decisions already made and why.

This file exists only because some tools look for `AGENTS.md` by
convention. **Do not duplicate content here.**

> Until September 2026 this file was a trimmed copy of CLAUDE.md, and
> it fell behind: it still said the Labeller and the Viewer were "under
> construction" —both shipped in August— and that PP fluoresces orange,
> when measurement showed it is yellowish-green. Keeping two copies of
> the same document is not free: one of them lies, and you cannot tell
> which.

## The minimum before writing code

The review checklist for these rules is
[`CODING_STANDARDS.md`](CODING_STANDARDS.md).

- **Code comments, docstrings and identifiers in English** — Spanish
  only survives inside the `tr()` layer and the UI strings it wraps.
- **Paths always with `pathlib.Path`**, never bare strings.
- **Colors from `polyx/gui/theme.py`** (`T.ACCENT`, `T.OK`…), never by hand.
- **Nothing OS-specific outside `polyx/core/platform.py`.**
- Images with `cv2.imdecode`/`imencode` over `np.fromfile`, not `imread`/`imwrite`:
  on Windows those fail silently with accented paths.
- Dev env: **`uv sync --extra gui`** — the suite imports Qt; plain
  `uv sync` prunes PySide6 from the `.venv`.
- **Docs ship with the behaviour** — manual text and CLAUDE.md updates
  ride in the same PR as the change that makes them true; the worktree
  never carries uncommitted doc updates across builds.
- **Run the gate with `python tools/check.py`** — suite, ruff and the ty
  engine check, from any checkout or worktree. Wait until the **full
  suite passes with zero failures and zero skips** before calling
  anything done.
- **Branch from `origin/main`**, never from a merged feature-branch tip:
  this repo squash-merges, so old tips are not ancestors and produce
  phantom PR conflicts.
