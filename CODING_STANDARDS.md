# Coding Standards — Poly-X

Read by the review agent. Implementation guidance lives in `CLAUDE.md`.

## Language

- Comments, docstrings, and identifiers in English. Spanish survives only
  inside `tr()` and the UI strings it wraps.
- Qt signals use snake_case: `models_changed`, `run_started`, `run_finished`.

## Paths

- Always `pathlib.Path`, never bare strings.

## Colors

- Use `T.ACCENT`, `T.OK`, etc. (from `polyx.gui.theme`), never hand-coded hex.
- As text: the `_TX` variant (`T.OK_TX`). As background of a filled button:
  the base one. In HTML reports: `T.DOC.*`.
- Domain colours (class palette) live in `polyx.core.annotate` and are
  re-exported by the theme. Fallback for unknown classes: `T.FALLBACK_COLOR_HEX`.
- Never hardcode a fallback like `"#33aaff"` or `"#888"` — use the shared constant.

## Methods

- `_` prefix for private (widget, signal callback).

## Image I/O

- `cv2.imdecode`/`imencode` over `np.fromfile`, never `imread`/`imwrite`
  (accented paths fail on Windows).

## OS-specific code

- Nothing OS-specific outside `polyx.core.platform`.

## Tests

- New test code uses English identifiers (grandfathered Spanish test files
  are not being migrated, but new code follows the rule).
- Run with the project `.venv`. Full suite must pass with zero failures
  and zero skips before calling anything done.

## Review checklist

Automated by the local hooks and the gate — these are the checks
`.pre-commit-config.yaml` and `tools/check.py` actually run:

- [ ] `pytest` passes with zero failures and zero skips.
- [ ] `ruff check` and `ruff format --check` are clean.
- [ ] `ty check src/polyx/core src/polyx/cli` is clean.
- [ ] `english-added-lines` — the lines a commit adds to `src/` or `tests/`
      carry no Spanish comments, docstrings or identifiers (the vocabulary
      in `tests/english_scan.py`).
- [ ] `no-hardcoded-fallback` — no literal fallback in a
      `CLASS_COLOR_HEX.get(...)` lookup; use `T.FALLBACK_COLOR_HEX`.

Review-only — the tooling cannot see these, so the reviewer must:

- [ ] No bare string paths (use `pathlib.Path`).
- [ ] No OS-specific code outside `platform.py`.
- [ ] `cv2.imdecode`/`imencode` used, not `imread`/`imwrite`.
- [ ] Qt signals use snake_case.
- [ ] No hardcoded colour hex outside the palette fallback (theme tokens or
      `polyx.core.annotate` constants).
- [ ] Docs (`CLAUDE.md`, `GLOSSARY.md`) updated in the same PR as behaviour
      changes.