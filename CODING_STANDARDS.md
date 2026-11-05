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

- [ ] No hardcoded colour hex values (use theme tokens or annotate constants)
- [ ] No bare string paths (use pathlib.Path)
- [ ] No OS-specific code outside platform.py
- [ ] No Spanish identifiers in new code
- [ ] cv2.imdecode/imencode used, not imread/imwrite
- [ ] Docs (CLAUDE.md, GLOSSARY.md) updated in the same PR as behaviour changes