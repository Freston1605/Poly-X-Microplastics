# Coding Standards — Poly-X

Read by the review agent. Implementation guidance lives in `CLAUDE.md`.

## Language

- Comments, docstrings, and identifiers in English.
- User-facing strings are written in English and wrapped in a gettext call
  from `polyx.core.i18n` (`_`, `ngettext`, `pgettext`, `npgettext`; `N_` or
  `lazy` when the string is resolved later). English is the source language:
  the English sentence is the message id.
- Spanish lives only in the catalogs under `src/polyx/locales/`, never in
  Python. A new string is added in English and translated in the `.po` file.
- Placeholders are named `%(name)s`. No f-strings, positional `{}` braces or
  `str.format` inside a translation call: no catalog validator can check them.
- Machine contracts stay English in every locale: CSV headers, `run.json`
  keys, log messages and event names are not translated (ADR-0004).
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
      in `tests/english_scan.py`). On a whole-tree run, a `src/` file with no
      staged changes is scanned in full; `tests/` keeps the added-lines rule
      so grandfathered Spanish names still pass.
- [ ] `i18n-audit` — the committed `messages.pot` matches the source tree;
      every shipped entry is translated with matching `%(name)s` placeholders,
      English placeholder names and no `(s)`/`(es)` parentheticals; no
      user-visible literal reaches a known Qt, CLI or HTML sink without a
      translation call; and no Spanish string constant survives anywhere in
      `src/polyx`. Runs as `python tools/check_i18n.py` and in pre-commit.
      Text that reaches the interface through a local variable is invisible to
      the static scan; the reviewer traces those by hand.
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