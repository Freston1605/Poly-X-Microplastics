<!-- PR title format: <type>(<scope>): <short description> (e.g. feat(detector): add per-image metrics) -->

## Summary

<!-- One or two sentences: what this PR does and why. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no functional change)
- [ ] Performance improvement
- [ ] Documentation
- [ ] CI / tooling / build
- [ ] Breaking change

## Related issues

<!-- "Fixes #123", "Closes #456", "Relates to #789". Linear IDs as [POLY-123] when applicable. -->

## Changes

<!-- Bullet list of concrete changes; group by area if large. -->

-

## Screenshots / recordings

<!-- For Qt/GUI changes: before & after. Delete if not applicable. -->

## Testing

- [ ] `uv run pytest -q` (or `.venv\Scripts\python.exe -m pytest -q`) — 0 failures, 0 skips
- [ ] `uv run ruff check src/ tests/ tools/ install/`
- [ ] `uv run ruff format --check src/ tests/ tools/ install/`
- [ ] `uv run ty check src/polyx/core src/polyx/cli` (engine) — no new errors
- [ ] GUI type check reviewed: `uv run ty check src/polyx/gui` (advisory)
- [ ] Regression test added for bug fixes; new tests for new behavior
- [ ] Manual smoke test (describe):

## Checklist

- [ ] Self-reviewed my own diff
- [ ] Comments, docstrings and identifiers are in English (Spanish only inside `tr()`)
- [ ] User-facing changes documented (README / docs/manual)
- [ ] `CHANGELOG.md` updated for user-facing changes (Keep a Changelog)
- [ ] No secrets, credentials or large binaries committed
- [ ] Breaking changes called out below

## Migration notes / breaking changes

<!-- Delete if none. -->

## Reviewer focus / reading order

<!-- For large PRs: suggested file order, risky areas, known limitations. Delete if small. -->
