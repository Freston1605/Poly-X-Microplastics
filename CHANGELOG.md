# Changelog

All notable changes to Poly-X are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-10-26

### Added
- Headless engines and CLIs: `polyx-train`, `polyx-eval`, `polyx-detect`,
  `polyx-report`, plus `polyx-download-models` for the non-bundled weights.
- Python packaging: `pyproject.toml` with hatchling, src layout, console
  scripts, optional extras `[gui]`, `[dev]`, `[sam2]`.
- Self-contained Spanish-remnant scanner (`tests/english_scan.py`) backing
  the four English-convention gates without LibCST.
- CI wheel build + smoke-test job (build, install wheel in a clean venv,
  import + asset check + `--help` on every headless CLI + `[gui]` import).
- Publish workflow using PyPI Trusted Publishing (OIDC), triggered by `v*` tags.
- `LICENSE` (MIT) and `.env.example`.

### Changed
- Package layout: source moved to `src/polyx/`; tooling under `tools/`
  (`tools/audit`, `tools/manual`, `tools/packaging`); documentation under
  `docs/`; end-user double-click scripts under `install/`.
- Bundled assets (`polyx.ico`) now ship inside the wheel.
- Single source of truth for the version (`src/polyx/__init__.py`,
  read by hatchling).
- PyPI metadata: urls, classifiers, keywords, author email, PEP 639 license.
- README install section documents `pip install polyx-microplastics[gui]`.

### Removed
- LibCST translation codemods, harness, manifests and the `libcst` pin.
- 12.4 MB of generated manual HTML from version control (restored after
  review: a ZIP download must carry the manual offline).
- `auditar_traduccion.py` (moved to `tools/audit/audit_translations.py`).

### Fixed
- Per-image metrics in rebuilt reports use the real `ap_class_index`.
- `empaquetar_para_llevar.bat` shipped an offline bundle without program
  code (it robocopy'ed a `polyx\` directory that no longer exists).
- `auditar_traduccion.py` walked a nonexistent `polyx/` path and always
  reported "-4 translated".
- Manual project-tree figure described the pre-rename layout.