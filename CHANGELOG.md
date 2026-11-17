# Changelog

All notable changes to Poly-X are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Fast `Ruff` CI workflow (`ruff check` + `ruff format --check`) that installs
  the ruff version pinned in `uv.lock` and needs no project dependencies, so a
  lint failure reports in seconds instead of after the gate's suite.
- `pre-commit` CI workflow running every hook in `.pre-commit-config.yaml`, so
  a hook that was never installed locally still fails the PR.

### Changed
- One operating-point module owns precision/recall/F1
  (`core/detection_run/operating_point.py`). The Results KPIs read the match
  the run decided under the localisation criterion; the confidence advice, the
  Compare table and the report's second F1 column use the with-class
  criterion, and every label on the page names the one it shows. The duplicate
  helper in `core/report_metrics.py`, the arithmetic in the Compare page and
  the report's threshold sweep were removed.
- Re-reading the ground truth of a finished run now goes through the
  detection-run engine (`reload_ground_truth` in
  `core/detection_run/reload.py`) instead of a GUI copy of the sequence. The
  reload measures with the run's own shape path, sizes each label with that
  photo's recorded calibration, matches with the run's own rule (mask IoU when
  the run did) and records that rule next to the new counts. It republishes the
  affected images in `run.json`, so the reloaded views and the run record can no
  longer disagree. `gui/detector/recargar_gt.py` is gone; the Results page is a
  progress dialog and one call.
- Model aliases are stable run-folder keys, separate from the label the
  interface shows: the GUI writes `model_1`/`model_2`, the CLI writes the
  weights stem unless `--alias` names one, and `run.json` records
  `alias_generated` so the report translates only a generated label. Changing
  the interface language no longer renames a run folder; old folders still load
  and show their recorded alias verbatim.
- The per-image result carries the match the run decided (`MatchResult` with
  the TP/FP/FN/misclassified indices) and the rule behind it (`iou_thr`, box
  or mask IoU). The Errors page, the Results KPIs, the report's error section
  and both threshold sweeps read that decision instead of re-matching with a
  rule of their own; `run.json` records the rule per image/model and
  `load_run` rebuilds the match from it. Old run folders still load.
- The HTML report package is internal: its public surface is
  `generate_report` plus the section catalog (`SECTIONS`, `IDS_SECTIONS`,
  `PRESETS`, `anchor`), the batch precompute is a named immutable result and
  every section has one uniform builder. Rendered reports are byte-identical.
- `ruff-pre-commit` bumped to `v0.17.2` to match the locked ruff version.

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