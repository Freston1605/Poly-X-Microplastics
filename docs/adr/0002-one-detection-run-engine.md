# ADR-0002: One detection-run engine

- **Status:** Accepted
- **Date:** 2026-11-06
- **Landed in:** #64

## Context

Poly-X carried two detection pipelines that had to agree but did not share
code. The GUI runner orchestrated model loading, decoding, ground-truth
reading, prediction, measurement, matching, drawing and persistence in
`polyx/gui/detector/runner.py`; the headless path reimplemented a subset in
`polyx.core.detection_run.run_batch`. The two produced different folders —
per-model `labels/` and view PNGs for the GUI, a flat `summary.json` +
`centroids.csv` + `images/` folder for the CLI — and their semantics drifted:
measurement, tiling, calibration, abstention and mask-aware matching could be
fixed on one side and stay wrong on the other.

The report was coupled to the GUI state too. Rebuilding a `DetectorState`
from a run folder was the only way to report a headless run, and that
reconstruction could never carry everything the original state held.

## Decision

`polyx.core.detection_run` is the only detection-run implementation. It is
synchronous, Qt-free and owns decode, per-photo calibration, GT reading,
prediction, measurement, abstention, size filtering, numbering, matching,
drawing and persistence. Its entry point is `run_detection(RunConfig)` and its
result is a `RunResult`; both are plain core data.

The run folder is canonical and self-describing:

```text
<run>/
  run.json               # schema_version 1: status, params, models,
                         #   calibrations, metrics, images, skipped/error
  detections.csv         # predictions, one row per detection
  <alias>/
    {index:04d}_{stem}_{annot,pred,gt}.png
    labels/{index:04d}_{stem}.txt
    classes.txt
```

`load_run` reopens that folder as a `RunResult` without Qt. The GUI is a thin
QThread adapter around the engine: it maps the engine's progress/results onto
the state the existing pages already read (model slots, previews and
calibrations) and keeps review verdicts GUI-side. `polyx-detect`,
`polyx-report --full-report` and `generate_report` all consume the same
`RunResult`; none of them rebuilds a GUI state.

Legacy folders are read, never migrated or rewritten. Old headless folders
(`summary.json` + `centroids.csv`) go through
`polyx.core.detection_run.load_legacy_run`, which rebuilds a `RunResult` from
the counts and predictions the old writer did record. Old GUI run folders keep
their per-model `labels/*.txt` and stay readable in the Viewer. The compact
report keeps its `summary.json`/`centroids.csv` fallback.

`run_batch` and the old headless writer are deleted. The old GUI
`run_state` loader no longer rebuilds a `DetectorState`: a thin compatibility
module keeps the `build_state_from_run` name and delegates to
`load_legacy_run`, returning the same `RunResult` the report consumes.
`polyx.core` exposes the canonical engine surface only.

## Considered options

- **Keep both engines and share helpers.** Rejected: the drift already showed
  up in measurement and calibration, and every new capability would still
  need two integrations plus a parity test.
- **Keep `run_batch` as a thin wrapper over the engine.** Rejected: it
  preserves a second public entry point, a second folder contract and the
  expectation that its old return value is a stable API. It would be a
  permanent adapter for callers that no longer exist.
- **Migrate old folders to the canonical layout on open.** Rejected: it
  rewrites user data, can fail halfway and destroys the ability to compare
  what the old version actually wrote.
- **Keep `run_state` to rebuild `DetectorState` for old reports.** Rejected:
  the report now consumes `RunResult`, so the reconstruction cannot represent
  the report's input faithfully. A Qt-free core reader is the smaller, honest
  seam.

## Consequences

- One implementation to fix, test and document. GUI runs, CLI runs and
  reports share measurement, calibration, matching and folder semantics by
  construction.
- New runs write only `run.json`, `detections.csv`, per-model views/labels and
  `report.html`; they never write `summary.json`, `metrics.json`,
  `resumen.json` or the flat `images/` folder.
- Existing folders remain readable without migration: the Viewer for old GUI
  runs, `load_legacy_run` for old headless runs and the compact report for
  both.
- Old headless GT boxes were never persisted, so a legacy `RunResult` carries
  the per-image TP/FP/FN counts but an empty GT box list. Reports from those
  folders cannot redraw the GT; they describe the counts the old run
  recorded.
- Segmentation masks are measurement inputs, not report artifacts: the
  canonical folder records their effect in the size fields, `size_source` and
  `has_mask`, but not the mask arrays. `load_run` rebuilds the report-facing
  `RunResult`, not a re-runnable inference state.
- ADR-0001 still governs where runs live. This ADR only decides what a run is
  and how it is written.
- The architecture is pinned by `tests/test_detection_engine.py`,
  `tests/test_legacy_runs.py`, `tests/test_cli_detect.py` and
  `tests/test_report_runresult.py`.
