---
title: English-only run artifact vocabulary
description: Specification for the canonical English machine vocabulary of Poly-X artifacts and the hard cut from legacy formats.
tags:
  - spec
  - artifacts
  - i18n
---

# English-only run artifact vocabulary

Source decisions: [Inventory the English machine vocabulary emitted by run artifacts](https://github.com/Freston1605/Poly-X-Microplastics/issues/175), [Locate legacy Spanish reads and Spanish tokens written into artifacts](https://github.com/Freston1605/Poly-X-Microplastics/issues/176), [Decide which artifact families the spec covers](https://github.com/Freston1605/Poly-X-Microplastics/issues/177), [Decide canonical English replacements for Spanish artifact tokens](https://github.com/Freston1605/Poly-X-Microplastics/issues/178), [Decide the compatibility policy for legacy Spanish read boundaries](https://github.com/Freston1605/Poly-X-Microplastics/issues/179), [How does the spec present the current-vs-target token mapping?](https://github.com/Freston1605/Poly-X-Microplastics/issues/180), [What schema_version and hard-cut release note does the spec recommend?](https://github.com/Freston1605/Poly-X-Microplastics/issues/181). The full source inventory is on the throwaway research branch `research/run-artifact-vocabulary`; that branch is never merged.

## Problem Statement

Poly-X writes machine-readable artifacts that a program reads: `run.json`, `detections.csv`, JSONL logs, Viewer exports, calibration data and report anchors. ADR-0004 requires these tokens to stay English in every locale, because a translated key is a schema change and makes the same run mean different things on two machines.

Older code wrote Spanish into these artifacts and still reads Spanish forms from disk: `sucio` and `versiones` in provenance, `placa` and `ninguna` as calibration origins, `media` and `avisos` in the calibration summary, `visor_` and `resumen.json` in Viewer exports, Spanish `data` keys in JSONL logs, Spanish prose in persisted warnings, Spanish report section ids and preset keys, and Spanish values in tiling, boolean and preference readers.

Two readers of the same artifact cannot agree on a spelling today. A script written against English keys fails on a Spanish cell. A saved run written by an older version is still read by current code through pre-canonical readers, so the English contract can never be enforced cleanly while those readers remain.

## Solution

Make the machine vocabulary English-only. Replace every Spanish machine token with a canonical English token, list the replacements in one place, remove all legacy readers, and bump `schema_version` from 1 to 2 so a new `run.json` announces the new contract. Old files stay on disk but new code does not open them. A release note in the CHANGELOG states the cut plainly.

The spec presents a replacement list, not a compatibility table. It is a record of what changed, not a promise that old spellings are accepted.

## User Stories

1. As a researcher, I want the same `run.json` keys on every machine, so that I can compare runs without translating their schema.
2. As a researcher, I want status, tiling, morphotype and size-source values in English, so that I can filter runs with one script.
3. As a researcher, I want calibration origins named `plate`, `index`, `manual` and `none`, so that I can tell where a scale came from without a dictionary.
4. As a researcher, I want `mean`, `median`, `variation`, `origins` and `warnings` in the calibration summary, so that statistics read the same in every locale.
5. As a reviewer, I want persisted warnings as stable English codes, so that I can group and count them.
6. As a reviewer, I want a warning sentence chosen at display time, so that I can read it in my own language without changing the artifact.
7. As a reviewer, I want numeric warning details in structured fields, so that a rounded percentage is not buried inside prose.
8. As a script author, I want `detections.csv` headers that never change with the interface language, so that my parser keeps working.
9. As a script author, I want English values in `morphotype` and `method_length`, so that I do not special-case two languages.
10. As a script author, I want `schema_version: 2` in every new run, so that I can tell at once whether a file uses the English contract.
11. As a script author, I want an actionable error for a version-1 run, so that I know the file is old rather than corrupt.
12. As a script author, I want no silent fallback to a legacy reader, so that I never analyse an old file by mistake.
13. As a lab technician, I want English headers in my calibration index CSV, so that my measurements use the same vocabulary as the run.
14. As a lab technician, I want the CLI to accept only the canonical index format, so that a stale Spanish CSV cannot silently set the wrong scale.
15. As a GUI user, I want Viewer exports named in English, so that I can hand them to a colleague without explaining the folder name.
16. As a GUI user, I want the Viewer export to refuse a run folder, so that it cannot overwrite a canonical prediction table.
17. As a GUI user, I want report section ids in English, so that a link into a report reads the same as the section title.
18. As a reviewer, I want stable HTML anchors in new reports, so that a link I copy still names the right section.
19. As a translator, I want Spanish only in the catalog, so that I never have to translate a machine token.
20. As a translator, I want hardcoded Spanish prose replaced by gettext calls, so that every sentence I can change appears in the template.
21. As an observability user, I want English `data` keys in JSONL logs, so that I can grep an event across modules.
22. As an observability user, I want `verb.object` event names with English values, so that alerts match on one spelling.
23. As a trainer, I want English progress keys in the event stream, so that a dashboard does not branch on language.
24. As a launcher user, I want English module and system keys, so that startup failures are searchable.
25. As an updater user, I want English reason tokens, so that a skipped update has one stable cause name.
26. As a report author, I want English figure folder names, so that an archived report has the same layout on every machine.
27. As a package consumer, I want the public `Morphotype` and `Method` values in English, so that my own code uses the same enum values as the artifacts.
28. As a maintainer, I want one replacement list grouped by artifact class, so that I can find the decision for any token.
29. As a maintainer, I want a write site beside each token, so that I can change the producer without guessing where it lives.
30. As a maintainer, I want an old-token index, so that a token found in a bug report leads to its canonical name.
31. As a maintainer, I want externally owned tokens in their own section, so that nobody renames a third-party schema by mistake.
32. As a maintainer, I want the `summary.json` collision called out, so that a new export is never mistaken for a legacy run.
33. As a maintainer, I want tests at public seams, so that a refactor inside a module does not break the contract.
34. As a maintainer, I want no dual-read path, so that the English contract is the only contract left in the code.
35. As a release manager, I want exact release-note wording in the spec, so that the CHANGELOG entry says the same thing as the code.
36. As a release manager, I want the ADR supersessions named, so that a reader is not told two opposite things.
37. As a user with old runs, I want my files left untouched, so that an upgrade does not rewrite or delete my data.
38. As a user with old runs, I want an honest failure message, so that I know to open those runs with an older version.
39. As a new contributor, I want the glossary to match the decision, so that I do not learn a rule that was already reversed.
40. As a reviewer of this spec, I want every decision traceable to a wayfinder ticket, so that I can see who chose it.

## Implementation Decisions

### Scope boundary

The boundary is the machine token, not the writer. Any structured, machine-readable artifact carrying a token read or written across a disk boundary is in scope. The four classes are **canonical run artifacts**, **cross-cutting logs**, **tool outputs** and **legacy inputs**, using the definitions in [GLOSSARY.md](../../GLOSSARY.md). Pixels and gettext-wrapped prose are not artifacts. Purely in-memory state and CLI stdout are out of scope. Ultralytics-owned tokens are catalogued but not changed.

### Replacement list

Old tokens are not read by new code. Each table has four columns: old token, canonical token, write site (module and symbol, no line numbers), and a note for exceptions. Rows are alphabetical by old token. Legacy-input rows say `removed — not read by new code` instead of giving a canonical token.

#### Canonical run artifacts

| Old token | Canonical token | Write site | Note |
|---|---|---|---|
| `avisos` | `warnings` | `core.measure.calibration.summary_batch` | warning codes, not prose |
| `Feret maximo` | `maximum_feret` | `core.measure.morphology_constants.Method` | wheel-visible enum value; needs the version bump |
| `geodesico` | `geodesic` | `core.measure.morphology_constants.Method` | wheel-visible enum value |
| `imagen` | `image` | `gui.viewer.window` export summary | export JSON key |
| `indeterminado` | `indeterminate` | `core.measure.morphology_constants.Morphotype` | wheel-visible enum value |
| `media` | `mean` | `core.measure.calibration.summary_batch` | `std`, `n`, `min` and `max` already English |
| `mediana` | `median` | `core.measure.calibration.summary_batch` | — |
| `metodos` | `methods` | `gui.report.registry.SECTIONS` | section id and HTML anchor |
| `n_detecciones` | `n_detections` | `gui.viewer.window` export summary | — |
| `ninguna` | `none` | `core.measure.calibration` origin constants | reuses the English calibration mode word |
| `origenes` | `origins` | `core.measure.calibration.summary_batch` | — |
| `placa` | `plate` | `core.measure.calibration` origin constants | origin values are compared as literals in the report |
| `por_clase` | `by_class` | `gui.viewer.window` export summary | — |
| `recorte` header | `image` header (canonical index) | `core.measure.calibration.read_calibration_index` | legacy header rejected; see legacy inputs |
| `resumen` | `summary` | `gui.report.registry.SECTIONS` | section id, preset key and anchor |
| `sucio` | boolean `dirty` | `core.provenance.commit_git` | `true`/`false`, not `si`/`no` |
| `um_por_px` | `um_per_px` | `core.measure.calibration.Calibration` | matches the CSV and JSONL spelling |
| `variacion` | `variation` | `core.measure.calibration.summary_batch` | — |
| `versiones` | `versions` | `core.provenance.versions` | inner key `so` becomes `os` |
| Spanish warning prose | English warning codes | `core.measure.calibration` and `core.measure.shape` | codes plus structured parameters |
| Spanish `Calibration.description()` prose | gettext English sentences | `core.measure.calibration.Calibration.description` | not a machine token |
| Spanish section ids | English section ids | `gui.report.registry.SECTIONS`, `gui.report.render.SECTION_BUILDERS` | `summary`, `methods`, `calibration`, `shape`, `size_by_folder`, `depth`, `cards`, `results`, `models`, `errors`, `comparison`, `gallery`, `counts`, `references` |
| Spanish preset keys | `complete`, `summary`, `methodological` | `gui.report.registry.PRESETS` | values are the new section ids |

#### Cross-cutting logs

| Old token | Canonical token | Write site | Note |
|---|---|---|---|
| `caja` | `box` | `gui.trainer.runner` | — |
| `epoca` | `epoch` | `gui.trainer.runner` | — |
| `familias` | `families` | `gui.trainer.runner` | — |
| `familias_completadas` | `families_completed` | `gui.trainer.runner` | — |
| `imgsz_usados` | `imgsz_used` | `gui.detector.runner` | already used in the model-execution payload |
| `medir_forma` | `measure_shape` | `gui.detector.runner` | matches `RunParams.measure_shape` |
| `modelo` | `model` | `gui.detector.runner`, `gui.trainer.runner` | — |
| `modelos` | `models` | `gui.detector.runner` | — |
| `modulo` | `module` | `gui.launcher` | — |
| `optimizador` | `optimizer` | `gui.trainer.runner` | — |
| `placas` | `plates` | `gui.detector.runner` | — |
| `remoto` | `remote` | `gui.updates` | — |
| `results_2` | `n_results` | `gui.report.render` | — |
| `sin_forma` | `shapeless` | `gui.detector.runner` | matches the `run.shapeless_high` event |
| `sin_respuesta_remota` | `no_remote_response` | `gui.updates` | reason value |
| `sin_version_instalada` | `no_installed_version` | `gui.updates` | reason value |
| `sistema` | `system` | `gui.launcher` | — |
| `solo_images` | `images_only` | `gui.report.render` | — |
| `um_por_px` | `um_per_px` | `gui.detector.runner` | — |

#### Tool outputs

| Old token | Canonical token | Write site | Note |
|---|---|---|---|
| `<stem>_anotada.png` | `<stem>_annotated.png` | `gui.viewer.window` export action | — |
| `ambas_superpuestas` | `both_overlay` | `gui.detector.pages.report` figure builder | name sits inside a gettext msgid; re-extract the catalog |
| `conteo_manual` | `manual_count` | `gui.detector.pages.report` figure builder | gettext msgid |
| `dataset_solo_real.yaml` | `dataset_real_only.yaml` | `gui.trainer.dominio` | — |
| `deteccion_modelo` | `model_detection` | `gui.detector.pages.report` figure builder | gettext msgid |
| `detecciones.csv` | `detections.csv` | `gui.viewer.window` export action, `gui.detector.pages.results` | export must refuse a folder containing `run.json` |
| `resumen.json` | `viewer_summary.json` | `gui.viewer.window` export action | not `summary.json`: the legacy reader uses that name |
| `visor_` prefix | `viewer_` prefix | `gui.viewer.window` export action | — |

#### Legacy inputs

| Old token | Canonical token | Write site | Note |
|---|---|---|---|
| `centroids.csv` | removed — not read by new code | `core.detection_run.legacy_reader` | pre-canonical run |
| `indice.csv`, `calibracion_placas.csv` | rejected by filename | external lab scripts | use `calibration_index.csv` instead; the reader never searches for a file |
| `recorte`, `placa`, `um_por_px` CSV headers | rejected by header | external lab scripts | use the canonical headers `image,um_per_px` |
| `si`, `sí` boolean cells | removed — not read by new code | `core.detection_run.schema._bool` | only `true`/`false` remain |
| `siempre`, `nunca` | removed — not read by new code | `core.detect.tiling.TilingMode.parse` | CLI hidden aliases removed |
| `summary.json` | removed — not read by new code | `core.detection_run.legacy_reader`, `core.report_simple` | — |
| `tema`, `animaciones`, `idioma` dotfiles | removed — not read by new code | `core.preferences._migrate` | — |
| legacy report anchors `abstract`, `methods`, `calib`, `results`, `models`, `errors`, `compare`, `gallery`, `refs` | removed — not emitted by new code | `gui.report.registry.anchor` | new reports use the English section ids as anchors |
| old GUI per-model `labels/*.txt` | removed — not read by new code | `core.annotate` legacy path | — |

**Canonical per-image index.** The one accepted index is the English file
`calibration_index.csv` with the headers `image,um_per_px` (positive numbers),
keyed by image basename and passed explicitly as
`--calibrate index --calibration-index PATH`. It supplies one µm/px per photo,
so crops without a visible dish keep their own scale instead of sharing a
batch-wide fallback; an image with no row is recorded with
`image_not_in_index` and no scale. Because the basename is the lookup key, an
index-calibrated batch must give every image a unique basename: the engine
(`core.detection_run.engine._reject_duplicate_image_names`) refuses two
same-named images in different folders before inference rather than letting one
row silently scale both. The strict reader
(`core.measure.calibration.read_calibration_index`) rejects the legacy names
and headers with an error that names the canonical format, rejects duplicates
and invalid values, and neither searches for nor rewrites the user's file.

#### Externally owned tokens — catalogue only

Ultralytics writes trainer `results.csv` headers and evaluation `metrics.json` keys such as `mAP50`, `mAP50-95`, `per_class`, `precision`, `recall` and `AP50`. These are outside our control. They stay as they are and get no replacement recommendation.

### Warning codes

Persisted warnings become stable English codes with structured parameters. Proposed codes: `image_unreadable`, `nominal_diameter_invalid`, `plate_not_found`, `ring_fit_failed`, `zero_adjusted_radius`, `plate_measure_failed`, `image_not_in_index` (a photo absent from the per-image index), and the matching shape-warning codes. Numbers and nested causes move to fields such as `detour_pct` and `cause`; no interpolated sentence is stored. The sentence a person reads is an English gettext message chosen at display time. The existing refusal-code pattern in calibration is the precedent.

### Hard cut

All legacy readers are removed. There is no deprecation period, no warning event, no report banner, no silent exception, no dual-read and no conversion tool. Old runs and archived reports stay on disk; new code cannot open them, and opening them needs an older Poly-X version. External lab scripts must switch to English headers in their index CSVs. The accepted cost is that saved scientific records from older versions become unreadable by new code.

This reverses [ADR-0001](../adr/0001-run-data-location.md), [ADR-0002](../adr/0002-one-detection-run-engine.md) and [ADR-0005](../adr/0005-report-package-is-internal.md); the implementation must supersede them. [ADR-0004](../adr/0004-locale-invariant-artifacts.md) stays in force.

### Schema version

`SCHEMA_VERSION` becomes 2. `load_run` keeps its strict equality check and rejects every other value. The error says old runs are not supported and points to this spec. The version marker stays only in `run.json`; logs, CSV headers, report HTML and calibration carry no version.

### Before and after

Before, in a run written by the old code:

```json
{"schema_version": 1, "provenance": {"sucio": "si", "versiones": {...}}, "calibrations": [{"origin": "placa", "um_por_px": 0.4}]}
```

After, in a run written by the new code:

```json
{"schema_version": 2, "provenance": {"dirty": true, "versions": {...}}, "calibrations": [{"origin": "plate", "um_per_px": 0.4}]}
```

### Old-token index

| Old token | Canonical token |
|---|---|
| `ambas_superpuestas` | `both_overlay` |
| `animaciones` | removed |
| `avisos` | `warnings` |
| `caja` | `box` |
| `calibracion` | `calibration` |
| `calibracion_placas.csv` | removed |
| `centroids.csv` | removed |
| `comparacion` | `comparison` |
| `completo` | `complete` |
| `conteo` | `counts` |
| `conteo_manual` | `manual_count` |
| `dataset_solo_real.yaml` | `dataset_real_only.yaml` |
| `deteccion_modelo` | `model_detection` |
| `detecciones.csv` | `detections.csv` |
| `epoca` | `epoch` |
| `errores` | `errors` |
| `familias` | `families` |
| `familias_completadas` | `families_completed` |
| `fibra` | `fibre` |
| `fichas` | `cards` |
| `forma` | `shape` |
| `fragmento` | `fragment` |
| `galeria` | `gallery` |
| `geodesico` | `geodesic` |
| `idioma` | removed |
| `imgsz_usados` | `imgsz_used` |
| `imagen` | `image` |
| `indeterminado` | `indeterminate` |
| `indice` | `index` |
| `indice.csv` | removed |
| `mediana` | `median` |
| `medir_forma` | `measure_shape` |
| `media` | `mean` |
| `metodologico` | `methodological` |
| `metodos` | `methods` |
| `modelo` | `model` |
| `modelos` | `models` |
| `modulo` | `module` |
| `ninguna` | `none` |
| `n_detecciones` | `n_detections` |
| `optimizador` | `optimizer` |
| `origenes` | `origins` |
| `placa` | `plate` |
| `placas` | `plates` |
| `por_clase` | `by_class` |
| `profundidad` | `depth` |
| `recorte` | removed |
| `referencias` | `references` |
| `remoto` | `remote` |
| `resumen` | `summary` |
| `resumen.json` | `viewer_summary.json` |
| `resultados` | `results` |
| `results_2` | `n_results` |
| `si`, `sí` | removed |
| `sistema` | `system` |
| `sin_forma` | `shapeless` |
| `sin_respuesta_remota` | `no_remote_response` |
| `sin_version_instalada` | `no_installed_version` |
| `so` | `os` |
| `solo_images` | `images_only` |
| `sucio` | `dirty` |
| `siempre`, `nunca` | removed |
| `summary.json` | removed |
| `talla_carpetas` | `size_by_folder` |
| `tema` | removed |
| `um_por_px` | `um_per_px` |
| `versiones` | `versions` |
| `visor_` | `viewer_` |
| `_anotada.png` | `_annotated.png` |
| Spanish warning prose | English warning codes |
| Spanish section ids and anchors | English section ids |

### Migration hazards the implementation must handle

- The report compares calibration origin values as literals; changing `placa` to `plate` changes those branches.
- `Morphotype` and `Method` values are exported from the package root; they are a wheel-visible contract.
- Tests assert JSONL `data` keys by literal name.
- Report figure folder names sit inside gettext msgids; the catalog must be re-extracted after the rename.
- `summary.json` marks a pre-canonical run; the Viewer export must never adopt that name.
- The version gate has no migration; a version-1 file fails immediately by design.

### Release note wording

The CHANGELOG entry goes under `## [Unreleased]` → `### Changed`, in one bold-lead bullet following the existing style. The exact wording is:

> - **English-only run artifacts; legacy readers removed** (ADR-0004; supersedes ADR-0001, ADR-0002 and ADR-0005). All machine tokens in `run.json`, `detections.csv`, logs, Viewer exports and report ids are now English: `sucio` → `dirty`, `placa` → `plate`, `visor_` → `viewer_`, `resumen` → `summary`. New files carry `schema_version: 2`; `load_run` rejects version 1 and pre-canonical folders (`summary.json` + `centroids.csv`) are no longer read. Old runs and archived reports stay on disk but cannot be opened by this version. Scripts must switch to the English tokens; there is no compatibility option. The full replacement list is in `docs/specs/run-artifact-vocabulary.md`.

## Testing Decisions

A good test observes external behaviour only: the bytes written to an artifact, the keys read back, the HTML emitted, or the event logged. It does not assert a private helper, an internal constant name or a line number. Existing contract tests are the prior art: the run-data round-trip tests, the detection-table contract tests, the report surface and section tests, the observability tests and the Viewer export tests.

Three seams are in scope, chosen because they already exist and sit as high as possible:

1. **Artifact I/O** through `load_run` and the run writer: assert English `run.json` and `detections.csv` tokens, `schema_version: 2`, and the error for a version-1 file or a pre-canonical folder.
2. **Report generation** through `generate_report(sections=...)`: assert English section ids, anchors, preset keys and figure folder names in the emitted HTML.
3. **Structured logs and exports** through `observability.event(...)` and the Viewer export action: assert English JSONL `data` keys, English export filenames and keys, and that an export into a folder containing `run.json` is refused.

No new seam is introduced for a private module. Warning-code rendering is tested at the report seam, not by calling a formatter directly.

## Out of Scope

- Implementing the migration in code. This spec hands decisions to a later effort.
- Keeping legacy readers, deprecation warnings, dual-read, conversion tools or migration on open.
- Ultralytics' `results.csv` headers and `metrics.json` keys: externally owned, catalogue only.
- Triptych PNG pixels and any other image content: no machine vocabulary.
- Purely in-memory state and CLI stdout: neither reaches a file.
- The sanctioned Spanish gettext catalog under `src/polyx/locales/`: the intended home for translated prose.
- OS code signing, distribution integrity and the other wayfinder maps.

## Further Notes

- The wayfinder map is [Map: run-artifact machine vocabulary and legacy Spanish read points](https://github.com/Freston1605/Poly-X-Microplastics/issues/174); all seven child tickets are closed.
- `GLOSSARY.md` already carries **machine-readable artifact**, the four artifact classes, **warning code**, and the revised **Canonical run folder** and **Legacy input** definitions.
- The implementation must supersede ADR-0001, ADR-0002 and ADR-0005 in the same change that makes the hard cut true; documentation may not lag the behaviour.
- The project gate is `uv run --locked --extra gui python tools/check.py`, or the project `.venv` interpreter when the pinned uv is unavailable.
- Branch from `origin/main`; every `gh` command carries `-R Freston1605/Poly-X-Microplastics`.
