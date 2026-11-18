# Findings: run-artifact machine vocabulary and legacy Spanish

Throwaway research branch for the wayfinder map
[Map: run-artifact machine vocabulary and legacy Spanish read points](https://github.com/Freston1605/Poly-X-Microplastics/issues/174).
Captured for tickets #175 (inventory) and #176 (legacy Spanish reads/writes). Never merged.

All paths are relative to the repo root unless stated otherwise.

## 1. Inventory: English machine vocabulary emitted by artifacts

### Canonical detection-run folder

Owner: `RunWriter`, `src/polyx/core/detection_run/writer.py`. Location:
`<user data dir>/runs/detect_YYYYMMDD_HHMMSS/`.

| File | Written at | Notes |
|---|---|---|
| `run.json` | `writer.py:72-80, 111, 148, 157`; failure paths `engine.py:334,390,410` | atomic via `.run.json.tmp` (`:78`); payload built in `schema.py:464-492` |
| `detections.csv` | `writer.py:49, 65-70, 108-110, 152-156` | `DictWriter` over `DETECTION_COLUMNS` |
| `<alias>/` per model | `writer.py:60-64, 92-94, 136-137` | folder key = stable model alias |
| `<alias>/classes.txt` | `writer.py:63-64` → `labels.py:535-546` | one per model |
| `<alias>/labels/{index:04d}_{stem}.txt` | `writer.py:100-104` | prediction labels, 6th column = confidence |
| `<alias>/{index:04d}_{stem}_{annot,pred,gt}.png` | `writer.py:139-144`; `annotate.py:71-98` | triptych views |
| `report.html` | `report_simple.py:52,108`; `cli/detect.py:324,358`; `cli/report.py:56` | after the run |

Legacy, read-only: `summary.json`, `centroids.csv`, flat `images/` (`legacy_reader.py:31-33, 90-188`); old GUI `labels/*.txt` (`annotate.py:101-148`). New runs must not write `summary.json`, `metrics.json`, `resumen.json` or flat `images/` (CLAUDE.md 538-541; ADR-0002:85-87).

### `run.json` top-level keys

`src/polyx/core/detection_run/schema.py:477-492`:
`schema_version` (value `1`), `status`, `started_at`, `finished_at`, `duration_s`,
`provenance`, `params`, `models`, `metrics`, `calibrations`,
`calibration_summary`, `images`, `error`, `skipped_images`.

- Status values (`result.py:24-42`): `running`, `completed`, `aborted`, `failed`.
- Params keys (`result.py:54-86`): `conf`, `iou_nms`, `iou_tp`, `imgsz`, `device`,
  `size_min_um`, `size_max_um`, `conf_assignment`, `measure_shape`, `tiling`,
  `tiling_threshold_px`, `tiling_tile`, `tiling_overlap`.
- Tiling values (`tiling.py:33-35`): `auto`, `always`, `never`; legacy `siempre`, `nunca` (§2).
- Calibration policy modes (`config.py:19`): `auto`, `manual`, `index`, `none`.
- Model payload (`schema.py:387-407`): `index`, `alias`, `alias_generated`, `path`,
  `model_type` (`"yolo"`), `names`, `sha256`,
  `execution{images,detections,tiled_images,imgsz_used,memory_fallback,shapeless}`, `metrics`.
  Generated alias token `model_<n>` (`identity.py:44-46`).
- Per-image payload (`schema.py:450-460`): `index`, `path`, `name`, `has_gt`, `skipped`, `gt`, `models`;
  per-model row (`:428-448`): `model`, `has_gt`, `predictions`, `tp`, `fp`, `fn`, `miscls`,
  `match_iou_thr`, `match_use_mask_iou`, `tiling_plan`, `tiling`.
- `tiling_plan` (`schema.py:212-218`): `tile`, `overlap`, `n_tiles`, `side`, `threshold_px`.
- Detection / GT object (`schema.py:168-193`): `class_id`, `class`, `conf`, `x1`, `y1`, `x2`, `y2`,
  `diam_um`, `area_um2`, `length_um`, `width_um`, `aspect`, `curvature`, `morphotype`,
  `feret_um`, `geodesic_um`, `equivalent_rect_length_um`, `method_length`, `size_source`,
  `number`, `assignable`, `review`, `warning_shape`, `has_mask`.
- Aggregate metrics (`schema.py:372-380`): `images`, `images_with_gt`, `detections`, `tp`, `fp`, `fn`, `miscls`.
- Calibration entry (`schema.py:473-476`; `calibration.py:85-100,153-158`): `image`, `um_por_px`,
  `origin`, `cx`, `cy`, `radius_px`, `n_points`, `detour_hough_pct`, `diameter_mm`,
  `factor_parallax`, `warning`, `valid`.
- Other enum values: `size_source` ∈ `mask`/`box`; `morphotype` ∈ `fibre`/`fragment`;
  classes `PET`/`PP`/`LDPE` with ids `0/1/2`; boolean cells accept `1/true/yes/y/si/sí` (`schema.py:259-269`).

### CSV headers

- `DETECTION_COLUMNS` (26) — `schema.py:29-56`: `model, image, number, class_id, class, conf,
  x1, y1, x2, y2, um_per_px, calibration_origin, length_um, method_length, feret_um,
  geodesic_um, equivalent_rect_length_um, width_um, area_um2, diam_um, aspect, curvature,
  morphotype, size_source, has_mask, assignable`.
- `PAGE_COLUMNS` — `schema.py:62-84`, used by `results.py:590-607`.
- `VIEWER_COLUMNS` (10) — `schema.py:90-101`: `class, conf, x1, y1, x2, y2, diam_px, diam_um,
  area_um2, size_source`.

### JSONL logs

`src/polyx/core/provenance/observability.py`. Envelope (`:88-112`): `ts`, `level`, `module`,
`pid`, `thread`, `event`, `run_id`, `ms`, `data`, `exc`. Events are `verb.object` (`:20-21`):
`module.launch`, `model.loaded`, `model.error`, `run.config`, `run.finish`, `run.error`,
`run.shapeless_high`, `train.progress`, `train.aborted`, `train.error`,
`update.skipped/error/available/ok`, `tiling.unknown_mode`, `image.undecodable`,
`log.format_error`, plus `*.start/*.finish/*.error` (`:335,344,353`).
Log files: `~/.polyx/logs/<module>.jsonl` (`observability.py:70-72,209-210`).

### Other run-adjacent artifacts

- Trainer: `<run>/weights/best.pt`, `results.csv`, `confusion_matrix.png` (`training.py:86-87`).
- Evaluation: `metrics.json` (`evaluation.py:100-104`) with `mAP50`, `mAP50-95`, `per_class`,
  `names`, per-class `precision`/`recall`/`AP50` (`evaluation.py:38-69`).
- Preferences: `~/.polyx_preferences.json` (`preferences.py:21,40-44`).
- Viewer export: `visor_<stem>_<timestamp>/` (`viewer/window.py:1425-1429`) with
  `<stem>_anotada.png`, `detecciones.csv`, `resumen.json` (`:1466-1505`).
- HTML report section ids/presets/legacy anchors: `gui/report/registry.py:16-75`.

## 2. Legacy Spanish: read sites and Spanish tokens written into artifacts

### Read sites (genuine Spanish parsed from disk)

1. `TilingMode.parse` reads `siempre` / `nunca` from saved `run.json`, old scripts and old callers —
   `src/polyx/core/detect/tiling.py:37-56`, legacy map `_LEGACY_MODES` at `:61-64`,
   `TILING_VALUES` at `:68-71`. CLI hidden aliases `cli/detect.py:222-234`.
   Documented in GLOSSARY.md:80-90 and CLAUDE.md 763-770.
2. `schema._bool` reads `si` / `sí` from CSV cells — `schema.py:259-269`.
3. Preferences migration reads Spanish keys from legacy dotfiles — `preferences.py:23-28`
   (`tema→theme`, `animaciones→animations`, `idioma→locale`); reads
   `~/.polyx_tema.json`, `~/.polyx_theme.json`, `~/.polyx_idioma.json` at `:78-91`.
4. `calibration.carry_index` reads Spanish column headers from CSV data files —
   `calibration.py:475-504`: `row.get("recorte") or row.get("placa")` (`:495`),
   `row.get("um_por_px")` (`:497`); file names `indice.csv`, `calibracion_placas.csv` (`:507`);
   discovery `search_index` (`:510-528`). Values feed `origin=ORIGIN_INDEX` written into artifacts.
5. Legacy run readers read older (English-named) headless files with an older schema —
   `legacy_reader.py:90-188`; compact-report fallback `report_simple.py:46-50`;
   old GUI label parsing `annotate.py:101-148`.

### Spanish tokens written into artifacts (tension with ADR-0004)

ADR-0004 and CLAUDE.md 455-462 require English, locale-invariant machine text. Violations:

1. Provenance block of every `run.json` — `src/polyx/core/provenance/provenance.py`:
   key `sucio` (`:75`, `:96`), key `versiones` (`:102`), values `si`/`no` (`:96`).
   Embedded via `schema.py:483`, `writer.py:39-48`.
2. Calibration origin tokens written to `detections.csv.calibration_origin`,
   `run.json.calibrations[].origin`, `calibration_summary.origenes` — `calibration.py:78-81`:
   `ORIGIN_PLATE = "placa"`, `ORIGIN_INDEX = "indice"`, `ORIGIN_MANUAL = "manual"`,
   `ORIGIN_NONE = "ninguna"`.
3. `run.json.calibration_summary` keys — `summary_batch`, `calibration.py:628-654`:
   `n`, `min`, `max`, `media`, `std`, `mediana`, `variacion`, `origenes`, `avisos` (`:636-653`).
4. Spanish warning strings persisted in artifacts (`Calibration.warning` →
   `run.json.calibrations[].warning`, `calibration_summary.avisos`) — `calibration.py:428,430,435,444,451,454,565,575`,
   e.g. `"imagen vacia o ilegible"`, `"diametro nominal invalido"`,
   `"no se encontro la placa en la foto"`, `"radio ajustado nulo"`,
   `"no se pudo leer la imagen"`. English `"the image could not be read"` at `engine.py:439` is inconsistent.
5. Hardcoded Spanish user-facing prose (not gettext-wrapped) — `Calibration.description()`,
   `calibration.py:106-118`: `"sin calibrar"`, `"placa de ... puntos de borde"`,
   `"heredada del recorte"`, `"introducida a mano"`.
6. Viewer export filenames and JSON keys — `viewer/window.py:1427` (`visor_<stem>_<ts>`),
   `:1466` (`<stem>_anotada.png`), `:1477` (`detecciones.csv`), `:1503` (`resumen.json`),
   keys `imagen`, `timestamp`, `um_per_px`, `n_detecciones`, `por_clase`, `n` (`:1490-1499`).
   Documented in CLAUDE.md 681-689 (headers are correctly English via `VIEWER_COLUMNS`).
7. Results-page CSV default filename `detecciones.csv` — `detector/pages/results.py:609-613`.
8. Spanish JSONL data keys / values against the English event contract:
   - `detector/runner.py:149-158`: `placas`, `imgsz_usados`, `sin_forma`; `:161-167`: `modelo`,
     `sin_forma`; `:177-188`: `placas`, `modelos`, `medir_forma`, `um_por_px`.
   - `trainer/runner.py:230-239`: `epoca`, `total`, `caja` (alongside `map50`/`precision`/`recall`).
   - `launcher.py:119`: `modulo=dotted`.
   - `updates.py:78,88,93`: `reason="sin_version_instalada"`, `reason="sin_respuesta_remota"`, `remoto=...`.
9. Spanish in-memory trainer state — `trainer/runner.py:198` `run_current = {"nombre": ..., "abortado": False}`.
10. Allowlisted leftovers — `english_scan.py:135-145` allowlists `"detecciones"`; its
    `SPANISH_SUBWORDS` list (`:703-797`) contains `versiones`, `origenes`, `ninguna`,
    `siempre`, `nunca`, `morfotipo`, `indice`, `recorte`, `placa`; `quality/assignment.py:51`
    docstring parameter `detecciones`; `gui/microscope_panel.py:97,255,335` key `"clase"`.

### Sanctioned Spanish (contrast)

Gettext catalog `src/polyx/locales/es/LC_MESSAGES/messages.po`, loaded by `i18n.py:240-284`.
GLOSSARY.md:241-249 confirms `translations_en.py` was deleted and Spanish is no longer a key.

## Bottom line

- Canonical run artifacts: `run.json`, `detections.csv`, `.run.json.tmp`, per-model `classes.txt`
  and labels, triptych PNGs, `report.html`; plus JSONL logs.
- Machine vocabulary is enumerated in `schema.py:25-101, 477-492`, `result.py:24-86`,
  `config.py:19`, `tiling.py:25-71`, `calibration.py:78-100`, `provenance.py:56-104`,
  `observability.py:88-112`, `registry.py:16-75`.
- Spanish is genuinely read at `tiling.py:61-64`, `schema.py:268`, `preferences.py:23-28`,
  `calibration.py:495-507`; Spanish machine tokens are written most notably in
  `provenance.py:75-102`, `calibration.py:78-81,628-654`, and `viewer/window.py:1427-1503`.
