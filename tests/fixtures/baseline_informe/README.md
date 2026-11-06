# English report baseline — pipeline report

## Purpose

This directory holds the **English** render of the two shared report fixtures
(`tests/report_fixtures.py`) produced by
`polyx.gui.report_html.generate_report` through the real pipeline.

It is a **semantic** reference, not a byte-identity gate. `informe.html` is
read by `tests/test_report_sections.py`, which compares each section builder's
body against the corresponding fragment of the document, and by the escaping
tests, which render their own documents. No test compares a whole-file hash,
so a whitespace change or a matplotlib font change does not fail the suite;
what fails is a section that changed meaning.

The old byte-identity gate (`informe.sha256`, the normalized digest and the
`TAMANO_BASELINE` pin) was retired in ticket #73. It failed on whitespace and
said nothing about what broke.

## Provenance

- **Original capture:** `tools/audit/capture_report_fixtures.py` in M001/T03,
  against pre-fix code.
- **Recapture:** M006/S05, after the twelve raw Spanish report headings became
  English literals (`Muestra (imagen)` -> `Sample (image)`, the
  confusion-matrix headers, the comparison header and the size-by-class
  headers).
- **Recapture (PR #16, report package split):** the `<style>` block left
  `report_html.py` for `polyx/gui/report/report.css`.
- **Ticket #73:** the `.sha256` files and the whole-file comparison were
  removed; the snapshots stay as the semantic reference for the per-section
  tests.

## Language contract

The baseline is in **English**. The capture script forces `POLYX_LOCALE=en`
before importing any fixture or `polyx` module, and `tests/test_report_sections.py`
selects English through `i18n.set_language` while it compares, so neither the
capture nor the comparison depends on the machine locale or the saved
preference.

## Contents

- `benigna/informe.html` — benign render, all sections.
- `hostil/informe.html` — the same measured data with HTML injection in the
  model alias and the image file name; the escaping tests render their own
  hostile documents, so this file documents the worst case next to the clean
  one.

## Regenerating the baseline

Regenerate only with a conscious change to report output:

```sh
python tools/audit/capture_report_fixtures.py
```

There is no digest to update and no size pin to keep in sync. Review the diff
of the HTML instead: the per-section tests will tell you which section changed.
