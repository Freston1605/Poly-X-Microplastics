# ADR-0005: The report package is internal

- **Status:** Accepted
- **Date:** 2026-11-08
- **Landed in:** #88

## Context

The HTML report package exported 76 names, about sixty of them private
(`_section_*`, `_fig_*`, `_batch_metrics`, …). A contract test pinned that
list by hand, so the package's internal wiring was treated as API: renaming a
helper inside a section module failed a test that simulated a consumer nobody
was. The assembler also passed its batch precompute as an 11-slot positional
tuple and split a 2-tuple coming out of the shape section by convention. A
field reorder was silent until the report rendered wrong, and adding a section
meant editing a hand-written name list.

The real seam was already small: `generate_report(run, output_path, …)` plus
the section catalog (`SECTIONS`, `IDS_SECTIONS`, `PRESETS`, `anchor`). Keeping
everything else on the surface made internal change expensive for no
user-visible reason.

## Decision

`polyx.gui.report` is an internal package. Its public surface is exactly the
entry point and the section catalog:

- `generate_report` — the call the GUI report page, `polyx-detect
  --full-report` and `polyx-report --full-report` render through.
- `SECTIONS`, `IDS_SECTIONS`, `PRESETS` and `anchor` — the catalog the report
  page and the section tests read.

The section builders, figures, precomputes and helpers are implementation and
live in their owning modules. `_generate_report_impl` stays importable from
the assembler for the partial-render test, but it is not exported.

The named inputs replace positional plumbing:

- `ReportContext` carries the run, the two precomputes, the shape bodies and
  the render caps, so a builder reads a named field instead of a tuple slot.
- `BatchMetrics` and `LotForms` are frozen dataclasses computed once per
  report; a misspelled field fails loudly instead of silently reading the
  wrong slot. `ShapeBodies` names the two bodies the shape pass produces, so
  only the section that owns them knows how they split.

Every catalog id has one `build_<id>(ctx) -> str` builder; the assembler owns
the id-to-builder mapping, and section order stays in `SECTIONS`. A new
section is one catalog entry plus one builder.

`polyx.gui.report_html` stays as a five-name compatibility alias whose
`__file__` points at the package, so existing importers do not change.

Section ids stay Spanish (`resumen`, `metodos`, …) as stable, locale-invariant
keys and HTML anchors. They are identifiers, not user-facing strings: the
visible titles are English `N_()` message ids resolved at render time.

## Considered options

- **Keep the 76-name surface.** Rejected: it pins private helpers as API,
  makes a rename fail a contract test, and forces a hand-maintained name list
  to change every time a section is added. The list protected wiring, not
  consumers.
- **Delete the compatibility shim.** Rejected: the GUI page, the CLI and the
  audit tool import `polyx.gui.report_html`. Deleting it would turn an
  internal refactor into a consumer-visible break for no benefit; the shim
  costs one module and keeps the five-name contract.
- **Move the package to `polyx.core`.** Rejected as out of scope: it is a
  separate decision about Qt-free importability, not a surface refactor, and
  it would widen the change far beyond the contract this ADR records.

## Consequences

- Internal renames are free: no test fails because a private helper moved or
  changed its name.
- A new report section is a catalog entry plus a builder; the builder-coverage
  test (`set(SECTION_BUILDERS) == set(IDS_SECTIONS)`) fails if only one half
  is added.
- Tests import submodules for the internal diagnostic seam
  (`ReportContext` + `build_<id>`); the public seam stays `generate_report`.
- The frozen benign baseline remains the behavioural contract, so the
  refactor cannot quietly change output. The render is byte-identical and the
  baseline is not recaptured.
- No message id changes: section titles stay English `N_()` ids and no
  translation catalog is edited.
- The batch precompute stays pure and Qt-free, and the calibration import
  stays lazy, so importing the report does not pull the scientific stack.

Related: #83 (card), #84 (batch precompute), #85 (builders and context), #86
(public surface), #87 (this record).
