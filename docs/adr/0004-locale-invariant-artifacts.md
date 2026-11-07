# ADR-0004: Reports and run artifacts do not follow the reader's locale

- **Status:** Accepted
- **Date:** 2026-11-06
- **Landed in:** #67

## Context

Poly-X produces two kinds of text. One is read by a person in the interface and
in the words of a report; the other is read by a program: CSV column headers,
`run.json` keys, run metadata, numbers, dates and event names. The second kind
is also compared by people — two researchers opening the same run on two
machines, a reviewer checking a table against a saved file, a script parsing a
batch of runs.

Translating the first kind is the point of the i18n work. Translating the
second would make the same run mean different things depending on who opens it:

- A decimal comma in a CSV breaks the parser that split on `,`, or forces the
  separator to change and every downstream reader with it.
- `12/03/2026` is 12 March in one convention and December 3 in another. A run
  folder does not carry enough context to disambiguate it after the fact.
- A translated column header renames a schema. `detections.csv` is read by the
  CLI, the Viewer, the report and by users' own scripts; changing `class` to
  `clase` is a breaking format change disguised as a translation.
- A saved report that re-renders in a new language changes a record underneath
  the person who archived it.

The interface has the opposite requirement: a Spanish-speaking user expects
numbers and dates in the shapes their system uses while they work.

## Decision

**Locale-invariant artifacts.** The following keep the same spelling and
formatting in every locale:

- Report numbers: dot decimal separator, no digit grouping in machine-readable
  fields.
- Dates and timestamps: ISO 8601 (`YYYY-MM-DD`, `YYYY-MM-DDTHH:MM:SS`), never a
  localised month name or a day/month order that depends on the reader.
- CSV column headers: fixed English tokens, because the file is a schema.
- `run.json` keys and run metadata keys: fixed English tokens.
- Log messages and event names: fixed English tokens, so a log line can be
  grepped and an alert can match it.

The words *around* the numbers are translated at render time: headings, table
captions, figure captions, methods prose and the labels a person reads. The
HTML `lang` attribute follows the interface language so spell checkers and
screen readers treat the document correctly, but it does not change a number
or a header. A report saved to disk keeps the language it was generated in: a
language switch applies to the windows and to the next report, never to a file
that was already written.

**Locale-aware formatting is allowed only in interactive widgets.** A spin box,
a live status line or a dialog may use the user's locale for decimal separators,
digit grouping and date display. That formatting is presentation of a value
while it is being edited; it never reaches a saved artifact.

## Considered options

- **Locale-aware numbers and dates everywhere.** Rejected: it makes the same
  run render differently on two machines, breaks CSV parsers and cannot be
  reversed reliably from the file alone. Scientific comparison depends on the
  digits being the same digits.
- **Translate CSV headers and `run.json` keys with the interface.** Rejected:
  it turns a display preference into a schema change. Every consumer would need
  to know the language of the file before it could read it, and existing runs
  would become unreadable by the new code.
- **Re-render saved reports when the language changes.** Rejected: the file is
  a record, not a view. A report archived for a paper must keep the language
  and the numbers it was generated with.
- **Use the system locale for the report and keep only the interface
  configurable.** Rejected: the report would then differ between the author's
  machine and the reader's, which is the problem, not a workaround.
- **Leave the interface untranslated to keep everything invariant.** Rejected:
  it solves the artifact problem by giving up the feature the users asked for.
  The two kinds of text are separable, so they are separated.

## Consequences

- `detections.csv`, `run.json`, log lines and event names are stable contracts.
  Changing a header or a key is a schema change with its own migration, not a
  translation change.
- Report prose is translated; report numbers, dates and headers are not.
  A translator never sees a CSV header or a `run.json` key in the catalog.
- The on-screen windows follow a live language switch; a report generated
  afterwards uses the new language, and a saved report keeps the one it was
  written with.
- Interactive widgets may use locale formatting, but the value they save is the
  invariant form.
- This decision is the reason the translation audit's scope excludes CSV
  headers, run metadata keys, log messages and event names, and why those
  strings are not extracted into `messages.pot`.
