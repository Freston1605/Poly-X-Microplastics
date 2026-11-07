# ADR-0003: English is the source language

- **Status:** Accepted
- **Date:** 2026-11-06
- **Landed in:** #67

## Context

Poly-X used Spanish sentences as translation keys. A dictionary in
`polyx/core/translations_en.py` mapped each whole Spanish sentence to its
English counterpart, so the Spanish text *was* the identity of a message. That
worked while the only two languages were Spanish and English, and it carried
costs that grew with every surface:

- A third language meant editing Python: the strings, the dictionary and the
  language list all lived in code. There was no file a translator could open
  and no way to hand work to someone who does not read the repository.
- A missing translation was invisible. Lookup silently returned the Spanish
  key, and the only signal was an audit script outside the gate.
- Plurals were hand-written conditionals and `(es)`/`(s)` parentheticals, so a
  language with different plural rules could not be represented.
- Over fifty strings were resolved at class-definition time, which froze the
  language at import and forced a restart to change it.
- The surfaces disagreed: the full report was bilingual, the compact report
  English-only, the training summary Spanish-only and the command line
  English-only.
- The codebase claimed to be English while 831 Spanish literals acted as keys,
  so the "Spanish survives only inside the translation layer" rule had to
  special-case the very strings it was checking.

The conventions already demanded English comments, docstrings and identifiers.
Keeping the source language Spanish contradicted that rule in the one place it
mattered most: the text every user reads.

## Decision

English is the source language. The English sentence is the message id, and
every user-facing string is wrapped in a gettext call:

```python
from polyx.core.i18n import _, ngettext, pgettext

button = QPushButton(_("Open folder…"))
label = ngettext("%(count)s image", "%(count)s images", count)
verb = pgettext("verb", "Close")
```

`N_()` marks a string for extraction without translating it now; `lazy()`
defers resolution to display time for module-level constants that would
otherwise freeze the language at import.

**One catalog directory per locale.** Translations live in plain gettext
catalogs under the package:

```text
src/polyx/locales/
  messages.pot                       # the committed template
  es/LC_MESSAGES/messages.po         # the Spanish translation
```

The `.po` file is the translator handoff: a standard interchange format every
CAT tool, translation service and plain editor can open. The `.mo` file is the
build artifact: it is compiled from the `.po`, gitignored, produced by the Hatch
build hook for the wheel and by the PyInstaller data list for the frozen
application, and compiled on demand into a temporary cache by a development
checkout. No hand-edited binary is reviewed.

The directory *is* the registration. Adding Portuguese is creating
`src/polyx/locales/pt/LC_MESSAGES/messages.po`, running the update command and
letting Babel supply the language's own display name; no Python changes.

**One owner for language state.** `polyx.core.i18n` exposes `_`, `ngettext`,
`pgettext`, `npgettext`, `N_`, `lazy`, `set_language`, `language`,
`available_languages`, `when_changing` and `resolve`. It uses the standard
library gettext runtime, with Babel for extraction, compilation and locale
display names. `polyx.core.i18n` stays Qt-free; `polyx.gui.i18n_qt` is the Qt
bridge.

**Resolution order.** `POLYX_LOCALE`, then the saved preference in
`~/.polyx_preferences.json` (owned by `polyx.core.preferences`), then the
system locale, then English. A regional tag falls back through its language
(`es_AR` → `es`) and then to English. An explicitly chosen language with no
catalog is a logged error, not a silent switch; a detected system locale with
no catalog falls back to English silently, because nobody chose it.

**Missing translations degrade to English.** The msgid is already the correct
English sentence, so an unfinished catalog renders English rather than a key
or a blank. The release gate is the opposite: the audit fails on any empty,
fuzzy or missing Spanish entry, on a placeholder mismatch, on an f-string or
brace-formatted translation call, and on a committed template that no longer
matches the source tree. Graceful degradation is for an unfinished language;
the audit is for a shipped one.

**Placeholders are named `%(name)s`.** Positional `{}` and f-strings inside a
translation call are rejected: `%` specifiers are the only form a catalog
validator can compare, and a literal brace in an HTML fragment would break
`str.format` at runtime.

**Live switching.** Because strings resolve when they are displayed instead of
when the module is imported, `set_language` can swap the catalog in a running
application. Qt's own catalogs move with it, and each top-level window rebuilds
its text from the new catalog.

## Considered options

- **Keep Spanish as the key and extend the dictionary.** Rejected: it is the
  status quo with a longer list of languages. A translator still cannot work
  without Python, a missing entry is still invisible, plurals still need code,
  and the source still contradicts the English convention.
- **Symbolic ids (`menu.file.open`) with separate source strings.** Rejected:
  an id hides the sentence from the person reading the code and from an agent
  navigating it, every message needs a second lookup to become text, and a
  translator sees `dialog.error.title` instead of the sentence to translate.
  The English sentence is both the readable code and the fallback, which is
  exactly the property symbolic ids spend a lookup to lose.
- **Qt's `.ts`/`.qm` catalogs.** Rejected: they only reach code that uses Qt.
  The HTML report, the compact report, the training summary and the command
  line are not Qt surfaces, and one catalog has to serve all of them. `.ts` is
  XML with poor diffs and `.qm` is a binary; neither is the interchange file a
  translator expects. Qt's own translations are still loaded alongside ours so
  native dialogs follow the interface.
- **A JSON or YAML catalog.** Rejected: no standard extraction or validation
  toolchain, no plural-rule support, and no CAT-tool interoperability. `.po`
  already is the format the ecosystem reads.
- **Translate at import and restart on a change.** Rejected: it is the
  behaviour this replaces, and it loses the user's loaded images, models and
  run for a display preference.

## Consequences

- Source code reads in English, including every user-facing string. Spanish
  exists only in `src/polyx/locales/es/LC_MESSAGES/messages.po`.
- `polyx/core/translations_en.py` is deleted, and the old `tr()` function is
  retired. Its name collided with Qt's own `tr()`, and it was the mechanism
  this ADR removes.
- Adding a language costs one catalog directory and one `update` run; no
  Python change and no second migration.
- A hardcoded English string is now a defect the static audit can find, because
  "it is already English" no longer means it is translated.
- The template `src/polyx/locales/messages.pot` is committed and must be
  regenerated whenever a translation call is added, removed or moved; the gate
  fails on a stale template.
- Numbers, dates, CSV headers and run metadata deliberately do not follow the
  catalog. That decision is recorded separately in ADR-0004.
- The extract → translate → compile → test loop is documented in
  [`CONTRIBUTING.md`](../../CONTRIBUTING.md); the vocabulary is in
  [`GLOSSARY.md`](../../GLOSSARY.md).
