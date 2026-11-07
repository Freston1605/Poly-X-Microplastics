"""Retired i18n vocabulary must not return to the agent-facing documents.

Before spec #67 these documents described the `tr()` layer, the
`translations_en` dictionary, the `POLYX_IDIOMA` override and the standalone
language file. The code moved to gettext and the prose outlived it: the stale
references cost a full documentation pass in #74. This test keeps them out.

ADRs and `GLOSSARY.md` may name a retired term to record its history, so they
are deliberately outside the scan. The documents here describe what the code
does now.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Documents an agent reads before touching the translation layer.
DOCUMENTS = (
    "CLAUDE.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "CODING_STANDARDS.md",
    "README.md",
)

#: Retired term -> the pattern that proves it came back.
RETIRED = {
    "tr()": re.compile(r"\btr\(\)"),
    "translations_en.py": re.compile(r"translations_en"),
    "POLYX_IDIOMA": re.compile(r"POLYX_IDIOMA"),
    "~/.polyx_language.json": re.compile(r"\.polyx_language\.json"),
}


def _findings(text: str) -> list[str]:
    """Retired terms present in ``text``, in dictionary order."""
    return [name for name, pattern in RETIRED.items() if pattern.search(text)]


def test_the_agent_documents_name_no_retired_i18n_term():
    for name in DOCUMENTS:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"{name} is empty; the scan would pass vacuously"
        found = _findings(text)
        assert not found, f"{name} names retired i18n term(s): {', '.join(found)}"


def test_the_scanner_catches_each_retired_term():
    assert _findings("Set POLYX_IDIOMA=en before running.") == ["POLYX_IDIOMA"]
    assert _findings("Call tr() for the label.") == ["tr()"]
    assert _findings("The translations_en dictionary is gone.") == ["translations_en.py"]
    assert _findings("Reads ~/.polyx_language.json at startup.") == ["~/.polyx_language.json"]


def test_the_scanner_does_not_fire_on_current_vocabulary():
    current = "Every label is an English msgid wrapped in `_()`; `str()` stays untouched."
    assert _findings(current) == []
