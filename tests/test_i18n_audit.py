"""The translation audit is a gate, so each failure mode has a test.

The audit exists because "the string is already English" stopped meaning
"nothing to translate": a hardcoded label, a stale template, an empty Spanish
entry and a placeholder typo all render as something the user did not ask for,
and none of them raises at runtime. These tests pin the five checks, the
gettext file reader they share, the extraction walker, and the wiring that
makes the gate run in ``tools/check.py`` and in pre-commit.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import yaml
from tools.i18n import audit, po
from tools.i18n.extract import iter_source_messages

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE_REL = "src/polyx/locales/messages.pot"
CATALOG_REL = "src/polyx/locales/es/LC_MESSAGES/messages.po"


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _header() -> str:
    return 'msgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=UTF-8\\n"\n\n'


def _pot(entries: str) -> str:
    return _header() + entries


def _catalog(entries: str) -> str:
    return _header() + entries


def _entry(msgid: str, msgstr: str, *, location: str = "src/polyx/app.py:3") -> str:
    return f'#: {location}\nmsgid "{msgid}"\nmsgstr "{msgstr}"\n\n'


# ---------------------------------------------------------------------------
# PO reader.
# ---------------------------------------------------------------------------


def test_po_reader_understands_plurals_contexts_locations_and_flags():
    text = (
        _header()
        + "#: src/polyx/app.py:3 src/polyx/other.py:9\n"
        + 'msgid "Open"\nmsgstr "Abrir"\n\n'
        + '#: src/polyx/app.py:5\nmsgctxt "verb"\nmsgid "Close"\nmsgstr "Cerrar"\n\n'
        + '#: src/polyx/app.py:7\nmsgid "%(count)d image"\nmsgid_plural "%(count)d images"\n'
        + 'msgstr[0] "%(count)d imagen"\nmsgstr[1] "%(count)d imágenes"\n\n'
        + '#, fuzzy\n#: src/polyx/app.py:9\nmsgid "Later"\nmsgstr "Luego"\n\n'
        + '#~ msgid "Gone"\n#~ msgstr "Ido"\n'
    )

    entries = po.parse_po(text)

    by_id = {entry.msgid: entry for entry in entries if entry.msgid}
    assert by_id["Open"].msgstr == "Abrir"
    assert by_id["Open"].locations == [("src/polyx/app.py", 3), ("src/polyx/other.py", 9)]
    assert by_id["Close"].context == "verb"
    assert by_id["%(count)d image"].msgid_plural == "%(count)d images"
    assert by_id["%(count)d image"].msgstr_plural == {
        0: "%(count)d imagen",
        1: "%(count)d imágenes",
    }
    assert by_id["Later"].fuzzy
    assert by_id["Gone"].obsolete


def test_po_reader_joins_multiline_strings_and_decodes_escapes():
    text = (
        _header() + 'msgid "A very long "\n"message\\nwith a tab\\there"\nmsgstr "Uno "\n"largo"\n'
    )

    entry = po.parse_po(text)[1]

    assert entry.msgid == "A very long message\nwith a tab\there"
    assert entry.msgstr == "Uno largo"


def test_po_reader_extracts_named_and_positional_placeholders():
    assert po.format_placeholders("%(count)d of %(total)d") == {"count", "total"}
    assert po.format_placeholders("%d images") == {"%d"}
    assert po.format_placeholders("100%% sure") == set()


# ---------------------------------------------------------------------------
# Extraction walker.
# ---------------------------------------------------------------------------


def test_extraction_walker_covers_every_keyword():
    source = (
        "from polyx.core.i18n import _, N_, lazy, ngettext, pgettext, npgettext\n"
        'a = _("Open")\n'
        'b = N_("Marked")\n'
        'c = lazy("Deferred")\n'
        'd = ngettext("%(n)d image", "%(n)d images", 2)\n'
        'e = pgettext("verb", "Close")\n'
        'f = npgettext("verb", "%(n)d box", "%(n)d boxes", 2)\n'
    )

    found = {
        (message.function, message.message, message.context)
        for message in iter_source_messages(source)
    }

    assert found == {
        ("_", "Open", None),
        ("N_", "Marked", None),
        ("lazy", "Deferred", None),
        ("ngettext", ("%(n)d image", "%(n)d images"), None),
        ("pgettext", "Close", "verb"),
        ("npgettext", ("%(n)d box", "%(n)d boxes"), "verb"),
    }


def test_extraction_walker_sees_calls_nested_in_f_strings():
    source = "html = f\"<h3>{_('Methods')}</h3>\"\n"

    assert [message.message for message in iter_source_messages(source)] == ["Methods"]


# ---------------------------------------------------------------------------
# Stale template.
# ---------------------------------------------------------------------------


def test_stale_template_names_missing_and_obsolete_messages(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/app.py": 'from polyx.core.i18n import _\n\nlabel = _("New string")\n',
            TEMPLATE_REL: _pot(_entry("Old string", "", location="src/polyx/app.py:1")),
        },
    )

    findings = audit.audit_template(tmp_path)

    assert {finding.kind for finding in findings} == {"stale-template"}
    detail = " ".join(finding.detail for finding in findings)
    assert "New string" in detail
    assert "Old string" in detail
    missing = next(finding for finding in findings if "New string" in finding.detail)
    assert missing.path == "src/polyx/app.py"
    assert missing.line == 3


def test_stale_template_names_location_drift(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/app.py": 'from polyx.core.i18n import _\n\n\n\nlabel = _("Open")\n',
            TEMPLATE_REL: _pot(_entry("Open", "", location="src/polyx/app.py:3")),
        },
    )

    findings = audit.audit_template(tmp_path)

    assert len(findings) == 1
    assert findings[0].kind == "stale-template"
    assert "location" in findings[0].detail
    assert "Open" in findings[0].detail


def test_missing_template_is_a_stale_template_finding(tmp_path):
    _write_tree(
        tmp_path, {"src/polyx/app.py": 'from polyx.core.i18n import _\n\nlabel = _("Open")\n'}
    )

    findings = audit.audit_template(tmp_path)

    assert findings
    assert findings[0].kind == "stale-template"
    assert "missing" in findings[0].detail


# ---------------------------------------------------------------------------
# Empty translations.
# ---------------------------------------------------------------------------


def test_empty_translation_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/app.py": 'from polyx.core.i18n import _\n\nlabel = _("Open")\n',
            TEMPLATE_REL: _pot(_entry("Open", "")),
            CATALOG_REL: _catalog(_entry("Open", "")),
        },
    )

    findings = audit.audit_catalogs(tmp_path)

    assert findings
    assert findings[0].kind == "empty-translation"
    assert findings[0].path == CATALOG_REL
    assert findings[0].line > 0
    assert "Open" in findings[0].detail


def test_translation_missing_from_the_catalog_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/app.py": 'from polyx.core.i18n import _\n\nlabel = _("Open")\n',
            TEMPLATE_REL: _pot(_entry("Open", "")),
            CATALOG_REL: _catalog(""),
        },
    )

    findings = audit.audit_catalogs(tmp_path)

    assert findings
    assert findings[0].kind == "empty-translation"
    assert "missing" in findings[0].detail
    assert "Open" in findings[0].detail


def test_a_plural_with_one_empty_form_fails_the_gate(tmp_path):
    plural = (
        '#: src/polyx/app.py:3\nmsgid "%(n)d image"\nmsgid_plural "%(n)d images"\n'
        'msgstr[0] "%(n)d imagen"\nmsgstr[1] ""\n\n'
    )
    _write_tree(
        tmp_path,
        {
            "src/polyx/app.py": (
                "from polyx.core.i18n import ngettext\n\n"
                'label = ngettext("%(n)d image", "%(n)d images", 2)\n'
            ),
            TEMPLATE_REL: _pot(plural),
            CATALOG_REL: _catalog(plural),
        },
    )

    findings = audit.audit_catalogs(tmp_path)

    assert findings
    assert findings[0].kind == "empty-translation"
    assert "plural" in findings[0].detail


# ---------------------------------------------------------------------------
# Placeholder mismatch.
# ---------------------------------------------------------------------------


def test_placeholder_mismatch_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/app.py": (
                'from polyx.core.i18n import _\n\nlabel = _("%(count)d images")\n'
            ),
            TEMPLATE_REL: _pot(_entry("%(count)d images", "")),
            CATALOG_REL: _catalog(_entry("%(count)d images", "%(total)d imágenes")),
        },
    )

    findings = audit.audit_catalogs(tmp_path)

    assert findings
    assert findings[0].kind == "placeholder-mismatch"
    assert "count" in findings[0].detail
    assert "total" in findings[0].detail


def test_positional_placeholder_in_a_translation_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/app.py": (
                'from polyx.core.i18n import _\n\nlabel = _("%(count)d images")\n'
            ),
            TEMPLATE_REL: _pot(_entry("%(count)d images", "")),
            CATALOG_REL: _catalog(_entry("%(count)d images", "%d imágenes")),
        },
    )

    findings = audit.audit_catalogs(tmp_path)

    assert findings
    assert findings[0].kind == "placeholder-mismatch"


def test_braces_in_a_translation_fail_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/app.py": (
                'from polyx.core.i18n import _\n\nlabel = _("%(count)d images")\n'
            ),
            TEMPLATE_REL: _pot(_entry("%(count)d images", "")),
            CATALOG_REL: _catalog(_entry("%(count)d images", "{count} imágenes")),
        },
    )

    findings = audit.audit_catalogs(tmp_path)

    assert findings
    assert findings[0].kind == "placeholder-mismatch"


# ---------------------------------------------------------------------------
# Dynamic translation calls.
# ---------------------------------------------------------------------------


def test_fstring_translation_call_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/page.py": (
                "from polyx.core.i18n import _\n\n\n"
                "def label(name):\n"
                '    return _(f"Hello {name}")\n'
            )
        },
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "dynamic-call"
    assert findings[0].path == "src/polyx/gui/page.py"
    assert findings[0].line == 5
    assert "f-string" in findings[0].detail


def test_positional_brace_translation_call_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/page.py": (
                "from polyx.core.i18n import _\n\n\n"
                "def label(name):\n"
                '    return _("Hello {}".format(name))\n'
            )
        },
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "dynamic-call"
    assert "format" in findings[0].detail


def test_literal_braces_in_a_translation_call_fail_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {"src/polyx/gui/page.py": 'from polyx.core.i18n import _\n\nlabel = _("Hello {}")\n'},
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "dynamic-call"
    assert "brace" in findings[0].detail


# ---------------------------------------------------------------------------
# Unwrapped strings that reach a widget.
# ---------------------------------------------------------------------------


def test_unwrapped_widget_string_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/page.py": (
                "from PySide6.QtWidgets import QLabel\n\n\n"
                "def build():\n"
                '    return QLabel("Open folder")\n'
            )
        },
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "unwrapped-string"
    assert findings[0].path == "src/polyx/gui/page.py"
    assert findings[0].line == 5
    assert "Open folder" in findings[0].detail


def test_a_wrapped_widget_string_passes(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/page.py": (
                "from PySide6.QtWidgets import QLabel\n"
                "from polyx.core.i18n import _\n\n\n"
                "def build():\n"
                '    return QLabel(_("Open folder"))\n'
            )
        },
    )

    assert audit.audit_sources(tmp_path) == []


def test_a_message_box_text_is_checked(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/page.py": (
                "from PySide6.QtWidgets import QMessageBox\n"
                "from polyx.core.i18n import _\n\n\n"
                "def warn(parent):\n"
                '    QMessageBox.warning(parent, _("Error"), "Could not open the file")\n'
            )
        },
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "unwrapped-string"
    assert "Could not open the file" in findings[0].detail


def test_an_f_string_with_an_unwrapped_tail_is_checked(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/page.py": (
                "from PySide6.QtWidgets import QLabel\n"
                "from polyx.core.i18n import _\n\n\n"
                "def build(count):\n"
                "    label = QLabel()\n"
                "    label.setText(f\"{_('Ready')} {count} items\")\n"
                "    return label\n"
            )
        },
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "unwrapped-string"
    assert "items" in findings[0].detail


def test_markup_and_punctuation_are_not_user_visible_text(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/page.py": (
                "from PySide6.QtWidgets import QLabel\n\n\n"
                "def build():\n"
                "    label = QLabel()\n"
                '    label.setText("<b>")\n'
                '    label.setText("—")\n'
                "    label.setText(\"<br><span style='font-size:9pt'>\")\n"
                "    return label\n"
            )
        },
    )

    assert audit.audit_sources(tmp_path) == []


def test_stylesheets_and_object_names_are_not_translated(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/page.py": (
                "from PySide6.QtWidgets import QLabel\n\n\n"
                "def build():\n"
                "    label = QLabel()\n"
                '    label.setStyleSheet("color: red")\n'
                '    label.setObjectName("main_label")\n'
                "    return label\n"
            )
        },
    )

    assert audit.audit_sources(tmp_path) == []


# ---------------------------------------------------------------------------
# Non-Qt surfaces: the command line and the HTML documents built by hand.
# ---------------------------------------------------------------------------


def test_cli_printed_message_without_translation_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/cli/tool.py": (
                'from polyx.core.i18n import _\n\n\ndef main():\n    print("Open folder")\n'
            )
        },
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "unwrapped-string"
    assert findings[0].path == "src/polyx/cli/tool.py"
    assert findings[0].line == 5
    assert "Open folder" in findings[0].detail


def test_cli_argument_help_without_translation_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/cli/tool.py": (
                "from argparse import ArgumentParser\n\n"
                'parser = ArgumentParser(description="Batch detect")\n'
            )
        },
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "unwrapped-string"
    assert "Batch detect" in findings[0].detail


def test_a_wrapped_cli_message_passes(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/cli/tool.py": (
                'from polyx.core.i18n import _\n\n\ndef main():\n    print(_("Open folder"))\n'
            )
        },
    )

    assert audit.audit_sources(tmp_path) == []


def test_cli_machine_contract_strings_pass(tmp_path):
    """CSV headers, metadata keys and log messages are not prose."""
    _write_tree(
        tmp_path,
        {
            "src/polyx/cli/tool.py": (
                "import logging\n\n"
                "log = logging.getLogger(__name__)\n\n\n"
                "def write(rows):\n"
                '    header = "image,class,conf"\n'
                '    key = "schema_version"\n'
                '    log.info("run started")\n'
                "    return rows\n"
            )
        },
    )

    assert audit.audit_sources(tmp_path) == []


def test_compact_report_prose_without_translation_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/core/report_simple.py": (
                'def generate():\n    return "<h1>Poly-X run report</h1>"\n'
            )
        },
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "unwrapped-string"
    assert "Poly-X run report" in findings[0].detail


def test_training_summary_prose_without_translation_fails_the_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/trainer/pages/summary.py": (
                'def build():\n    return "<h2>Referencias</h2>"\n'
            )
        },
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "unwrapped-string"
    assert "Referencias" in findings[0].detail


def test_bibliography_citations_are_not_flagged(tmp_path):
    """Citations are names and titles, not interface prose."""
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/trainer/pages/summary.py": (
                'def build():\n    return "<li>Pérez M (2024). Development of a new method.</li>"\n'
            )
        },
    )

    assert audit.audit_sources(tmp_path) == []


def test_html_machine_contract_strings_pass(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/core/report_simple.py": (
                "CSS = '*{box-sizing:border-box;}'\n"
                "HEADER = 'image,class,conf'\n"
                "PATH = 'run.json'\n"
            )
        },
    )

    assert audit.audit_sources(tmp_path) == []


def test_report_html_raw_label_fails_the_gate(tmp_path):
    """The full report builds HTML by hand; a text node is still visible text."""
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/report/sections.py": (
                "def breadcrumb(station, stretch, plate):\n"
                '    return f"<td>{station} &middot; tramo {stretch}'
                ' &middot; placa {plate}</td>"\n'
            )
        },
    )

    findings = audit.audit_sources(tmp_path)

    assert findings
    assert findings[0].kind == "unwrapped-string"
    assert findings[0].path == "src/polyx/gui/report/sections.py"
    assert findings[0].line == 2
    assert "tramo" in findings[0].detail


def test_report_html_neutral_tokens_are_not_flagged(tmp_path):
    """CSS, metric abbreviations, units and citations are not prose to translate."""
    _write_tree(
        tmp_path,
        {
            "src/polyx/gui/report/sections.py": (
                'STYLE = "<style>.kpi { color: red; }</style>"\n'
                'ROW = "<tr><th>TP</th><th>FP</th><th>FN</th><th>MISCLS</th></tr>"\n'
                'SCALE = "<p>100 mm &divide; 2 px = 50 µm/px</p>"\n'
                'REF = "<li>Pérez M (2024). Development of a new method.</li>"\n'
                'CREDIT = "<footer>© Cristofher Ferrada</footer>"\n'
            )
        },
    )

    assert audit.audit_sources(tmp_path) == []


# ---------------------------------------------------------------------------
# The real tree.
# ---------------------------------------------------------------------------


def test_the_real_tree_passes_the_audit():
    findings = audit.audit_tree(ROOT)

    assert not findings, "\n".join(str(finding) for finding in findings)


# ---------------------------------------------------------------------------
# Pseudo-locale: development only, never a gate, never shipped.
# ---------------------------------------------------------------------------


def test_the_pseudo_locale_is_ignored_by_the_catalog_gate(tmp_path):
    _write_tree(
        tmp_path,
        {
            "src/polyx/app.py": 'from polyx.core.i18n import _\n\nlabel = _("Open")\n',
            TEMPLATE_REL: _pot(_entry("Open", "")),
            CATALOG_REL: _catalog(_entry("Open", "Abrir")),
            "src/polyx/locales/qps_ploc/LC_MESSAGES/messages.po": _catalog(_entry("Open", "")),
        },
    )

    assert audit.audit_catalogs(tmp_path) == []


def test_the_pseudo_locale_cannot_be_committed_or_shipped():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "src/polyx/locales/qps_ploc/" in ignore

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excludes = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"].get("exclude", [])
    assert any("qps_ploc" in pattern for pattern in excludes)
    sdist_excludes = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"].get("exclude", [])
    assert any("qps_ploc" in pattern for pattern in sdist_excludes)


def test_the_pseudo_catalog_mangles_text_and_keeps_placeholders(tmp_path):
    from tools.i18n import pseudo

    _write_tree(tmp_path, {TEMPLATE_REL: _pot(_entry("Open %(name)s", ""))})

    output = pseudo.generate(tmp_path)

    entry = next(item for item in po.parse_po(output.read_text(encoding="utf-8")) if item.msgid)
    assert entry.msgstr != entry.msgid
    assert "%(name)s" in entry.msgstr
    assert "Open" not in entry.msgstr


# ---------------------------------------------------------------------------
# Packaging: compiled catalogs reach the wheel and the frozen application.
# ---------------------------------------------------------------------------


def test_compiled_catalogs_are_in_the_wheel(tmp_path):
    from tools.i18n import packaging

    _write_tree(tmp_path, {CATALOG_REL: _catalog(_entry("Open", "Abrir"))})

    compiled = packaging.compile_catalogs(tmp_path)

    assert compiled
    assert compiled[0].suffix == ".mo"
    assert compiled[0].is_file()
    assert packaging.wheel_catalog_entries(tmp_path)


def test_the_frozen_application_bundles_compiled_catalogs_without_the_pseudo_locale(tmp_path):
    from tools.i18n import packaging

    _write_tree(
        tmp_path,
        {
            CATALOG_REL: _catalog(_entry("Open", "Abrir")),
            "src/polyx/locales/qps_ploc/LC_MESSAGES/messages.po": _catalog(_entry("Open", "")),
        },
    )

    datas = packaging.frozen_locale_datas(tmp_path)
    flattened = " ".join(source for source, _destination in datas)

    assert "messages.mo" in flattened
    assert "qps_ploc" not in flattened


def test_the_build_hook_is_wired_into_pyproject():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    hooks = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["hooks"]

    assert hooks["custom"]["path"] == "tools/i18n/hatch_hook.py"


# ---------------------------------------------------------------------------
# Wiring: the local gate and pre-commit run the audit.
# ---------------------------------------------------------------------------


def test_the_local_gate_runs_the_audit():
    import tools.check as gate

    commands = gate._commands(Path("/usr/bin/python"), [], [])

    assert any("check_i18n.py" in part for command in commands for part in command)


def test_pre_commit_runs_the_audit():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    local = next(repo for repo in config["repos"] if repo["repo"] == "local")
    hook = next(item for item in local["hooks"] if item["id"] == "i18n-audit")

    assert hook["entry"] == "python tools/check_i18n.py"
    assert hook["pass_filenames"] is False
    assert hook["language"] == "python"
    assert not hook.get("additional_dependencies")
    pattern = __import__("re").compile(hook["files"])
    assert pattern.search("src/polyx/gui/page.py")
    assert pattern.search(CATALOG_REL)
    assert pattern.search(TEMPLATE_REL)


# ---------------------------------------------------------------------------
# Runtime: a missing catalog is an error, and languages are discovered.
# ---------------------------------------------------------------------------


def test_a_missing_explicit_language_is_logged_and_falls_back(monkeypatch, caplog):
    import polyx.core.i18n as i18n

    monkeypatch.setenv("POLYX_LOCALE", "fr")
    with caplog.at_level("ERROR", logger="polyx.core.i18n"):
        importlib.reload(i18n)

    assert i18n.language() == "en"
    assert "fr" in caplog.text
    monkeypatch.delenv("POLYX_LOCALE", raising=False)
    importlib.reload(i18n)


def test_available_languages_discovers_a_new_catalog_directory(tmp_path, monkeypatch):
    import polyx.core.i18n as i18n

    (tmp_path / "pt" / "LC_MESSAGES").mkdir(parents=True)
    (tmp_path / "pt" / "LC_MESSAGES" / "messages.po").write_text(
        _catalog(_entry("Open", "Abrir")), encoding="utf-8"
    )
    (tmp_path / "xx" / "LC_MESSAGES").mkdir(parents=True)
    (tmp_path / "xx" / "LC_MESSAGES" / "messages.po").write_text(
        _catalog(_entry("Open", "Abrir")), encoding="utf-8"
    )
    monkeypatch.setattr(i18n, "LOCALES_DIR", tmp_path)

    languages = i18n.available_languages()

    assert "pt" in languages
    assert languages["pt"] != "pt"
    assert languages["xx"] == "xx"


def test_a_fresh_checkout_compiles_a_catalog_on_demand(tmp_path, monkeypatch):
    import polyx.core.i18n as i18n

    (tmp_path / "es" / "LC_MESSAGES").mkdir(parents=True)
    (tmp_path / "es" / "LC_MESSAGES" / "messages.po").write_text(
        _catalog(_entry("Open", "Abrir")), encoding="utf-8"
    )
    monkeypatch.setattr(i18n, "LOCALES_DIR", tmp_path)

    assert i18n.set_language("es") is True
    assert i18n._("Open") == "Abrir"
    monkeypatch.setattr(i18n, "_language", "en")
    monkeypatch.setattr(i18n, "_catalog", importlib.import_module("gettext").NullTranslations())
