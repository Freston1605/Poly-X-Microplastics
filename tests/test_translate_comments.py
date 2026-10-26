"""Tests for the LibCST comment codemod (M006/S03/T01).

The comment core under ``tools/translate_comments.py`` and the shared
harness under ``tests/translate_code_text.py`` are loaded by file path
because neither module is part of a package. Fixtures are inline sources
written into ``tmp_path``: no git-ignored or machine-local paths are
involved.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    """Load a module by path, registering it before exec (MEM105)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_module("translate_code_text", ROOT / "tests" / "translate_code_text.py")
tc = harness.tc

SOURCE = '''\
# header comment
import os  # inline comment


def function(
    value,  # parenthesized parameter comment
):
    # indented comment
    return value  # tail comment
'''


def _write(tmp_path: Path, source: str, name: str = "sample.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Enumeration and prefix handling
# ---------------------------------------------------------------------------


def test_enumerates_comments_in_document_order():
    enumeration = tc.enumerate_comments(SOURCE)
    tokens = harness._comment_tokens(SOURCE)
    # Index sequence is contiguous and starts at zero.
    assert [index for index, _ in enumeration] == list(range(len(enumeration)))
    # The core enumeration and tokenize agree value by value (document order).
    assert [raw for _, raw in enumeration] == [token.string for token in tokens]
    assert enumeration[0][1] == "# header comment"
    assert "# inline comment" in [raw for _, raw in enumeration]
    assert "# parenthesized parameter comment" in [raw for _, raw in enumeration]
    assert "# indented comment" in [raw for _, raw in enumeration]
    assert "# tail comment" in [raw for _, raw in enumeration]


def test_prefixes_are_preserved_including_multiple_spaces():
    source = "#: doc comment\n#   aligned table\n#\ttabbed comment\n#tight\n"
    translations = {
        0: "Documented.",
        1: "aligned table",
        2: "tabbed comment",
        3: "tight",
    }
    new_source, changes, _ = tc.translate_source(source, translations)
    assert "#: Documented." in new_source
    assert "#   aligned table" in new_source
    assert "#\ttabbed comment" in new_source
    assert "#tight" in new_source
    assert [change.index for change in changes] == [0]


def test_multiple_spaces_and_trailing_whitespace_survive_rewrite():
    source = "#   old body   \nVALUE = 1\n"
    new_source, changes, _ = tc.translate_source(source, {0: "new body"})
    assert new_source == "#   new body   \nVALUE = 1\n"
    assert len(changes) == 1


def test_split_comment_exposes_prefix_body_and_trailing_whitespace():
    assert tc.split_comment("#   body  ") == ("#   ", "body", "  ")
    assert tc.split_comment("#: body") == ("#: ", "body", "")
    assert tc.split_comment("#") == ("#", "", "")


# ---------------------------------------------------------------------------
# Safety envelope
# ---------------------------------------------------------------------------


def test_only_comment_tokens_change():
    source = '''\
"""Docstring must stay."""
VALUE = "string literal must stay"


def function():
    return f"{VALUE}"  # old comment


# another old comment
'''
    new_source, _, _ = tc.translate_source(
        source, {0: "new comment", 1: "another new comment"}
    )
    old_tokens = harness._tokens_de(source)
    new_tokens = harness._tokens_de(new_source)
    assert [token for token in old_tokens if token[0] != tokenize.COMMENT] == [
        token for token in new_tokens if token[0] != tokenize.COMMENT
    ]
    assert (
        "Docstring must stay." in new_source
        and 'VALUE = "string literal must stay"' in new_source
    )


def test_second_apply_is_idempotent(tmp_path):
    path = _write(tmp_path, SOURCE)
    first = tc.process_file(path, {0: "Header translated."}, apply=True)
    after_first = path.read_text(encoding="utf-8")
    second = tc.process_file(path, {0: "Header translated."}, apply=True)
    assert len(first) == 1
    assert second == []
    assert path.read_text(encoding="utf-8") == after_first


def test_dry_run_writes_nothing(tmp_path):
    path = _write(tmp_path, SOURCE)
    original = path.read_text(encoding="utf-8")
    changes = tc.process_file(path, {0: "Header translated."}, apply=False)
    assert len(changes) == 1
    assert path.read_text(encoding="utf-8") == original


def test_translation_with_newline_is_rejected():
    with pytest.raises(ValueError, match="newline"):
        tc.translate_source(SOURCE, {0: "first line\nsecond line"})
    with pytest.raises(ValueError, match="newline"):
        tc.translate_source(SOURCE, {0: "first line\rsecond line"})


def test_protected_comment_translation_is_rejected():
    protected_sources = {
        "noqa": "# noqa: F401\nVALUE = 1\n",
        "type": "# type: ignore[attr-defined]\nVALUE = 1\n",
        "shebang": "#!/usr/bin/env python3\nVALUE = 1\n",
        "coding": "# -*- coding: utf-8 -*-\nVALUE = 1\n",
        "separator": "# ----------------\nVALUE = 1\n",
        "empty": "#\nVALUE = 1\n",
    }
    for label, source in protected_sources.items():
        with pytest.raises(ValueError, match="protected"):
            tc.translate_source(source, {0: "Translated."})


def test_protected_helper_covers_directives_and_separators():
    protected = [
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        "# noqa: F401",
        "# type: ignore[attr-defined]",
        "# pragma: no cover",
        "# fmt: off",
        "# isort: skip",
        "# pylint: disable=invalid-name",
        "# mypy: ignore-errors",
        "# nosec",
        "# %%",
        "#",
        "#   ",
        "# ----",
        "# ════════",
    ]
    for raw in protected:
        assert tc.is_protected_comment(raw), raw
    assert not tc.is_protected_comment("#: documented attribute")
    assert not tc.is_protected_comment("# a real comment")


def test_empty_translation_is_rejected():
    with pytest.raises(ValueError, match="erase"):
        tc.translate_source(SOURCE, {0: "   "})


def test_out_of_range_manifest_index_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        tc.translate_source(SOURCE, {999: "Translated."})


def test_missing_comment_translations_fails(tmp_path):
    manifest = tmp_path / "manifest_without_table.py"
    manifest.write_text("DOCSTRING_TRANSLATIONS = {}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="COMMENT_TRANSLATIONS"):
        tc.load_manifest(manifest)


def test_manifest_apply_and_core_cli_dry_run(tmp_path):
    path = _write(tmp_path, "# old comment\nVALUE = 1\n")
    manifest = tmp_path / "comment_manifest.py"
    manifest.write_text(
        f"COMMENT_TRANSLATIONS = {{ {path.as_posix()!r}: {{0: 'new comment'}} }}\n",
        encoding="utf-8",
    )
    original = path.read_text(encoding="utf-8")
    assert tc.main(["--dry-run", "--manifest", str(manifest), str(path)]) == 0
    assert path.read_text(encoding="utf-8") == original
    assert tc.main(["--apply", "--manifest", str(manifest), str(path)]) == 0
    assert path.read_text(encoding="utf-8") == "# new comment\nVALUE = 1\n"


# ---------------------------------------------------------------------------
# Spanish comment scanner (D023)
# ---------------------------------------------------------------------------


def test_unaccented_spanish_sentence_is_flagged():
    findings = harness.spanish_comment_findings(
        "la mala clasificacion penaliza en ambos lados.", set()
    )
    assert findings


def test_backticked_spanish_literal_passes_while_bare_text_fails():
    assert (
        harness.spanish_comment_findings(
            "The dialog shows `Guardar como` before closing.", set()
        )
        == []
    )
    assert harness.spanish_comment_findings(
        "The dialog shows Guardar como before closing.", set()
    )


def test_code_shaped_identifiers_and_paths_are_not_prose():
    findings = harness.spanish_comment_findings(
        "Uses SECCIONES_INFORME, TAMANO_TOTAL and polyx/gui/theme.py here.",
        {"SECCIONES_INFORME", "TAMANO_TOTAL"},
    )
    assert findings == []
    assert (
        harness.spanish_comment_findings(
            "The detector writes NNNN_<stem>.txt under labels/.", set()
        )
        == []
    )


def test_english_comments_are_clean():
    assert (
        harness.spanish_comment_findings(
            "Build the report from the current run.", set()
        )
        == []
    )
    assert (
        harness.spanish_comment_findings(
            "Values come from the detector output, not from the cache.", set()
        )
        == []
    )


def test_check_comments_exit_codes(tmp_path):
    spanish = _write(
        tmp_path,
        "# la mala clasificacion penaliza en ambos lados\nVALUE = 1\n",
        name="spanish.py",
    )
    english = _write(
        tmp_path,
        "# Build the report from the current run.\n"
        "# Values come from the detector output, not from the cache.\n"
        "VALUE = 1\n",
        name="english.py",
    )
    assert harness.main(["check-comments", str(spanish)]) == 1
    assert harness.main(["check-comments", str(english)]) == 0


def test_check_comments_skips_protected_directives(tmp_path):
    protected = _write(
        tmp_path,
        "# noqa: F401\n# type: ignore[attr-defined]\n# ----------------\nVALUE = 1\n",
        name="protected.py",
    )
    assert harness.main(["check-comments", str(protected)]) == 0


def test_list_comments_json_pairs_tokenize_lines(tmp_path, capsys):
    path = _write(tmp_path, SOURCE)
    assert harness.main(["list-comments", "--json", str(path)]) == 0
    entries = json.loads(capsys.readouterr().out)
    tokens = harness._comment_tokens(SOURCE)
    assert [entry["index"] for entry in entries] == list(range(len(tokens)))
    assert [entry["line"] for entry in entries] == [
        token.start[0] for token in tokens
    ]
    assert [entry["text"] for entry in entries] == [
        token.string for token in tokens
    ]
    assert all(
        entry["protected"] == tc.is_protected_comment(entry["text"])
        for entry in entries
    )


# ---------------------------------------------------------------------------
# Token audit against git HEAD (D024)
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_audit_passes_after_codemod_and_flags_non_comment_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    path = repo / "sample.py"
    path.write_text('# old comment\nVALUE = "old string"\n', encoding="utf-8")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-m", "base")

    # A COMMENT-only change is accepted.
    tc.process_file(path, {0: "new comment"}, apply=True)
    assert harness.audit_comment_file(path) == []

    # A changed code token is flagged.
    path.write_text(
        '# new comment\nRENAMED = "old string"\n', encoding="utf-8"
    )
    assert harness.audit_comment_file(path)

    # A changed string literal is flagged.
    path.write_text(
        '# new comment\nVALUE = "new string"\n', encoding="utf-8"
    )
    assert harness.audit_comment_file(path)


def test_audit_untracked_file_is_clean(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    path = repo / "untracked.py"
    path.write_text("# comment\nVALUE = 1\n", encoding="utf-8")
    assert harness.audit_comment_file(path) == []


def test_audit_comments_expands_directories(tmp_path):
    repo = tmp_path / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    first = package / "first.py"
    second = package / "second.py"
    first.write_text("# old first\nVALUE = 1\n", encoding="utf-8")
    second.write_text("# old second\nOTHER = 2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    # The CLI receives the package directory, as the slice verify command does.
    tc.process_file(first, {0: "new first"}, apply=True)
    assert harness.main(["audit-comments", str(package)]) == 0

    # A code-token change inside the directory is still flagged.
    second.write_text("# old second\nOTHER = 3\n", encoding="utf-8")
    assert harness.main(["audit-comments", str(package)]) == 1


# ---------------------------------------------------------------------------
# Line endings and package-wide alignment
# ---------------------------------------------------------------------------


def test_crlf_line_endings_survive(tmp_path):
    path = tmp_path / "crlf.py"
    path.write_bytes(b"# old comment\r\nVALUE = 1\r\n")
    tc.process_file(path, {0: "new comment"}, apply=True)
    assert path.read_bytes() == b"# new comment\r\nVALUE = 1\r\n"


def test_enumeration_matches_tokenize_on_every_polyx_module():
    files = sorted((ROOT / "src" / "polyx").rglob("*.py"))
    assert len(files) == 96
    for path in files:
        source = path.read_text(encoding="utf-8")
        enumeration = tc.enumerate_comments(source)
        tokens = harness._comment_tokens(source)
        assert len(enumeration) == len(tokens), path
        assert [raw for _, raw in enumeration] == [
            token.string for token in tokens
        ], path


def test_theme_inventory_and_non_vacuous_scanner(tmp_path, capsys):
    theme = ROOT / "src" / "polyx" / "gui" / "theme.py"
    assert harness.main(["list-comments", "--json", str(theme)]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert len(entries) == 65
    assert all("line" in entry for entry in entries)
    assert any(entry["protected"] for entry in entries)
    # The scanner must still see Spanish: an inline fixture keeps this check
    # alive once every real module has been translated (S03/T05).
    spanish = tmp_path / "spanish.py"
    spanish.write_text(
        "# Los comentarios en espanol se detectan sin problema.\nVALUE = 1\n",
        encoding="utf-8",
    )
    assert harness.main(["check-comments", str(spanish)]) == 1
