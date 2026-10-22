"""Tests for the identifier codemod core and harness (M006/S04/T01).

Covers ``tools/translate_identifiers.py`` (the LibCST transformer, Spanish
scanner, suggestion engine and preflight) and the ``*-identifiers`` modes of
``tests/translate_code_text.py``. Fixtures are inline sources written into
``tmp_path``; the audit tests build a throwaway git repository there.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    """Load a module from file path, registering it before execution."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ti = _load("translate_identifiers", REPO_ROOT / "tools" / "translate_identifiers.py")
tct = _load("translate_code_text", REPO_ROOT / "tests" / "translate_code_text.py")

RENAMES: Dict[str, str] = {
    "Caja": "Box",
    "medir": "measure",
    "ruta": "path",
    "clase": "class_id",
    "g2": "grid2",
    "local": "local_value",
    "texto": "text",
    "filas": "rows",
}


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _manifest(tmp_path: Path, mapping: Dict[str, str]) -> Path:
    body = ", ".join(f"{old!r}: {new!r}" for old, new in mapping.items())
    return _write(tmp_path / "map.py", f"IDENTIFIER_TRANSLATIONS = {{{body}}}\n")


def _init_repo(tmp_path: Path, path: Path) -> None:
    """Create a toy repository with ``path`` already committed at HEAD."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", path.name],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=tmp_path,
        check=True,
    )


# ---------------------------------------------------------------------------
# Transformer behavior.
# ---------------------------------------------------------------------------

SAMPLE = '''\
"""Module docstring mentions ruta and caja."""

# Comment mentions ruta and caja.
MODULE_VALUE = 1
PLAIN = "ruta is just a string"


class Caja:
    """Class docstring for Caja."""

    ruta: str

    def medir(self, ruta, clase=1, g2=None):
        local = ruta
        texto = "ruta"
        self.ruta = local
        return otro(medida=local), texto, PLAIN
'''


def test_renames_module_class_function_parameter_local_attribute_and_keyword():
    result, changes = ti.translate_source(SAMPLE, RENAMES, label="sample.py")
    assert "class Box:" in result
    assert "def measure(self, path, class_id=1, grid2=None):" in result
    assert "local_value = path" in result
    assert "self.path = local_value" in result
    assert "otro(medida=local_value)" in result
    assert "PLAIN = \"ruta is just a string\"" in result
    kinds = {change.kind for change in changes}
    assert "name" in kinds


def test_strings_docstrings_comments_numbers_and_operators_are_untouched():
    result, _changes = ti.translate_source(SAMPLE, RENAMES, label="sample.py")
    assert '"""Module docstring mentions ruta and caja."""' in result
    assert "# Comment mentions ruta and caja." in result
    assert '"""Class docstring for Caja."""' in result
    assert 'text = "ruta"' in result
    assert "MODULE_VALUE = 1" in result
    assert "PLAIN = \"ruta is just a string\"" in result


def test_import_module_paths_and_module_stems_are_preserved():
    source = (
        "from .iconos import icono_conmutable\n"
        "import polyx.gui.iconos\n"
        "from . import iconos\n"
        "iconos = 1\n"
    )
    mapping = {
        "iconos": "icons",
        "icono_conmutable": "toggleable_icon",
        "polyx": "package",
        "gui": "ui",
    }
    result, _changes = ti.translate_source(
        source, mapping, stems=ti.discover_module_stems(), label="pkg.py"
    )
    assert "from .iconos import toggleable_icon" in result
    assert "import polyx.gui.iconos" in result
    # The imported alias that is itself a module stem is not renamed, and
    # neither is the local binding it creates.
    assert "from . import iconos" in result
    assert "iconos = 1" in result

    # Without the stem import, a local variable is renamed normally.
    local, _ = ti.translate_source(
        "iconos = 1\n", mapping, stems=ti.discover_module_stems(), label="local.py"
    )
    assert local == "icons = 1\n"


def test_imported_module_stem_keeps_its_local_binding():
    source = "from polyx.gui import iconos\n\nvalue = iconos.icono\n"
    result, changes = ti.translate_source(
        source,
        {"iconos": "icons", "icono": "icon"},
        stems=ti.discover_module_stems(),
        label="pkg.py",
    )
    assert "from polyx.gui import iconos" in result
    assert "value = iconos.icon" in result
    assert all(change.old != "iconos" for change in changes)


def test_getattr_and_dunder_all_update_while_plain_strings_stay():
    source = (
        '__all__ = ["ruta", "otro"]\n'
        "__all__ += ('ruta',)\n"
        'record = "ruta"\n'
        'first = getattr(obj, "ruta")\n'
        'setattr(obj, "ruta", 1)\n'
        'second = hasattr(obj, "ruta")\n'
        'third = delattr(obj, "ruta")\n'
    )
    result, changes = ti.translate_source(source, {"ruta": "path"}, label="dynamic.py")
    assert '__all__ = ["path", "otro"]' in result
    assert "__all__ += ('path',)" in result
    assert 'record = "ruta"' in result
    assert 'getattr(obj, "path")' in result
    assert 'setattr(obj, "path", 1)' in result
    assert 'hasattr(obj, "path")' in result
    assert 'delattr(obj, "path")' in result
    kinds = [change.kind for change in changes]
    assert kinds.count("__all__") == 2
    assert kinds.count("getattr") == 4


def test_translation_is_idempotent():
    first, _changes = ti.translate_source(SAMPLE, RENAMES, label="sample.py")
    second, changes = ti.translate_source(first, RENAMES, label="sample.py")
    assert second == first
    assert changes == []


def test_crlf_line_endings_are_preserved_byte_for_byte():
    source = "ruta = 1\r\nfilas = [ruta]\r\n"
    result, _changes = ti.translate_source(
        source, {"ruta": "path", "filas": "rows"}, label="crlf.py"
    )
    assert result == "path = 1\r\nrows = [path]\r\n"
    assert result.count("\r\n") == 2
    assert "\n" not in result.replace("\r\n", "")


# ---------------------------------------------------------------------------
# Preflight.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"lista": "list"}, "builtin"),
        ({"clase": "class"}, "keyword"),
        ({"ruta": "path", "camino": "path"}, "both target"),
        ({"ruta": "path", "path": "route"}, "idempotency"),
        ({"ruta": "not valid"}, "valid identifier"),
    ],
)
def test_invalid_rename_maps_are_refused(mapping, message):
    with pytest.raises(ValueError, match=message):
        ti.validate_rename_map(mapping)


def test_same_scope_collision_is_refused():
    source = "nombres = 1\nnames = 2\n"
    with pytest.raises(ValueError, match="already has both"):
        ti.translate_source(source, {"nombres": "names"}, label="collision.py")


def test_distinct_scopes_do_not_collide():
    source = "def first():\n    nombres = 1\n\ndef second():\n    return names\n"
    source = source.replace("return names", "return 0")
    result, changes = ti.translate_source(
        source, {"nombres": "names"}, label="scopes.py"
    )
    assert "names = 1" in result
    assert changes


# ---------------------------------------------------------------------------
# Scanner and suggestions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "mascara",
        "umbral_px",
        "calibraciones",
        "morfotipo",
        "N_ANGULOS",
        "DialogoPreferencias",
        "clasificacion",
        "confianza",
        "número",
        "ruta_imagen",
        "refrescar",
        "comun",
        "usa_lista",
        "pide_marcadas",
    ],
)
def test_scanner_flags_spanish_identifiers(name):
    findings = ti.spanish_identifier_findings(name)
    assert findings, f"scanner missed Spanish identifier {name!r}"
    assert all(
        finding.startswith("spanish-identifier:") for finding in findings
    ), findings


@pytest.mark.parametrize(
    "name",
    [
        "manual",
        "total",
        "version",
        "DetectorPage",
        "MicroscopePanel",
        "LEVEL_ERROR",
        "EN",
        "px",
        "um",
        "gt",
        "metrics",
        "model",
        "canvas",
        "state",
        "page",
        "trainer",
        "QRadioButton",
        "SizeVerCursor",
        "mac_ver",
    ],
)
def test_scanner_accepts_homographs_units_and_api_names(name):
    assert ti.spanish_identifier_findings(name) == []


def test_scanner_splits_snake_camel_and_acronyms():
    assert ti.split_identifier("N_ANGULOS") == ["N", "ANGULOS"]
    assert ti.split_identifier("DialogoPreferencias") == [
        "Dialogo",
        "Preferencias",
    ]
    assert ti.split_identifier("_TarjetaTema") == ["Tarjeta", "Tema"]
    assert ti.split_identifier("g2") == ["g2"]
    assert ti.split_identifier("HTTPServer") == ["HTTP", "Server"]


def test_import_context_is_excluded_but_local_iconos_is_flagged():
    source = "from .iconos import icono_conmutable\niconos = 1\n"
    occurrences = ti.identifier_occurrences(source)
    names = [name for _line, name in occurrences]
    assert "iconos" in names  # the local assignment
    # The import statement contributes no occurrence at all.
    assert occurrences == [(2, "iconos")]


def test_module_stem_binding_suppresses_its_local_uses():
    imported = "from polyx.gui import iconos\n\nvalue = iconos.pixmap(1)\n"
    stems = ti.discover_module_stems()
    names = [name for _line, name in ti.identifier_occurrences(imported, stems)]
    assert "iconos" not in names
    assert "value" in names

    local = "iconos = 1\nvalue = iconos + 1\n"
    names = [name for _line, name in ti.identifier_occurrences(local, stems)]
    assert names.count("iconos") == 2


def test_suggestions_preserve_digits_and_never_emit_builtins_or_keywords():
    suggestions = ti.suggest_identifier_map(
        [
            "ruta_imagen",
            "g2",
            "maximo",
            "minimo",
            "lista",
            "clase",
            "alto_caja",
            "largo_rect_eq_um",
            "DialogoPreferencias",
            "manual",
        ]
    )
    assert suggestions["ruta_imagen"] == "image_path"
    assert suggestions["g2"] == "grid2"
    assert suggestions["maximo"] == "maximum"
    assert suggestions["minimo"] == "minimum"
    assert suggestions["lista"] == "items"
    assert suggestions["clase"] == "class_id"
    assert suggestions["alto_caja"] == "box_height"
    assert suggestions["largo_rect_eq_um"] == "equivalent_rect_length_um"
    assert suggestions["DialogoPreferencias"] == "PreferencesDialog"
    assert "manual" not in suggestions
    import builtins
    import keyword

    for target in suggestions.values():
        assert target.isidentifier()
        assert not keyword.iskeyword(target)
        assert target not in dir(builtins)


# ---------------------------------------------------------------------------
# Consumer scope.
# ---------------------------------------------------------------------------


def test_consumer_mode_updates_imports_but_not_local_definitions():
    source = (
        "from polyx.ml.morfologia import medir\n"
        "from polyx.ml.morfologia import medir as medir2\n"
        "\n"
        "def medir(ruta):\n"
        "    return ruta\n"
        "\n"
        "def medir2(ruta):\n"
        "    return ruta\n"
        "\n"
        "def use(obj):\n"
        "    return medir(obj.ruta)\n"
    )
    result, changes = ti.translate_source(
        source,
        {"medir": "measure", "ruta": "path"},
        scope="consumers",
        label="test_consumer.py",
    )
    # Local definitions and their parameters are test-local source.
    assert "def medir(ruta):" in result
    assert "def medir2(ruta):" in result
    # The dead import is left alone because the local def shadows it.
    assert "from polyx.ml.morfologia import medir\n" in result
    # Attribute access in the map is still updated.
    assert "medir(obj.path)" in result


def test_consumer_mode_consumes_polyx_imports_and_spares_local_medir(tmp_path):
    """A test-local ``def medir`` and a polyx import are treated per file."""
    local_path = _write(
        tmp_path / "test_local.py",
        "def medir(ruta):\n    return ruta\n",
    )
    import_path = _write(
        tmp_path / "test_import.py",
        "from polyx.ml.morfologia import medir\n\nresult = medir(1)\n",
    )
    manifest = _manifest(tmp_path, {"medir": "measure", "ruta": "path"})
    assert (
        tct.main(
            [
                "apply-identifiers",
                "--scope",
                "consumers",
                "--manifest",
                str(manifest),
                str(local_path),
                str(import_path),
            ]
        )
        == 0
    )
    assert local_path.read_text(encoding="utf-8") == "def medir(ruta):\n    return ruta\n"
    assert "from polyx.ml.morfologia import measure" in import_path.read_text(
        encoding="utf-8"
    )
    assert "result = measure(1)" in import_path.read_text(encoding="utf-8")


def test_consumer_mode_renames_imported_names_attributes_and_keywords():
    source = (
        "from polyx.ml.morfologia import medir\n"
        "\n"
        "def use(obj):\n"
        "    return medir(obj.ruta, umbral=1)\n"
    )
    result, changes = ti.translate_source(
        source,
        {"medir": "measure", "ruta": "path", "umbral": "threshold"},
        scope="consumers",
        label="test_consumer.py",
    )
    assert "from polyx.ml.morfologia import measure" in result
    assert "measure(obj.path, threshold=1)" in result
    changed = {(change.old, change.new, change.kind) for change in changes}
    assert ("medir", "measure", "import") in changed
    assert ("medir", "measure", "name") in changed
    assert ("ruta", "path", "attribute") in changed
    assert ("umbral", "threshold", "keyword") in changed


def test_consumer_mode_keeps_imported_module_stem_and_its_uses():
    source = "from polyx.gui import iconos\n\nvalue = iconos.icono\n"
    result, _changes = ti.translate_source(
        source,
        {"iconos": "icons", "icono": "icon"},
        scope="consumers",
        stems=ti.discover_module_stems(),
        label="consumer.py",
    )
    assert "from polyx.gui import iconos" in result
    assert "value = iconos.icon" in result


def test_consumer_mode_ignores_non_polyx_imports():
    source = "from other.module import ruta\n\ndef use(obj):\n    return ruta(obj)\n"
    result, changes = ti.translate_source(
        source, {"ruta": "path"}, scope="consumers", label="other.py"
    )
    assert result == source
    assert changes == []


# ---------------------------------------------------------------------------
# CLI: list / check / suggest / apply.
# ---------------------------------------------------------------------------


def test_list_identifiers_enumerates_a_real_polyx_file(capsys):
    path = REPO_ROOT / "polyx" / "ml" / "calibracion.py"
    assert tct.main(["list-identifiers", "--json", str(path)]) == 0
    entries = json.loads(capsys.readouterr().out)
    names = {entry["name"] for entry in entries}
    # S04/T02 renamed this module through the generated manifest.
    assert "Calibration" in names
    assert "N_ANGLES" in names
    assert not any(entry["findings"] for entry in entries)


def test_check_identifiers_exits_1_on_spanish_and_0_on_clean(tmp_path):
    dirty = _write(tmp_path / "dirty.py", "def medir(ruta):\n    return ruta\n")
    assert tct.main(["check-identifiers", str(dirty)]) == 1
    clean = _write(tmp_path / "clean.py", "def measure(path):\n    return path\n")
    assert tct.main(["check-identifiers", str(clean)]) == 0


def test_check_identifiers_skips_import_paths(tmp_path, capsys):
    source = "from .iconos import icono_conmutable\n"
    path = _write(tmp_path / "imports.py", source)
    assert tct.main(["check-identifiers", str(path)]) == 0
    capsys.readouterr()
    local = _write(tmp_path / "local.py", "from .iconos import icono_conmutable\niconos = 1\n")
    assert tct.main(["check-identifiers", str(local)]) == 1
    assert "iconos" in capsys.readouterr().err


def test_suggest_identifiers_reports_a_draft_map(tmp_path, capsys):
    path = _write(tmp_path / "suggest.py", "ruta = 1\ng2 = ruta\n")
    assert tct.main(["suggest-identifiers", "--json", str(path)]) == 0
    suggestions = json.loads(capsys.readouterr().out)
    assert suggestions["ruta"] == "path"
    assert suggestions["g2"] == "grid2"


def test_dry_run_writes_nothing(tmp_path, capsys):
    path = _write(tmp_path / "toy.py", "ruta = 1\n")
    manifest = _manifest(tmp_path, {"ruta": "path"})
    before = path.read_text(encoding="utf-8")
    assert (
        tct.main(
            ["apply-identifiers", "--manifest", str(manifest), "--dry-run", str(path)]
        )
        == 0
    )
    assert path.read_text(encoding="utf-8") == before
    assert "detected" in capsys.readouterr().out


def test_apply_writes_and_second_run_is_a_noop(tmp_path, capsys):
    path = _write(tmp_path / "toy.py", "ruta = 1\n")
    manifest = _manifest(tmp_path, {"ruta": "path"})
    assert tct.main(["apply-identifiers", "--manifest", str(manifest), str(path)]) == 0
    assert path.read_text(encoding="utf-8") == "path = 1\n"
    capsys.readouterr()
    assert tct.main(["apply-identifiers", "--manifest", str(manifest), str(path)]) == 0
    assert "0 change(s) applied." in capsys.readouterr().out


def test_apply_identifiers_consumers_scope_cli(tmp_path, capsys):
    path = _write(
        tmp_path / "consumer.py",
        "from polyx.ml.morfologia import medir\n\ndef use(obj):\n    return medir(obj.ruta)\n",
    )
    manifest = _manifest(tmp_path, {"medir": "measure", "ruta": "path"})
    assert (
        tct.main(
            [
                "apply-identifiers",
                "--scope",
                "consumers",
                "--manifest",
                str(manifest),
                str(path),
            ]
        )
        == 0
    )
    result = path.read_text(encoding="utf-8")
    assert "from polyx.ml.morfologia import measure" in result
    assert "measure(obj.path)" in result


# ---------------------------------------------------------------------------
# Audit.
# ---------------------------------------------------------------------------

AUDIT_SOURCE = '''\
from .iconos import icono_conmutable

__all__ = ["ruta"]

def medir(ruta):
    """Docstring keeps ruta."""
    # Comment keeps ruta.
    texto = "ruta"
    return getattr(obj, "ruta"), texto
'''

AUDIT_MAP = {
    "medir": "measure",
    "ruta": "path",
    "texto": "text",
    "icono_conmutable": "toggleable_icon",
}


def test_audit_passes_after_identifier_codemod(tmp_path, capsys):
    path = _write(tmp_path / "audit.py", AUDIT_SOURCE)
    _init_repo(tmp_path, path)
    manifest = _manifest(tmp_path, AUDIT_MAP)
    assert tct.main(["apply-identifiers", "--manifest", str(manifest), str(path)]) == 0
    capsys.readouterr()
    assert tct.main(["audit-identifiers", "--manifest", str(manifest), str(path)]) == 0


def test_audit_rejects_a_comment_change(tmp_path, capsys):
    path = _write(tmp_path / "audit.py", AUDIT_SOURCE)
    _init_repo(tmp_path, path)
    manifest = _manifest(tmp_path, AUDIT_MAP)
    tct.main(["apply-identifiers", "--manifest", str(manifest), str(path)])
    path.write_text(
        path.read_text(encoding="utf-8").replace("# Comment keeps ruta.", "# Changed."),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert tct.main(["audit-identifiers", "--manifest", str(manifest), str(path)]) == 1
    assert "non-name token changes" in capsys.readouterr().err


def test_audit_rejects_an_unmapped_name_change(tmp_path, capsys):
    path = _write(tmp_path / "audit.py", AUDIT_SOURCE)
    _init_repo(tmp_path, path)
    manifest = _manifest(tmp_path, AUDIT_MAP)
    tct.main(["apply-identifiers", "--manifest", str(manifest), str(path)])
    path.write_text(
        path.read_text(encoding="utf-8").replace("obj", "target"),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert tct.main(["audit-identifiers", "--manifest", str(manifest), str(path)]) == 1
    assert "NAME change not in manifest" in capsys.readouterr().err


def test_audit_rejects_a_plain_string_change(tmp_path, capsys):
    path = _write(tmp_path / "audit.py", AUDIT_SOURCE)
    _init_repo(tmp_path, path)
    manifest = _manifest(tmp_path, AUDIT_MAP)
    tct.main(["apply-identifiers", "--manifest", str(manifest), str(path)])
    path.write_text(
        path.read_text(encoding="utf-8").replace('text = "ruta"', 'text = "otra"'),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert tct.main(["audit-identifiers", "--manifest", str(manifest), str(path)]) == 1
    assert "outside an approved dynamic context" in capsys.readouterr().err


def test_audit_rejects_a_dynamic_string_outside_the_manifest(tmp_path, capsys):
    path = _write(tmp_path / "audit.py", AUDIT_SOURCE)
    _init_repo(tmp_path, path)
    manifest = _manifest(tmp_path, AUDIT_MAP)
    tct.main(["apply-identifiers", "--manifest", str(manifest), str(path)])
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'getattr(obj, "path")', 'getattr(obj, "other")'
        ),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert tct.main(["audit-identifiers", "--manifest", str(manifest), str(path)]) == 1
    assert "dynamic STRING change not in manifest" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Failure modes (Q5).
# ---------------------------------------------------------------------------


def test_missing_file_is_reported_and_exits_1(tmp_path, capsys):
    manifest = _manifest(tmp_path, {"ruta": "path"})
    missing = tmp_path / "does_not_exist.py"
    assert (
        tct.main(["apply-identifiers", "--manifest", str(manifest), str(missing)])
        == 1
    )
    assert "ERROR in" in capsys.readouterr().err


def test_unparsable_source_is_reported_and_exits_1(tmp_path, capsys):
    path = _write(tmp_path / "broken.py", "def (:\n")
    manifest = _manifest(tmp_path, {"ruta": "path"})
    assert tct.main(["apply-identifiers", "--manifest", str(manifest), str(path)]) == 1
    assert "ERROR in" in capsys.readouterr().err


def test_non_utf8_source_is_reported_and_exits_1(tmp_path, capsys):
    path = tmp_path / "latin1.py"
    path.write_bytes(b"ruta = '\xff'\n")
    manifest = _manifest(tmp_path, {"ruta": "path"})
    assert tct.main(["apply-identifiers", "--manifest", str(manifest), str(path)]) == 1
    assert "ERROR in" in capsys.readouterr().err


def test_manifest_without_translations_is_refused(tmp_path):
    manifest = _write(tmp_path / "bad_manifest.py", "OTHER = {}\n")
    with pytest.raises(SystemExit):
        ti.load_manifest(manifest)


def test_manifest_with_invalid_target_is_refused_before_any_write(tmp_path):
    manifest = _write(
        tmp_path / "bad_manifest.py",
        "IDENTIFIER_TRANSLATIONS = {'ruta': 'list'}\n",
    )
    with pytest.raises(SystemExit):
        ti.load_manifest(manifest)


def test_audit_without_git_history_is_a_noop(tmp_path, capsys):
    """An untracked (or non-repository) file cannot be audited against HEAD."""
    path = _write(tmp_path / "untracked.py", "ruta = 1\n")
    manifest = _manifest(tmp_path, {"ruta": "path"})
    before = path.read_text(encoding="utf-8")
    assert tct.main(["apply-identifiers", "--manifest", str(manifest), str(path)]) == 0
    assert path.read_text(encoding="utf-8") == "path = 1\n"
    capsys.readouterr()
    assert tct.main(["audit-identifiers", "--manifest", str(manifest), str(path)]) == 0


# ---------------------------------------------------------------------------
# Translation-software bridge (D033).
# ---------------------------------------------------------------------------


class _FakeBackend(ti.TranslationBackend):
    """Deterministic backend used to prove cache/glossary precedence."""

    name = "fake"
    model = "fake-1"

    def __init__(self, translations=None, *, fail=False):
        self.translations = dict(translations or {})
        self.fail = fail
        self.calls = []

    def translate_many(self, texts):
        if self.fail:
            raise AssertionError("the backend must not be called")
        self.calls.append(list(texts))
        return [self.translations.get(text, text) for text in texts]


def _bridge_plan(tmp_path, names, cache=None, backend=None, offline=False):
    cache = cache or ti.TranslationCache(tmp_path / "translation_cache.json")
    return ti.build_identifier_map(
        {tmp_path / "sample.py": set(names)},
        cache,
        backend=backend,
        offline=offline,
    )


def test_cache_hit_performs_no_backend_call(tmp_path):
    cache = ti.TranslationCache(tmp_path / "cache.json")
    cache.store("candidatas", "candidates", "fake", "fake-1")
    backend = _FakeBackend(fail=True)
    plan = _bridge_plan(tmp_path, ["candidatas"], cache=cache, backend=backend)
    assert plan.mapping == {"candidatas": "candidates"}
    assert backend.calls == []
    assert plan.stats.cache_hits == 1
    assert plan.stats.backend_calls == 0


def test_glossary_wins_over_the_backend(tmp_path):
    backend = _FakeBackend(fail=True)
    plan = _bridge_plan(
        tmp_path, ["ruta", "N_ANGULOS", "_ARCHIVO"], backend=backend
    )
    assert plan.mapping == {
        "ruta": "path",
        "N_ANGULOS": "N_ANGLES",
        "_ARCHIVO": "_FILE",
    }
    assert backend.calls == []
    assert plan.stats.glossary_hits >= 3


def test_digit_suffix_and_casing_survive(tmp_path):
    backend = _FakeBackend(fail=True)
    plan = _bridge_plan(
        tmp_path, ["ANGULOS2", "_ARCHIVO", "ResumenAsignacion"], backend=backend
    )
    assert plan.mapping == {
        "ANGULOS2": "ANGLES2",
        "_ARCHIVO": "_FILE",
        "ResumenAsignacion": "SummaryAssignment",
    }
    assert backend.calls == []


def test_backend_rejects_builtin_and_keyword_candidates(tmp_path):
    backend = _FakeBackend({"candidatas": "class"})
    plan = _bridge_plan(tmp_path, ["candidatas"], backend=backend)
    target = plan.mapping["candidatas"]
    assert target == "class_value"
    assert target.isidentifier()
    assert target not in ti._BUILTIN_NAMES
    assert plan.stats.rejected_candidates == 1


def test_offline_cache_miss_fails_loudly(tmp_path):
    cache = ti.TranslationCache(tmp_path / "cache.json")
    with pytest.raises(ti.OfflineCacheMiss, match="candidatas"):
        _bridge_plan(tmp_path, ["candidatas"], cache=cache, offline=True)


def test_phrase_fallback_uses_the_full_identifier(tmp_path):
    backend = _FakeBackend(
        {"entorno": "environment", "python del entorno": "python environment"}
    )
    plan = _bridge_plan(tmp_path, ["python_del_entorno"], backend=backend)
    assert plan.mapping == {"python_del_entorno": "python_environment"}
    assert ["python del entorno"] in backend.calls
    assert plan.stats.phrase_hits == 1


def test_cache_miss_is_written_with_provenance(tmp_path):
    path = tmp_path / "cache.json"
    cache = ti.TranslationCache(path)
    backend = _FakeBackend({"candidatas": "candidates"})
    _bridge_plan(tmp_path, ["candidatas"], cache=cache, backend=backend)
    cache.save()
    reloaded = ti.TranslationCache.load(path)
    entry = reloaded.lookup("candidatas")
    assert entry is not None
    assert entry.translation == "candidates"
    assert entry.backend == "fake"
    assert entry.model == "fake-1"
    assert entry.timestamp


def test_write_manifest_loads_through_the_existing_loader(tmp_path):
    plan = _bridge_plan(tmp_path, ["ruta", "N_ANGULOS"])
    manifest = tmp_path / "generated.py"
    ti.write_manifest(manifest, plan, command="translate-identifiers polyx/model")
    assert ti.load_manifest(manifest) == plan.mapping
    text = manifest.read_text(encoding="utf-8")
    assert "IDENTIFIER_TRANSLATION_SOURCES" in text
    assert "do not edit by hand" in text


def test_consumer_mode_renames_imports_inside_a_function():
    source = (
        "def use():\n"
        "    from polyx.ml.observabilidad import cronometro\n"
        "    return cronometro(1)\n"
    )
    result, changes = ti.translate_source(
        source, {"cronometro": "timer"}, scope="consumers", label="nested.py"
    )
    assert "from polyx.ml.observabilidad import timer" in result
    assert "return timer(1)" in result
    assert any(change.kind == "import" for change in changes)


def test_consumer_mode_keeps_argparse_namespace_attributes():
    source = (
        "from polyx.ml.morfologia import medir\n"
        "\n"
        "def run(parser):\n"
        "    args = parser.parse_args()\n"
        "    return medir(args.ruta, salida=args.salida)\n"
    )
    result, _changes = ti.translate_source(
        source,
        {"medir": "measure", "ruta": "path", "salida": "output"},
        scope="consumers",
        label="script.py",
    )
    assert "measure(args.ruta, output=args.salida)" in result


def test_monkeypatch_setattr_string_is_rewritten():
    source = 'monkeypatch.setattr(ob, "TAMANO_TOTAL", 1)\n'
    result, changes = ti.translate_source(
        source, {"TAMANO_TOTAL": "SIZE_TOTAL"}, label="monkeypatch.py"
    )
    assert result == 'monkeypatch.setattr(ob, "SIZE_TOTAL", 1)\n'
    assert changes[0].kind == "getattr"


def test_consumer_mode_protects_non_model_polyx_module_attributes():
    source = (
        "from polyx.gui import theme as T\n"
        "\n"
        "monkeypatch.setattr(T, '_ARCHIVO', 1)\n"
        "value = T._ARCHIVO\n"
    )
    result, changes = ti.translate_source(
        source, {"_ARCHIVO": "_FILE"}, scope="consumers", label="tests/test_temas.py"
    )
    assert "T._ARCHIVO" in result
    assert changes == []


def test_consumer_mode_renames_model_module_dynamic_attributes():
    source = (
        "from polyx.ml import observabilidad as ob\n"
        "\n"
        "monkeypatch.setattr(ob, 'TAMANO_TOTAL', 1)\n"
    )
    result, changes = ti.translate_source(
        source,
        {"TAMANO_TOTAL": "SIZE_TOTAL"},
        scope="consumers",
        label="tests/test_observabilidad.py",
    )
    assert "SIZE_TOTAL" in result
    assert changes[0].kind == "getattr"


def test_consumer_mode_attribute_module_renames_module_alias_attributes():
    source = (
        "from polyx.gui import report_html as rh\n"
        "\n"
        "html = rh._seccion_resumen(state, results)\n"
    )
    result, changes = ti.translate_source(
        source,
        {"_seccion_resumen": "_section_summary"},
        scope="consumers",
        label="tests/test_secciones_informe.py",
        attribute_modules=["polyx.gui.report_html"],
    )
    assert "rh._section_summary(state, results)" in result
    assert changes[0].kind == "attribute"


def test_consumer_mode_attribute_module_requires_explicit_opt_in():
    source = (
        "from polyx.gui import report_html as rh\n"
        "\n"
        "html = rh._seccion_resumen(state, results)\n"
    )
    result, changes = ti.translate_source(
        source,
        {"_seccion_resumen": "_section_summary"},
        scope="consumers",
        label="tests/test_secciones_informe.py",
    )
    assert "rh._seccion_resumen" in result
    assert changes == []


def test_format_keyword_is_neither_renamed_nor_scanned():
    source = 'x = tr("{metodo}").format(metodo=m.method)\n'
    result, changes = ti.translate_source(source, {"metodo": "method_2"})
    assert result == source
    assert changes == []
    names = [name for _line, name in ti.identifier_occurrences(source, stems=())]
    assert "metodo" not in names


def test_observability_data_keys_are_neither_renamed_nor_scanned():
    source = (
        'from polyx.ml.observabilidad import event\n'
        '\n'
        'def f(log):\n'
        '    event(log, "bootstrap.finish", sistema=1, modulo="m")\n'
    )
    result, changes = ti.translate_source(
        source, {"sistema": "system", "modulo": "module"}
    )
    assert result == source
    assert changes == []
    names = [name for _line, name in ti.identifier_occurrences(source, stems=())]
    assert "sistema" not in names
    assert "modulo" not in names


def test_consumer_scope_keeps_observability_data_key_keywords(tmp_path, capsys):
    path = _write(
        tmp_path / "consumer.py",
        'from polyx.ml.observabilidad import event\n'
        '\n'
        'def f(log):\n'
        '    event(log, "x", sistema=1)\n',
    )
    manifest = _manifest(tmp_path, {"sistema": "system"})
    assert (
        tct.main(
            [
                "apply-identifiers",
                "--scope",
                "consumers",
                "--manifest",
                str(manifest),
                str(path),
            ]
        )
        == 0
    )
    assert "sistema=1" in path.read_text(encoding="utf-8")


def test_nested_module_stem_binding_is_protected():
    source = (
        'def f():\n'
        '    from polyx.ml import observabilidad\n'
        '    return observabilidad.event\n'
    )
    stems = ti.discover_module_stems()
    names = [name for _line, name in ti.identifier_occurrences(source, stems)]
    assert "observabilidad" not in names
    result, changes = ti.translate_source(
        source, {"observabilidad": "observability"}, stems=stems
    )
    assert result == source
    assert changes == []


def test_audit_accepts_a_structural_string_reference(tmp_path, capsys):
    path = _write(tmp_path / "estructura.py", 'CONSTRUCTORES = ("_seccion_resumen",)\n')
    _init_repo(tmp_path, path)
    manifest = _manifest(tmp_path, {"_seccion_resumen": "_section_summary"})
    # Structural strings are updated by the task, not by the codemod.
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"_seccion_resumen"', '"_section_summary"'
        ),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert tct.main(["audit-identifiers", "--manifest", str(manifest), str(path)]) == 0
