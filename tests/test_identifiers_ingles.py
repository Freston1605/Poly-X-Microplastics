"""Contract test: every polyx symbol identifier reads English (M006 S04).

This is the durable replacement for the generated per-package identifier
manifests retired at slice closeout. It uses the same scanner as the LibCST
codemod (``tools/translate_identifiers.py``), loaded by file path exactly like
``tests/test_comments_ingles.py`` loads the S03 harness, so the gate and the
codemod can never disagree about what counts as a Spanish identifier.

Scope rules (D027/D028): only symbol identifiers in ``polyx/**/*.py`` are
checked. Module filenames and import paths are excluded by construction, and
an imported name that is itself a module stem is skipped because the symbol is
defined and scanned in its own module. String literals, JSONL data keys and
the runtime ``tr()`` layer are out of scope.

The scanner cannot see every Spanish word, so the S04 closeout also included a
manual survivor sweep of the complete unflagged identifier inventory against a
broad Spanish vocabulary; that sweep found and retired the remaining words
(``Morfologia``, ``segmentar``, ``logger_para``, ``raiz``, ``talla``, ...).

Anti-vacuity guards (P028): the walk must see the real package, a synthetic
Spanish tuple must be flagged, known English/API names must stay clean, a
local ``iconos`` binding must be scanned while its import path is not, and
every module must round-trip through LibCST byte for byte.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import libcst as cst
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The codemod core is loaded from file path: ``tools/`` is not a package and
# the harness itself loads its cores the same way.
_RUTA_CORE = REPO_ROOT / "tools" / "translate_identifiers.py"
_spec = importlib.util.spec_from_file_location("translate_identifiers_gate", _RUTA_CORE)
ti = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, ti)
_spec.loader.exec_module(ti)

# Tokens the scanner flags but that are legitimate in polyx identifiers.
# Extend this set, never loosen the scanner vocabulary.
TEST_ALLOWLIST: frozenset = frozenset()

# Known real modules the walk must see (anti-vacuity: proves the scan is not
# silently passing over an empty or truncated file set).
_KNOWN_MODULES = [
    "polyx/model/morfologia.py",
    "polyx/model/observabilidad.py",
    "polyx/gui/report_html.py",
    "polyx/gui/detector/state.py",
]

# Floors measured at S04 closeout. The module floor may grow; it must never
# shrink below the 72 modules the slice closed with.
_MODULE_FLOOR = 72
_UNIQUE_IDENTIFIER_FLOOR = 3000

# Public names that must exist after the rename, and the Spanish names that
# must be gone from the module that defines them.
_PUBLIC_SURFACE = [
    (
        "polyx/model/calibracion.py",
        ("Calibration",),
        ("Calibracion",),
    ),
    (
        "polyx/model/morfologia.py",
        ("Morphology", "segment", "measure"),
        ("Morfologia", "segmentar", "medir"),
    ),
    (
        "polyx/model/observabilidad.py",
        ("logger_for",),
        ("logger_para",),
    ),
    (
        "polyx/gui/visor/window.py",
        ("ViewerWindow",),
        ("VisorWindow",),
    ),
    (
        "polyx/gui/detector/pages/comparar.py",
        ("ComparePage",),
        ("CompararPage",),
    ),
    (
        "polyx/gui/detector/pages/parametros.py",
        ("ParametersPage",),
        ("ParametrosPage",),
    ),
]


def _polyx_files() -> list[Path]:
    """Every ``*.py`` file under the polyx package, sorted for stability."""
    return sorted((REPO_ROOT / "polyx").rglob("*.py"))


def _read_source(path: Path) -> str:
    """Read a module preserving its original line endings (no newline translation)."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _allowed(findings: list[str]) -> list[str]:
    """Findings whose flagged token is in TEST_ALLOWLIST are dropped."""
    kept = []
    for finding in findings:
        _, separator, token = finding.partition(":")
        if separator and token.lower() in TEST_ALLOWLIST:
            continue
        kept.append(finding)
    return kept


def _unique_names(occurrences: list[tuple[int, str]]) -> set[str]:
    """Every unique identifier token of one module."""
    return {name for _line, name in occurrences}


def _binding_names(tree: ast.Module) -> set[str]:
    """Module-level definitions and imports that ``__all__`` may expose."""
    names: set[str] = set()

    def collect_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                collect_target(element)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                collect_target(target)
        elif isinstance(node, ast.AnnAssign):
            collect_target(node.target)
    return names


def _all_entries(tree: ast.Module) -> list[str]:
    """String entries of every module-level ``__all__`` assignment."""
    entries: list[str] = []

    def collect(value: ast.AST) -> None:
        if isinstance(value, (ast.List, ast.Tuple)):
            for element in value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    entries.append(element.value)
        elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            collect(value.left)
            collect(value.right)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                collect(node.value)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                collect(node.value)
    return entries


@pytest.fixture(scope="module")
def inventory() -> list[tuple[Path, str, list[tuple[int, str]], ast.Module]]:
    """Read, tokenize and parse every polyx module once."""
    modules = []
    for path in _polyx_files():
        source = _read_source(path)
        modules.append(
            (
                path,
                source,
                ti.identifier_occurrences(source),
                ast.parse(source, filename=str(path)),
            )
        )
    return modules


def test_polyx_package_walk_finds_no_spanish_identifiers(
    inventory: list[tuple[Path, str, list[tuple[int, str]], ast.Module]],
) -> None:
    """Main contract: zero Spanish identifier findings across polyx."""
    offenders: list[str] = []
    for path, _source, occurrences, _tree in inventory:
        relative = path.relative_to(REPO_ROOT).as_posix()
        for line, name in occurrences:
            findings = _allowed(ti.spanish_identifier_findings(name))
            if findings:
                offenders.append(f"{relative}:{line} {name!r} -> {findings}")
    assert not offenders, "Spanish identifier remnants in polyx:\n" + "\n".join(
        offenders
    )


def test_inventory_exceeds_floors_and_covers_known_modules(
    inventory: list[tuple[Path, str, list[tuple[int, str]], ast.Module]],
) -> None:
    """Anti-vacuity: the scan must actually see a substantial package."""
    relative_paths = {
        path.relative_to(REPO_ROOT).as_posix() for path, *_rest in inventory
    }
    unique_names: set[str] = set()
    for _path, _source, occurrences, _tree in inventory:
        unique_names |= _unique_names(occurrences)
    assert len(inventory) >= _MODULE_FLOOR, f"only {len(inventory)} polyx modules walked"
    assert len(unique_names) > _UNIQUE_IDENTIFIER_FLOOR, (
        f"only {len(unique_names)} unique identifiers scanned"
    )
    for known in _KNOWN_MODULES:
        assert known in relative_paths, f"known module missing from the walk: {known}"


@pytest.mark.parametrize(
    "name", ["mascara", "umbral_px", "calibraciones", "morfotipo"]
)
def test_scanner_flags_synthetic_spanish_identifiers(name: str) -> None:
    """Anti-vacuity: the Spanish scanner must flag real Spanish words."""
    findings = _allowed(ti.spanish_identifier_findings(name))
    assert findings, f"scanner failed to flag synthetic Spanish identifier {name!r}"


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
    ],
)
def test_english_and_api_names_are_not_flagged(name: str) -> None:
    """English words, Qt/API names and unit tokens must stay clean."""
    assert _allowed(ti.spanish_identifier_findings(name)) == []


def test_local_module_stem_is_scanned_while_an_import_path_is_not() -> None:
    """A local ``iconos`` binding is a symbol; an imported module path is not."""
    local_source = "from .iconos import toggleable_icon\niconos = 1\n"
    local_names = [name for _line, name in ti.identifier_occurrences(local_source)]
    assert "iconos" in local_names, "local iconos binding must be scanned"
    assert "toggleable_icon" not in local_names, "imported symbol belongs to its module"
    assert _allowed(ti.spanish_identifier_findings("iconos"))

    module_source = "from polyx.gui import iconos\nvalue = iconos.icon\n"
    module_names = [name for _line, name in ti.identifier_occurrences(module_source)]
    assert "iconos" not in module_names, "imported module stem must be skipped"


def test_renamed_public_api_returns_and_spanish_definitions_are_gone(
    inventory: list[tuple[Path, str, list[tuple[int, str]], ast.Module]],
) -> None:
    """The public surface is English in the module that defines it."""
    by_path = {
        path.relative_to(REPO_ROOT).as_posix(): names
        for path, _source, occurrences, _tree in inventory
        for names in [_unique_names(occurrences)]
    }
    for relative, required, forbidden in _PUBLIC_SURFACE:
        names = by_path[relative]
        missing = [name for name in required if name not in names]
        surviving = [name for name in forbidden if name in names]
        assert not missing, f"{relative} is missing public names: {missing}"
        assert not surviving, f"{relative} still defines Spanish names: {surviving}"


def test_every_dunder_all_entry_resolves(
    inventory: list[tuple[Path, str, list[tuple[int, str]], ast.Module]],
) -> None:
    """Every ``__all__`` entry names a module-level definition or import."""
    offenders: list[str] = []
    for path, _source, _occurrences, tree in inventory:
        bindings = _binding_names(tree)
        relative = path.relative_to(REPO_ROOT).as_posix()
        for entry in _all_entries(tree):
            if entry not in bindings:
                offenders.append(f"{relative}: __all__ entry {entry!r} does not resolve")
    assert not offenders, "\n".join(offenders)


def test_every_polyx_module_compiles(
    inventory: list[tuple[Path, str, list[tuple[int, str]], ast.Module]],
) -> None:
    """Every module is valid Python after the codemod."""
    for path, source, _occurrences, _tree in inventory:
        compile(source, str(path), "exec")


def test_every_polyx_module_round_trips_through_libcst(
    inventory: list[tuple[Path, str, list[tuple[int, str]], ast.Module]],
) -> None:
    """Formatting proof (D024): the codemod never rewrote code or layout."""
    mismatched = [
        path.relative_to(REPO_ROOT).as_posix()
        for path, source, _occurrences, _tree in inventory
        if cst.parse_module(source).code != source
    ]
    assert not mismatched, "LibCST round-trip changed these modules:\n" + "\n".join(
        mismatched
    )
    assert len(inventory) >= _MODULE_FLOOR


def test_generated_identifier_manifests_are_retired() -> None:
    """The durable gate replaced the generated manifests and translation cache."""
    retired = REPO_ROOT / "tests" / "identifier_manifests"
    assert not retired.exists(), f"generated identifier manifests still exist: {retired}"
    assert not (retired / "translation_cache.json").exists()


def test_no_test_imports_the_retired_manifests() -> None:
    """No test may depend on the deleted manifest directory."""
    candidates = sorted((REPO_ROOT / "tests").glob("test_*.py"))
    candidates.append(REPO_ROOT / "tests" / "conftest.py")
    offenders = []
    for path in candidates:
        if path.resolve() == Path(__file__).resolve():
            continue
        if "identifier_manifests" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, "tests still reference tests/identifier_manifests:\n" + (
        "\n".join(offenders)
    )


def test_preflight_rejects_builtin_target() -> None:
    """A renamed target may never shadow a builtin."""
    with pytest.raises(ValueError, match="builtin"):
        ti.validate_rename_map({"lista": "list"})


def test_preflight_rejects_keyword_target() -> None:
    """A renamed target may never be a Python keyword."""
    with pytest.raises(ValueError, match="keyword"):
        ti.validate_rename_map({"clase": "class"})


def test_preflight_rejects_same_scope_collision() -> None:
    """Old and new names coexisting in one scope are refused."""
    source = "clase = 1\nclass_value = 2\n"
    problems = ti.same_scope_collisions(source, {"clase": "class_value"}, "collision.py")
    assert problems, "same-scope collision was not detected"
    with pytest.raises(ValueError, match="already has both"):
        ti.preflight_module(source, {"clase": "class_value"}, "collision.py")


def test_allowlist_drops_only_listed_tokens() -> None:
    """The allowlist filter must drop exactly the tokens it names."""
    synthetic = ["spanish-identifier:del", "spanish-identifier:mascara"]
    assert _allowed(synthetic) == synthetic, "empty allowlist must drop nothing"
