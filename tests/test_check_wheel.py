"""The wheel check catches the packaging defects the source gate cannot see."""

from __future__ import annotations

import tomllib
import zipfile
from pathlib import Path

from babel.messages.catalog import Catalog
from babel.messages.mofile import write_mo
from tools.check_wheel import CATALOG, check

ROOT = Path(__file__).resolve().parents[1]


def _wheel(
    tmp_path: Path,
    *,
    include_catalog: bool = True,
    include_pseudo: bool = False,
    translation: str = "Error al guardar",
) -> Path:
    wheel = tmp_path / "demo-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        if include_catalog:
            compiled = tmp_path / "messages.mo"
            catalog = Catalog(locale="es")
            catalog.add("Error while saving", translation)
            with compiled.open("wb") as handle:
                write_mo(handle, catalog)
            archive.write(compiled, CATALOG)
        if include_pseudo:
            archive.writestr("polyx/locales/qps_ploc/LC_MESSAGES/messages.mo", b"")
    return wheel


def test_a_wheel_with_a_working_catalog_passes(tmp_path):
    assert check(_wheel(tmp_path)) == []


def test_a_wheel_without_the_catalog_fails(tmp_path):
    problems = check(_wheel(tmp_path, include_catalog=False))

    assert problems
    assert CATALOG in problems[0]


def test_a_wheel_with_the_pseudo_locale_fails(tmp_path):
    problems = check(_wheel(tmp_path, include_pseudo=True))

    assert problems
    assert any("pseudo-locale" in problem for problem in problems)


def test_a_catalog_that_lost_its_translation_fails(tmp_path):
    problems = check(_wheel(tmp_path, translation=""))

    assert problems
    assert "expected" in problems[0]


def test_the_sdist_ships_the_build_hook():
    """`uv build` builds the wheel from the sdist, so the hook must travel."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]

    assert "tools/i18n" in include
