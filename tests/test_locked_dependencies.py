"""Every declared dependency is present in the committed ``uv.lock``.

A developer's existing ``.venv`` can still contain a package after it is added
to ``pyproject.toml``, so the suite can pass while the lock is stale and a
fresh ``uv sync --locked`` fails. This test turns that drift into a red suite.
It parses the two files instead of running uv, so it works in any checkout.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "uv.lock"

_REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalise(name: str) -> str:
    """PEP 503 normalisation, enough for the names this project uses."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement.strip())
    assert match is not None, f"unparsable requirement: {requirement!r}"
    return _normalise(match.group(1))


def _declared_requirements(pyproject: dict) -> set[str]:
    project = pyproject["project"]
    requirements: list[str] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        requirements.extend(group)
    for group in pyproject.get("dependency-groups", {}).values():
        requirements.extend(item for item in group if isinstance(item, str))
    return {_requirement_name(requirement) for requirement in requirements}


def _locked_packages(lock: dict) -> set[str]:
    return {_normalise(package["name"]) for package in lock.get("package", [])}


def test_every_declared_dependency_is_in_the_lock():
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))

    missing = _declared_requirements(pyproject) - _locked_packages(lock)

    assert not missing, (
        "pyproject.toml declares packages absent from uv.lock: "
        + ", ".join(sorted(missing))
        + ". Run `uv lock` and commit the result."
    )


def test_requirement_names_are_normalised():
    assert _requirement_name("PySide6==6.7.2") == "pyside6"
    assert _requirement_name("types-PyYAML>=6") == "types-pyyaml"
    assert _requirement_name("pre-commit[foo]; python_version >= '3.11'") == "pre-commit"


def test_a_missing_dependency_is_detected():
    pyproject = {
        "project": {"dependencies": ["declared"]},
        "dependency-groups": {"dev": ["dev-only"]},
    }
    lock = {"package": [{"name": "declared"}]}

    assert _declared_requirements(pyproject) - _locked_packages(lock) == {"dev-only"}
