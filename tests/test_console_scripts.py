"""Every console script the package declares is installed and resolves.

The suite imports ``src/`` through ``PYTHONPATH``, so a wrong entry-point
target — ``polyx-train = polyx.cli.train:train`` — survives the source tree
and breaks only the installed wheel, where a user meets it after a release.
This test is the distribution boundary for that: it reads the console scripts
from installed distribution metadata, compares them with the ones
``pyproject.toml`` declares, and imports every target.

It cannot pass vacuously: an empty declaration, an empty install or a
distribution that is not installed at all is a failure, not a skip.

The module is also the seam where the installed distribution's metadata is
compared with the declaration, so ``Requires-Python`` is checked here for the
same reason: ``pip`` reads the wheel, not ``pyproject.toml``.
"""

from __future__ import annotations

import sys
from importlib.metadata import EntryPoint, distribution, metadata

from packaging.specifiers import SpecifierSet

from polyx.cli.main import main as dispatcher
from tests.test_pre_commit_config import pyproject

DISTRIBUTION = "micra"
GROUP = "console_scripts"
COMMAND_NAMES = ("micra", "polyx")


def declared_scripts(pyproject: dict) -> dict[str, str]:
    """The console scripts ``[project.scripts]`` declares, name to target."""
    scripts = pyproject.get("project", {}).get("scripts", {})
    return {str(name): str(target) for name, target in scripts.items()}


def installed_scripts() -> dict[str, str]:
    """The console scripts this install exposes, name to target."""
    return {
        entry.name: entry.value
        for entry in distribution(DISTRIBUTION).entry_points
        if entry.group == GROUP
    }


def parity_problems(declared: dict[str, str], installed: dict[str, str]) -> list[str]:
    """Every way the installed entry points disagree with the declaration."""
    problems: list[str] = []
    # An empty side would make the difference checks below pass by saying
    # nothing, which is the one way this test could be vacuous.
    if not declared:
        problems.append("pyproject.toml declares no console scripts")
    if not installed:
        problems.append(f"{DISTRIBUTION} installs no console scripts")
    problems.extend(
        f"{name} is declared but not installed"
        for name in sorted(declared.keys() - installed.keys())
    )
    problems.extend(
        f"{name} is installed but not declared"
        for name in sorted(installed.keys() - declared.keys())
    )
    problems.extend(
        f"{name}: declared {declared[name]}, installed {installed[name]}"
        for name in sorted(declared.keys() & installed.keys())
        if declared[name] != installed[name]
    )
    return problems


def _load_target(name: str, value: str) -> object:
    """Load one target the way the generated console script does."""
    return EntryPoint(name=name, value=value, group=GROUP).load()


def resolve_problems(scripts: dict[str, str], load=_load_target) -> list[str]:
    """Every installed console script that does not load a callable target."""
    problems: list[str] = []
    if not scripts:
        problems.append("there are no installed console scripts to load")
    for name, value in sorted(scripts.items()):
        try:
            target = load(name, value)
        except Exception as error:  # noqa: BLE001 - any load failure is the finding
            problems.append(f"{name} ({value}) does not import: {error}")
            continue
        if not callable(target):
            problems.append(f"{name} ({value}) does not name a callable")
    return problems


def test_every_console_script_is_installed_and_resolves():
    declared = declared_scripts(pyproject())
    installed = installed_scripts()

    assert not parity_problems(declared, installed)
    assert not resolve_problems(installed)


def test_both_command_names_reach_the_same_dispatcher():
    """``micra`` is the product name; ``polyx`` is the alias old scripts type."""
    declared = declared_scripts(pyproject())
    installed = installed_scripts()

    for name in COMMAND_NAMES:
        assert name in declared, f"pyproject.toml does not declare {name}"
        assert name in installed, f"{DISTRIBUTION} does not install {name}"
    assert declared["micra"] == declared["polyx"], "the two names declare different targets"
    assert installed["micra"] == installed["polyx"], "the two names dispatch differently"

    micra = _load_target("micra", installed["micra"])
    polyx = _load_target("polyx", installed["polyx"])
    assert micra is polyx, "micra and polyx do not resolve to one dispatcher"


def test_the_dispatcher_prints_the_pipeline_usage(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["micra"])

    status = dispatcher(["pipeline", "--help"])

    assert status == 0
    assert "micra pipeline" in capsys.readouterr().out


def test_the_dispatcher_rejects_an_unknown_command(capsys):
    status = dispatcher(["teleport"])

    assert status == 2
    assert "teleport" in capsys.readouterr().err


def test_the_installed_python_range_matches_the_declaration():
    declared = pyproject()["project"]["requires-python"]
    installed = metadata(DISTRIBUTION).get("Requires-Python")

    assert installed, f"{DISTRIBUTION} does not declare Requires-Python"
    # as PEP 440 specifier sets: the wheel may reorder or re-space the same
    # clauses, which is not a different claim.
    assert SpecifierSet(installed) == SpecifierSet(declared)


def test_an_empty_side_fails_instead_of_passing_vacuously():
    assert parity_problems({}, {}) == [
        "pyproject.toml declares no console scripts",
        "micra installs no console scripts",
    ]


def test_nothing_to_load_is_a_problem():
    assert resolve_problems({}) == ["there are no installed console scripts to load"]


def test_a_script_the_install_lacks_is_reported():
    declared = {
        "polyx": "polyx.cli.main:main",
        "polyx-ghost": "polyx.cli.ghost:main",
    }
    installed = {"polyx": "polyx.cli.main:main"}

    assert parity_problems(declared, installed) == ["polyx-ghost is declared but not installed"]


def test_a_stale_installed_script_is_reported():
    declared = {"polyx": "polyx.cli.main:main"}
    installed = {"polyx": "polyx.cli.main:main", "polyx-old": "polyx.cli.old:main"}

    assert parity_problems(declared, installed) == ["polyx-old is installed but not declared"]


def test_a_target_that_moved_is_reported():
    declared = {"polyx": "polyx.cli.main:main"}
    installed = {"polyx": "polyx.cli.main:run"}

    assert parity_problems(declared, installed) == [
        "polyx: declared polyx.cli.main:main, installed polyx.cli.main:run"
    ]


def test_a_target_that_does_not_import_is_reported():
    def absent(name, value):
        raise ImportError(f"No module named {value.partition(':')[0]!r}")

    problems = resolve_problems({"polyx-ghost": "polyx.cli.ghost:main"}, absent)

    assert len(problems) == 1
    assert "polyx-ghost" in problems[0]
    assert "does not import" in problems[0]


def test_a_target_that_is_not_callable_is_reported():
    problems = resolve_problems(
        {"polyx": "polyx.cli.main:main"}, lambda name, value: "not a function"
    )

    assert problems == ["polyx (polyx.cli.main:main) does not name a callable"]
