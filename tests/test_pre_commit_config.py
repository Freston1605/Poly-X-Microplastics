"""The pre-commit configuration is portable and orders the local checks.

Every local hook is a Python script that uses only the standard library and
in-repo modules — no third-party dependency — launched through the interpreter
pre-commit provisions, never through a Unix shell or an absolute path. This is
a file contract: it pins the YAML, not pre-commit's internals.
"""

from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".pre-commit-config.yaml"
PYPROJECT = ROOT / "pyproject.toml"

LOCAL_IDS = {"english-added-lines", "i18n-audit", "no-hardcoded-fallback"}
SHELL_TOKENS = ("&&", "||", "|", ";", ">", "<", "$(", "`")
RUFF_REPOSITORY = "https://github.com/astral-sh/ruff-pre-commit"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def pyproject() -> dict:
    """The parsed ``pyproject.toml``.

    ``tests/test_console_scripts.py`` reads the same file, so the parse lives
    here once instead of in a second copy that could drift.
    """
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_config_file_ends_with_a_newline():
    assert CONFIG.read_text(encoding="utf-8").endswith("\n")


def _local_hooks(config: dict) -> list[dict]:
    repos = [repo for repo in config["repos"] if repo["repo"] == "local"]
    assert len(repos) == 1, "there must be exactly one local repository"
    return repos[0]["hooks"]


def _hook(hooks: list[dict], hook_id: str) -> dict:
    matches = [hook for hook in hooks if hook["id"] == hook_id]
    assert len(matches) == 1, f"expected exactly one {hook_id!r} hook"
    return matches[0]


def _dependency_name(requirement: str) -> str:
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].lower()


def _ruff_hook_version() -> str:
    """The version the ruff hooks install, leading ``v`` included."""
    repos = [repo for repo in _config()["repos"] if repo["repo"] == RUFF_REPOSITORY]
    assert len(repos) == 1, "there must be exactly one ruff hook repository"
    return repos[0]["rev"]


def test_every_local_hook_uses_only_stdlib_and_in_repo_modules():
    hooks = _local_hooks(_config())

    assert {hook["id"] for hook in hooks} == LOCAL_IDS
    assert len({hook["id"] for hook in hooks}) == len(hooks)

    for hook in hooks:
        entry = hook["entry"]
        assert hook["language"] == "python", hook["id"]

        parts = shlex.split(entry)
        assert parts[0] == "python", f"{hook['id']} must name the interpreter"
        assert len(parts) == 2, f"{hook['id']} must invoke exactly one script"

        script = parts[1]
        assert script.startswith("tools/") and script.endswith(".py"), script
        assert not Path(script).is_absolute(), script
        assert not re.match(r"^[A-Za-z]:", script), script
        assert not script.startswith(("/", "\\", "~", "$", "%")), script
        assert (ROOT / script).is_file(), f"{hook['id']} names missing {script}"

        for token in SHELL_TOKENS:
            assert token not in entry, f"{hook['id']} entry uses shell token {token!r}"

        assert not hook.get("additional_dependencies"), hook["id"]


def test_english_hook_scans_staged_python_files():
    hook = _hook(_local_hooks(_config()), "english-added-lines")

    assert hook["entry"] == "python tools/check_english_ids.py"
    assert hook["pass_filenames"] is True
    assert hook["require_serial"] is True

    pattern = re.compile(hook["files"])
    assert pattern.match("src/polyx/core/model.py")
    assert pattern.match("tests/test_model.py")
    assert not pattern.match("docs/model.py")
    assert not pattern.match("src/polyx/core/model.txt")


def test_palette_hook_keeps_its_whole_tree_scope():
    hook = _hook(_local_hooks(_config()), "no-hardcoded-fallback")

    assert hook["entry"] == "python tools/check_palette_fallbacks.py"
    assert hook["pass_filenames"] is False

    pattern = re.compile(hook["files"])
    assert pattern.search("src/polyx/gui/theme.py")
    assert pattern.search("tools/check_palette_fallbacks.py")
    assert not pattern.search("docs/theme.md")


def test_ruff_hooks_are_present_with_the_expected_ids():
    config = _config()
    repos = [repo for repo in config["repos"] if repo["repo"] == RUFF_REPOSITORY]
    assert len(repos) == 1
    assert [hook["id"] for hook in repos[0]["hooks"]] == ["ruff-check", "ruff-format"]


def _development_requirements() -> list[str]:
    """Every development requirement the pyproject declares.

    The same list appears twice: the dependency group the gate syncs and the
    optional ``dev`` extra the README installs. Both are doors into a
    development environment, so both must agree on ruff.
    """
    data = pyproject()
    return [
        *data["dependency-groups"]["dev"],
        *data["project"]["optional-dependencies"]["dev"],
    ]


def test_ruff_is_pinned_to_the_version_the_hooks_install():
    """The preview and the gate run one ruff version.

    The gate runs ``python -m ruff`` from the development group; the hooks
    install their own interpreter from ``ruff-pre-commit``. Nothing else ties
    the two, so an exact pin plus this test is what keeps a hook that passes
    locally from failing the gate on the version alone.
    """
    hook_version = _ruff_hook_version().removeprefix("v")

    requirements = [
        requirement
        for requirement in _development_requirements()
        if _dependency_name(requirement) == "ruff"
    ]

    assert requirements, "the pyproject declares no ruff requirement"
    assert all(requirement == f"ruff=={hook_version}" for requirement in requirements), (
        f"pin ruff exactly at the hooks' {hook_version}: {requirements}"
    )


def test_dev_dependencies_declare_pre_commit_once():
    data = pyproject()

    groups = {
        "dependency-groups.dev": data["dependency-groups"]["dev"],
        "project.optional-dependencies.dev": data["project"]["optional-dependencies"]["dev"],
    }
    for label, requirements in groups.items():
        names = [_dependency_name(requirement) for requirement in requirements]
        assert names.count("pre-commit") == 1, label
        assert names.count("pytest-xdist") == 1, label


def test_english_hook_has_no_shell_or_absolute_git():
    source = (ROOT / "tools" / "check_english_ids.py").read_text(encoding="utf-8")

    assert "/usr/bin/git" not in source
    assert "shell=True" not in source
    assert "bash -c" not in source
    assert 'shutil.which("git")' in source
