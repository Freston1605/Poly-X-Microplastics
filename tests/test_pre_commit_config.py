"""The pre-commit configuration is portable and orders the local checks.

Every local hook must be a standard-library Python script launched through the
interpreter pre-commit provisions, never through a Unix shell or an absolute
path. This is a file contract: it pins the YAML, not pre-commit's internals.
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

LOCAL_IDS = {"english-added-lines", "no-hardcoded-fallback"}
SHELL_TOKENS = ("&&", "||", "|", ";", ">", "<", "$(", "`")


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


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


def test_every_local_hook_is_a_stdlib_python_script():
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


def test_ruff_hooks_are_unchanged():
    config = _config()
    repos = [
        repo
        for repo in config["repos"]
        if repo["repo"] == "https://github.com/astral-sh/ruff-pre-commit"
    ]
    assert len(repos) == 1
    assert repos[0]["rev"] == "v0.17.0"
    assert [hook["id"] for hook in repos[0]["hooks"]] == ["ruff-check", "ruff-format"]


def test_dev_dependencies_declare_pre_commit_once():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

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
