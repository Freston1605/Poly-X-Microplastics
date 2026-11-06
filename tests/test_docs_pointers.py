"""The two agent-facing documents point at one review checklist.

``CLAUDE.md`` is the project reference and ``AGENTS.md`` the minimum a
contributor reads first; both link ``CODING_STANDARDS.md`` instead of growing a
third copy of the conventions prose. That checklist names the checks the local
hooks and the gate run, so the documentation and the tooling cannot disagree.

This is a file contract: it reads the files in the tree and talks to nothing.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDARDS = ROOT / "CODING_STANDARDS.md"
POINTER = "[`CODING_STANDARDS.md`](CODING_STANDARDS.md)"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_agent_documents_point_at_the_checklist():
    for name in ("CLAUDE.md", "AGENTS.md"):
        assert POINTER in _read(name), f"{name} must link {POINTER}"


def test_checklist_names_the_checks_the_hooks_and_gate_run():
    text = STANDARDS.read_text(encoding="utf-8")

    # The two local hooks, by their ids in .pre-commit-config.yaml. The
    # config must own the id and the checklist must name it.
    config = _read(".pre-commit-config.yaml")
    for hook in ("english-added-lines", "no-hardcoded-fallback"):
        assert f"id: {hook}" in config, f"missing {hook} in .pre-commit-config.yaml"
        assert hook in text, f"CODING_STANDARDS.md must name the {hook} hook"

    # The commands the gate runs: ruff lint and format, pytest, and ty.
    for command in ("ruff check", "ruff format", "pytest", "ty check"):
        assert command in text, f"CODING_STANDARDS.md must name {command!r}"


def test_checklist_separates_automated_from_review_only_rules():
    text = STANDARDS.read_text(encoding="utf-8")

    assert "## Review checklist" in text
    assert "Review-only" in text, "the checklist must say what the tooling cannot enforce"


def test_contributor_guide_documents_the_local_hooks():
    text = _read("CONTRIBUTING.md")

    assert "uv run pre-commit install" in text
    assert "uv run pre-commit run --all-files" in text
    for hook in ("english-added-lines", "no-hardcoded-fallback"):
        assert hook in text, f"CONTRIBUTING.md must name the {hook} hook"
