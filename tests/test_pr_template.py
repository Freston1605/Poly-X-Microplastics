"""The PR template is the single owner of the PR body contract.

Every PR body names its originating spec (``Spec: #N``) or states there is
none (``Spec: none``), and carries Summary, Evidence and Merge Danger. Those
fields are defined in the template only; nothing here talks to the network or
the GitHub API.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"


def test_pr_template_is_the_body_contract():
    assert TEMPLATE.is_file(), f"missing {TEMPLATE}"

    text = TEMPLATE.read_text(encoding="utf-8")

    assert "Spec: #N" in text, "the template must require a spec reference"
    assert "Spec: none" in text, "the template must name the no-spec form"
    for heading in ("## Summary", "## Evidence", "## Merge Danger"):
        assert heading in text, f"the template must require {heading!r}"


def test_contributing_points_at_the_template():
    text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert ".github/PULL_REQUEST_TEMPLATE.md" in text, (
        "CONTRIBUTING must point at the template instead of restating its fields"
    )
