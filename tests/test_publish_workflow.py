"""The release job is complete without PyPI, and that is deliberate.

#203 requires desktop delivery not to depend on PyPI credentials or
availability, so the GitHub release does not wait for the upload. This file is
the contract for that decision: the release still attaches every distribution
file, and only the upload itself is independent. Review asked for this to be
documented and tested rather than left implicit.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _needs(job: dict) -> set[str]:
    needs = job.get("needs", [])
    return {needs} if isinstance(needs, str) else set(needs)


def _release_command(job: dict) -> str:
    steps = job["steps"]
    (create,) = [step for step in steps if step.get("name") == "Create the GitHub release"]
    return create["run"]


def test_the_release_waits_for_the_desktop_zip_and_the_built_distributions():
    jobs = _jobs()

    assert {"desktop", "build"} <= _needs(jobs["release"])


def test_the_release_does_not_wait_for_the_pypi_upload():
    jobs = _jobs()

    # Independence is the decision (#203); a red `publish` job still appears on
    # the same tag run, so silence is not the failure mode here.
    assert "publish" not in _needs(jobs["release"])


def test_the_release_attaches_the_wheel_sdist_and_desktop_zip():
    command = _release_command(_jobs()["release"])

    for pattern in ("dist/*.whl", "dist/*.tar.gz", "dist/desktop/*.zip", "dist/desktop/*.sha256"):
        assert pattern in command, f"the release must attach {pattern}"


def test_the_desktop_zip_is_checksummed_and_proven_before_it_is_published():
    build_steps = _jobs()["desktop"]["steps"]
    names = [step.get("name", "") for step in build_steps]

    assert any("Build the desktop ZIP" in name for name in names)
    assert any("opens and carries the program" in name for name in names)


def test_publishing_still_waits_for_the_built_distributions():
    assert "build" in _needs(_jobs()["publish"])
