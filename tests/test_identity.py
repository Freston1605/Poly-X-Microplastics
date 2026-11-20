"""The project's identity and credit are true and single-sourced (ADR-0011).

The distribution is ``micra``, the repository is ``Freston1605/micra``, and
every surface that names either reads the same value. This is a file contract:
it parses ``pyproject.toml``, the documents and the package constant, and it
imports the CLI module that consumes it. Nothing here reaches into a private
helper or asserts a line of prose; it asserts the claims a package, a licence
and a security policy make.

The upstream project is named on purpose in three places — the README's Origin
paragraph, the glossary term and the ADR that records the decision — and
nowhere else.
"""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import metadata
from pathlib import Path

from polyx.cli import download_models
from polyx.repository import REPO_SLUG, REPO_URL

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SLUG = "Freston1605/micra"
CANONICAL_URL = f"https://github.com/{CANONICAL_SLUG}"
UPSTREAM = "CrissFerrada/"

#: Documents that describe the project as it is today.
IDENTITY_DOCUMENTS = (
    "README.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/agents/issue-tracker.md",
    "docs/agents/repository-settings.md",
)

#: Where the upstream project may still be named: credit, vocabulary and the
#: record of decisions already taken.
UPSTREAM_ALLOWED = {"README.md", "GLOSSARY.md", "CHANGELOG.md"}

#: Python surfaces the identity ticket owns. ``gui/updates.py`` is retired with
#: the consumer-distribution track by a separate ticket; it keeps the upstream
#: owner until that ticket deletes it rather than this one rewriting code with
#: an expiry date.
RETIRED_BY_DISTRIBUTION_TICKET = {Path("src/polyx/gui/updates.py")}


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _python_surfaces() -> list[Path]:
    """The package and the manual generators, minus the retired updater."""
    surfaces = [*(ROOT / "src" / "polyx").rglob("*.py"), *(ROOT / "tools" / "manual").glob("*.py")]
    return [
        path
        for path in sorted(surfaces)
        if path.relative_to(ROOT) not in RETIRED_BY_DISTRIBUTION_TICKET
    ]


def _mentions_upstream(path: Path) -> bool:
    return UPSTREAM in path.read_text(encoding="utf-8", errors="replace")


def test_the_distribution_is_named_micra() -> None:
    assert _pyproject()["project"]["name"] == "micra"


def test_every_package_url_points_at_the_one_repository() -> None:
    urls = _pyproject()["project"]["urls"]

    assert set(urls) == {"Homepage", "Repository", "Issues", "Changelog"}
    stray = {name: url for name, url in urls.items() if not url.startswith(CANONICAL_URL)}
    assert not stray, f"package URLs outside {CANONICAL_URL}: {stray}"


def test_the_python_range_ends_where_the_gui_stack_ends() -> None:
    declared = _pyproject()["project"]["requires-python"]
    # An independent source of truth: the declared range of the Qt version the
    # ``gui`` extra pins, read from its installed metadata. A wider claim than
    # the one PySide6 supports resolves into a set that cannot install.
    gui_range = metadata("PySide6").get("Requires-Python")
    assert gui_range, "PySide6 does not declare Requires-Python"
    upper = re.search(r"<\s*([0-9][0-9.]*)", gui_range)
    assert upper, f"no upper bound in PySide6's {gui_range!r}"

    assert declared == f">=3.11,<{upper.group(1)}"
    assert re.fullmatch(r">=3\.11,<3\.1[0-9]", declared), declared


def test_the_author_stays_and_the_maintainer_is_name_only() -> None:
    project = _pyproject()["project"]

    assert [author["name"] for author in project["authors"]] == ["Cristofher Ferrada"]
    maintainers = project["maintainers"]
    assert maintainers == [{"name": "Franco Ignacio Valdés Navarro"}]
    assert "email" not in maintainers[0], "the repository is public: no personal address"


def test_one_constant_is_the_python_source_of_the_repository() -> None:
    assert REPO_SLUG == CANONICAL_SLUG
    assert REPO_URL == CANONICAL_URL

    assert not hasattr(download_models, "DEFAULT_URL"), (
        "no default weight URL until a release exists: a missing URL must be an "
        "explicit error, not a 404"
    )

    source = Path(download_models.__file__).read_text(encoding="utf-8")
    assert UPSTREAM not in source
    assert "Freston1605" not in source, "the owner belongs in polyx.repository, not here"


def test_the_readme_names_the_repository_and_credits_upstream_once() -> None:
    text = _read("README.md")

    assert CANONICAL_URL in text
    upstream_lines = [
        line for line in text.splitlines() if "CrissFerrada/Poly-X-Microplastics" in line
    ]
    assert len(upstream_lines) == 1, "the upstream belongs in the Origin paragraph and nowhere else"
    assert "Franco Ignacio Valdés Navarro" in text
    assert "Laboratorio de Química Ambiental" in text


def test_the_licence_keeps_the_original_notice_and_adds_the_modifications() -> None:
    text = _read("LICENSE")

    assert "Copyright (c) 2026 Cristofher Ferrada" in text
    assert re.search(
        r"^Copyright \(c\) 2026 Franco Ignacio Valdés Navarro.*modifications", text, re.MULTILINE
    )


def test_security_reporting_is_github_only() -> None:
    text = _read("SECURITY.md")

    assert f"{CANONICAL_URL}/security/advisories/new" in text
    assert "mailto:" not in text
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), "no personal email in a public repo"


def test_a_new_distribution_channel_requires_an_adr() -> None:
    text = _read("CONTRIBUTING.md")

    collapsed = re.sub(r"\s+", " ", text)
    assert re.search(r"new distribution channel.*?requires a new ADR", collapsed, re.IGNORECASE), (
        "CONTRIBUTING.md does not rule on a new distribution channel"
    )


def test_the_identity_documents_name_one_repository_and_one_product() -> None:
    for name in IDENTITY_DOCUMENTS:
        text = _read(name)
        assert "Freston1605/Poly-X-Microplastics" not in text, f"{name} keeps the old slug"
        assert "MICRA" in text, f"{name} does not name the project"
    for name in ("CLAUDE.md", "CONTRIBUTING.md", "docs/agents/repository-settings.md"):
        assert CANONICAL_SLUG in _read(name), f"{name} does not name the canonical repository"


def test_the_runbook_says_publication_is_deferred() -> None:
    text = _read("docs/agents/repository-settings.md")

    assert re.search(r"publication is deferred", text, re.IGNORECASE)
    assert CANONICAL_SLUG in text


def test_the_upstream_is_named_only_as_credit_or_history() -> None:
    candidates = [
        *(ROOT / name for name in IDENTITY_DOCUMENTS),
        *(ROOT / name for name in ("GLOSSARY.md", "CHANGELOG.md")),
        *(ROOT / "docs" / "adr").glob("*.md"),
        *_python_surfaces(),
    ]
    found = {str(path.relative_to(ROOT)) for path in candidates if _mentions_upstream(path)}

    allowed = UPSTREAM_ALLOWED
    unexpected = {
        name for name in found if name not in allowed and not name.startswith("docs/adr/")
    }
    assert not unexpected, f"upstream project named outside credit and history: {unexpected}"
