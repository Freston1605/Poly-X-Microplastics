"""The one Python source for where this project lives (ADR-0011).

The package metadata, the README and the model-weight downloader all name the
same repository. Keeping the slug here means a rename is one edit instead of a
search for the URL that was missed, and it is why no other module spells the
owner out.
"""

from __future__ import annotations

REPO_SLUG = "Freston1605/micra"

REPO_URL = f"https://github.com/{REPO_SLUG}"

__all__ = ["REPO_SLUG", "REPO_URL"]
