"""Where a released Poly-X comes from.

Distribution moved to the fork that supersedes upstream. This module is the
one place the desktop surface names it: the Launcher sends the user to the
Releases page, and the update check no longer reads a branch head whose
commit SHA cannot say whether the installed dependency set is patched.

The release protocol itself — what is published, what is signed and how an
update is verified — is tracked in #153 and is not decided here.
"""

from __future__ import annotations

REPOSITORY = "Freston1605/Poly-X-Microplastics"
RELEASES_URL = f"https://github.com/{REPOSITORY}/releases"


def desktop_release_url() -> str:
    """The page that lists the downloadable desktop releases."""
    return RELEASES_URL
