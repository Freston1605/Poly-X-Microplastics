"""Assert a built wheel ships a working Spanish catalog and no pseudo-locale.

Usage:
    python tools/check_wheel.py dist/polyx_microplastics-*.whl

The local gate imports the source tree, so it cannot see packaging. Hatchling
also excludes VCS-ignored files by default, which is how a compiled `.mo`
silently missed the wheel once. This check opens the wheel, loads the catalog
with the standard library and reads a known translation — what an installed
copy does — so the packaging invariant is tested, not assumed.
"""

from __future__ import annotations

import gettext
import sys
import tempfile
import zipfile
from pathlib import Path

CATALOG = "polyx/locales/es/LC_MESSAGES/messages.mo"
PSEUDO = "qps_ploc"
KNOWN_MESSAGE = "Error while saving"
KNOWN_TRANSLATION = "Error al guardar"


def check(wheel: Path) -> list[str]:
    """Problems with ``wheel``'s packaged catalogs; empty means usable."""
    if not wheel.is_file():
        return [f"{wheel}: not a file"]
    problems: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if CATALOG not in names:
            problems.append(f"{wheel}: missing {CATALOG}")
        if any(PSEUDO in name for name in names):
            problems.append(f"{wheel}: ships the development pseudo-locale")
        if CATALOG in names:
            with tempfile.TemporaryDirectory() as directory:
                archive.extract(CATALOG, directory)
                with (Path(directory) / CATALOG).open("rb") as handle:
                    translations = gettext.GNUTranslations(handle)
                actual = translations.gettext(KNOWN_MESSAGE)
                if actual != KNOWN_TRANSLATION:
                    problems.append(
                        f"{CATALOG}: {KNOWN_MESSAGE!r} -> {actual!r}, "
                        f"expected {KNOWN_TRANSLATION!r}"
                    )
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: check_wheel.py <wheel>", file=sys.stderr)
        return 2
    wheel = Path(argv[0])
    problems = check(wheel)
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(f"{wheel.name}: packaged catalog OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
