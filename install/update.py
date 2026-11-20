"""Point an installed Poly-X at a tagged release from the fork.

Releases come from ``Freston1605/Poly-X-Microplastics``, which supersedes the
upstream project. This script does not overwrite the running installation:
an older install that still tracks upstream keeps working where it is, and the
user installs the fork alongside it.

Which channel replaces this manual route — and how verification, rollback and
the CPU fallback are decided — is #161. Nothing here claims that a dependency
was patched: only the release it was installed from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = "Freston1605/Poly-X-Microplastics"
RELEASES = f"https://github.com/{REPO}/releases"


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _pause() -> None:
    import contextlib

    with contextlib.suppress(EOFError):
        input("  Pulsa ENTER para salir...")


def main() -> int:
    ap = argparse.ArgumentParser(description="Point Poly-X at a fork release")
    ap.add_argument("--root", default=None, help="installation root (read only)")
    ap.add_argument("--no-pause", action="store_true", help="do not wait for ENTER at the end")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else here.parent

    _say()
    _say("=" * 60)
    _say("  Poly-X  -  Releases")
    _say("=" * 60)
    _say()
    _say(f"Instalacion actual (sin tocar): {root}")
    _say()
    _say("Las versiones nuevas se publican en:")
    _say(f"  {RELEASES}")
    _say()
    _say("Descarga la version para tu sistema, comprueba su SHA-256 y sigue")
    _say("las instrucciones del release para instalarla junto a esta copia.")
    _say("Esta instalacion y tus datos no se modifican.")
    _say()
    if not args.no_pause:
        _pause()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
