"""Unified ``polyx`` command-line entry point.

Dispatches each command to its module. The GUI path imports Qt lazily, so the
headless commands keep working without the ``gui`` extra.
"""

from __future__ import annotations

import sys

from polyx.core.i18n import _

_COMMANDS = {
    "gui": "polyx.gui.__main__",
    "train": "polyx.cli.train",
    "detect": "polyx.cli.detect",
    "eval": "polyx.cli.eval",
    "report": "polyx.cli.report",
    "download-models": "polyx.cli.download_models",
}


def _usage() -> str:
    """The help screen, translated when it is printed."""
    return "\n".join(
        [
            _("Poly-X command line."),
            "",
            _("Usage:"),
            f"    polyx                      # {_('launch the GUI')}",
            f"    polyx gui                  # {_('launch the GUI')}",
            f"    polyx train ...            # {_('headless training')}",
            f"    polyx detect ...           # {_('headless detection')}",
            f"    polyx eval ...             # {_('headless evaluation')}",
            f"    polyx report ...           # {_('compact HTML report')}",
            f"    polyx download-models ...  # {_('download weights')}",
        ]
    )


def _launch_gui() -> int:
    from polyx.gui.qt import require_qt

    require_qt()
    from polyx.gui.launcher import main as _launcher_main

    _launcher_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        print()
        print(_("Commands:"))
        for name in _COMMANDS:
            print(f"  {name}")
        print()
        print(_("Run 'polyx <command> --help' for command-specific options."))
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "help":
        if rest and rest[0] in _COMMANDS:
            import importlib

            mod = importlib.import_module(_COMMANDS[rest[0]])
            return mod.main(["--help"])
        print(_usage())
        return 0

    if cmd not in _COMMANDS:
        # Backwards-friendly: unknown first token -> try GUI? No, error.
        print(_("Unknown command: %(command)s") % {"command": cmd}, file=sys.stderr)
        print(_("Run 'polyx --help' for the list of commands."), file=sys.stderr)
        return 2

    if cmd == "gui":
        return _launch_gui()

    import importlib

    mod = importlib.import_module(_COMMANDS[cmd])
    return mod.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
