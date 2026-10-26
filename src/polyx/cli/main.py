"""Unified ``polyx`` command-line entry point.

Usage:
    polyx                      # launch the GUI
    polyx gui                  # launch the GUI
    polyx train ...            # headless training
    polyx detect ...           # headless detection
    polyx eval ...             # headless evaluation
    polyx report ...           # compact HTML report
    polyx download-models ...  # download weights
"""

from __future__ import annotations

import sys

_COMMANDS = {
    "gui": "polyx.gui.__main__",
    "train": "polyx.cli.train",
    "detect": "polyx.cli.detect",
    "eval": "polyx.cli.eval",
    "report": "polyx.cli.report",
    "download-models": "polyx.cli.download_models",
}


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
        print(__doc__.strip())
        print()
        print("Commands:")
        for name in _COMMANDS:
            print(f"  {name}")
        print()
        print("Run 'polyx <command> --help' for command-specific options.")
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "help":
        if rest and rest[0] in _COMMANDS:
            import importlib

            mod = importlib.import_module(_COMMANDS[rest[0]])
            return mod.main(["--help"])
        print(__doc__.strip())
        return 0

    if cmd not in _COMMANDS:
        # Backwards-friendly: unknown first token -> try GUI? No, error.
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print("Run 'polyx --help' for the list of commands.", file=sys.stderr)
        return 2

    if cmd == "gui":
        return _launch_gui()

    import importlib

    mod = importlib.import_module(_COMMANDS[cmd])
    return mod.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
