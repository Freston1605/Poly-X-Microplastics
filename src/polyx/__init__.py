"""Poly-X — Microplastics detection suite."""

import os as _os
import sys as _sys

__version__ = "2.0.0"
__author__ = "Cristofher Ferrada"


def _ensure_standard_output() -> None:
    """Guarantees that sys.stdout and sys.stderr exist.

    When Poly-X is opened from the Desktop shortcut, it starts under
    ``pythonw.exe`` so no black console window is left open. pythonw leaves
    ``sys.stdout`` and ``sys.stderr`` as ``None``, and the launcher starts each
    module with ``sys.executable``, so the children inherit the same.

    That kills training: Ultralytics draws a ``tqdm`` bar while caching the
    labels and tqdm writes to ``sys.stderr``, which raises
    ``AttributeError: 'NoneType' object has no attribute 'write'``.

    The failure only shows up the **first** time a dataset is used, because on
    later runs the ``.cache`` already exists and that bar is never drawn. That
    is why it goes unnoticed on the development machine and blows up on a
    clean one.

    They are redirected to ``os.devnull``: with no console there is nowhere to
    show that text, and training metrics reach the interface through
    callbacks, not through standard output.
    """
    for name in ("stdout", "stderr"):
        if getattr(_sys, name, None) is None:
            try:
                destination = open(_os.devnull, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
            except OSError:
                continue
            setattr(_sys, name, destination)
            # Some libraries look at the originals instead of the current ones.
            if getattr(_sys, f"__{name}__", None) is None:
                setattr(_sys, f"__{name}__", destination)


_ensure_standard_output()
