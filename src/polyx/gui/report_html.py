"""Compatibility shim over the ``polyx.gui.report`` package."""

import polyx.gui.report as report
from polyx.gui.report import *  # noqa: F403

__all__ = report.__all__
__file__ = report.__file__
