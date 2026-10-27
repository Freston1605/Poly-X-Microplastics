"""Compatibility shim over the ``polyx.gui.report`` package."""

import polyx.gui.report as report

from polyx.gui.report import *  # noqa: F401,F403

__all__ = report.__all__

# Keep __file__ pointing to the report package directory for CSS resolution
__file__ = report.__file__
