"""Compatibility shim."""

import os

import polyx.gui.report as _report
import polyx.gui.report._common as _common

globals().update({k: v for k, v in vars(_report).items() if not k.startswith("__")})
# Keep __file__ pointing at the module that actually holds _generate_report_impl
# so structural tests that read the source keep working after the split.
__file__ = _common.__file__
