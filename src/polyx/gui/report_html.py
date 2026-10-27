"""Compatibility shim."""

import polyx.gui.report as _report
import polyx.gui.report._common as _common

globals().update({k: v for k, v in vars(_report).items() if not k.startswith("__")})
__file__ = _common.__file__
