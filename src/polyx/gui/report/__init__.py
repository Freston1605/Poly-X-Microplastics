"""Report package."""

from ._common import *
from .figures import *
from .sections import *
import polyx.gui.report._common as _common
import polyx.gui.report.figures as _figures
import polyx.gui.report.sections as _sections

_common.__dict__.update({k: v for k, v in vars(_figures).items() if not k.startswith("__")})
_common.__dict__.update({k: v for k, v in vars(_sections).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_common).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_figures).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_sections).items() if not k.startswith("__")})
