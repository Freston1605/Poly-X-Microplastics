"""Compatibility shim: re-exports the morphology submodules.

The implementation now lives in ``segmentation``, ``geodesic``, ``shape`` and
``drawing``; this module keeps the historical import paths working
(``from polyx.core import morphology as M``, ``from polyx.core.measure.morphology import ...``).
"""

from __future__ import annotations

from .morphology_constants import *
from .segmentation import *
from .geodesic import *
from .shape import *
from .drawing import *

# Re-export everything the submodules expose.
from .morphology_constants import __all__ as _c
from .segmentation import __all__ as _s
from .geodesic import __all__ as _g
from .shape import __all__ as _sh
from .drawing import __all__ as _d

__all__ = _c + _s + _g + _sh + _d