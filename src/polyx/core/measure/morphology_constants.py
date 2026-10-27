# Morphology constants (extracted).

from __future__ import annotations

from enum import Enum

__all__ = [
    "FIBER",
    "FRAGMENT",
    "INDETERMINATE",
    "Morphotype",
    "Method",
    "ASPECT_FIBER",
    "THINNESS_FOR_GEODESIC",
    "SOLIDITY_CONVEX",
    "THRESHOLD_CORE",
    "CURVE_FROM",
    "ELONGATION_ELONGATED",
    "CONTRAST_MINIMUM",
    "SLACK_COMPACT",
    "SLACK_ELONGATED",
    "DRAIN_X_DIAGONAL",
]

import numpy as np

class Morphotype(str, Enum):
    """Morphotype of a particle, stored as a string for report compatibility."""

    FIBER = "fibra"
    FRAGMENT = "fragmento"
    INDETERMINATE = "indeterminado"


FIBER = Morphotype.FIBER
FRAGMENT = Morphotype.FRAGMENT
INDETERMINATE = Morphotype.INDETERMINATE


class Method(str, Enum):
    """How the reported length was obtained."""

    GEODESIC = "geodesico"
    FERET = "Feret maximo"


ASPECT_FIBER = 3.0


THINNESS_FOR_GEODESIC = 4.0


SOLIDITY_CONVEX = 0.90


THRESHOLD_CORE = 0.80


CURVE_FROM = 1.15


ELONGATION_ELONGATED = 3.0


CONTRAST_MINIMUM = 12


SLACK_COMPACT = 0.10


SLACK_ELONGATED = 0.25


DRAIN_X_DIAGONAL = 1.5

