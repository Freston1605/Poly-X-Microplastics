"""Constants for the morphology measurement layer."""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ASPECT_FIBER",
    "CONTRAST_MINIMUM",
    "CURVE_FROM",
    "DRAIN_X_DIAGONAL",
    "ELONGATION_ELONGATED",
    "Method",
    "Morphotype",
    "SLACK_COMPACT",
    "SLACK_ELONGATED",
    "SOLIDITY_CONVEX",
    "THINNESS_FOR_GEODESIC",
    "THRESHOLD_CORE",
]


class Morphotype(StrEnum):
    """Morphotype of a particle, stored as a string for report compatibility."""

    FIBER = "fibra"
    FRAGMENT = "fragmento"
    INDETERMINATE = "indeterminado"


class Method(StrEnum):
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