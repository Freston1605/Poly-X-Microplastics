"""Stable warning codes persisted in artifacts, with structured parameters.

Mirrors the refusal-code pattern in :mod:`polyx.core.measure.calibration`: the
core names the warning with a stable English code and keeps numbers and nested
causes in named fields. No translated sentence is ever stored in an artifact;
the interface chooses the wording at display time from :func:`warning_sentence`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from polyx.core.i18n import _

# ── Calibration warning codes ─────────────────────────────────────────────

IMAGE_UNREADABLE = "image_unreadable"
NOMINAL_DIAMETER_INVALID = "nominal_diameter_invalid"
PLATE_NOT_FOUND = "plate_not_found"
RING_FIT_FAILED = "ring_fit_failed"
ZERO_ADJUSTED_RADIUS = "zero_adjusted_radius"
DETOUR_HOUGH = "detour_hough"
PLATE_MEASURE_FAILED = "plate_measure_failed"
MANUAL_SCALE_MISSING = "manual_scale_missing"

# ── Shape warning codes ───────────────────────────────────────────────────

NO_MASK = "no_mask"
NO_CONTOUR = "no_contour"
PARTICLE_TOO_SMALL = "particle_too_small"
SEGMENTATION_FAILED = "segmentation_failed"
MASK_CLIPPED = "mask_clipped"
OPENCV_ERROR = "opencv_error"

#: Every machine warning code this module can produce. A caller that maps codes
#: to sentences iterates this tuple, so a new code cannot silently fall through
#: to a generic sentence (the same contract as ``FAILURE_CODES``).
WARNING_CODES = (
    IMAGE_UNREADABLE,
    NOMINAL_DIAMETER_INVALID,
    PLATE_NOT_FOUND,
    RING_FIT_FAILED,
    ZERO_ADJUSTED_RADIUS,
    DETOUR_HOUGH,
    PLATE_MEASURE_FAILED,
    MANUAL_SCALE_MISSING,
    NO_MASK,
    NO_CONTOUR,
    PARTICLE_TOO_SMALL,
    SEGMENTATION_FAILED,
    MASK_CLIPPED,
    OPENCV_ERROR,
)

# Fields carried beside the code. Numbers and nested causes travel here, never
# interpolated into the code or into a prose sentence.
_PARAM_FIELDS = ("detour_pct", "cause", "excess")


@dataclass(frozen=True)
class Warning:
    """One warning: a stable code and the structured values behind it."""

    code: str
    detour_pct: float | None = None
    cause: str | None = None
    excess: float | None = None

    def __bool__(self) -> bool:
        return bool(self.code)

    def to_dict(self) -> dict:
        """JSON form; only the parameters that actually occurred are written."""
        data: dict = {"code": self.code}
        for name in _PARAM_FIELDS:
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data


def warning_from_dict(value: object) -> Warning | None:
    """Rebuild a :class:`Warning` from its JSON form (``None`` stays ``None``)."""
    if value is None:
        return None
    if isinstance(value, Warning):
        return value
    if isinstance(value, str):
        # A pre-canonical string was prose; it is no longer read as a warning.
        return None
    if isinstance(value, dict):
        code = value.get("code")
        if not isinstance(code, str) or not code:
            return None
        params = {name: value[name] for name in _PARAM_FIELDS if name in value}
        return Warning(code, **params)
    return None


def warning_to_dict(value: Warning | dict | None) -> dict | None:
    """The JSON form of a warning held in memory or read from a summary list."""
    if value is None:
        return None
    if isinstance(value, Warning):
        return value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    return None


def count_by_code(warnings: Iterable[Warning | dict]) -> dict[str, int]:
    """Group and count warnings by their stable code."""
    counts: dict[str, int] = {}
    for warning in warnings:
        code = warning["code"] if isinstance(warning, dict) else warning.code
        counts[code] = counts.get(code, 0) + 1
    return counts


def warning_sentence(warning: Warning | dict | str | None) -> str:
    """The English gettext sentence for a warning, chosen at display time.

    The mapping is built inside the function so the call resolves through the
    active catalog: switching language takes effect without reloading code.
    Numbers are formatted into named placeholders; the stored code never
    contains prose.
    """
    if warning is None:
        return ""
    if isinstance(warning, str):
        code, params = warning, {}
    elif isinstance(warning, dict):
        code, params = (
            str(warning.get("code") or ""),
            {key: value for key, value in warning.items() if key != "code"},
        )
    else:
        code, params = (
            warning.code,
            {
                name: getattr(warning, name)
                for name in _PARAM_FIELDS
                if getattr(warning, name) is not None
            },
        )

    messages = {
        IMAGE_UNREADABLE: _("The image could not be read or is empty."),
        NOMINAL_DIAMETER_INVALID: _("The nominal plate diameter must be greater than zero."),
        PLATE_NOT_FOUND: _("The plate was not found in the photograph."),
        RING_FIT_FAILED: _(
            "The fine ring fit failed; the scale was taken from Hough and is unreliable."
        ),
        ZERO_ADJUSTED_RADIUS: _("The adjusted radius is zero."),
        DETOUR_HOUGH: _("Hough differed from the fine fit by %(detour_pct)s%%."),
        PLATE_MEASURE_FAILED: _("The plate could not be measured: %(cause)s."),
        MANUAL_SCALE_MISSING: _(
            "The manual calibration needs a positive micrometres-per-pixel value."
        ),
        NO_MASK: _("No mask was produced for the particle."),
        NO_CONTOUR: _("No contour was found in the particle mask."),
        PARTICLE_TOO_SMALL: _("The particle is too small to measure its shape."),
        SEGMENTATION_FAILED: _("The particle could not be separated from the background."),
        MASK_CLIPPED: _(
            "The mask extended %(excess)s times its box diagonal and was clipped to the box; "
            "it may be a touching neighbour, so the size is underestimated."
        ),
        OPENCV_ERROR: _("OpenCV failed while segmenting: %(cause)s"),
    }
    message = messages.get(code)
    if message is None:
        return _("An unrecognised warning was recorded.")

    # The per-code formatters sit beside the messages and are built inside the
    # call for the same reason: the active catalog is resolved at display time.
    # A code with no structured fields maps to no formatter and keeps its
    # message verbatim.
    formatters = {
        PLATE_MEASURE_FAILED: lambda msg, values: (
            msg % {"cause": warning_sentence(values.get("cause"))}
        ),
        DETOUR_HOUGH: lambda msg, values: (
            msg % {"detour_pct": f"{float(values.get('detour_pct') or 0.0):.1f}"}
        ),
        MASK_CLIPPED: lambda msg, values: (
            msg % {"excess": f"{float(values.get('excess') or 0.0):.1f}"}
        ),
        OPENCV_ERROR: lambda msg, values: msg % {"cause": str(values.get("cause") or "")},
    }
    formatter = formatters.get(code)
    return formatter(message, params) if formatter is not None else message


__all__ = [
    "DETOUR_HOUGH",
    "IMAGE_UNREADABLE",
    "MANUAL_SCALE_MISSING",
    "MASK_CLIPPED",
    "NOMINAL_DIAMETER_INVALID",
    "NO_CONTOUR",
    "NO_MASK",
    "OPENCV_ERROR",
    "PARTICLE_TOO_SMALL",
    "PLATE_MEASURE_FAILED",
    "PLATE_NOT_FOUND",
    "RING_FIT_FAILED",
    "SEGMENTATION_FAILED",
    "WARNING_CODES",
    "ZERO_ADJUSTED_RADIUS",
    "Warning",
    "count_by_code",
    "warning_from_dict",
    "warning_sentence",
    "warning_to_dict",
]
