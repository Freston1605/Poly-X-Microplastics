"""Plate annotation — how Poly-X draws detections (and ground truth) on a plate.

One owner for what the drawing paths re-derive per copy today: the class
palette (theme-independent domain data — it must never change with the
interface theme), the scale-aware thickness policy (fixed 2 px lines went
invisible on full plates: a 4096 px photo is rescaled to 1100 to embed in
the report, and the line ended up measuring half a pixel), the run-view
naming contract (written and parsed from one place so the Viewer cannot
desync from the runner), and the drawn views themselves: the primitive
draws boxes + labels under a style; the triptych composes the three views
of a run — predictions, ground truth, combined. The Detector runner, the
GT-reload path, the Viewer's export and the headless batch all draw
through this owner.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from polyx.core.detect.protocol import Instance

# ════════════════════════════════════════════════════════════════════
#  Domain colours: the same in every theme
# ════════════════════════════════════════════════════════════════════
CLASS_COLOR_HEX = {
    "PET": "#e3342f",  # drawn red
    "PP": "#ff8c00",  # drawn orange
    "LDPE": "#ffd700",  # drawn yellow
    "PE": "#ffd700",  # alias
}

# The one fallback for classes outside the palette (unknown ids, free-form
# names). Before this module there were four; unknown meant differently-
# coloured depending on the door that drew the plate.
FALLBACK_COLOR_HEX = "#33aaff"

# The ground-truth overlay in the combined view: its colour and label
# prefix belong here, not to whichever caller drew it second.
GT_COLOR_BGR = (255, 200, 80)


def bgr_for_class(name: str) -> tuple[int, int, int]:
    """Hex palette → BGR, because the plates are BGR.

    Unknown classes get :data:`FALLBACK_COLOR_HEX` everywhere, so a
    missing class id is recognisable instead of differently-coloured per
    module.
    """
    hex_ = CLASS_COLOR_HEX.get(name, FALLBACK_COLOR_HEX).lstrip("#")
    r, g, b = int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)
    return (b, g, r)


def line_width(img_shape: tuple[int, ...]) -> int:
    """Box outline thickness, scaled with the plate's longer side.

    A fixed 2 px width went invisible on full plates; scaling with the
    longer side, a 3260 px plate draws ~5 px, a 4096 px one ~6, and a
    1630 px crop keeps 2. It never drops below 2.
    """
    side = max(img_shape[:2])
    return max(2, int(round(side / 700)))  # noqa: RUF046


# The suffixes the runner writes for the three views of a run. Stripped
# longest-first when parsing. The format has always had one wrinkle: a
# stem that itself ends in a view suffix parses as that suffix — inherent,
# not new.
VIEW_SUFFIXES = ("_annot", "_pred", "_gt", "")

_RUN_VIEW_RE = re.compile(r"^(\d+)_(.+)\.png$")
_RUN_LABEL_RE = re.compile(r"^(\d+)_(.+)\.txt$")


def _run_stem(index: int, stem: str) -> str:
    """The index/stem prefix shared by every file of a run."""
    return f"{index:04d}_{stem}"


def run_view_path(folder: Path, index: int, stem: str, suffix: str = "") -> Path:
    """The name of one annotated view inside a run folder.

    One owner for the contract the Viewer parses back: per run view,
    ``{index:04d}_{stem}{suffix}.png``. The index comes before everything
    else because two photos from different folders can share a stem.
    """
    return folder / f"{_run_stem(index, stem)}{suffix}.png"


def run_label_path(folder: Path, index: int, stem: str) -> Path:
    """The prediction label of one image inside a run folder.

    Same index/stem contract as :func:`run_view_path`; the label is the
    ``.txt`` the Viewer loads, not a drawn view.
    """
    return folder / f"{_run_stem(index, stem)}.txt"


def parse_run_label_name(name: str) -> tuple[int, str] | None:
    """Parse a run label's file name back into (index, stem).

    Returns ``None`` for anything that is not one: the caller decides
    whether that is an error or something to skip.
    """
    match = _RUN_LABEL_RE.match(name)
    if match is None:
        return None
    return (int(match.group(1)), match.group(2))


def parse_run_view_name(name: str) -> tuple[int, str, str] | None:
    """Parse a run view's file name back into (index, stem, suffix).

    Returns ``None`` for anything that is not one: the caller decides
    whether that is an error or something to skip.
    """
    match = _RUN_VIEW_RE.match(name)
    if match is None:
        return None
    index, rest = int(match.group(1)), match.group(2)
    # Longest first, so a stem that itself ends in a suffix is stripped
    # unambiguously rather than relying on the tuple's declaration order.
    for suffix in sorted(VIEW_SUFFIXES, key=len, reverse=True):
        if rest.endswith(suffix):
            return (index, rest[: len(rest) - len(suffix)], suffix)
    return None


# ════════════════════════════════════════════════════════════════════
#  Drawing: the primitive and the triptych
# ════════════════════════════════════════════════════════════════════

import cv2  # noqa: E402
import numpy as np  # noqa: E402


class Triptych(NamedTuple):
    """The three views of a run over one plate."""

    pred: np.ndarray
    gt: np.ndarray | None
    combined: np.ndarray


def draw_detections(
    img_bgr: np.ndarray,
    dets: list[Instance],
    color_for=None,
    label_for=None,
) -> np.ndarray:
    """Draw boxes + labels over one plate, returning a new image.

    The single thickness policy of the program: scale-aware, because a
    fixed 2 px width went invisible on full plates. Labels picture a tag
    above the box, white text on the class colour, and the particle's
    number comes before everything else — it is what one looks for when
    coming from the report table.

    ``color_for`` / ``label_for`` are the style: by default every
    detection gets its class colour and its class name as label. The
    triptych composes the run-view conventions on top; other callers
    (the Viewer's measured sizes) inject theirs.
    """
    color_map = color_for or (lambda d: bgr_for_class(d.class_name))
    label_map = label_for or (lambda d: d.class_name)

    out = img_bgr.copy()
    side = max(out.shape[:2])
    thickness = line_width(out.shape)
    scale_txt = max(0.5, side / 3000)
    thickness_txt = max(1, int(round(scale_txt * 2)))  # noqa: RUF046

    for d in dets:
        c = color_map(d)
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)
        label = label_map(d)
        # The number comes before everything else: it is what one looks for when coming
        # from the report table.
        if d.number:
            label = f"{d.number}. {label}" if label else str(d.number)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale_txt, thickness_txt)
            margin = max(4, thickness * 2)
            cv2.rectangle(out, (x1, y1 - th - margin), (x1 + tw + margin, y1), c, -1)
            cv2.putText(
                out,
                label,
                (x1 + margin // 2, y1 - margin // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale_txt,
                (255, 255, 255),
                thickness_txt,
                cv2.LINE_AA,
            )
    return out


def triptych(
    img_bgr: np.ndarray,
    dets: list[Instance],
    gt: list[Instance],
    color_for=None,
    pred_label_for=None,
    gt_label_for=None,
    gt_prefix: str = "GT",
    has_gt: bool | None = None,
) -> Triptych:
    """The three views of a run: predictions, ground truth, combined.

    The conventions are the run's: the predictions view labels ``class
    conf``, the GT view labels ``class`` in the class colours, and the
    combined view overlays the GT style — the GT colour and a
    ``{gt_prefix} `` label — over the predictions view. Without ground
    truth there is no GT view and the combined view IS the predictions
    view, the same array.

    ``has_gt`` distinguishes "no ground-truth file" from "a ground-truth
    file with zero boxes": a reviewed plate with nothing to count still
    has a GT view (the untouched plate), and its combined view is the
    predictions view. The default infers it from ``gt`` for callers that
    do not know the difference.

    Callers can restyle the predictions view (``color_for``,
    ``pred_label_for``) and the GT label text (``gt_label_for``); the GT
    view keeps the class colours and the combined view keeps the GT colour
    and the ``{gt_prefix} `` prefix regardless of injection, so the run
    conventions survive restyling.
    """
    pred = draw_detections(
        img_bgr,
        dets,
        color_for=color_for,
        label_for=pred_label_for or (lambda d: f"{d.class_name} {d.conf:.2f}"),
    )
    if has_gt is None:
        has_gt = bool(gt)
    if not has_gt:
        return Triptych(pred=pred, gt=None, combined=pred)
    # The GT view keeps the class colours regardless of any injected colour
    # mapping: the spec assigns class colours to the ground truth.
    gt_base_label = gt_label_for or (lambda d: d.class_name)
    gt_view = draw_detections(
        img_bgr,
        gt,
        label_for=gt_base_label,
    )
    combined = draw_detections(
        pred,
        gt,
        color_for=lambda d: GT_COLOR_BGR,
        label_for=lambda d: f"{gt_prefix} {gt_base_label(d)}",
    )
    return Triptych(pred=pred, gt=gt_view, combined=combined)


def save_view(
    img_bgr: np.ndarray, folder: Path, index: int, stem: str, suffix: str = ""
) -> Path | None:
    """Encode one view as PNG and write it under the run-view naming.

    Returns the path, or ``None`` when the encoding fails — callers keep
    working without the PNG.
    """
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return None
    out = run_view_path(folder, index, stem, suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(buf))
    return out


__all__ = [
    "CLASS_COLOR_HEX",
    "FALLBACK_COLOR_HEX",
    "GT_COLOR_BGR",
    "VIEW_SUFFIXES",
    "Triptych",
    "bgr_for_class",
    "draw_detections",
    "line_width",
    "parse_run_label_name",
    "parse_run_view_name",
    "run_label_path",
    "run_view_path",
    "save_view",
    "triptych",
]
