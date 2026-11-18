"""One tiling decision, one plan and one description.

Whether to tile a photo, at which size, and how to phrase the result used to
live in three modules: the policy in ``protocol``, the sentence in the run
engine and a second sentence in each preview. A change to the plan or to the
wording had to be made in three places, and the run record stored prose
instead of the decision.

This module owns the geometry (:func:`policy_tiling`), the decision
(:func:`decide_tiling`) and the sentence (:func:`describe_tiling`). Callers
either get a :class:`TilingPlan` in hand, or call ``describe()`` while
rendering; nothing else formats the tile count, the tile side or the overlap.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass

from polyx.core.i18n import ngettext
from polyx.core.provenance import logger_for


class TilingMode(enum.StrEnum):
    """What the run is allowed to decide about tiling.

    A ``str`` enum so ``json.dumps`` keeps writing ``"always"`` and a caller
    can still pass a plain string. The three English values are the only modes;
    there are no legacy spellings and :meth:`parse` never translates one.
    """

    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"

    @classmethod
    def parse(cls, value: object) -> TilingMode:
        """Normalise a mode from any boundary; an unknown value falls back to AUTO.

        Only the three English values are modes. A legacy spelling is not
        translated: it is an unknown value, logged with the canonical choices
        and treated as ``AUTO`` so a bad cell cannot crash the program.
        """
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower() if value is not None else ""
        if text in cls._value2member_map_:
            return cls(text)
        if text:
            logger_for("detector").warning(
                "tiling.unknown_mode value=%r (choose %s)",
                value,
                ", ".join(mode.value for mode in cls),
            )
        return cls.AUTO


#: Every text a boundary accepts as a mode: the English values, nothing else.
#: The command line validates against this exact tuple.
TILING_VALUES: tuple[str, ...] = tuple(mode.value for mode in TilingMode)


@dataclass(frozen=True)
class TilingPlan:
    """The geometry of one tiled photo.

    ``threshold_px`` is the threshold the policy actually compared against:
    ``0`` means the mode forced the decision and the tile side came from
    ``imgsz`` alone. ``side`` is the photo's longer side.
    """

    tile: int
    overlap: float
    n_tiles: int
    side: int
    threshold_px: int

    def describe(self) -> str:
        """Phrase this plan on its own, for a record that kept the numbers.

        The :class:`TilingDecision` also carries the photo's two dimensions and
        phrases them as ``W×H``; a persisted plan only has the longer side,
        which is the one the policy compares. Translated when called, never at
        import time.
        """
        return ngettext(
            "%(side)d px longest side → %(tiles)d tile of %(tile)d px, %(overlap)d%% overlap",
            "%(side)d px longest side → %(tiles)d tiles of %(tile)d px, %(overlap)d%% overlap",
            self.n_tiles,
        ) % {
            "side": int(self.side),
            "tiles": int(self.n_tiles),
            "tile": int(self.tile),
            "overlap": round(self.overlap * 100),
        }


@dataclass(frozen=True)
class TilingDecision:
    """What tiling one photo got, and why."""

    mode: TilingMode
    width: int
    height: int
    plan: TilingPlan | None  # ``None`` = a single pass over the whole photo

    @property
    def tiled(self) -> bool:
        return self.plan is not None

    @classmethod
    def from_record(
        cls, record: Mapping[str, object] | None, mode: TilingMode | str
    ) -> TilingDecision:
        """Rebuild the decision a backend reported through ``predict_auto``.

        The record is the contract between the engine and a tiling backend; a
        backend that does not tile leaves it empty, which reads as a single
        pass. A record whose ``plan`` is not a :class:`TilingPlan` is also a
        single pass: the engine must never record half a decision.
        """
        data = record or {}
        plan = data.get("plan")
        width = data.get("width")
        height = data.get("height")
        tiled = bool(data.get("tiled")) and isinstance(plan, TilingPlan)
        return cls(
            mode=TilingMode.parse(mode),
            width=width if isinstance(width, int) else 0,
            height=height if isinstance(height, int) else 0,
            plan=plan if tiled else None,
        )

    def describe(self) -> str:
        """The one sentence that phrases this decision (see :func:`describe_tiling`)."""
        return describe_tiling(self)


def policy_tiling(
    width: int,
    height: int,
    imgsz: int,
    threshold_px: int = 2000,
    tile: int = 0,
    overlap: float = 0.25,
) -> TilingPlan | None:
    """Decides whether the photo gets tiled, and with which geometry, from its
    size alone.

        ``None`` means a single pass over the full image.

        The threshold is not compared against the area but against the **longer
        side**, because what makes a particle disappear is the rescale to
        ``imgsz``: a 3260 px photo entering at 2080 shrinks every particle to
        0.64x, and below the network's stride it stops existing. Tiling avoids
        that rescale.

        With ``tile=0`` the tile side comes from ``min(threshold_px, imgsz)``:
        this way the crop enters the network without being shrunk, which is
        exactly the goal.

        WHETHER to tile and AT WHICH SIZE are two different questions, and
        mixing them up caused a failure: to force tiling, ``decide_tiling``
        passes ``threshold_px=0`` -- its way of saying "do not decide, tile" --,
        and that yielded ``min(0, imgsz) = 0``, which the clamp raised to 256.
        On a 4096x3072 photo that is 336 tiles of 256 px: extremely slow and
        with ZERO detections, because each tile gets rescaled 8x up to imgsz and
        the model sees nothing resembling what it was trained on. With
        ``threshold_px`` at 0 the size is governed by imgsz, the only thing that
        truly determines it.
    """
    side = max(int(width), int(height))
    if side <= int(threshold_px):
        return None
    if int(tile) > 0:
        t = int(tile)
    else:
        candidate_sides = [v for v in (int(threshold_px), int(imgsz)) if v > 0]
        t = min(candidate_sides) if candidate_sides else 1280
    t = max(256, min(t, side))
    overlap = min(max(float(overlap), 0.0), 0.9)
    step = max(1, int(round(t * (1.0 - overlap))))  # noqa: RUF046
    n = len(_tile_starts(int(width), t, step)) * len(_tile_starts(int(height), t, step))
    return TilingPlan(
        tile=t,
        overlap=overlap,
        n_tiles=n,
        side=side,
        threshold_px=int(threshold_px),
    )


def decide_tiling(
    width: int,
    height: int,
    imgsz: int,
    *,
    mode: TilingMode | str,
    threshold_px: int,
    tile: int = 0,
    overlap: float = 0.25,
) -> TilingDecision:
    """Resolve the mode and the photo size into one :class:`TilingDecision`.

    ``ALWAYS`` forces the decision by passing ``threshold_px=0`` to the
    geometry, exactly as the old backend did, so forcing keeps skipping only
    the decision and never changes the crop. ``NEVER`` skips the geometry.
    """
    resolved = TilingMode.parse(mode)
    if resolved is TilingMode.NEVER:
        return TilingDecision(mode=resolved, width=int(width), height=int(height), plan=None)
    effective_threshold = 0 if resolved is TilingMode.ALWAYS else int(threshold_px)
    plan = policy_tiling(
        width,
        height,
        imgsz,
        threshold_px=effective_threshold,
        tile=tile,
        overlap=overlap,
    )
    return TilingDecision(mode=resolved, width=int(width), height=int(height), plan=plan)


def describe_tiling(decision: TilingDecision) -> str:
    """The sentence the engine and both previews show for a decision.

    Translated when called, never at import time (the language can change
    while the program runs). An untiled decision has no plan to phrase, so the
    sentence is empty; each page adds its own reason in its own context
    (threshold, disabled, ...), but none of them formats the tile count, the
    tile side or the overlap again.
    """
    plan = decision.plan
    if plan is None:
        return ""
    return ngettext(
        "%(width)d×%(height)d px → %(tiles)d tile of %(tile)d px, %(overlap)d%% overlap",
        "%(width)d×%(height)d px → %(tiles)d tiles of %(tile)d px, %(overlap)d%% overlap",
        plan.n_tiles,
    ) % {
        "width": int(decision.width),
        "height": int(decision.height),
        "tiles": int(plan.n_tiles),
        "tile": int(plan.tile),
        "overlap": round(plan.overlap * 100),
    }


def _tile_starts(total: int, tile: int, step: int) -> list[int]:
    """Start positions of the tiles along one axis (the last one flush with the edge)."""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, step))
    if not starts or starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


__all__ = [
    "TILING_VALUES",
    "TilingDecision",
    "TilingMode",
    "TilingPlan",
    "decide_tiling",
    "describe_tiling",
    "policy_tiling",
]
