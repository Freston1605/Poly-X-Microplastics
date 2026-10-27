from __future__ import annotations

import cv2
import numpy as np
from .morphology_constants import *


def _spread_geodesic(inside: np.ndarray, seed: tuple) -> np.ndarray:
    """Geodesic distance from a seed, staying inside the mask.

    chamfer 3x3 successive-minima sweep over the 8 neighbours, weight 1 on
    straight and sqrt(2)-ish on diagonal. Vectorized: each iteration is eight
    shifted adds, and as many are needed as the longest path has pixels.

    Hot-loop buffer design (S02 rewrite): the loop allocates NOTHING on the
    steady state. All scratch is preallocated once (candidate buffer ``p``,
    update buffer ``e``, cached ``fuera`` mask, three slab temps for the
    shifted adds), every shifted add writes into a preallocated ``out=`` view,
    and the convergence update happens in-place into ``e`` followed by a
    reference swap instead of a full copy. Allocation churn was the dominant
    avoidable cost at ~5.9 ms/particle measured on a 10-plate batch.

    Bit-identical contract, pinned by tests/test_geodesia.py:
      * float32 weights np.float32(0.95509) / np.float32(1.36930), UNCHANGED;
      * the 8 shifted minimums accumulate into ``p`` in the SAME arange order
        as the pre-rewrite body (rectos then diagonals, bottom-up then top-down)
        — reordering is not free: float minimum is exact but the accumulator
        order is part of the frozen behavior;
      * ``tope = int(dentro.sum()) + 2`` iteration budget, unchanged;
      * convergence breaks on ``np.array_equal`` between successive sweeps,
        bit-exactly as before.

    Done this way rather than with scipy.sparse.csgraph because scipy is not
    a declared project dependency: it arrives transitively with ultralytics,
    and relying on it would break the install the day ultralytics stops
    shipping it.
    """
    INF = np.inf
    H, W = inside.shape
    d = np.full(inside.shape, INF, dtype=np.float32)
    d[seed] = 0.0
    # Optimal 3x3 chamfer weights, not 1 and sqrt(2). With the naive weights the
    # staircase path overestimates the Euclidean distance by up to 8% -- measured
    # on a circle of known diameter, it gave 108 instead of 100 --. These
    # two minimize that maximum error and leave it around 1.4%.
    UNO = np.float32(0.95509)
    R2 = np.float32(1.36930)
    # Safety bound: no simple path can have more steps than pixels.
    max_size = int(inside.sum()) + 2
    # Buffers preallocated once: p (candidates), e (update), the
    # cached inverse of `dentro` and three slabs for the shifted addends
    # (row, column and diagonal). None of this is allocated inside the loop.
    p = np.full_like(d, INF)
    e = np.empty_like(d)
    outside = ~inside  # cached: `~dentro` is constant throughout the sweep
    t_row = np.empty((max(H - 1, 0), W), dtype=np.float32)
    t_col = np.empty((H, max(W - 1, 0)), dtype=np.float32)
    t_diag = np.empty((max(H - 1, 0), max(W - 1, 0)), dtype=np.float32)
    for _ in range(max_size):
        p.fill(INF)
        # Straight
        np.add(d[:-1, :], UNO, out=t_row[: H - 1])
        np.minimum(p[1:, :], t_row[: H - 1], out=p[1:, :])
        np.add(d[1:, :], UNO, out=t_row[: H - 1])
        np.minimum(p[:-1, :], t_row[: H - 1], out=p[:-1, :])
        np.add(d[:, :-1], UNO, out=t_col[:, : W - 1])
        np.minimum(p[:, 1:], t_col[:, : W - 1], out=p[:, 1:])
        np.add(d[:, 1:], UNO, out=t_col[:, : W - 1])
        np.minimum(p[:, :-1], t_col[:, : W - 1], out=p[:, :-1])
        # Diagonal
        np.add(d[:-1, :-1], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[1:, 1:], t_diag[: H - 1, : W - 1], out=p[1:, 1:])
        np.add(d[:-1, 1:], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[1:, :-1], t_diag[: H - 1, : W - 1], out=p[1:, :-1])
        np.add(d[1:, :-1], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[:-1, 1:], t_diag[: H - 1, : W - 1], out=p[:-1, 1:])
        np.add(d[1:, 1:], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[:-1, :-1], t_diag[: H - 1, : W - 1], out=p[:-1, :-1])
        p[outside] = INF
        # In-place update with reference swap, without copying the array.
        np.minimum(d, p, out=e)
        # Checking against the previous step is enough: the propagation is monotonically
        # decreasing, so if nothing went down, it has converged.
        if np.array_equal(e, d):
            break
        d, e = e, d
    return d


def length_geodesic(mask: np.ndarray) -> float:
    """Longest path that fits INSIDE the particle, following its shape.

    It is the geodesic diameter. It does the right thing in the two cases that
    matter, without changing the formula:

      * Curved fiber: the path cannot leave the particle, so it follows the
        curve and returns its length, not the chord between the endpoints.
      * Compact particle: the longest path is the straight one, and it
        coincides with the maximum Feret diameter.

    It is obtained with the double-propagation technique, the same one that
    gives the diameter of a tree: from any point the farthest one is found,
    and from that one it propagates again; the maximum of that second
    propagation is the diameter. It does not depend on the perimeter, so a
    jagged edge does not inflate it the way the rectangle model does.
    """
    idx = np.argwhere(mask > 0)
    if len(idx) < 2:
        return 0.0
    # Crop to the minimum rectangle containing the particle. The cost of the
    # propagation goes with the canvas area, and a 30 px particle inside
    # a 4096 plate would cost the same as the whole plate.
    y0, x0 = idx.min(axis=0)
    y1_, x1_ = idx.max(axis=0) + 1
    inside = mask[y0:y1_, x0:x1_] > 0
    idx = np.argwhere(inside)

    d1 = _spread_geodesic(inside, tuple(idx[0]))
    finite = np.isfinite(d1)
    if not finite.any():
        return 0.0
    a = np.unravel_index(np.argmax(np.where(finite, d1, -1)), d1.shape)

    d2 = _spread_geodesic(inside, a)
    finite2 = np.isfinite(d2)
    return float(d2[finite2].max()) if finite2.any() else 0.0


def path_geodesic(mask: np.ndarray):
    """The points of the longest geodesic path, so it can be drawn.

    Same double propagation as ``largo_geodesico``, but returning the route
    instead of only its length. It is reconstructed by always stepping to the
    lowest-distance neighbour from the farthest endpoint.

    It exists so the report can SHOW along which route each fiber was
    measured. A size that cannot be seen measured cannot be verified.
    """
    idx = np.argwhere(mask > 0)
    if len(idx) < 2:
        return []
    y0, x0 = idx.min(axis=0)
    y1, x1 = idx.max(axis=0) + 1
    inside = mask[y0:y1, x0:x1] > 0
    ii = np.argwhere(inside)
    d1 = _spread_geodesic(inside, tuple(ii[0]))
    if not np.isfinite(d1).any():
        return []
    a = np.unravel_index(np.argmax(np.where(np.isfinite(d1), d1, -1)), d1.shape)
    d2 = _spread_geodesic(inside, a)
    if not np.isfinite(d2).any():
        return []
    b = np.unravel_index(np.argmax(np.where(np.isfinite(d2), d2, -1)), d2.shape)

    path, p = [b], b
    for _ in range(int(inside.sum()) + 2):
        if p == a:
            break
        best, minor = None, d2[p]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (p[0] + dy, p[1] + dx)
                if 0 <= q[0] < d2.shape[0] and 0 <= q[1] < d2.shape[1] and d2[q] < minor:
                    minor, best = d2[q], q
        if best is None:
            break
        p = best
        path.append(p)
    return [(int(c[1] + x0), int(c[0] + y0)) for c in path]

__all__ = ['_spread_geodesic', 'length_geodesic', 'path_geodesic']
