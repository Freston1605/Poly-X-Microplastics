"""Lossless, compact storage for the raster masks a segmenter produces.

A 4K plate holds hundreds of candidates and a validation split holds every
candidate of every image, so a boolean mask at one byte per pixel costs 16 MiB
and a whole epoch's predictions cost tens of gigabytes once the GPU has
released them. Packed at one bit per pixel the same set is 16 times smaller and
still exact, which matters here: a polygon round-trip loses up to half a pixel
per edge and can turn a perfect mask into a miss at IoU 0.50.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class PackedMask:
    """One boolean image, bit-packed; ``shape`` is the unpacked ``(h, w)``.

    ``eq=False`` on purpose: a generated ``__eq__`` would compare the two numpy
    arrays pixel by pixel and raise instead of answering, so these objects are
    compared by identity and never hashed.
    """

    bits: np.ndarray
    shape: tuple[int, int]

    @classmethod
    def pack(cls, mask: np.ndarray) -> PackedMask:
        """Pack a boolean (or 0/1, or probability) mask without losing a pixel."""
        array = np.asarray(mask)
        if array.dtype != np.bool_:
            array = array >= 0.5
        array = array.reshape(array.shape[-2:])
        return cls(
            bits=np.packbits(array),
            shape=(int(array.shape[0]), int(array.shape[1])),
        )

    def unpack(self) -> np.ndarray:
        """The boolean mask this object packs."""
        height, width = self.shape
        flat = np.unpackbits(self.bits, count=height * width)
        return flat.reshape(self.shape).astype(bool)

    def any(self) -> bool:
        """Whether any pixel is set, without unpacking the whole mask."""
        return bool(self.bits.any())


def as_mask(value: PackedMask | np.ndarray | None) -> np.ndarray | None:
    """A boolean array from either representation, or ``None``."""
    if value is None:
        return None
    if isinstance(value, PackedMask):
        return value.unpack()
    return np.asarray(value)
