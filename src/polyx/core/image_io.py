"""Small helpers for reading/writing images through OpenCV.

Centralised here so the ``cv2.imdecode`` over ``np.fromfile`` idiom (the only
thing that survives non-ASCII paths on Windows) lives in one place instead of
being copy-pasted at every call site.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image from disk, returning a BGR ``ndarray`` or ``None``."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def decode_image_bytes(data: bytes | bytearray | memoryview) -> np.ndarray | None:
    """Decode an in-memory buffer into a BGR ``ndarray`` (or ``None``)."""
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None
