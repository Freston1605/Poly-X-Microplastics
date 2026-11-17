"""Decode an image into the pixmap a canvas paints — one frame everywhere.

Qt's ``QPixmap(path)`` ignores the EXIF orientation tag (``QImageReader``
applies it only when ``setAutoTransform(True)`` is requested), while OpenCV's
``imdecode`` applies it by default. :func:`polyx.core.detect.protocol.size_image`
reports the rotated frame. A canvas that painted the Qt pixmap therefore drew
in a frame the label codec did not share with it: a photo tagged orientation 6
looked right on screen and its boxes normalised in another geometry — displaced
on the detector, sometimes outside the plate.

Every canvas decodes here instead, so the painted pixels, the editor's
dimensions and the saved box share one frame. ``read_image`` is also the only
idiom that reliably opens accented paths on Windows, which is why the rest of
the project already reads that way.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

from polyx.core.image_io import read_image


def qpixmap_from_file(path: Path) -> QPixmap:
    """The image as the frame ``size_image`` measures, or a null pixmap."""
    bgr = read_image(path)
    if bgr is None:
        return QPixmap()
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    # .copy() so the QImage owns its bytes: the numpy array is freed on
    # return, and without the copy the pixmap would point to dead memory.
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
