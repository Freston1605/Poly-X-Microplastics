"""One list of supported image formats, and ``.webp`` on every path.

``core.paths.IMAGE_EXTS`` is the only place ``src/polyx`` spells which files
are images. The Viewer and the Labeller each carried their own set without
``.webp``, so a file the Detector had analysed was invisible to them: the
folder listing dropped it and the Viewer's drag & drop refused it. A second
copy of the list cannot disagree with the Detector if there is only one, so
these tests pin the behaviour (both states list and open a ``.webp``) and the
structure (nobody re-spells the extensions anywhere else).
"""

from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
import pytest

from polyx.core.paths import IMAGE_EXTS

pytest.importorskip("PySide6")

from polyx.gui.labeler import state as labeler_state
from polyx.gui.trainer import dominio
from polyx.gui.viewer import state as viewer_state

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "src" / "polyx" / "core" / "paths.py"


def _image(path: Path, suffix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(suffix, np.zeros((40, 60, 3), dtype=np.uint8))
    assert ok
    path.write_bytes(bytes(buf))
    return path


def _extension_literals(path: Path) -> list[tuple[int, set[str]]]:
    """Every collection literal of three or more dotted strings in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Set, ast.Tuple, ast.List)) or not node.elts:
            continue
        values = [e.value for e in node.elts if isinstance(e, ast.Constant)]
        if len(values) != len(node.elts):
            continue
        if len(values) >= 3 and all(isinstance(v, str) and v.startswith(".") for v in values):
            found.append((node.lineno, set(values)))
    return found


def _rebinds_image_exts(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "IMAGE_EXTS" for target in targets):
            return True
    return False


def test_the_viewer_lists_and_opens_a_webp(tmp_path: Path):
    folder = tmp_path / "run"
    _image(folder / "a.webp", ".webp")
    _image(folder / "b.png", ".png")

    state = viewer_state.ViewerState()
    state.load_folder(folder)

    assert folder / "a.webp" in state.images
    assert state.current_image == folder / "a.webp", "the Viewer opens what it lists"

    state.load_single(folder / "a.webp")

    assert state.current_image == folder / "a.webp"


def test_the_labeller_lists_and_opens_a_webp(tmp_path: Path):
    folder = tmp_path / "run"
    _image(folder / "a.webp", ".webp")
    _image(folder / "b.png", ".png")

    state = labeler_state.LabelerState()
    state.load_images(folder)

    assert folder / "a.webp" in state.images
    assert state.current_image == folder / "a.webp", "the Labeller opens what it lists"


def test_the_extensions_are_spelled_once_in_the_package():
    sources = sorted((ROOT / "src" / "polyx").rglob("*.py"))
    redefined = [
        str(p.relative_to(ROOT)) for p in sources if p != CANONICAL and _rebinds_image_exts(p)
    ]
    assert not redefined, f"IMAGE_EXTS must only be defined in core/paths.py: {redefined}"

    respelled = [
        f"{p.relative_to(ROOT)}:{line}"
        for p in sources
        if p != CANONICAL
        for line, extensions in _extension_literals(p)
        if extensions == set(IMAGE_EXTS)
    ]
    assert not respelled, f"spell the extensions through IMAGE_EXTS, not by hand: {respelled}"


def test_the_three_readers_share_the_canonical_tuple():
    assert viewer_state.IMAGE_EXTS is IMAGE_EXTS
    assert labeler_state.IMAGE_EXTS is IMAGE_EXTS
    assert dominio.IMAGE_EXTS is IMAGE_EXTS
