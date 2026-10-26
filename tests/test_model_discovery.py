"""Model discovery seam (M011/S01/T01)."""
from pathlib import Path

import pytest

from polyx.core import paths


def test_root_resolves_to_project_root():
    assert paths.ROOT.name != "src"
    assert (paths.ROOT / "src" / "polyx").is_dir()


def test_default_model_path():
    assert paths.DEFAULT_MODEL == paths.MODELS_DIR / "bestdetectormedium.pt"


def test_discover_default_model_absent(monkeypatch, tmp_path):
    # point DEFAULT_MODEL to a nonexistent path
    monkeypatch.setattr(paths, "DEFAULT_MODEL", tmp_path / "missing.pt")
    assert paths.discover_default_model() is None


def test_discover_default_model_present(tmp_path):
    model = tmp_path / "bestdetectormedium.pt"
    model.write_bytes(b"x")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(paths, "DEFAULT_MODEL", model)
    try:
        assert paths.discover_default_model() == model
    finally:
        monkeypatch.undo()