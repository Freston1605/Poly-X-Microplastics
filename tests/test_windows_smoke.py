"""Smoke tests for GUI windows: catch broken late imports (e.g. renamed pages)."""

from polyx.gui.detector.window import DetectorWindow
from polyx.gui.trainer.window import TrainerWindow


def test_detector_window_builds(qapp):
    window = DetectorWindow()
    assert window is not None


def test_trainer_window_builds(qapp):
    window = TrainerWindow()
    assert window is not None