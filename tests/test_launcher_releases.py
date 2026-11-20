"""Launcher export behavior for the manual release route."""

from __future__ import annotations

import pytest


@pytest.fixture
def opened(monkeypatch):
    from polyx.gui import launcher

    calls: list[str] = []
    monkeypatch.setattr(
        launcher, "open_in_system", lambda target: calls.append(str(target)) or True
    )
    return calls


def test_the_launcher_sends_the_user_to_the_fork_releases(qapp, opened):
    from polyx.gui.launcher import LauncherWindow
    from polyx.release import desktop_release_url

    window = LauncherWindow()
    try:
        window.btn_update.click()
    finally:
        window.close()

    assert opened == [desktop_release_url()]
    assert "Freston1605/Poly-X-Microplastics" in opened[0]


def test_the_export_button_is_not_a_self_replacing_updater(qapp, opened):
    from polyx.gui.launcher import LauncherWindow

    window = LauncherWindow()
    try:
        window.btn_update.click()
    finally:
        window.close()

    assert not hasattr(window, "_sha_version_new")
    assert len(opened) == 1


def test_a_page_that_cannot_be_opened_still_tells_the_user_where_to_go(qapp, monkeypatch):
    from polyx.gui import launcher

    warnings: list[tuple] = []
    monkeypatch.setattr(launcher, "open_in_system", lambda _target: False)
    monkeypatch.setattr(launcher.QMessageBox, "warning", lambda *args: warnings.append(args))

    window = launcher.LauncherWindow()
    try:
        window.btn_update.click()
    finally:
        window.close()

    assert warnings
    assert "Freston1605/Poly-X-Microplastics/releases" in warnings[0][2]
