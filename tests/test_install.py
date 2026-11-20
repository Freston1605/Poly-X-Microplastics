"""Smoke tests for the cross-platform installer helpers."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "install"))

import gpu  # noqa: E402
import versions  # noqa: E402


def test_versions_pins() -> None:
    assert versions.TORCH_VERSION
    assert versions.TORCHVISION_VERSION
    assert versions.REQUIREMENTS == "requirements.txt"


def test_versions_publishes_the_pinned_pillow_from_requirements() -> None:
    assert versions.pinned_version("Pillow", REPO / "requirements.txt") == "13.0.0"
    assert versions.pinned_version("PySide6", REPO / "requirements.txt") == "6.7.2"
    assert versions.pinned_version("not-declared", REPO / "requirements.txt") is None


def _fake_interpreter(tmp_path: Path, body: str) -> Path:
    fake = tmp_path / "python"
    fake.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    fake.chmod(0o755)
    return fake


def test_the_installer_reports_the_pillow_it_actually_left_installed(tmp_path: Path) -> None:
    import bootstrap

    fake = _fake_interpreter(
        tmp_path,
        "import pathlib, sys\n"
        "state = pathlib.Path(__file__).with_suffix('.state')\n"
        "log = pathlib.Path(__file__).with_suffix('.argv')\n"
        "if '-c' in sys.argv:\n"
        "    print(state.read_text() if state.exists() else '12.3.0')\n"
        "else:\n"
        "    log.write_text(' '.join(sys.argv[1:]))\n"
        "    state.write_text('13.0.0')\n",
    )

    assert bootstrap.enforce_pillow(fake, REPO / "requirements.txt") is True
    assert fake.with_suffix(".state").read_text(encoding="utf-8") == "13.0.0"
    # The repair resolves through the same requirements file as the first
    # install, so it cannot pick a different build for the same pin.
    repair = fake.with_suffix(".argv").read_text(encoding="utf-8")
    assert "-r" in repair.split()
    assert str(REPO / "requirements.txt") in repair


def test_a_pillow_that_cannot_be_replaced_is_reported_as_a_failure(tmp_path: Path) -> None:
    import bootstrap

    fake = _fake_interpreter(
        tmp_path,
        "import sys\nif '-c' in sys.argv:\n    print('12.3.0')\nelse:\n    raise SystemExit(1)\n",
    )

    assert bootstrap.enforce_pillow(fake, REPO / "requirements.txt") is False


def test_the_install_manifest_names_pillow_and_the_backend() -> None:
    import bootstrap

    manifest = bootstrap.build_manifest({"backend": "cpu", "cc": None}, "13.0.0")

    assert manifest["pillow_version"] == "13.0.0"
    assert manifest["backend"] == "cpu"
    assert manifest["torch_version"] == versions.TORCH_VERSION


def test_pillow_security_pin_matches_desktop_and_package_resolution() -> None:
    expected = "Pillow==13.0.0"
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
    metadata = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))

    assert expected in requirements.splitlines()
    assert expected in metadata["project"]["dependencies"]
    assert {package["version"] for package in lock["package"] if package["name"] == "pillow"} == {
        "13.0.0"
    }


def test_update_command_points_to_fork_without_modifying_old_install(tmp_path: Path) -> None:
    old_install = tmp_path / "Old Poly-X"
    old_install.mkdir()
    (old_install / ".polyx_version").write_text("old-marker", encoding="ascii")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "install" / "update.py"),
            "--root",
            str(old_install),
            "--no-pause",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "https://github.com/Freston1605/Poly-X-Microplastics/releases" in result.stdout
    assert (old_install / ".polyx_version").read_text(encoding="ascii") == "old-marker"
    assert sorted(p.name for p in old_install.iterdir()) == [".polyx_version"]


def test_gpu_detect_cpu_when_no_tools() -> None:
    with mock.patch("shutil.which", return_value=None):
        info = gpu.detect_gpu()
    assert info["backend"] == "cpu"


def test_gpu_detect_cuda(monkeypatch) -> None:
    def fake_which(name):
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    def fake_run(cmd, **kw):
        class R:
            returncode = 0
            stdout = "8.6\n" if "compute_cap" in " ".join(cmd) else "RTX 3080\n"
            stderr = ""

        return R()

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)
    info = gpu.detect_gpu()
    assert info["backend"] == "cuda"
    assert info["cc_major"] == 8


def test_bootstrap_imports() -> None:
    import bootstrap

    assert hasattr(bootstrap, "main")
    assert hasattr(bootstrap, "_detect_gpu")
