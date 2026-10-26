# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Poly-X (one-folder build).

Run via:  python tools/packaging/build.py
Or:       pyinstaller tools/packaging/polyx.spec
"""
from pathlib import Path

# Project root (two levels up from this spec file).
ROOT = Path(SPECPATH).resolve().parents[2]
SRC = ROOT / "src"

block_cipher = None

a = Analysis(
    [str(SRC / "polyx" / "launcher.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=[
        # Bundled (read-only) assets only. Weights live in USER_DATA_DIR.
        (str(ROOT / "assets"), "assets"),
    ],
    hiddenimports=[
        "polyx.core",
        "polyx.gui",
        "polyx.gui.qt",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch.distributed",
        "torch.testing",
        "tensorboard",
        "pytest",
        "libcst",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Poly-X",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX can corrupt PySide6 binaries
    console=False,      # windowed app
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "polyx.ico") if (ROOT / "assets" / "polyx.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Poly-X",
)