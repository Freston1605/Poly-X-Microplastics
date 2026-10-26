"""Cross-platform updater for Poly-X.

Downloads the latest version from GitHub, overwrites program files while
preserving .venv, models, runs and local data, and refreshes dependencies.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "CrissFerrada/Poly-X-Microplastics"
BRANCH = "main"
UA = "PolyX-Updater"
EXCLUDED_NAMES = {
    ".git",
    ".venv",
    ".polyx_version",
    "__pycache__",
    "actualizar.bat",
    "actualizar.ps1",
    "actualizar_linux.sh",
    "actualizar_macOS.command",
    "update.py",
}


def _run(cmd, **kw):
    return subprocess.run(cmd, check=False, **kw)  # noqa: S603


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _pause() -> None:
    import contextlib

    with contextlib.suppress(EOFError):
        input("  Pulsa ENTER para salir...")


def _fetch_sha() -> str:
    url = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
        data = json.loads(resp.read().decode())
    return data["sha"]


def _download_zip(tmp: Path) -> Path:
    url = f"https://github.com/{REPO}/archive/refs/heads/{BRANCH}.zip"
    dest = tmp / "src.zip"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:  # noqa: S310
        shutil.copyfileobj(resp, f)
    return dest


def _extracted_root(tmp: Path) -> Path:
    with zipfile.ZipFile(tmp / "src.zip") as zf:
        zf.extractall(tmp)
    candidates = [p for p in tmp.iterdir() if p.is_dir() and p.name.endswith(f"-{BRANCH}")]
    if not candidates:
        raise RuntimeError("ZIP sin la estructura esperada")
    return candidates[0]


def _ignore(dir_, names):
    out = set()
    for n in names:
        if n in EXCLUDED_NAMES:
            out.add(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Update Poly-X from GitHub")
    ap.add_argument("--root", default=None, help="project root")
    ap.add_argument("--no-pause", action="store_true", help="do not wait for ENTER at the end")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else here.parent

    _say()
    _say("=" * 60)
    _say("  Poly-X  -  Buscar actualizaciones")
    _say("=" * 60)
    _say()
    _say(f"Carpeta: {root}")
    _say("Comprobando en GitHub...")

    try:
        remote = _fetch_sha()
    except Exception as exc:  # noqa: BLE001
        _say(f"[ERROR] No se pudo consultar GitHub: {exc}")
        if not args.no_pause:
            _pause()
        return 1

    ver_file = root / ".polyx_version"
    local = ver_file.read_text(encoding="utf-8").strip() if ver_file.exists() else ""

    if local and local == remote:
        _say()
        _say("[OK] Ya tienes la ultima version. No hay nada que actualizar.")
        if not args.no_pause:
            _pause()
        return 0

    _say()
    _say("Hay una version nueva disponible. Descargando...")

    tmp = Path(tempfile.mkdtemp(prefix="polyx_update_"))
    try:
        _download_zip(tmp)
        src_dir = _extracted_root(tmp)
        _say("Aplicando actualizacion (se conservan .venv, models, runs y tus datos)...")
        shutil.copytree(src_dir, root, dirs_exist_ok=True, ignore=_ignore)
    except Exception as exc:  # noqa: BLE001
        _say(f"[ERROR] Fallo la actualizacion: {exc}")
        shutil.rmtree(tmp, ignore_errors=True)
        if not args.no_pause:
            _pause()
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    import contextlib

    with contextlib.suppress(OSError):
        ver_file.write_text(remote, encoding="ascii", newline="")

    _say("[OK] Codigo actualizado a la ultima version.")

    venv_py = root / (
        ".venv/Scripts/python.exe" if sys.platform.startswith("win") else ".venv/bin/python"
    )
    if venv_py.is_file():
        _say("Revisando dependencias (pip)...")
        r = _run(
            [str(venv_py), "-m", "pip", "install", "-r", str(root / "requirements.txt")],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            _say("[OK] Dependencias al dia.")
        else:
            _say("[AVISO] Algo fallo al revisar dependencias; el codigo si se actualizo.")
    else:
        _say("[AVISO] No hay entorno .venv todavia. Ejecuta el instalador para crearlo.")

    _say()
    _say("=" * 60)
    _say(f"  Actualizado a {remote[:7]}")
    _say("  Abre el launcher para usar Poly-X.")
    _say("=" * 60)
    if not args.no_pause:
        _pause()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
