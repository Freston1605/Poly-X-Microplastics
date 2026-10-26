"""Cross-platform installer for Poly-X.

This is the single source of truth for the install pipeline. The thin
``SETUP.bat`` / ``setup.sh`` wrappers only locate a Python interpreter
and call this script; all the real work (copy, venv, GPU detection, torch,
dependencies, verification) happens here.

Run with the system Python:
    python install/bootstrap.py
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from gpu import detect_gpu as _detect_gpu
from versions import REQUIREMENTS, TORCH_VERSION, TORCHVISION_VERSION


def _expected_torch() -> tuple[str, str]:
    if sys.platform == "darwin" and platform.machine().lower() in ("x86_64", "i386"):
        return "2.2.2", "0.17.2"
    return TORCH_VERSION, TORCHVISION_VERSION


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, **kw)  # noqa: S603


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{text}{suffix}: ").strip()
    except EOFError:
        answer = ""
    if not answer and default is not None:
        return default
    return answer


def _locate_python() -> str:
    # bootstrap is already running under a Python interpreter.
    return sys.executable


def _copy_tree(src: Path, dst: Path) -> None:
    def ignore(dir_, names):
        out = set()
        for n in names:
            if n in {
                ".venv",
                ".git",
                "__pycache__",
                "runs",
                "runs_train",
                ".ruff_cache",
                ".gsd",
                ".codegraph",
                ".project",
                "node_modules",
                ".opencode",
            }:
                out.add(n)
            if n.endswith(".log"):
                out.add(n)
        return out

    if dst.exists():
        # merge: copy only missing files
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                if target.exists():
                    _copy_tree(item, target)
                else:
                    shutil.copytree(item, target, ignore=ignore)
            else:
                if not target.exists():
                    shutil.copy2(item, target)
    else:
        shutil.copytree(src, dst, ignore=ignore)


def _remove_venv(venv: Path) -> None:
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)


def _venv_python(venv: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _venv_broken(venv: Path) -> bool:
    py = _venv_python(venv)
    if not py.is_file():
        return True
    try:
        r = _run([str(py), "--version"], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return True
    except Exception:  # noqa: BLE001
        return True
    # torch import check
    try:
        r = _run([str(py), "-c", "import torch"], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            # only broken if torch was supposed to be installed
            r2 = _run(
                [
                    str(py),
                    "-c",
                    "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') is None else 1)",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r2.returncode == 1:
                return True
    except Exception:  # noqa: BLE001, S110
        pass
    return False


def _install_torch(venv_py: Path, gpu: dict) -> None:
    backend = gpu["backend"]
    if backend == "cuda":
        cc_major = gpu.get("cc_major")
        whl = "cu128" if (cc_major is not None and cc_major >= 12) else "cu118"
        cudanom = "12.8" if whl == "cu128" else "11.8"
        _say(f"[OK] GPU NVIDIA detectada (compute capability: {gpu.get('cc')}).")
        _say(f"[INFO] Instalando PyTorch con CUDA {cudanom}...")
        idx = f"https://download.pytorch.org/whl/{whl}"
    elif backend == "rocm":
        _say("[OK] GPU AMD (ROCm) detectada.")
        _say("[INFO] Instalando PyTorch con ROCm 6.1...")
        idx = "https://download.pytorch.org/whl/rocm6.1"
    else:
        _say("[INFO] Sin GPU compatible. Instalando PyTorch CPU...")
        idx = "https://download.pytorch.org/whl/cpu"

    torch_ver, vision_ver = _expected_torch()
    cmd = [
        str(venv_py),
        "-m",
        "pip",
        "install",
        f"torch=={torch_ver}",
        f"torchvision=={vision_ver}",
        "--index-url",
        idx,
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError("PyTorch install failed")


def _torch_already_ok(venv_py: Path, gpu: dict) -> bool:
    r = _run(
        [
            str(venv_py),
            "-c",
            (
                "import torch,sys; print(torch.__version__); "
                "print(getattr(torch.version,'cuda',None)); "
                "print(getattr(torch.version,'hip',None))"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        return False
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    if len(lines) < 3 or lines[0] != _expected_torch()[0]:
        return False
    if gpu["backend"] == "cuda":
        return _verify_gpu(venv_py, gpu)
    return True


def _write_manifest(install_dir: Path, gpu: dict) -> None:
    import contextlib
    import json
    from datetime import UTC, datetime

    manifest = {
        "torch_version": TORCH_VERSION,
        "torchvision_version": TORCHVISION_VERSION,
        "backend": gpu["backend"],
        "compute_capability": gpu.get("cc"),
        "installed_at": datetime.now(UTC).isoformat(),
    }
    with contextlib.suppress(OSError):
        (install_dir / ".polyx_install.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def _verify_imports(venv_py: Path) -> bool:
    r = _run(
        [
            str(venv_py),
            "-c",
            "import PySide6, ultralytics, cv2, numpy, matplotlib; import PySide6.QtWebEngineWidgets",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return r.returncode == 0


def _verify_gpu(venv_py: Path, gpu: dict) -> bool:
    if gpu["backend"] != "cuda":
        return True
    code = (
        "import sys, torch; "
        "ok=torch.cuda.is_available(); "
        "archs=torch.cuda.get_arch_list() if ok else []; "
        "cap='sm_%d%d'%torch.cuda.get_device_capability(0) if ok else ''; "
        "print('   GPU  :', torch.cuda.get_device_name(0) if ok else 'no disponible'); "
        "print('   arch :', cap); "
        "print('   torch:', torch.__version__, '| soporta:', ' '.join(archs) if archs else 'nada'); "
        "sys.exit(0 if (ok and cap in archs) else 1)"
    )
    r = _run([str(venv_py), "-c", code], capture_output=True, text=True, timeout=300)
    _say(r.stdout.strip())
    if r.returncode != 0:
        _say(r.stderr.strip())
    return r.returncode == 0


def _version_stamp(install_dir: Path) -> None:
    try:
        with urllib.request.urlopen(
            "https://api.github.com/repos/CrissFerrada/Poly-X-Microplastics/commits/main",
            timeout=15,
        ) as resp:
            import json

            data = json.loads(resp.read().decode())
            sha = data.get("sha")
            if sha:
                (install_dir / ".polyx_version").write_text(sha, encoding="ascii", newline="")
                _say("[OK] Version registrada: el aviso de actualizaciones queda activo.")
                return
    except Exception:  # noqa: BLE001, S110
        pass
    _say("[AVISO] No se pudo consultar GitHub. Poly-X funciona igual, pero")
    _say("no avisara de versiones nuevas hasta la primera actualizacion manual.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Poly-X cross-platform installer")
    ap.add_argument("--source", default=None, help="source tree (default: parent of install/)")
    ap.add_argument("--dest", default=None, help="install destination (default: prompt)")
    ap.add_argument("--yes", action="store_true", help="accept defaults, no prompts")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    src = Path(args.source).resolve() if args.source else here.parent
    _say()
    _say("=" * 60)
    _say("  Poly-X  -  Instalacion del entorno")
    _say("=" * 60)
    _say()
    _say(f"Carpeta de origen: {src}")
    _say()

    if args.dest:
        dest = Path(args.dest).resolve()
    elif args.yes:
        dest = src
    else:
        raw = _prompt("Carpeta de instalacion (ENTER = aqui mismo)", str(src))
        raw = raw.strip().strip('"')
        dest = Path(raw).resolve() if raw else src

    if dest != src:
        _say(f"[INFO] Se instalara en: {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        _say("[INFO] Copiando archivos del programa...")
        _copy_tree(src, dest)
    install_dir = dest
    os.chdir(install_dir)
    _say(f"[OK] Carpeta de instalacion: {install_dir}")
    _say()

    # Python
    pyexe = _locate_python()
    _say(f"[OK] Python: {pyexe}")
    _say()

    venv = install_dir / ".venv"
    # repair
    if venv.exists() and _venv_broken(venv):
        _say("[INFO] El entorno .venv no sirve. Se rehace desde cero...")
        _remove_venv(venv)
        if venv.exists():
            _say("[ERROR] No se pudo borrar .venv. Cierra Poly-X y vuelve a ejecutar.")
            return 1

    if not venv.exists():
        _say("[INFO] Creando entorno virtual .venv ...")
        r = _run([pyexe, "-m", "venv", str(venv)])
        if r.returncode != 0:
            _say("[ERROR] No se pudo crear el venv.")
            return 1

    venv_py = _venv_python(venv)
    _say("[INFO] Actualizando pip...")
    _run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    gpu = _detect_gpu()
    if _torch_already_ok(venv_py, gpu):
        _say(f"[OK] PyTorch {TORCH_VERSION} ya esta instalado y compatible. Se omite.")
    else:
        try:
            _install_torch(venv_py, gpu)
        except RuntimeError as exc:
            _say(f"[ERROR] {exc}")
            return 1

    _say("[INFO] Instalando dependencias del proyecto...")
    r = _run([str(venv_py), "-m", "pip", "install", "-r", REQUIREMENTS])
    if r.returncode != 0:
        _say("[ERROR] Hubo problemas instalando dependencias.")
        return 1

    _say("[INFO] Verificando la instalacion...")
    if _verify_imports(venv_py):
        _say(
            "[OK] Todas las dependencias se importan correctamente (incluida la exportacion a PDF)."
        )
    else:
        _say("[AVISO] Algunas dependencias no se importaron correctamente.")

    # GPU kernel compat check (CUDA only)
    if gpu["backend"] == "cuda":
        _say("[INFO] Verificando compatibilidad GPU/PyTorch...")
        if not _verify_gpu(venv_py, gpu):
            _say("[ERROR] PyTorch NO trae kernels para esta GPU.")
            _say("[INFO] Reinstalando PyTorch con la rueda correcta...")
            whl = (
                "cu128" if (gpu.get("cc_major") is not None and gpu["cc_major"] >= 12) else "cu118"
            )
            idx = f"https://download.pytorch.org/whl/{whl}"
            _run(
                [
                    str(venv_py),
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    f"torch=={TORCH_VERSION}",
                    f"torchvision=={TORCHVISION_VERSION}",
                    "--index-url",
                    idx,
                ]
            )
            _say("[INFO] Restaurando las versiones fijadas en requirements.txt...")
            _run([str(venv_py), "-m", "pip", "install", "-r", REQUIREMENTS])
            _say("[INFO] Verificando de nuevo...")
            if _verify_gpu(venv_py, gpu):
                _say("[OK] Corregido: PyTorch ya tiene kernels para esta GPU.")
            else:
                _say("[ERROR] Sigue sin coincidir. Revisa la rueda e instala a mano.")
        else:
            _say("[OK] PyTorch tiene kernels para esta GPU.")

    _write_manifest(install_dir, gpu)

    if not (install_dir / ".polyx_version").exists():
        _say("[INFO] Registrando la version instalada...")
        _version_stamp(install_dir)

    # work dirs
    (install_dir / "models").mkdir(exist_ok=True)
    (install_dir / "data").mkdir(exist_ok=True)
    _say("[OK] Carpetas models/ y data/ listas.")

    _say()
    _say("=" * 60)
    _say("  Instalacion completada")
    _say("=" * 60)
    _say()
    _say(f"  Poly-X quedo instalado en: {install_dir}")
    _say()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
