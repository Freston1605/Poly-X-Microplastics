"""Build an offline wheelhouse from the locked environment.

The lockfile is the single source of truth for every package except the deep
learning framework. Torch and torchvision are replaced by the CUDA 12.6
wheels from the pinned PyTorch index: those are the builds that still compile
the V100 ``sm_70`` kernel set. The default public index is never used for
those two packages.

Everything the compute node needs is downloaded here, on a machine with
internet access, and verified by a ``SHA256SUMS`` manifest. Installation on
the node is then purely offline::

    uv pip install --no-index --find-links dist/wheelhouse \
        -r dist/wheelhouse/requirements.txt

Usage::

    python tools/hpc/build_wheelhouse.py [--wheelhouse dist/wheelhouse] [--dry-run]

``--dry-run`` prints the plan without downloading anything; it is safe to run
offline and is what the test suite exercises.

The builder never installs into the environment that invoked it: it downloads,
builds and verifies, and the image (or the node) does the offline install. A
self-check that ran `uv pip install` here would need a virtual environment
inside the image's build stage and, when it found none, could write into an
unrelated one.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

UV_VERSION = "0.13.7"
PIP_VERSION = "26.3"
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
CUDA_PACKAGES = ("torch==2.7.1+cu126", "torchvision==0.22.1+cu126")
REPLACED_PACKAGES = frozenset({"torch", "torchvision"})
_NAME = re.compile(r"^[A-Za-z0-9_.-]+")
DEFAULT_WHEELHOUSE = Path("dist") / "wheelhouse"


@dataclass(frozen=True)
class Plan:
    """The concrete steps needed to build and verify a wheelhouse."""

    wheelhouse: Path
    default_requirements: tuple[str, ...]
    cuda_requirements: tuple[str, ...]
    cuda_index_args: tuple[str, ...]
    steps: tuple[tuple[str, ...], ...]

    @property
    def requirements_file(self) -> Path:
        return self.wheelhouse / "requirements.txt"

    @property
    def checksum_file(self) -> Path:
        return self.wheelhouse / "SHA256SUMS"


def split_requirements(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep the locked pins, minus the two packages replaced by CUDA wheels."""
    default: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-")):
            continue
        match = _NAME.match(line)
        name = match.group(0).lower() if match else ""
        if name in REPLACED_PACKAGES:
            continue
        default.append(line)
    return tuple(default), CUDA_PACKAGES


def make_plan(requirements_text: str, wheelhouse: Path) -> Plan:
    """Turn an exported requirements list into ordered build steps."""
    default, cuda = split_requirements(requirements_text)
    wheelhouse = Path(wheelhouse)
    exported = wheelhouse / "lock-requirements.txt"
    pip = (
        "uv",
        "run",
        "--no-sync",
        "--with",
        f"pip=={PIP_VERSION}",
        "python",
        "-m",
        "pip",
        "download",
    )
    # The lockfile already contains every transitive pin, including the CUDA
    # runtime libraries the cu126 wheels need. Both downloads are therefore
    # dependency-free; resolving again would pull newer numpy/Pillow wheels
    # that contradict the lock.
    index_args = ("--index-url", CUDA_INDEX)
    steps = (
        (
            "uv",
            "export",
            "--frozen",
            "--no-emit-project",
            "--no-hashes",
            "--format",
            "requirements-txt",
            "-o",
            str(exported),
        ),
        ("write", str(exported), "locked pins without default torch/torchvision"),
        (*pip, "--no-deps", "--dest", str(wheelhouse), "-r", str(exported)),
        (*pip, "--no-deps", *index_args, "--dest", str(wheelhouse), *cuda),
        # The project's own wheel is built here, where the build backend and the
        # network are available, and ships in the wheelhouse. A compute node
        # then installs it with --no-index and never needs hatchling, Babel or
        # the source tree; building it there also required NOTICE, which the
        # force-include in pyproject.toml makes mandatory.
        ("uv", "build", "--wheel", "--out-dir", str(wheelhouse)),
        ("write", str(wheelhouse / "requirements.txt"), "locked pins + cu126 wheels"),
        ("write", str(wheelhouse / "SHA256SUMS"), "digest of every wheel in the directory"),
    )
    return Plan(
        wheelhouse=wheelhouse,
        default_requirements=default,
        cuda_requirements=cuda,
        cuda_index_args=index_args,
        steps=steps,
    )


def _export_lockfile() -> str:
    command = [
        "uv",
        "export",
        "--frozen",
        "--no-emit-project",
        "--no-hashes",
        "--format",
        "requirements-txt",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "uv export failed")
    return result.stdout


def _write_index(plan: Plan) -> None:
    plan.wheelhouse.mkdir(parents=True, exist_ok=True)
    header = [
        "# Generated from uv.lock by tools/hpc/build_wheelhouse.py; do not edit.",
        "# Every pin comes from the lockfile except the two CUDA 12.6 wheels, which",
        "# replace its default-index torch/torchvision pins.",
        "# The project's own wheel sits beside this file and is installed separately:",
        "#   uv pip install --no-index --no-deps micra-*.whl",
        "# Install offline with: uv pip install --no-index --find-links <dir> -r <dir>/requirements.txt",
    ]
    pins = [*plan.default_requirements, *plan.cuda_requirements]
    plan.requirements_file.write_text("\n".join([*header, *pins]) + "\n", encoding="utf-8")


def _write_checksums(plan: Plan) -> None:
    # The wheelhouse stage has no installed dependencies yet: importing the
    # ``polyx`` package would pull in cv2 through its scientific stack. Load
    # the standalone hashing module by path instead, with no heavy imports.
    import importlib.util

    path = ROOT / "src" / "polyx" / "hpc" / "hashing.py"
    spec = importlib.util.spec_from_file_location("polyx_hpc_hashing", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    hashing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hashing)

    wheels = sorted(plan.wheelhouse.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"no wheels were downloaded into {plan.wheelhouse}")
    hashing.write_manifest(plan.checksum_file, wheels)
    problems = hashing.verify_manifest(plan.wheelhouse, plan.checksum_file)
    if problems:
        raise RuntimeError("wheelhouse verification failed: " + "; ".join(problems))


def _run(step: tuple[str, ...]) -> int:
    print("+ " + " ".join(step), flush=True)
    if step[0] == "write":
        return 0
    return subprocess.run(step, cwd=ROOT, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an offline wheelhouse from uv.lock")
    parser.add_argument("--wheelhouse", type=Path, default=DEFAULT_WHEELHOUSE)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; download nothing")
    args = parser.parse_args(argv)

    try:
        requirements_text = _export_lockfile()
    except (OSError, RuntimeError) as exc:
        print(f"cannot read uv.lock: {exc}", file=sys.stderr)
        return 1
    plan = make_plan(requirements_text, args.wheelhouse)

    if args.dry_run:
        for step in plan.steps:
            print("+ " + " ".join(step))
        print(f"default pins: {len(plan.default_requirements)}")
        print(f"cuda wheels: {', '.join(plan.cuda_requirements)}")
        print(
            f"offline install: uv pip install --no-index --find-links {plan.wheelhouse} -r {plan.requirements_file}"
        )
        print(f"project wheel: uv pip install --no-index --no-deps {plan.wheelhouse}/micra-*.whl")
        return 0

    for step in plan.steps:
        if step[0] == "write":
            target = Path(step[1])
            if target == plan.wheelhouse / "lock-requirements.txt":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("\n".join(plan.default_requirements) + "\n", encoding="utf-8")
                print("+ " + " ".join(step), flush=True)
                continue
            if target == plan.requirements_file:
                _write_index(plan)
            elif target == plan.checksum_file:
                _write_checksums(plan)
            print("+ " + " ".join(step), flush=True)
            continue
        code = _run(step)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
