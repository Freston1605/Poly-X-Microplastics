"""Report whether a machine can run the pinned V100 training stack.

The probe is deliberately safe to run without a GPU. On a machine with no
visible CUDA device it says so plainly and marks the AMP pass as *not run*:
it never reports a pass it did not observe. The hardware portion (CUDA
visibility and a real FP16 pass) is meant to be run on the GPU node as a smoke
job; ticket #199 verifies it on OCEANO.

``probe`` takes the torch module as an argument so the positive path can be
exercised offline with a torch-shaped test double. Production callers pass
``None`` and let it import torch itself.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

TARGET_ARCHITECTURE = "sm_70"
TARGET_CUDA_VERSION = "12.6"

AMP_PASS = "pass"  # noqa: S105 - a status token, not a credential
AMP_FAIL = "fail"
AMP_NOT_RUN = "not-run"

NO_VALUE = "(none)"


@dataclass(frozen=True)
class EnvironmentReport:
    """The observable result of an environment probe."""

    cuda_available: bool
    architectures: tuple[str, ...]
    compiled_architectures: tuple[str, ...]
    cuda_version: str | None
    device_name: str | None
    amp_status: str
    amp_dtype: str | None
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every checked condition held."""
        return not self.problems

    def lines(self) -> tuple[str, ...]:
        """Plain, machine-readable lines describing the probe result."""
        result = "PASS" if self.ok else "FAIL"
        lines = [
            f"CUDA available: {'yes' if self.cuda_available else 'no'}",
            f"CUDA version: {self.cuda_version or NO_VALUE}",
            f"Reported architectures: {' '.join(self.architectures) if self.architectures else NO_VALUE}",
            f"Compiled architectures: {' '.join(self.compiled_architectures) if self.compiled_architectures else NO_VALUE}",
            f"Device: {self.device_name or NO_VALUE}",
        ]
        if self.amp_status == AMP_PASS:
            lines.append(f"AMP pass: {AMP_PASS} ({self.amp_dtype})")
        else:
            lines.append(f"AMP pass: {self.amp_status.replace('-', ' ')}")
        lines.append(
            f"Result: {result}" + (f": {'; '.join(self.problems)}" if self.problems else "")
        )
        lines.append("GPU smoke check: ticket #199 on OCEANO")
        return tuple(lines)


def _import_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except Exception:  # noqa: BLE001 - a broken install is a report, not a crash
        return None


def _compiled_architectures(torch: Any) -> tuple[str, ...]:
    """Architectures the wheel was built for, via the internal flags accessor.

    ``torch.cuda.get_arch_list()`` returns an empty list when no GPU is
    visible, so this secondary signal is the only way to inspect the wheel's
    build offline. It is diagnostic only: the pass/fail decision uses the
    public API on a machine where CUDA is actually available.
    """
    try:
        flags = torch._C._cuda_getArchFlags()
    except (AttributeError, TypeError, RuntimeError, OSError):
        return ()
    if not flags:
        return ()
    return tuple(str(flags).split())


def _amp_pass(torch: Any) -> tuple[str, str | None, str | None]:
    """Run a tiny FP16 matmul under autocast, or explain why it was skipped.

    Returns ``(status, dtype, problem)``. Nothing is inferred from the wheel
    alone: a pass requires an actual CUDA tensor to have been produced.
    """
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            left = torch.randn((4, 4), device="cuda", dtype=torch.float16)
            right = torch.randn((4, 4), device="cuda", dtype=torch.float16)
            result = left @ right
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(result).all().item())
    except Exception as exc:  # noqa: BLE001 - any CUDA error is a failed pass
        return AMP_FAIL, None, f"AMP pass failed: {exc}"
    if result.dtype != torch.float16 or not finite:
        return AMP_FAIL, None, "AMP pass produced a non-finite or wrong-dtype result"
    return AMP_PASS, "float16", None


def probe(torch_module: Any | None = None) -> EnvironmentReport:
    """Inspect the torch installation and, when possible, the CUDA device."""
    torch = torch_module if torch_module is not None else _import_torch()
    if torch is None:
        return EnvironmentReport(
            cuda_available=False,
            architectures=(),
            compiled_architectures=(),
            cuda_version=None,
            device_name=None,
            amp_status=AMP_NOT_RUN,
            amp_dtype=None,
            problems=("PyTorch is not installed or cannot be imported",),
        )

    compiled = _compiled_architectures(torch)
    try:
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:  # noqa: BLE001
        cuda_version = None

    try:
        available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        available = False

    if not available:
        return EnvironmentReport(
            cuda_available=False,
            architectures=(),
            compiled_architectures=compiled,
            cuda_version=str(cuda_version) if cuda_version else None,
            device_name=None,
            amp_status=AMP_NOT_RUN,
            amp_dtype=None,
            problems=("CUDA is not available on this machine",),
        )

    try:
        architectures = tuple(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001
        architectures = ()
    try:
        device_name = str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        device_name = None

    problems: list[str] = []
    if TARGET_ARCHITECTURE not in architectures:
        reported = " ".join(architectures) or NO_VALUE
        problems.append(
            f"CUDA build does not report {TARGET_ARCHITECTURE} (V100); reported: {reported}"
        )
    if str(cuda_version) != TARGET_CUDA_VERSION:
        problems.append(
            f"CUDA runtime is {cuda_version!r}, expected {TARGET_CUDA_VERSION!r} (driver 560)"
        )

    amp_status, amp_dtype, amp_problem = _amp_pass(torch)
    if amp_problem:
        problems.append(amp_problem)

    return EnvironmentReport(
        cuda_available=True,
        architectures=architectures,
        compiled_architectures=compiled,
        cuda_version=str(cuda_version) if cuda_version else None,
        device_name=device_name,
        amp_status=amp_status,
        amp_dtype=amp_dtype,
        problems=tuple(problems),
    )
