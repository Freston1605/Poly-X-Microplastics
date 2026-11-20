"""Resolve a family's weights to a local, hash-verified file.

A compute node has no internet access and a downloaded PyTorch pickle executes
code when loaded, so the training path never lets a framework fetch a weight it
did not pin. ``family.weights`` stays the framework's own identifier; this
module maps it to a file in the configured weights directory and verifies that
file against a ``sha256sum``-style manifest before any adapter sees it.

Two modes are possible:

- **pinned** — a weights directory and manifest are configured (directly or
  through ``POLYX_WEIGHTS_DIR``/``POLYX_WEIGHTS_MANIFEST``). The file must be
  present and match its digest, or the execution fails before training. Cluster
  jobs always run in this mode.
- **framework default** — nothing is configured, so the adapter asks the
  framework for its default weights. That may download on a networked machine
  and is reported as unverified; it is for local debugging only.

An architecture declaration (``*.yaml``), an explicitly untrained detector
(``weights: none``) and an empty name have nothing to verify and are returned
unchanged.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path

from polyx.hpc.hashing import HashError, parse_manifest, verify_file
from polyx.pipeline.config import Family

#: Extensions that name a PyTorch weight file rather than an architecture.
WEIGHT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})

#: Extensions that name an architecture declaration, which carries no weights.
ARCHITECTURE_SUFFIXES = frozenset({".yaml", ".yml"})

#: Names that explicitly ask for an untrained model.
UNTRAINED_NAMES = frozenset({"", "none", "random"})


class WeightResolutionError(RuntimeError):
    """A pinned weight is absent, unpinned or does not match its digest."""


@dataclass(frozen=True)
class ResolvedWeights:
    """What the adapter should hand to its framework, and whether it is trusted."""

    name: str
    path: Path | None
    verified: bool

    def describe(self) -> str:
        """The value recorded in ``training_state.json``."""
        return str(self.path) if self.path is not None else self.name


def _configured(value: Path | str | None, variable: str) -> Path | None:
    """An explicit configuration value, else the environment, else nothing."""
    if value:
        return Path(str(value)).expanduser()
    raw = os.environ.get(variable)
    return Path(raw).expanduser() if raw else None


def _torchvision_candidates(name: str) -> list[str]:
    """File names a torchvision factory's pretrained state dict is published as.

    The public weights enum knows the release URL, so its basename is the
    filename a maintainer stages. ``<factory>.pth`` is accepted as the simpler
    spelling. The enum is read defensively: a framework double or an older
    torchvision without the enum simply yields no extra candidate.
    """
    candidates: list[str] = []
    try:
        from torchvision.models import detection

        factory = getattr(detection, name, None)
        if factory is None:
            return candidates
        signature = inspect.signature(factory)
        default = signature.parameters.get("weights")
        url = getattr(default.default if default is not None else None, "url", "")
        filename = Path(str(url)).name if url else ""
        if filename:
            candidates.append(filename)
    except (ImportError, TypeError, ValueError):
        return candidates
    candidates.append(f"{name}{'.pth'}")
    return candidates


def resolve_weights(
    family: Family,
    *,
    weights_dir: Path | str | None = None,
    manifest: Path | str | None = None,
) -> ResolvedWeights:
    """Map ``family.weights`` to a verified local file, or fail closed.

    Raises :class:`WeightResolutionError` when a pinned file is missing, has no
    manifest entry, or fails its digest. Never downloads anything itself:
    fetching belongs to the staging step on a networked machine.
    """
    name = str(family.weights or "").strip()
    suffix = Path(name).suffix.lower()
    if name.lower() in UNTRAINED_NAMES or suffix in ARCHITECTURE_SUFFIXES:
        return ResolvedWeights(name=name, path=None, verified=False)
    if suffix in WEIGHT_SUFFIXES:
        candidates = [name]
    elif family.framework == "torchvision":
        candidates = _torchvision_candidates(name)
    else:
        candidates = [name]

    directory = _configured(weights_dir, "POLYX_WEIGHTS_DIR")
    manifest_path = _configured(manifest, "POLYX_WEIGHTS_MANIFEST")
    if directory is None and manifest_path is None:
        # Local debugging: the framework's own default is used unverified.
        return ResolvedWeights(name=name, path=None, verified=False)
    if directory is None:
        raise WeightResolutionError(
            "a weights manifest was configured without a weights directory; set "
            "weights_dir or POLYX_WEIGHTS_DIR to the staged directory"
        )
    pinned = manifest_path if manifest_path is not None else directory / "SHA256SUMS"
    if not pinned.is_file():
        raise WeightResolutionError(
            f"no weights manifest at {pinned}; stage one with fetch_weights or set "
            "weights_manifest/POLYX_WEIGHTS_MANIFEST"
        )
    entries = parse_manifest(pinned)
    if not candidates:
        raise WeightResolutionError(
            f"cannot map family.weights={name!r} to a staged file name for {family.framework}"
        )
    staged = next((candidate for candidate in candidates if candidate in entries), None)
    if staged is None:
        raise WeightResolutionError(
            f"weights {name!r} is not pinned in {pinned}; expected one of "
            f"{candidates} to have a manifest entry"
        )
    path = directory / staged
    if not path.is_file():
        raise WeightResolutionError(
            f"weight {staged!r} is declared in {pinned} but not staged at {path}; "
            "fetch it on a networked machine before submitting the job"
        )
    try:
        verify_file(path, entries[staged])
    except HashError as exc:
        raise WeightResolutionError(
            f"weight {staged!r} is present but not verified: {exc}"
        ) from exc
    return ResolvedWeights(name=name, path=path.resolve(), verified=True)
