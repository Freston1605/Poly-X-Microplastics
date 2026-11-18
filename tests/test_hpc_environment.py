"""The environment contract for the V100 training stack.

These tests drive the public reporting surface, not the private probe helpers.
A fake torch module stands in for the GPU: it lets the "hardware present" path
be exercised offline, while the default fake has no CUDA so the honest
"absent" path is the one a GPU-less machine actually takes. The real hardware
smoke check is ticket #199 on OCEANO, not this suite.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from polyx.hpc import check_env, environment, hashing, weights

ROOT = Path(__file__).resolve().parents[1]

FLOAT16 = object()


class _Tensor:
    """Just enough of a torch tensor for the AMP probe."""

    def __init__(self, dtype: object, finite: bool = True) -> None:
        self.dtype = dtype
        self.finite = finite

    def __matmul__(self, other: _Tensor) -> _Tensor:
        return _Tensor(self.dtype, self.finite and other.finite)


class _BoolTensor:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> _BoolTensor:
        return self

    def item(self) -> bool:
        return self.value


def _fake_torch(
    *,
    available: bool = True,
    architectures: tuple[str, ...] = ("sm_50", "sm_70", "sm_80"),
    cuda_version: str | None = "12.6",
    finite: bool = True,
    compiled_flags: str | None = "sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90",
) -> SimpleNamespace:
    """A torch-shaped object whose answers are controlled by the test."""
    cuda = SimpleNamespace(
        is_available=lambda: available,
        get_arch_list=lambda: list(architectures) if available else [],
        get_device_name=lambda _index: "Tesla V100-PCIE-32GB" if available else "",
        synchronize=lambda: None,
    )

    def randn(_shape: tuple[int, ...], *, device: str, dtype: object) -> _Tensor:
        assert device == "cuda"
        return _Tensor(dtype, finite)

    def autocast(device_type: str, *, dtype: object) -> object:
        assert device_type == "cuda"
        assert dtype is FLOAT16
        return contextlib.nullcontext()

    internal = SimpleNamespace(_cuda_getArchFlags=lambda: compiled_flags)
    return SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda=cuda_version),
        float16=FLOAT16,
        randn=randn,
        autocast=autocast,
        isfinite=lambda tensor: _BoolTensor(tensor.finite),
        _C=internal,
    )


# ── CUDA / architecture / AMP reporting ──


def test_no_gpu_reports_absence_and_does_not_fake_an_amp_pass() -> None:
    report = environment.probe(_fake_torch(available=False))

    assert report.cuda_available is False
    assert report.amp_status == environment.AMP_NOT_RUN
    assert report.amp_dtype is None
    assert report.ok is False
    assert any("CUDA is not available" in problem for problem in report.problems)

    lines = "\n".join(report.lines())
    assert "CUDA available: no" in lines
    assert "Reported architectures: (none)" in lines
    assert "AMP pass: not run" in lines
    assert "Result: FAIL" in lines


def test_gpu_with_v100_architecture_and_amp_passes() -> None:
    report = environment.probe(_fake_torch())

    assert report.cuda_available is True
    assert "sm_70" in report.architectures
    assert report.cuda_version == "12.6"
    assert report.device_name == "Tesla V100-PCIE-32GB"
    assert report.amp_status == environment.AMP_PASS
    assert report.amp_dtype == "float16"
    assert report.problems == ()
    assert report.ok is True

    lines = "\n".join(report.lines())
    assert "CUDA available: yes" in lines
    assert "AMP pass: pass (float16)" in lines
    assert "Result: PASS" in lines


def test_gpu_without_v100_architecture_is_rejected() -> None:
    report = environment.probe(_fake_torch(architectures=("sm_80", "sm_90")))

    assert report.cuda_available is True
    assert report.ok is False
    assert any("sm_70" in problem for problem in report.problems)
    assert report.amp_status == environment.AMP_PASS


def test_wrong_cuda_build_is_rejected() -> None:
    report = environment.probe(_fake_torch(cuda_version="12.8"))

    assert report.ok is False
    assert any("12.6" in problem for problem in report.problems)


def test_failed_amp_probe_is_reported_as_a_failure() -> None:
    report = environment.probe(_fake_torch(finite=False))

    assert report.amp_status == environment.AMP_FAIL
    assert report.ok is False
    assert any("AMP" in problem for problem in report.problems)


def test_missing_torch_is_reported_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment, "_import_torch", lambda: None)
    report = environment.probe(None)

    assert report.cuda_available is False
    assert report.amp_status == environment.AMP_NOT_RUN
    assert report.ok is False
    assert any("PyTorch" in problem for problem in report.problems)


def test_check_command_exits_nonzero_and_prints_clear_report_without_gpu(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert check_env.main([], torch_module=_fake_torch(available=False)) == 1

    output = capsys.readouterr().out
    assert "CUDA available: no" in output
    assert "AMP pass: not run" in output
    assert "GPU smoke check: ticket #199 on OCEANO" in output


def test_check_command_exits_zero_with_a_valid_gpu(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert check_env.main([], torch_module=_fake_torch()) == 0

    output = capsys.readouterr().out
    assert "CUDA available: yes" in output
    assert "Result: PASS" in output


# ── Hash verification and the offline wheelhouse ──


def test_sha256_manifest_round_trip(tmp_path: Path) -> None:
    first = tmp_path / "alpha-1.0-py3-none-any.whl"
    second = tmp_path / "beta-2.0-py3-none-any.whl"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    manifest = tmp_path / "SHA256SUMS"

    hashing.write_manifest(manifest, [first, second])

    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert hashing.verify_manifest(tmp_path, manifest) == []

    second.write_bytes(b"tampered")
    problems = hashing.verify_manifest(tmp_path, manifest)
    assert len(problems) == 1
    assert "beta-2.0-py3-none-any.whl" in problems[0]


def test_verify_file_accepts_a_matching_digest_and_rejects_a_wrong_one(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.pt"
    artifact.write_bytes(b"trusted bytes")
    digest = hashing.sha256_file(artifact)

    hashing.verify_file(artifact, digest)

    with pytest.raises(weights.WeightHashError, match="SHA-256 mismatch"):
        hashing.verify_file(artifact, "0" * 64)


def test_fetch_weights_requires_a_manifest_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "weights.sha256"
    manifest.write_text("# only a comment, no pinned digest\n", encoding="utf-8")
    called: list[str] = []

    def downloader(url: str, destination: Path) -> None:
        called.append(url)
        destination.write_bytes(b"unverified")

    with pytest.raises(weights.WeightHashError, match="no weights are pinned"):
        weights.fetch_weights(
            manifest,
            tmp_path / "models",
            base_url="https://example.invalid/weights/",
            downloader=downloader,
        )
    assert called == []


def test_fetch_weights_verifies_each_download_and_removes_bad_files(tmp_path: Path) -> None:
    good = b"trusted bytes"
    manifest = tmp_path / "weights.sha256"
    manifest.write_text(
        f"{hashing.sha256_bytes(good)}  model-a.pt\n{'1' * 64}  model-b.pt\n",
        encoding="utf-8",
    )
    destination = tmp_path / "models"
    called: list[str] = []

    def downloader(url: str, path: Path) -> None:
        called.append(url)
        path.write_bytes(good)

    with pytest.raises(weights.WeightHashError, match="SHA-256 mismatch"):
        weights.fetch_weights(
            manifest,
            destination,
            base_url="https://example.invalid/weights/",
            downloader=downloader,
        )

    assert (destination / "model-a.pt").read_bytes() == good
    assert not (destination / "model-b.pt").exists()
    assert not (destination / "model-b.pt.part").exists()
    assert called == [
        "https://example.invalid/weights/model-a.pt",
        "https://example.invalid/weights/model-b.pt",
    ]


def test_existing_weight_is_reverified_before_it_is_trusted(tmp_path: Path) -> None:
    payload = b"already on disk"
    manifest = tmp_path / "weights.sha256"
    manifest.write_text(f"{'2' * 64}  cached.pt\n", encoding="utf-8")
    destination = tmp_path / "models"
    destination.mkdir()
    cached = destination / "cached.pt"
    cached.write_bytes(payload)

    def downloader(_url: str, _path: Path) -> None:
        pytest.fail("a cached weight must not be downloaded again")

    with pytest.raises(weights.WeightHashError, match="SHA-256 mismatch"):
        weights.fetch_weights(
            manifest,
            destination,
            base_url="https://example.invalid/weights/",
            downloader=downloader,
        )
    assert not cached.exists()


# ── Wheelhouse plan (no network) ──


def test_wheelhouse_plan_uses_cuda_index_and_no_index_install() -> None:
    from tools.hpc import build_wheelhouse

    requirements = "\n".join(
        [
            "numpy==1.26.4",
            "torch==2.7.1",
            "torchvision==0.22.1",
            "ultralytics==8.3.40",
        ]
    )
    plan = build_wheelhouse.make_plan(requirements, Path("/wheelhouse"))

    assert plan.default_requirements == ("numpy==1.26.4", "ultralytics==8.3.40")
    assert plan.cuda_requirements == ("torch==2.7.1+cu126", "torchvision==0.22.1+cu126")
    assert build_wheelhouse.CUDA_INDEX in plan.cuda_index_args
    assert any("--no-index" in " ".join(step) for step in plan.steps)
    assert any("--find-links" in " ".join(step) for step in plan.steps)


def test_dry_run_prints_the_offline_install_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tools.hpc import build_wheelhouse

    assert build_wheelhouse.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "cu126" in output
    assert "--no-index" in output
    assert "wheelhouse" in output


# ── Static evidence for the container definitions ──


def test_container_definitions_pin_the_base_and_the_cuda_build() -> None:
    containerfile = (ROOT / "container" / "Containerfile").read_text(encoding="utf-8")
    definition = (ROOT / "container" / "hpc.def").read_text(encoding="utf-8")

    for text in (containerfile, definition):
        assert "@sha256:" in text
        assert "cu126" in text
        assert "torch==2.7.1+cu126" in text
        assert "--no-index" in text
    assert "Bootstrap: docker" in definition
