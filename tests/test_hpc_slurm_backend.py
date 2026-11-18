"""The SLURM backend contract for ticket #199.

These tests read the files the backend is made of: the submission scripts, the
documented checklist and the small helpers that resolve the work/scratch layout
and sync results back. They are file contracts and small, pure functions, so
they run without a cluster, a GPU or a scheduler. The real smoke job on OCEANO
is a manual acceptance step and is deliberately not faked here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ── Work and scratch layout ──


def test_layout_places_user_data_and_results_on_work_and_temporary_files_on_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polyx.hpc.layout import ClusterLayout

    work = tmp_path / "work"
    scratch = tmp_path / "scratch"
    user_data = work / "userdata"
    monkeypatch.setenv("POLYX_HPC_WORK", str(work))
    monkeypatch.setenv("POLYX_HPC_SCRATCH", str(scratch))
    monkeypatch.setenv("POLYX_HOME", str(user_data))

    layout = ClusterLayout.from_env()

    assert layout.work == work
    assert layout.scratch == scratch
    assert layout.user_data == user_data
    assert layout.user_data.is_relative_to(layout.work)
    assert layout.data.is_relative_to(work)
    assert layout.results.is_relative_to(work)
    assert layout.checkpoints.is_relative_to(work)
    assert layout.tmp.is_relative_to(scratch)


def test_layout_defaults_user_data_to_work_when_polyx_home_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polyx.hpc.layout import ClusterLayout

    monkeypatch.setenv("POLYX_HPC_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("POLYX_HPC_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.delenv("POLYX_HOME", raising=False)

    layout = ClusterLayout.from_env()

    assert layout.user_data == layout.work / "userdata"


def test_layout_rejects_a_user_data_location_outside_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polyx.hpc.layout import ClusterLayout

    monkeypatch.setenv("POLYX_HPC_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("POLYX_HPC_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("POLYX_HOME", str(tmp_path / "elsewhere"))

    with pytest.raises(ValueError, match="work filesystem"):
        ClusterLayout.from_env()


def test_layout_requires_work_and_scratch_to_be_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polyx.hpc.layout import ClusterLayout

    monkeypatch.setenv("POLYX_HPC_WORK", str(tmp_path / "work"))
    monkeypatch.delenv("POLYX_HPC_SCRATCH", raising=False)

    with pytest.raises(ValueError, match="POLYX_HPC_SCRATCH"):
        ClusterLayout.from_env()


# ── SLURM submission ──


def test_submit_script_requests_one_gpu_and_the_short_test_quality_of_service() -> None:
    text = (ROOT / "cluster" / "submit_smoke.sh").read_text(encoding="utf-8")

    assert "--gres=gpu:1" in text
    assert "--qos=" in text
    assert "POLYX_HPC_QOS" in text
    assert "short" in text
    assert "--account=" in text
    assert "sbatch" in text


def test_job_body_runs_the_environment_check_the_short_training_and_the_sync_back() -> None:
    text = (ROOT / "cluster" / "smoke_job.sh").read_text(encoding="utf-8")

    assert "python -m polyx.hpc.check_env" in text
    assert "polyx.pipeline.command" in text
    assert "protocol.epochs=1" in text
    assert "backend=slurm" in text
    assert "polyx.hpc.sync_back" in text or "sync_back" in text


def test_same_command_composes_the_slurm_backend() -> None:
    from polyx.pipeline.config import load_config

    config = load_config(
        ["backend=slurm", "data_yaml=dataset.yaml", "output_dir=execution", "protocol.epochs=1"]
    )

    assert config.backend.name == "slurm"
    assert config.backend.device == "cuda"
    assert config.protocol.epochs == 1


# ── Small-artifact sync-back ──


def test_sync_back_copies_only_the_agreed_small_artifacts(tmp_path: Path) -> None:
    from polyx.hpc.sync_back import sync_back
    from polyx.pipeline.artifacts import ArtifactLayout, artifact_manifest

    run = tmp_path / "run"
    layout = ArtifactLayout(run)
    layout.create()
    expected = {Path(path).relative_to(run) for path in artifact_manifest(layout).values()}
    for relative in expected:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative.name, encoding="utf-8")
    # Large inputs and full checkpoints stay on the cluster.
    (run / "checkpoints" / "epoch_099.pt").write_bytes(b"y" * 5000)
    (run / "dataset").mkdir()
    (run / "dataset" / "train.zip").write_bytes(b"x" * 5000)

    destination = tmp_path / "local"
    copied = sync_back(run, destination, max_bytes=1024)

    assert set(copied) == expected
    for relative in expected:
        assert (destination / relative).is_file(), f"not synced back: {relative}"
    assert not (destination / "dataset").exists()
    assert not (destination / "checkpoints" / "epoch_099.pt").exists()


def test_sync_back_refuses_an_oversized_artifact_instead_of_silently_skipping_it(
    tmp_path: Path,
) -> None:
    from polyx.hpc.sync_back import SyncError, sync_back

    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "metrics.json").write_bytes(b"z" * 2000)

    with pytest.raises(SyncError, match="too large"):
        sync_back(run, tmp_path / "local", max_bytes=1024)


# ── Publication acknowledgement ──


def test_the_required_acknowledgement_wording_is_recorded() -> None:
    text = (ROOT / "docs" / "hpc-submission.md").read_text(encoding="utf-8")
    expected = (
        "This research/project/thesis/course was partially supported by "
        "HPC OC\u00c9ANO (FONDEQUIP N\u00baEQM170214)."
    )

    assert expected in text
    assert "Only the resulting publication is emailed to the cluster contacts" in text


# ── Documentation contracts ──


def test_administrator_checklist_covers_every_confirmation() -> None:
    text = (ROOT / "docs" / "hpc-submission.md").read_text(encoding="utf-8")

    for phrase in (
        "container runtime",
        "GPU request syntax",
        "login-node internet",
        "account category",
        "quotas",
        "data-origin declaration",
        "sponsor letter",
        "long quality-of-service",
    ):
        assert phrase in text, f"checklist misses: {phrase}"


def test_smoke_procedure_is_exact_and_marked_pending() -> None:
    text = (ROOT / "docs" / "hpc-submission.md").read_text(encoding="utf-8")

    for phrase in (
        "GPU visibility",
        "architecture list",
        "AMP pass",
        "short training execution",
        "python -m polyx.hpc.check_env",
        "sbatch",
        "Pending",
    ):
        assert phrase in text, f"smoke procedure misses: {phrase}"


def test_no_credentials_usernames_or_personal_paths_are_tracked() -> None:
    files = (
        "cluster/submit_smoke.sh",
        "cluster/smoke_job.sh",
        "docs/hpc-submission.md",
        "docs/hpc-environment.example.env",
        "src/polyx/hpc/layout.py",
        "src/polyx/hpc/sync_back.py",
    )
    for relative in files:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "/home/" not in text, f"absolute personal path in {relative}"
        assert "/Users/" not in text, f"absolute personal path in {relative}"
        assert not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text), (
            f"email address in {relative}"
        )

    example = (ROOT / "docs" / "hpc-environment.example.env").read_text(encoding="utf-8")
    for line in example.splitlines():
        if line.startswith("POLYX_") and "=" in line:
            assert line.rstrip().endswith("="), f"committed value in the example config: {line}"
