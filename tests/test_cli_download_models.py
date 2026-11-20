"""Weight downloads are verified or refused, never fetched blind.

The public boundary is ``polyx.cli.download_models.download``: it streams a URL
into a destination and keeps the file only when its SHA-256 matches the digest
published with the release. The tests point ``file://`` URLs at bytes written
into ``tmp_path``, so nothing reaches the network and no release has to exist.

The matching digest is the FIPS 180-2 test vector for ``b"abc"`` — a published
constant, not a digest recomputed by the test and not an invented artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polyx.cli import download_models

PAYLOAD = b"abc"
#: FIPS 180-2 / NIST example for the ASCII message ``abc``.
PAYLOAD_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
WRONG_SHA256 = "0" * 64


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def test_a_matching_digest_writes_the_file(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(PAYLOAD)
    destination = tmp_path / "models" / "weights.pt"

    result = download_models.download(
        _file_url(source), destination, expected_sha256=PAYLOAD_SHA256
    )

    assert result == destination
    assert destination.read_bytes() == PAYLOAD
    assert [path.name for path in destination.parent.iterdir()] == ["weights.pt"]


def test_a_wrong_digest_raises_and_leaves_no_partial_file(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(PAYLOAD)
    destination = tmp_path / "models" / "weights.pt"

    with pytest.raises(RuntimeError) as failure:
        download_models.download(_file_url(source), destination, expected_sha256=WRONG_SHA256)

    assert "SHA-256" in str(failure.value)
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()
    assert list(destination.parent.iterdir()) == []


def test_download_without_a_digest_raises(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(PAYLOAD)
    destination = tmp_path / "models" / "weights.pt"

    with pytest.raises(TypeError):
        download_models.download(_file_url(source), destination)

    assert not destination.exists()


def test_the_cli_refuses_to_download_without_a_digest(capsys) -> None:
    exit_status = download_models.main(["--url", "https://example.invalid/weights.pt"])

    assert exit_status != 0
    assert "sha256" in capsys.readouterr().err.lower()


def test_the_cli_names_the_missing_url_instead_of_guessing_one(capsys) -> None:
    exit_status = download_models.main(["--sha256", PAYLOAD_SHA256])

    assert exit_status != 0
    assert "url" in capsys.readouterr().err.lower()
