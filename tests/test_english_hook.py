"""The English pre-commit hook reports staged lines and whole ``src/`` files.

It runs as a command in a throwaway repository, so the contract exercised is
the one pre-commit sees: repo-relative staged files, no Unix shell, git found
through normal executable lookup. Without a staged diff, a ``src/`` file is
scanned in full while a ``tests/`` file keeps the added-lines-only rule.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from tests.english_scan import scan_python_source

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "tools" / "check_english_ids.py"
GIT = shutil.which("git")

SPANISH_SOURCE = (
    '"""Módulo de cálculo de partículas."""\n'
    "\n"
    "# Esta función calcula la ruta\n"
    "def calcular_ruta():\n"
    "    return 1\n"
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    assert GIT is not None, "the test suite needs git on PATH"
    result = subprocess.run(
        [GIT, *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def _write(repo: Path, relative: str, source: str) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _stage(repo: Path, relative: str) -> None:
    _git(repo, "add", "--", relative)


def _run_hook(repo: Path, *paths: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(HOOK), *paths],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_staged_spanish_identifier_comment_and_docstring_fail(tmp_path):
    repo = _repo(tmp_path)
    _write(
        repo,
        "sample.py",
        '"""Calcula la ruta del archivo."""\n'
        "\n"
        "\n"
        "def calcular_ruta():\n"
        "    # Esta función calcula la ruta del archivo\n"
        "    return 1\n",
    )
    _stage(repo, "sample.py")

    result = _run_hook(repo, "sample.py")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "sample.py:1:" in result.stdout
    assert "sample.py:4:" in result.stdout
    assert "sample.py:5:" in result.stdout
    assert "spanish-identifier:" in result.stdout
    assert "spanish-word:" in result.stdout


def test_staged_english_file_passes(tmp_path):
    repo = _repo(tmp_path)
    _write(
        repo,
        "sample.py",
        '"""Return the file path."""\n'
        "\n"
        "\n"
        "def compute_path():\n"
        "    # Compute the file path for the caller.\n"
        "    return 1\n",
    )
    _stage(repo, "sample.py")

    result = _run_hook(repo, "sample.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""


def test_only_newly_added_lines_are_reported(tmp_path):
    repo = _repo(tmp_path)
    committed = "# Esta función calcula la ruta del archivo\n"
    _write(repo, "sample.py", committed)
    _stage(repo, "sample.py")
    _git(repo, "commit", "-q", "-m", "seed")

    # An English addition keeps the committed Spanish line silent.
    _write(repo, "sample.py", committed + "VALUE = 1\n")
    _stage(repo, "sample.py")
    clean = _run_hook(repo, "sample.py")

    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert clean.stdout == ""

    # A new Spanish line is reported, and still not the committed one.
    _write(repo, "sample.py", committed + "VALUE = 1\ndef obtener_estado():\n    return VALUE\n")
    _stage(repo, "sample.py")
    dirty = _run_hook(repo, "sample.py")

    assert dirty.returncode != 0, dirty.stdout + dirty.stderr
    assert "sample.py:3:" in dirty.stdout
    assert "sample.py:1:" not in dirty.stdout


def test_committed_spanish_source_file_fails_without_staged_diff(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "src/sample.py", "def calcular_ruta():\n    return 1\n")
    _stage(repo, "src/sample.py")
    _git(repo, "commit", "-q", "-m", "seed")

    result = _run_hook(repo, "src/sample.py")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "src/sample.py:1:" in result.stdout
    assert "spanish-identifier:" in result.stdout


def test_committed_spanish_test_file_passes_without_staged_diff(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "tests/sample.py", "def calcular_ruta():\n    return 1\n")
    _stage(repo, "tests/sample.py")
    _git(repo, "commit", "-q", "-m", "seed")

    result = _run_hook(repo, "tests/sample.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""


def test_scan_python_source_full_scan_reports_every_matcher():
    found = scan_python_source(SPANISH_SOURCE)

    assert {1, 3, 4} <= {line for line, _finding in found}
    findings = [finding for _line, finding in found]
    assert any(finding.startswith("spanish-word:") for finding in findings)
    assert any(finding.startswith("comment-word:") for finding in findings)
    assert any(finding.startswith("spanish-identifier:") for finding in findings)


def test_scan_python_source_only_lines_restricts_findings():
    found = scan_python_source(SPANISH_SOURCE, only_lines={4})

    assert found
    assert {line for line, _finding in found} == {4}
    assert all(finding.startswith("spanish-identifier:") for _line, finding in found)
