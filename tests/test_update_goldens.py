"""CLI seam for ``tools/update_goldens.py``.

The updater is exercised as a subprocess, the way an agent runs it. A
simulated pytest report (``--report``) selects which assertion failed so the
test never has to wait for a real render to drift.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "tools" / "update_goldens.py"

_OLD_FIRST = "1" * 64
_OLD_SECOND = "2" * 64
_NEW_FIRST = "a" * 64
_NEW_SECOND = "b" * 64


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _line_of(source: str, needle: str) -> int:
    for number, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} is not in the test source")


def _block(path: Path, line: int, test: str, old: str, new: str) -> str:
    return (
        f"__________________________________ {test} __________________________________\n"
        f"{path}:{line}: in {test}\n"
        "    assert hashlib.sha256(view).hexdigest() == _GOLDEN\n"
        "E   AssertionError: assert 'actual' == 'expected'\n"
        "E     \n"
        f"E     - {old}\n"
        f"E     + {new}\n"
    )


def _report(*blocks: str) -> str:
    return (
        "============================= test session starts ==============================\n"
        + "".join(blocks)
        + "=========================== short test summary info ============================\n"
    )


def _write_test(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "test_goldens_under_test.py"
    path.write_text(source, encoding="utf-8")
    return path


_TWO_CONSTANTS = f"""\
import hashlib

_GOLDEN_FIRST = "{_OLD_FIRST}"
_GOLDEN_SECOND = "{_OLD_SECOND}"


def test_first():
    view = "first"
    assert hashlib.sha256(view.encode()).hexdigest() == _GOLDEN_FIRST


def test_second():
    view = "second"
    assert hashlib.sha256(view.encode()).hexdigest() == _GOLDEN_SECOND
"""


def test_only_the_failed_goldens_literal_is_rewritten(tmp_path):
    path = _write_test(tmp_path, _TWO_CONSTANTS)
    original = path.read_bytes()
    report = tmp_path / "report.txt"
    report.write_text(
        _report(
            _block(
                path,
                _line_of(_TWO_CONSTANTS, "== _GOLDEN_SECOND"),
                "test_second",
                _OLD_SECOND,
                _NEW_SECOND,
            )
        ),
        encoding="utf-8",
    )

    result = _run(str(path), "--apply", "--report", str(report))

    assert result.returncode == 0, result.stderr
    assert path.read_bytes() == original.replace(_OLD_SECOND.encode(), _NEW_SECOND.encode())
    assert _OLD_FIRST.encode() in path.read_bytes()
    assert _NEW_SECOND in result.stdout
    assert "test_second" in result.stdout
    assert "_GOLDEN_SECOND" in result.stdout


def test_two_failures_are_mapped_independently_whatever_the_report_order(tmp_path):
    path = _write_test(tmp_path, _TWO_CONSTANTS)
    original = path.read_bytes()
    report = tmp_path / "report.txt"
    # Reversed relative to the file: zipping hashes in report order would
    # swap the goldens. Mapping by assertion must not.
    report.write_text(
        _report(
            _block(
                path,
                _line_of(_TWO_CONSTANTS, "== _GOLDEN_SECOND"),
                "test_second",
                _OLD_SECOND,
                _NEW_SECOND,
            ),
            _block(
                path,
                _line_of(_TWO_CONSTANTS, "== _GOLDEN_FIRST"),
                "test_first",
                _OLD_FIRST,
                _NEW_FIRST,
            ),
        ),
        encoding="utf-8",
    )

    result = _run(str(path), "--apply", "--report", str(report))

    assert result.returncode == 0, result.stderr
    assert path.read_bytes() == original.replace(_OLD_FIRST.encode(), _NEW_FIRST.encode()).replace(
        _OLD_SECOND.encode(), _NEW_SECOND.encode()
    )
    assert "test_first" in result.stdout and "test_second" in result.stdout


def test_ambiguous_assertion_refuses_and_leaves_the_file_byte_identical(tmp_path):
    source = f"""\
import hashlib

_GOLDEN_LEFT = "{_OLD_FIRST}"
_GOLDEN_RIGHT = "{_OLD_SECOND}"


def test_ambiguous():
    view = "both"
    assert hashlib.sha256(view.encode()).hexdigest() in (_GOLDEN_LEFT, _GOLDEN_RIGHT)
"""
    path = _write_test(tmp_path, source)
    original = path.read_bytes()
    report = tmp_path / "report.txt"
    report.write_text(
        _report(
            _block(
                path,
                _line_of(source, "in (_GOLDEN_LEFT"),
                "test_ambiguous",
                _OLD_FIRST,
                _NEW_FIRST,
            )
        ),
        encoding="utf-8",
    )

    result = _run(str(path), "--apply", "--report", str(report))

    assert result.returncode != 0
    assert path.read_bytes() == original
    assert "test_ambiguous" in result.stderr
    assert "_GOLDEN_LEFT" in result.stderr and "_GOLDEN_RIGHT" in result.stderr


def test_unmapped_failure_refuses_and_leaves_the_file_byte_identical(tmp_path):
    source = """\
import hashlib


def test_plain():
    view = "plain"
    assert hashlib.sha256(view.encode()).hexdigest() == "not-a-golden"
"""
    path = _write_test(tmp_path, source)
    original = path.read_bytes()
    report = tmp_path / "report.txt"
    report.write_text(
        _report(
            _block(
                path,
                _line_of(source, '"not-a-golden"'),
                "test_plain",
                "not-a-golden",
                _NEW_FIRST,
            )
        ),
        encoding="utf-8",
    )

    result = _run(str(path), "--apply", "--report", str(report))

    assert result.returncode != 0
    assert path.read_bytes() == original
    assert "test_plain" in result.stderr


def test_inline_golden_is_rewritten_and_dry_run_does_not_write(tmp_path):
    source = f"""\
import hashlib


def test_first():
    assert hashlib.sha256(b"first").hexdigest() == "{_OLD_FIRST}"


def test_second():
    assert hashlib.sha256(b"second").hexdigest() == "{_OLD_SECOND}"
"""
    path = _write_test(tmp_path, source)
    original = path.read_bytes()
    report = tmp_path / "report.txt"
    report.write_text(
        _report(
            _block(
                path,
                _line_of(source, f'"{_OLD_SECOND}"'),
                "test_second",
                _OLD_SECOND,
                _NEW_SECOND,
            )
        ),
        encoding="utf-8",
    )

    dry = _run(str(path), "--report", str(report))

    assert dry.returncode == 0, dry.stderr
    assert path.read_bytes() == original
    assert "test_second" in dry.stdout
    assert _NEW_SECOND in dry.stdout

    applied = _run(str(path), "--apply", "--report", str(report))

    assert applied.returncode == 0, applied.stderr
    assert path.read_bytes() == original.replace(_OLD_SECOND.encode(), _NEW_SECOND.encode())
