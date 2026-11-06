"""Helper to regenerate characterization golden hashes.

Usage:
    python tools/update_goldens.py <test_file> [--apply] [--report <path>]

The updater runs pytest for one test file, reads each failure's location and
the single actual hash in its diff, resolves the assertion at that location
with ``ast``, and rewrites only the golden literal that assertion names. The
association must be one-to-one: an assertion that names zero or several
goldens, a failure with several actual hashes, or two failures claiming the
same literal all abort the run. A refusal exits non-zero and writes nothing,
so a failed update is never a partial update.

``--report <path>`` reads an existing pytest report instead of running pytest.
This is the seam the CLI tests use. With ``--apply`` the mapped literals are
rewritten in place; without it the mapping is only printed.

Goldens are string literals of 64 lowercase hex characters, either inline in
the assertion or behind one module-level constant (``_GOLDEN = "..."``).
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_HEX_TEXT = re.compile(r"[0-9a-f]{64}")
_HEX_BYTES = re.compile(rb"[0-9a-f]{64}")
_ACTUAL = re.compile(r"^\s*(?:E\s+)?\+\s+([0-9a-f]{64})\s*$")
_FRAME = re.compile(r"^(?P<path>.+?):(?P<line>\d+): in (?P<test>\S+)\s*$")
_HEADING = re.compile(r"^_{3,}\s+.+?\s+_{3,}$")


@dataclass
class Failure:
    """One pytest failure and the assertion it came from."""

    test: str
    path: str
    line: int
    hashes: list[str]
    frames: list[tuple[str, int, str]]


@dataclass
class Replacement:
    """A single golden literal to rewrite."""

    test: str
    label: str
    assertion_line: int
    literal_line: int
    old: str
    new: str
    span: tuple[int, int]


def _blocks(report: str) -> list[list[str]]:
    """Split a pytest report into per-test failure blocks."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in report.splitlines():
        if _HEADING.match(line):
            if current:
                blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _same_file(reported: str, target: Path) -> bool:
    """Whether a report frame points at the test file we were given."""
    candidate = Path(reported)
    resolved_target = target.resolve()
    if candidate.is_absolute():
        return candidate.resolve() == resolved_target
    if resolved_target == (Path.cwd() / candidate).resolve():
        return True
    # A report frame is often relative to pytest's rootdir, not our cwd, so
    # match its path components against the tail of the resolved target.
    parts = candidate.parts
    return bool(parts) and resolved_target.parts[-len(parts) :] == parts


def _failures(report: str, target: Path) -> list[Failure]:
    """Extract the target file's failures from a pytest report."""
    failures: list[Failure] = []
    for block in _blocks(report):
        frames: list[tuple[str, int, str]] = []
        hashes: list[str] = []
        for line in block:
            frame = _FRAME.match(line)
            if frame is not None:
                frames.append((frame.group("path"), int(frame.group("line")), frame.group("test")))
                continue
            actual = _ACTUAL.match(line)
            if actual is not None:
                hashes.append(actual.group(1))
        matching = [frame for frame in frames if _same_file(frame[0], target)]
        if not matching:
            continue
        # The outermost frame identifies the test; the innermost one is the
        # assertion that raised, so report its location when mapping refuses.
        outermost = matching[0]
        innermost = matching[-1]
        failures.append(
            Failure(
                test=outermost[2],
                path=innermost[0],
                line=innermost[1],
                hashes=hashes,
                frames=matching,
            )
        )
    return failures


def _report_has_failures(report: str) -> bool:
    """Whether a pytest report contains at least one failure frame."""
    return any(_FRAME.match(line) is not None for block in _blocks(report) for line in block)


def _module_constants(tree: ast.Module) -> dict[str, ast.Constant]:
    """Module-level names bound to a 64-hex string literal."""
    constants: dict[str, ast.Constant] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not _is_hex(value.value):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
    return constants


def _is_hex(value: object) -> bool:
    return isinstance(value, str) and _HEX_TEXT.fullmatch(value) is not None


def _golden_refs(
    node: ast.Assert, constants: dict[str, ast.Constant]
) -> dict[int, tuple[str, ast.Constant]]:
    """Golden literals named by one assertion, keyed by node identity."""
    refs: dict[int, tuple[str, ast.Constant]] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and _is_hex(child.value):
            refs[id(child)] = ("<inline>", child)
        elif isinstance(child, ast.Name) and child.id in constants:
            constant = constants[child.id]
            refs[id(constant)] = (child.id, constant)
    return refs


def _assertion_at(asserts: list[ast.Assert], line: int) -> ast.Assert | None:
    """The assertion statement whose source range contains ``line``."""
    containing = [
        node for node in asserts if node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if not containing:
        return None
    # A frame points at the assert keyword; prefer the nearest start.
    return min(containing, key=lambda node: abs(node.lineno - line))


def _span(source: bytes, node: ast.Constant) -> tuple[int, int]:
    """Byte range of a literal node in the UTF-8 source."""
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset
    end = sum(len(line) for line in lines[: (node.end_lineno or node.lineno) - 1])
    end += node.end_col_offset or 0
    return start, end


def _replace_literal(segment: bytes, new: str) -> bytes | None:
    """Replace the single golden literal in ``segment``, or return ``None``."""
    replaced, count = _HEX_BYTES.subn(new.encode(), segment, count=1)
    return replaced if count == 1 else None


def _map_failure(
    failure: Failure,
    source: bytes,
    tree: ast.Module,
    constants: dict[str, ast.Constant],
) -> tuple[Replacement | None, str | None]:
    """Resolve one failure to exactly one golden literal, or explain why not."""
    if len(failure.hashes) != 1:
        return None, (f"expected exactly one actual hash, found {len(failure.hashes)}")

    asserts = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    # The innermost assertion frame is the one that raised; an outer assertion
    # that merely called it must not be rewritten in its place.
    assertion: ast.Assert | None = None
    for _path, line, _test in reversed(failure.frames):
        assertion = _assertion_at(asserts, line)
        if assertion is not None:
            break

    if assertion is None:
        return None, "no assertion statement found at the failure location"
    refs = _golden_refs(assertion, constants)
    if not refs:
        return None, "assertion references no golden literal"
    if len(refs) != 1:
        labels = ", ".join(sorted(label for label, _ in refs.values()))
        return None, f"assertion references {len(refs)} goldens: {labels}"

    label, literal = next(iter(refs.values()))
    start, end = _span(source, literal)
    segment = source[start:end]
    if _replace_literal(segment, failure.hashes[0]) is None:
        return None, f"could not locate the literal for {label} on line {literal.lineno}"
    return (
        Replacement(
            test=failure.test,
            label=label,
            assertion_line=assertion.lineno,
            literal_line=literal.lineno,
            old=literal.value,
            new=failure.hashes[0],
            span=(start, end),
        ),
        None,
    )


def _plan_replacements(
    failures: list[Failure], source: bytes, tree: ast.Module
) -> tuple[list[Replacement], list[str]]:
    """Map every failure, refusing the whole batch on any ambiguity."""
    constants = _module_constants(tree)
    replacements: list[Replacement] = []
    errors: list[str] = []
    claimed: dict[tuple[int, int], Replacement] = {}

    for failure in failures:
        replacement, reason = _map_failure(failure, source, tree, constants)
        if replacement is None:
            errors.append(f"  {failure.test} ({failure.path}:{failure.line}): {reason}")
            continue
        key = replacement.span
        if key in claimed:
            previous = claimed[key]
            errors.append(
                f"  {failure.test} ({failure.path}:{failure.line}): golden "
                f"{replacement.label} is also claimed by {previous.test}"
            )
            continue
        claimed[key] = replacement
        replacements.append(replacement)

    return replacements, errors


def _rewrite(source: bytes, replacements: list[Replacement]) -> bytes:
    """Splice only the mapped literal ranges, leaving every other byte alone."""
    updated = source
    for replacement in sorted(replacements, key=lambda item: item.span[0], reverse=True):
        start, end = replacement.span
        segment = updated[start:end]
        replaced = _replace_literal(segment, replacement.new)
        updated = updated[:start] + (segment if replaced is None else replaced) + updated[end:]
    return updated


def _print_mapping(test_file: Path, replacements: list[Replacement]) -> None:
    print(f"{test_file}: {len(replacements)} golden mismatch(es)")
    for replacement in replacements:
        print(
            f"  {replacement.test} (assertion at line {replacement.assertion_line}, "
            f"{replacement.label} on line {replacement.literal_line}): "
            f"{replacement.old} -> {replacement.new}"
        )


def _pytest_report(test_file: Path) -> tuple[str, int]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-q",
            "--no-header",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr, result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate golden hashes")
    parser.add_argument("test_file", help="Test file containing golden hashes")
    parser.add_argument("--apply", action="store_true", help="Apply new hashes")
    parser.add_argument(
        "--report",
        type=Path,
        help="Read a pytest failure report from this file instead of running pytest",
    )
    args = parser.parse_args(argv)

    test_file = Path(args.test_file)
    if not test_file.is_file():
        print(f"File not found: {test_file}", file=sys.stderr)
        return 1

    pytest_code: int | None = None
    if args.report is not None:
        report = args.report.read_text(encoding="utf-8")
    else:
        report, pytest_code = _pytest_report(test_file)

    failures = _failures(report, test_file)
    if not failures:
        if pytest_code not in (None, 0):
            print(
                f"Refusing to update {test_file}: pytest exited with {pytest_code} "
                "and none of its failures point at this file. No bytes were written.",
                file=sys.stderr,
            )
            return pytest_code or 1
        if _report_has_failures(report):
            print(
                f"Refusing to update {test_file}: the report contains failures but "
                "none of their frames belong to this file, so no golden can be "
                "mapped. No bytes were written.",
                file=sys.stderr,
            )
            return 1
        print("No hash mismatches found — goldens are up to date.")
        return 0

    source = test_file.read_bytes()
    tree = ast.parse(source.decode("utf-8"), filename=str(test_file))
    replacements, errors = _plan_replacements(failures, source, tree)
    if errors:
        print(
            f"Refusing to update {test_file}: {len(errors)} failure(s) cannot be "
            "mapped to exactly one golden.",
            file=sys.stderr,
        )
        print("\n".join(errors), file=sys.stderr)
        print("No bytes were written.", file=sys.stderr)
        return 1

    _print_mapping(test_file, replacements)
    if not args.apply:
        print("Dry run — re-run with --apply to rewrite the mapped literals.")
        return 0

    test_file.write_bytes(_rewrite(source, replacements))
    print(f"Updated {len(replacements)} golden(s) in {test_file}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
