"""Hatch build hook: compile the gettext catalogs before the wheel is built.

The ``.mo`` files are generated artifacts (gitignored), so a clean checkout
has none. Compiling them in place during ``initialize`` lets the existing
``packages = ["src/polyx"]`` target carry them into the wheel without a second
file list. The pseudo-locale is excluded by ``[tool.hatch.build.targets.wheel]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # pragma: no cover - only importable inside a Hatch build
    BuildHookInterface = object  # type: ignore[assignment,misc]


class CustomBuildHook(BuildHookInterface):  # type: ignore[misc]
    """Compile every shipped catalog into its ``.mo`` sibling."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        from tools.i18n.packaging import compile_catalogs

        compile_catalogs(Path(self.root))
