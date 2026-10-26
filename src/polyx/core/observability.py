"""Observability — JSONL logging and run instrumentation (re-export from provenance)."""

import sys

from .provenance import observability as _observability

# Re-export the canonical module object, not a copy of its names: consumer
# code does ``from polyx.core import observability as ob`` and then patche
# module attributes (e.g. ``ob.SIZE_TOTAL = ...`` in tests). A name-by-name
# re-export would copy the value into this shim's namespace, so the patch
# would land here while ``prune`` reads the global in the provenance module.
# Aliasing ``sys.modules[__name__]`` makes both names resolve to the very
# same module object, and the patch reaches the implementation.
sys.modules[__name__] = _observability
