"""Stats helpers (re-exported from polyx.core.stats)."""

from polyx.core import stats as _core_stats

globals().update({k: v for k, v in vars(_core_stats).items() if not k.startswith("__")})
