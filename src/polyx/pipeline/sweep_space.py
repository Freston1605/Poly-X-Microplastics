"""The one search space and direction a sweep may declare.

ADR-0010 keeps the benchmark's fixed fields out of the search space: a sweep may
tune hyperparameters, never the split, seed, augmentation subset or the
evaluator. Both entry points check their declared space here — the composed
``hydra.sweeper`` node when the study identity is written, and the plugin's own
resolved distributions before it asks Optuna for a trial — so a ``*_sweep``
override cannot reintroduce a field the config refuses. Keeping the list in one
module is what stops the two checks from drifting apart.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

#: Protocol hyperparameters a trial may change. Split, seed, augmentation
#: subset and the evaluator are outside this set and cannot be suggested.
SEARCHABLE_PROTOCOL_FIELDS = frozenset(
    {
        "epochs",
        "batch",
        "image_size",
        "optimizer",
        "learning_rate",
        "momentum",
        "weight_decay",
        "schedule",
        "warmup_epochs",
    }
)

#: The shared with-class score is maximized; no sweep optimizes a loss.
SWEEP_DIRECTION = "maximize"


def validate_search_space(names: Iterable[str]) -> None:
    """Refuse any declared name outside ``protocol.<searchable field>``."""
    for name in names:
        parts = str(name).split(".")
        if len(parts) != 2 or parts[0] != "protocol" or parts[1] not in SEARCHABLE_PROTOCOL_FIELDS:
            raise ValueError(
                "search space may only declare protocol hyperparameters: "
                f"{name} (allowed: {', '.join(sorted(SEARCHABLE_PROTOCOL_FIELDS))})"
            )


def validate_sampled_parameters(trials: Iterable[Any]) -> None:
    """Refuse a parameter a trial actually sampled outside the allowlist.

    ``validate_search_space`` sees the declared distributions. The plugin's
    ``custom_search_space`` callback is invoked afterwards and may suggest any
    name directly (``trial.suggest_categorical("protocol.split", ...)``), so the
    trial's own sampled parameters are the last place the rule can be enforced
    before a job is launched.
    """
    for trial in trials:
        validate_search_space(getattr(trial, "params", None) or {})


def validate_direction(value: Any) -> str:
    """Return the normalized direction, refusing anything but maximize."""
    name = getattr(value, "name", value)
    text = str(name).lower()
    if text.startswith("direction."):
        text = text.split(".", 1)[1]
    if text != SWEEP_DIRECTION:
        raise ValueError(
            f"the sweep direction must be {SWEEP_DIRECTION!r}: the objective is the "
            "shared evaluator's with-class score and no loss is ever optimized"
        )
    return text
