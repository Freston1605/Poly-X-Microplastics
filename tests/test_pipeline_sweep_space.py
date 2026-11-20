"""The search space a sweep may declare, and the fields it may never touch."""

from __future__ import annotations

from typing import Any

import pytest

from polyx.pipeline.sweep_space import (
    SEARCHABLE_PROTOCOL_FIELDS,
    validate_direction,
    validate_sampled_parameters,
    validate_search_space,
)


class _Trial:
    """The part of an Optuna trial the guard reads."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params


def test_every_declared_hyperparameter_is_accepted() -> None:
    validate_search_space(f"protocol.{field}" for field in sorted(SEARCHABLE_PROTOCOL_FIELDS))


@pytest.mark.parametrize(
    "name",
    [
        "protocol.split",
        "protocol.seed",
        "protocol.augmentation_subset",
        "protocol.early_stopping.patience",
        "protocol.confidence",
        "evaluator",
        "family.name",
    ],
)
def test_a_fixed_benchmark_field_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="only declare protocol hyperparameters"):
        validate_search_space([name])


def test_a_custom_search_space_cannot_smuggle_a_fixed_field() -> None:
    """The plugin's callback suggests names after the declared space is checked."""
    with pytest.raises(ValueError, match="only declare protocol hyperparameters"):
        validate_sampled_parameters(
            [_Trial({"protocol.learning_rate": 0.01, "protocol.split": "val"})]
        )


def test_declared_only_trial_parameters_pass() -> None:
    validate_sampled_parameters([_Trial({"protocol.learning_rate": 0.01})])
    validate_sampled_parameters([_Trial({})])


def test_the_direction_is_normalized_and_must_maximize() -> None:
    assert validate_direction("maximize") == "maximize"
    assert validate_direction("MAXIMIZE") == "maximize"

    with pytest.raises(ValueError, match="must be 'maximize'"):
        validate_direction("minimize")
