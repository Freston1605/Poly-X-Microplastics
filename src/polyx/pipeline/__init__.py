"""Headless, model- and platform-agnostic training pipeline.

A parallel path to the shipped GUI Trainer: Hydra composes a framework-neutral
protocol, one Lightning loop runs it for every family adapter, and one shared
evaluator scores every family in the same COCO currency.
"""

from polyx.pipeline.config import ExecutionConfig, Protocol, load_config
from polyx.pipeline.execution import run_execution

__all__ = ["ExecutionConfig", "Protocol", "load_config", "run_execution"]
