"""Quality metrics and dataset audit."""

from .metrics import (
    iou,
    match_image,
    aggregate_per_class,
    confusion_matrix,
    MatchResult,
    ClassMetrics,
)
from .assignment import (
    mark_assignables,
    class_id_reported,
    SummaryAssignment,
    NO_ASSIGNABLE,
)
from .dataset_audit import (
    audit_dataset,
    Finding,
    DatasetAudit,
    LEVEL_ERROR,
    LEVEL_WARN,
)

__all__ = [
    "iou",
    "match_image",
    "aggregate_per_class",
    "confusion_matrix",
    "MatchResult",
    "ClassMetrics",
    "mark_assignables",
    "class_id_reported",
    "SummaryAssignment",
    "NO_ASSIGNABLE",
    "audit_dataset",
    "Finding",
    "DatasetAudit",
    "LEVEL_ERROR",
    "LEVEL_WARN",
]
