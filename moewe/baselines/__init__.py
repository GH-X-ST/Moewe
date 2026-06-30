"""Smoke-scale baseline and ablation utilities."""

from .governor_ablation import (
    GovernorAblationComparison,
    UnfilteredCandidateScore,
    UnfilteredPrimitiveSelector,
    UnfilteredPrimitiveSelectorConfig,
    UnfilteredSelectionDecision,
    compare_governor_to_unfiltered,
)
from .reference_tracking import (
    ReferenceTrackingConfig,
    ReferenceTrackingController,
    ReferenceTrackingRolloutRecord,
    build_gate_tracking_target,
    run_reference_tracking_rollout,
)

__all__ = [
    "GovernorAblationComparison",
    "ReferenceTrackingConfig",
    "ReferenceTrackingController",
    "ReferenceTrackingRolloutRecord",
    "UnfilteredCandidateScore",
    "UnfilteredPrimitiveSelector",
    "UnfilteredPrimitiveSelectorConfig",
    "UnfilteredSelectionDecision",
    "build_gate_tracking_target",
    "compare_governor_to_unfiltered",
    "run_reference_tracking_rollout",
]
