"""Returnability graph construction from structured primitive evidence."""

from .build import build_returnability_graph, build_returnability_graph_from_report, transitions_from_structured_report
from .certificate import (
    ReturnabilityCertificate,
    ReturnabilityEdgeEvidence,
    ReturnabilityThresholds,
    compute_empirical_returnability_certificate,
)
from .graph import PrimitiveTransition, ReturnabilityGraph, ReturnabilityGraphConfig
from .report import ReturnabilityReport, map_compressed_tier_to_returnability
from .sets import ReturnabilityClassSets, compute_recoverable_classes, compute_returnability_class_sets

__all__ = [
    "PrimitiveTransition",
    "ReturnabilityCertificate",
    "ReturnabilityClassSets",
    "ReturnabilityEdgeEvidence",
    "ReturnabilityGraph",
    "ReturnabilityGraphConfig",
    "ReturnabilityReport",
    "ReturnabilityThresholds",
    "build_returnability_graph",
    "build_returnability_graph_from_report",
    "compute_empirical_returnability_certificate",
    "compute_recoverable_classes",
    "compute_returnability_class_sets",
    "map_compressed_tier_to_returnability",
    "transitions_from_structured_report",
]
