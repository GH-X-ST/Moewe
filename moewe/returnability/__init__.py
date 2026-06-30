"""Returnability graph construction from structured primitive evidence."""

from .build import build_returnability_graph, build_returnability_graph_from_report, transitions_from_structured_report
from .graph import PrimitiveTransition, ReturnabilityGraph, ReturnabilityGraphConfig
from .report import ReturnabilityReport, map_compressed_tier_to_returnability
from .sets import ReturnabilityClassSets, compute_recoverable_classes, compute_returnability_class_sets

__all__ = [
    "PrimitiveTransition",
    "ReturnabilityClassSets",
    "ReturnabilityGraph",
    "ReturnabilityGraphConfig",
    "ReturnabilityReport",
    "build_returnability_graph",
    "build_returnability_graph_from_report",
    "compute_recoverable_classes",
    "compute_returnability_class_sets",
    "map_compressed_tier_to_returnability",
    "transitions_from_structured_report",
]
