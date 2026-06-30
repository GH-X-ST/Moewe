"""Build returnability graphs from structured library-design evidence."""

from __future__ import annotations

from moewe.primitives.structured import StructuredLibraryBuildReport, build_structured_primitive_library

from .graph import PrimitiveTransition, ReturnabilityGraph, ReturnabilityGraphConfig
from .sets import compute_returnability_class_sets


def transitions_from_structured_report(report: StructuredLibraryBuildReport) -> tuple[PrimitiveTransition, ...]:
    """Extract primitive-labelled graph transitions from a structured build report."""

    transitions: list[PrimitiveTransition] = []
    case_set_by_id = {case.case_id: case.case_set for case in report.design_cases}
    for result in report.validation_report.results:
        evidence = result.evidence
        design_case_id = result.scenario_id
        transitions.append(
            PrimitiveTransition(
                primitive_id=result.primitive_id,
                design_case_id=design_case_id,
                case_set=case_set_by_id[design_case_id],
                family=result.family,
                controller_type=result.controller_type,
                entry_class=result.entry_class.label,
                exit_class=result.exit_class.label,
                retained=result.retention.retained,
                rollout_success=evidence.rollout_success,
                min_safety_margin_m=float(evidence.min_safety_margin_m),
                terminal_specific_energy_change_j_kg=float(evidence.terminal_specific_energy_change_j_kg),
                terminal_specific_energy_margin_j_kg=evidence.terminal_specific_energy_margin_j_kg,
                max_angle_of_attack_rad=float(evidence.max_angle_of_attack_rad),
                max_command_abs_rad=float(evidence.max_command_abs_rad),
                failure_reason=evidence.failure_reason,
                retention_reason=result.retention.reason,
                scenario_id=result.scenario_id,
                seed=result.seed,
            )
        )
    return tuple(sorted(transitions, key=lambda item: item.transition_id))


def build_returnability_graph_from_report(
    report: StructuredLibraryBuildReport,
    config: ReturnabilityGraphConfig | None = None,
) -> ReturnabilityGraph:
    """Build a returnability graph from a library-design structured primitive-library report."""

    transitions = transitions_from_structured_report(report)
    class_sets = compute_returnability_class_sets(transitions, config=config)
    return ReturnabilityGraph(
        transitions=transitions,
        safe_classes=class_sets.safe_classes,
        forbidden_classes=class_sets.forbidden_classes,
        terminal_success_classes=class_sets.terminal_success_classes,
        recoverable_classes=class_sets.recoverable_classes,
        dead_end_classes=class_sets.dead_end_classes,
        entry_supported_classes=class_sets.entry_supported_classes,
        exit_observed_classes=class_sets.exit_observed_classes,
    )


def build_returnability_graph(
    report: StructuredLibraryBuildReport | None = None,
    config: ReturnabilityGraphConfig | None = None,
    max_primitives: int | None = None,
) -> ReturnabilityGraph:
    """Build the default library-design returnability graph without random challenge cases."""

    structured_report = build_structured_primitive_library(max_primitives=max_primitives) if report is None else report
    return build_returnability_graph_from_report(structured_report, config=config)
