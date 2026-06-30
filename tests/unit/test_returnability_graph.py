from __future__ import annotations

from moewe.primitives import build_structured_library_design_cases, build_structured_primitive_library
from moewe.returnability import build_returnability_graph_from_report


def _small_graph():
    cases = build_structured_library_design_cases()[:2]
    report = build_structured_primitive_library(design_cases=cases, max_primitives=3)
    return report, build_returnability_graph_from_report(report)


def test_returnability_graph_is_deterministic_and_traced_to_design_cases() -> None:
    first_report, first = _small_graph()
    _, second = _small_graph()

    assert first.to_records() == second.to_records()
    assert len(first.transitions) == first_report.validated_rollout_count
    assert {transition.design_case_id for transition in first.transitions} == {"design_still_air_trim_recovery", "design_still_air_gate_alignment"}
    assert {transition.case_set for transition in first.transitions} == {"library_design"}
    assert all(transition.primitive_id for transition in first.transitions)
    assert all(transition.entry_class for transition in first.transitions)
    assert all(transition.exit_class for transition in first.transitions)


def test_returnability_graph_exposes_required_class_sets() -> None:
    _, graph = _small_graph()

    assert graph.entry_supported_classes
    assert graph.exit_observed_classes
    assert graph.safe_classes
    assert graph.recoverable_classes <= graph.safe_classes
    assert graph.dead_end_classes <= graph.safe_classes
    assert graph.forbidden_classes.isdisjoint(graph.safe_classes)
    assert graph.retained_transitions()
    assert graph.to_summary()["transition_count"] == len(graph.transitions)
