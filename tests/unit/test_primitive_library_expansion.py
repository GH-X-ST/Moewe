from __future__ import annotations

from moewe.primitives import (
    build_structured_library_design_cases,
    dense_smoke_grammar,
    expand_primitive_grammar,
    run_validation_sweep,
)


def test_dense_grammar_expansion_is_deterministic() -> None:
    expanded = expand_primitive_grammar(dense_smoke_grammar())
    first = expanded.generate(max_primitives=5)
    second = expanded.generate(max_primitives=5)

    assert expanded.candidate_count == 400
    assert [candidate.primitive_id for candidate in first] == [candidate.primitive_id for candidate in second]
    assert len({candidate.primitive_id for candidate in first}) == len(first)


def test_structured_design_cases_have_required_ids_and_reproducible_records() -> None:
    cases = build_structured_library_design_cases()
    case_ids = [case.case_id for case in cases]

    assert case_ids == [
        "design_still_air_trim_recovery",
        "design_still_air_gate_alignment",
        "design_lift_before_gate",
        "design_lift_at_gate",
        "design_lift_after_gate",
        "design_lateral_lift",
        "design_strong_lift",
        "design_broad_lift",
        "design_compact_lift",
        "design_two_source_symmetric_lift",
        "design_two_source_asymmetric_lift",
        "design_recovery_only",
    ]
    assert all(case.case_set == "library_design" for case in cases)
    assert all(not case.randomized for case in cases)
    assert all(case.seed is None for case in cases)

    scenarios = tuple(case.to_validation_scenario() for case in cases[:2])
    first = run_validation_sweep(scenarios=scenarios, max_primitives=2)
    second = run_validation_sweep(scenarios=scenarios, max_primitives=2)

    assert first.to_records() == second.to_records()
    assert first.to_summary() == second.to_summary()
