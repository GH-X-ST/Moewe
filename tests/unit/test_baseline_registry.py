from __future__ import annotations

import json

from moewe.baselines import (
    BASELINE_COMMON_SCHEMA_FIELDS,
    BASELINE_METHOD_NAMES,
    BASELINE_REGISTRY,
    B1_UNGOVERNED_PRIMITIVE_SELECTOR,
    B2_FILTER_ONLY_NO_DEGRADATION,
    B3_NO_RETURNABILITY_SELECTOR,
    B6_WIND_AWARE_GUIDANCE,
    B8_LIFT_EVIDENCE_REMOVED,
    baseline_spec,
    instantiate_baseline,
)


def test_all_required_baselines_instantiate_and_return_common_schema() -> None:
    assert len(BASELINE_REGISTRY) == 9
    for baseline_id in BASELINE_REGISTRY:
        baseline = instantiate_baseline(baseline_id)
        record = baseline.evaluate_smoke_case().to_record()
        assert set(BASELINE_COMMON_SCHEMA_FIELDS) <= set(record)
        assert record["method_name"] in BASELINE_METHOD_NAMES
        json.dumps(record, sort_keys=True)


def test_semantic_baseline_properties_cover_required_ablations() -> None:
    assert baseline_spec(B1_UNGOVERNED_PRIMITIVE_SELECTOR).uses_returnability is False
    assert baseline_spec(B2_FILTER_ONLY_NO_DEGRADATION).uses_returnability is True
    assert baseline_spec(B2_FILTER_ONLY_NO_DEGRADATION).uses_degradation is False
    assert baseline_spec(B3_NO_RETURNABILITY_SELECTOR).uses_returnability is False
    assert baseline_spec(B8_LIFT_EVIDENCE_REMOVED).uses_returnability is True
    assert "useful_lift_exposure" in baseline_spec(B8_LIFT_EVIDENCE_REMOVED).removed_objective_terms


def test_wind_aware_baseline_receives_flow_information() -> None:
    baseline = instantiate_baseline(B6_WIND_AWARE_GUIDANCE)
    record = baseline.evaluate_smoke_case(case_family="still_air", scenario_seed=3).to_record()

    assert record["wind_information_available"] is True
    assert record["same_initial_state"] is True
    assert record["same_scenario_seed"] is True
    assert record["same_actuator_limits"] is True
