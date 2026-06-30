from __future__ import annotations

import numpy as np
import pytest

from moewe.primitives import (
    AcceptanceThresholds,
    EntryPerturbationSpec,
    PrimitiveGrammarSpec,
    PrimitiveRolloutConfig,
    ValidationScenario,
    generate_primitives,
    validate_primitive,
)


def _primitive():
    return generate_primitives(PrimitiveGrammarSpec.smoke())[0]


def test_validation_scenario_rejects_invalid_thresholds() -> None:
    scenario = ValidationScenario(
        scenario_id="bad_thresholds",
        thresholds=AcceptanceThresholds(max_command_abs_rad=0.0),
    )

    with pytest.raises(ValueError, match="max_command_abs_rad"):
        scenario.validate()


def test_entry_perturbation_rejects_non_finite_offsets() -> None:
    offset = [0.0] * 15
    offset[0] = float("nan")
    spec = EntryPerturbationSpec(offset=tuple(offset))

    with pytest.raises(ValueError, match="entry offset"):
        spec.validate()


def test_validation_record_retains_and_rejects_with_explicit_reasons() -> None:
    primitive = _primitive()
    pass_scenario = ValidationScenario(
        scenario_id="pass_smoke",
        rollout_config=PrimitiveRolloutConfig(dt_s=0.01, max_duration_s=0.05),
        thresholds=AcceptanceThresholds(
            min_safety_margin_m=-10.0,
            max_angle_of_attack_rad=10.0,
            max_command_abs_rad=10.0,
            min_terminal_specific_energy_change_j_kg=-100.0,
        ),
    )
    strict_scenario = ValidationScenario(
        scenario_id="reject_energy",
        rollout_config=PrimitiveRolloutConfig(dt_s=0.01, max_duration_s=0.05),
        thresholds=AcceptanceThresholds(min_terminal_specific_energy_change_j_kg=100.0),
    )

    retained = validate_primitive(primitive, pass_scenario)
    rejected = validate_primitive(primitive, strict_scenario)

    assert retained.retention.retained
    assert retained.retention.reason == "retained"
    assert retained.evidence.retained is True
    assert rejected.retention.retained is False
    assert rejected.retention.reason == "terminal_specific_energy_change"
    assert rejected.evidence.retention_reason == "terminal_specific_energy_change"


def test_validation_record_uses_units_and_canonical_command_order() -> None:
    primitive = _primitive()
    result = validate_primitive(
        primitive,
        ValidationScenario(
            scenario_id="record_units",
            rollout_config=PrimitiveRolloutConfig(dt_s=0.01, max_duration_s=0.05),
        ),
    )
    record = result.to_record()

    assert primitive.metadata["command_order"] == "[aileron, elevator, rudder]"
    for key in (
        "min_safety_margin_m",
        "terminal_specific_energy_change_j_kg",
        "max_angle_of_attack_rad",
        "max_command_abs_rad",
        "rollout_duration_s",
    ):
        assert key in record
    assert np.isfinite(record["terminal_specific_energy_change_j_kg"])
