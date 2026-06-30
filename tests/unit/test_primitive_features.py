from __future__ import annotations

import numpy as np

from moewe.primitives import (
    AcceptanceThresholds,
    PrimitiveRolloutConfig,
    ValidationScenario,
    extract_behaviour_feature,
    feature_vector,
    run_validation_sweep,
)


def _report():
    scenario = ValidationScenario(
        scenario_id="feature_smoke",
        rollout_config=PrimitiveRolloutConfig(dt_s=0.01, max_duration_s=0.05),
        thresholds=AcceptanceThresholds(
            min_safety_margin_m=-10.0,
            max_angle_of_attack_rad=10.0,
            max_command_abs_rad=10.0,
            min_terminal_specific_energy_change_j_kg=-100.0,
        ),
    )
    return run_validation_sweep(scenarios=(scenario,), max_primitives=1)


def test_feature_extraction_has_stable_keys_and_no_raw_arrays() -> None:
    feature = extract_behaviour_feature(_report().results[0])
    record = feature.to_record()

    assert list(record) == [
        "primitive_id",
        "family",
        "controller_type",
        "entry_class",
        "exit_class",
        "scenario_id",
        "seed",
        "retained",
        "retention_reason",
        "rollout_duration_s",
        "terminal_displacement_w_m",
        "terminal_velocity_delta_b_m_s",
        "terminal_specific_energy_change_j_kg",
        "terminal_specific_energy_margin_j_kg",
        "min_safety_margin_m",
        "max_angle_of_attack_rad",
        "max_command_abs_rad",
        "gate_miss_distance_m",
        "mean_positive_vertical_wind_m_s",
        "velocity_frame",
        "displacement_frame",
    ]
    assert not any(isinstance(value, np.ndarray) for value in record.values())
    assert record["velocity_frame"] == "body"
    assert record["displacement_frame"] == "world_z_up"


def test_feature_vector_is_finite_and_deterministic() -> None:
    feature = extract_behaviour_feature(_report().results[0])

    first = feature_vector(feature)
    second = feature_vector(feature)

    assert first.shape == second.shape
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)
