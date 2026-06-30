from __future__ import annotations

import numpy as np

from moewe.primitives import PrimitiveGrammarSpec, PrimitiveRolloutConfig, generate_primitives, rollout_primitive
from moewe.sim.state import FlightState
from moewe.tasks import FlightVolume, GatePlane, GateTraversalTask


def _straight_primitive():
    for primitive in generate_primitives(PrimitiveGrammarSpec.smoke()):
        factors = primitive.metadata["grammar_factors"]
        if (
            factors["target_bank_rad"] == 0.0
            and factors["delta_pitch_rad"] == 0.0
            and factors["dwell_duration_s"] == 0.1
        ):
            return primitive
    raise AssertionError("straight smoke primitive not found")


def _gate_task() -> GateTraversalTask:
    return GateTraversalTask(
        gate=GatePlane(
            centre_w_m=np.array([0.2, 0.0, 1.0]),
            normal_w=np.array([1.0, 0.0, 0.0]),
            width_m=2.0,
            height_m=2.0,
        ),
        flight_volume=FlightVolume(
            x_min_m=-1.0,
            x_max_m=2.0,
            y_min_m=-1.0,
            y_max_m=1.0,
            z_min_m=0.0,
            z_max_m=2.0,
        ),
        timeout_s=0.2,
        angle_of_attack_limit_rad=1.2,
    )


def test_primitive_smoke_rollout_returns_finite_history_and_evidence() -> None:
    primitive = _straight_primitive()
    result = rollout_primitive(
        primitive,
        task=_gate_task(),
        config=PrimitiveRolloutConfig(dt_s=0.01, max_duration_s=0.2, scenario_id="unit_gate_smoke", seed=0),
    )

    assert len(result.states) > 1
    assert all(state.finite() for state in result.states)
    assert result.commands_rad.shape[1] == 3
    assert np.isfinite(result.commands_rad).all()
    assert result.evidence.primitive_id == primitive.primitive_id
    assert result.evidence.family == primitive.family
    assert result.evidence.controller_type == "pd"
    assert np.isfinite(result.evidence.min_safety_margin_m)
    assert np.isfinite(result.evidence.terminal_specific_energy_margin_j_kg)
    assert np.isfinite(result.evidence.max_angle_of_attack_rad)
    assert result.evidence.gate_miss_distance_m is not None
    assert result.evidence.failure_reason is None or isinstance(result.evidence.failure_reason, str)

    record = result.evidence.to_record()
    for key in (
        "primitive_id",
        "family",
        "controller_type",
        "min_safety_margin_m",
        "terminal_specific_energy_change_j_kg",
        "terminal_specific_energy_margin_j_kg",
        "max_angle_of_attack_rad",
        "failure_reason",
    ):
        assert key in record


def test_primitive_rollout_rejects_invalid_entry_state_with_reason() -> None:
    primitive = _straight_primitive()
    invalid_state = FlightState(
        position_w_m=np.array([0.0, 0.0, 0.05]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([7.0, 0.0, 0.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )

    result = rollout_primitive(primitive, initial_state=invalid_state)

    assert not result.evidence.rollout_success
    assert result.evidence.failure_reason == "entry_altitude"
    assert result.commands_rad.shape == (0, 3)
