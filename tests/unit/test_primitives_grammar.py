from __future__ import annotations

import numpy as np

from moewe.primitives import (
    PrimitiveGrammarSpec,
    PrimitiveSafetyLimits,
    generate_primitives,
)
from moewe.sim.state import FlightState


def _state(
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity: tuple[float, float, float] = (7.0, 0.0, 0.0),
) -> FlightState:
    return FlightState(
        position_w_m=np.array(position),
        euler_rad=np.array(euler),
        velocity_b_m_s=np.array(velocity),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def test_smoke_grammar_generation_is_deterministic_and_unique() -> None:
    first = generate_primitives(PrimitiveGrammarSpec.smoke())
    second = generate_primitives(PrimitiveGrammarSpec.smoke())

    assert len(first) == 18
    assert [primitive.primitive_id for primitive in first] == [primitive.primitive_id for primitive in second]
    assert len({primitive.primitive_id for primitive in first}) == len(first)
    assert all(np.isfinite(primitive.reference.total_duration_s) for primitive in first)
    assert all(primitive.reference.total_duration_s > 0.0 for primitive in first)


def test_generated_primitive_records_are_inspectable_without_rollout() -> None:
    primitive = generate_primitives(PrimitiveGrammarSpec.smoke())[0]
    summary = primitive.summary()

    assert summary["primitive_id"] == primitive.primitive_id
    assert summary["controller_type"] == "pd"
    assert summary["phase_names"] == ["hold", "bank_transition", "pitch_pulse", "dwell", "recovery"]
    assert primitive.metadata["command_order"] == "[aileron, elevator, rudder]"


def test_generated_primitive_metadata_contains_degradation_links() -> None:
    primitives = generate_primitives(PrimitiveGrammarSpec.smoke())
    by_id = {primitive.primitive_id: primitive for primitive in primitives}

    assert all("aggressiveness_level" in primitive.metadata for primitive in primitives)
    assert all("entry_trim_mode" in primitive.metadata for primitive in primitives)
    assert all("bank_transition_label" in primitive.metadata for primitive in primitives)
    assert all("pitch_pulse_label" in primitive.metadata for primitive in primitives)
    assert all("dwell_label" in primitive.metadata for primitive in primitives)
    assert all("recovery_label" in primitive.metadata for primitive in primitives)
    assert all("task_intent_label" in primitive.metadata for primitive in primitives)

    linked = [primitive for primitive in primitives if primitive.metadata["same_family_degrade_id"] is not None]
    assert linked
    for primitive in linked:
        degrade_id = primitive.metadata["same_family_degrade_id"]
        assert degrade_id in by_id
        assert by_id[degrade_id].family == primitive.family
        assert by_id[degrade_id].metadata["aggressiveness_level"] < primitive.metadata["aggressiveness_level"]

    recovery_linked = [primitive for primitive in primitives if primitive.metadata["recovery_degrade_id"] is not None]
    assert recovery_linked
    assert all(by_id[primitive.metadata["recovery_degrade_id"]].metadata["task_intent_label"] == "safe_exit" for primitive in recovery_linked)


def test_entry_condition_accepts_nominal_and_rejects_impossible_states() -> None:
    primitive = generate_primitives(PrimitiveGrammarSpec.smoke())[0]
    nominal = primitive.reference.state_at(0.0)

    assert primitive.entry_condition.accepts(nominal)
    assert primitive.entry_condition.rejection_reason(_state(velocity=(0.1, 0.0, 0.0))) == "entry_airspeed"
    assert primitive.entry_condition.rejection_reason(_state(position=(0.0, 0.0, -1.0))) == "entry_altitude"
    assert primitive.entry_condition.rejection_reason(_state(euler=(2.0, 0.0, 0.0))) == "entry_bank"


def test_safety_limits_report_floor_ceiling_wall_and_aoa_failures() -> None:
    limits = PrimitiveSafetyLimits(
        x_min_m=-1.0,
        x_max_m=1.0,
        y_min_m=-1.0,
        y_max_m=1.0,
        z_min_m=0.2,
        z_max_m=2.0,
        max_abs_angle_of_attack_rad=0.4,
    )

    assert limits.check_state(_state(position=(0.0, 0.0, 0.1))).failure_reason == "floor"
    assert limits.check_state(_state(position=(0.0, 0.0, 2.5))).failure_reason == "ceiling"
    assert limits.check_state(_state(position=(2.0, 0.0, 1.0))).failure_reason == "wall"
    assert limits.check_state(_state(velocity=(1.0, 0.0, 1.0))).failure_reason == "stall_limit"
