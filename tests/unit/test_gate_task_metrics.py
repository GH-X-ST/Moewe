from __future__ import annotations

import numpy as np

from moewe.sim.glider_model import nominal_glider
from moewe.sim.state import FlightState
from moewe.tasks import FailureReason, FlightVolume, GatePlane, GateTraversalTask


def _state(x_m: float, y_m: float = 0.0, z_m: float = 1.0, w_b_m_s: float = 0.0) -> FlightState:
    return FlightState(
        position_w_m=np.array([x_m, y_m, z_m]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([5.0, 0.0, w_b_m_s]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def _task() -> GateTraversalTask:
    return GateTraversalTask(
        gate=GatePlane(
            centre_w_m=np.array([1.0, 0.0, 1.0]),
            normal_w=np.array([1.0, 0.0, 0.0]),
            width_m=1.0,
            height_m=0.6,
        ),
        flight_volume=FlightVolume(
            x_min_m=-1.0,
            x_max_m=2.0,
            y_min_m=-1.0,
            y_max_m=1.0,
            z_min_m=0.2,
            z_max_m=2.0,
        ),
        timeout_s=1.0,
        required_terminal_specific_energy_j_kg=0.0,
        angle_of_attack_limit_rad=0.5,
    )


def test_gate_crossing_success_metrics() -> None:
    metrics = _task().evaluate([_state(0.0), _state(2.0)], dt_s=0.1, model=nominal_glider())

    assert metrics.success
    assert metrics.gate_crossed
    assert metrics.gate_miss_distance_m == 0.0
    assert metrics.failure_reason == FailureReason.NONE
    assert metrics.flight_time_s == 0.1
    assert metrics.terminal_specific_energy_margin_j_kg > 0.0


def test_exit_gate_at_volume_boundary_terminates_before_outside_state_counts_as_wall() -> None:
    task = GateTraversalTask(
        gate=GatePlane(
            centre_w_m=np.array([1.0, 0.0, 1.0]),
            normal_w=np.array([1.0, 0.0, 0.0]),
            width_m=1.0,
            height_m=0.6,
        ),
        flight_volume=FlightVolume(
            x_min_m=-1.0,
            x_max_m=1.0,
            y_min_m=-1.0,
            y_max_m=1.0,
            z_min_m=0.2,
            z_max_m=2.0,
        ),
        timeout_s=1.0,
        angle_of_attack_limit_rad=0.5,
    )

    metrics = task.evaluate([_state(0.9), _state(1.1)], dt_s=0.1)

    assert metrics.success
    assert metrics.gate_crossed
    assert metrics.failure_reason == FailureReason.NONE
    assert metrics.min_safety_margin_m == 0.0


def test_gate_crossing_requires_airframe_clearance_not_only_cg_position() -> None:
    task = GateTraversalTask(
        gate=GatePlane(
            centre_w_m=np.array([1.0, 0.0, 1.0]),
            normal_w=np.array([1.0, 0.0, 0.0]),
            width_m=0.6,
            height_m=1.0,
        ),
        flight_volume=FlightVolume(
            x_min_m=-1.0,
            x_max_m=1.0,
            y_min_m=-1.0,
            y_max_m=1.0,
            z_min_m=0.2,
            z_max_m=2.0,
        ),
        timeout_s=1.0,
        angle_of_attack_limit_rad=0.5,
    )

    metrics = task.evaluate([_state(0.9), _state(1.1)], dt_s=0.1, model=nominal_glider())

    assert not metrics.success
    assert not metrics.gate_crossed
    assert metrics.gate_miss_distance_m > 0.0
    assert metrics.failure_reason == FailureReason.WALL


def test_gate_miss_distance_and_timeout_failure() -> None:
    metrics = _task().evaluate([_state(0.0, y_m=0.8), _state(2.0, y_m=0.8)], dt_s=0.1)

    assert not metrics.success
    assert not metrics.gate_crossed
    assert metrics.gate_miss_distance_m > 0.0
    assert metrics.failure_reason == FailureReason.TIMEOUT


def test_flight_volume_failures_are_explicit() -> None:
    metrics = _task().evaluate([_state(0.0), _state(0.5, z_m=0.1)], dt_s=0.1)

    assert not metrics.success
    assert metrics.failure_reason == FailureReason.FLOOR
    assert metrics.min_safety_margin_m < 0.0


def test_angle_of_attack_limit_failure_is_explicit() -> None:
    metrics = _task().evaluate([_state(0.0, w_b_m_s=10.0), _state(2.0, w_b_m_s=10.0)], dt_s=0.1)

    assert metrics.failure_reason == FailureReason.STALL_LIMIT
    assert metrics.max_angle_of_attack_rad > 0.5


def test_non_finite_state_failure_is_explicit() -> None:
    bad = _state(0.0)
    bad_vector = bad.as_vector()
    bad_vector[0] = np.nan

    metrics = _task().evaluate([bad, FlightState.from_vector(bad_vector)], dt_s=0.1)

    assert metrics.failure_reason == FailureReason.NON_FINITE_STATE
