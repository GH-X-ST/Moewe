"""Tests for the motion-only state-flow observer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from control.flow import FlowBounds
from control.observer import (
    IssuedCommandHistory,
    MotionFlowObserver,
    ObserverCalibration,
    _body_flow_bounds,
)
from control.interval import Interval
from control.predictor import GeneratedAircraft, _generate_aircraft
from control.uncertainty import Bounds
from models.aircraft import Aircraft
from models.geometry import RigidBodyGeometry, body_to_world


@pytest.fixture(scope="module")
def generated() -> GeneratedAircraft:
    aircraft = Aircraft()
    bounds = Bounds(
        flow=FlowBounds(
            np.full(3, -0.2),
            np.full(3, 0.2),
            np.full((3, 3), -0.004),
            np.full((3, 3), 0.004),
            np.full(3, 0.003),
        ),
        density_kg_m3=(1.224, 1.226),
        aerodynamic_scale=(0.999, 1.001),
        force_residual_abs_n=np.full(3, 1.0e-5),
        moment_residual_abs_n_m=np.full(3, 1.0e-7),
        mass_kg=(0.1429, 0.1431),
        cg_residual_abs_m=np.full(3, 1.0e-5),
        inertia_residual_abs_kg_m2=np.full((3, 3), 1.0e-8),
        actuator_tau_lower_s=np.full(3, 0.059),
        actuator_tau_upper_s=np.full(3, 0.061),
        command_error_abs_rad=np.full(3, 1.0e-5),
        state_estimation_abs=np.array(
            [1.0e-3] * 3
            + [1.0e-4] * 3
            + [1.0e-2] * 3
            + [1.0e-3] * 3
            + [1.0e-4] * 3
        ),
        command_delay_s=(0.0, 0.073),
        nonlinear_remainder_abs=np.array(
            [2.0e-5] * 3
            + [2.0e-6] * 3
            + [2.0e-4] * 3
            + [2.0e-4] * 3
            + [2.0e-5] * 3
        ),
        numerical_remainder_abs=np.full(15, 1.0e-12),
        body_inflation_m=1.0e-4,
        mission_position_error_abs_m=1.0e-4,
        mission_attitude_error_abs_rad=1.0e-4,
        roll_abs_max_rad=np.deg2rad(60.0),
        pitch_abs_max_rad=np.deg2rad(60.0),
        airspeed_m_s=(1.0, 15.0),
        alpha_abs_max_rad=np.deg2rad(45.0),
        body_rate_abs_max_rad_s=10.0,
    )
    body = np.array(
        (
            (0.1055, 0.382, 0.0),
            (0.1055, -0.382, 0.0),
            (-0.38875, 0.182, 0.014),
            (-0.38875, -0.182, 0.014),
            (-0.37175, 0.0, -0.116),
        )
    )
    geometry = RigidBodyGeometry(body, body[2:4], body[2:4])
    anchor = np.zeros(15)
    anchor[:3] = (2.0, 2.0, 2.0)
    anchor[6] = 6.0
    model = _generate_aircraft(
        aircraft,
        geometry,
        bounds,
        anchor,
        np.zeros(3),
    )
    return GeneratedAircraft(
        model.aircraft,
        model.geometry,
        model.bounds,
        model.state_scale,
        model.domain_anchor,
        model.reference_center,
        model.reference_scale,
        model.cells,
    )


def _calibration(generated: GeneratedAircraft) -> ObserverCalibration:
    integrity_abs = np.concatenate(
        (0.5 * generated.bounds.state_estimation_abs, np.full(3, 0.01))
    )
    return ObserverCalibration(
        process_variance_per_s=np.full(18, 1.0e-5),
        pose_variance=np.full(6, 1.0e-6),
        initial_variance=np.concatenate((np.full(15, 0.01), np.ones(3))),
        integrity_generators=np.diag(integrity_abs),
        innovation_abs=np.full(6, 0.5),
        flow_change_abs_m_s=np.full(3, 1.0e-4),
        latency_max_s=0.02,
        sample_gap_max_s=0.006,
        nominal_delay_s=0.04,
        initialization_samples=1,
    )


def _history(commands: np.ndarray | None = None) -> IssuedCommandHistory:
    timestamps = np.arange(-0.2, 0.101, 0.02)
    values = np.zeros((timestamps.size, 3)) if commands is None else commands
    return IssuedCommandHistory(timestamps, values)


def _observer(
    generated: GeneratedAircraft,
    calibration: ObserverCalibration | None = None,
) -> MotionFlowObserver:
    return MotionFlowObserver(
        generated,
        _calibration(generated) if calibration is None else calibration,
        generated.cells[0].anchor,
        0.0,
    )


def test_integrity_set_is_independent_of_filter_covariance(
    generated: GeneratedAircraft,
) -> None:
    calibration = _calibration(generated)
    history = _history()
    first_observer = _observer(generated, calibration)
    second_observer = _observer(
        generated,
        replace(calibration, initial_variance=100.0 * calibration.initial_variance),
    )
    pose = generated.cells[0].anchor[:6]
    assert first_observer.update(pose, 0.0, history)
    assert second_observer.update(pose, 0.0, history)
    first = first_observer.estimate(0.0, history)
    second = second_observer.estimate(0.0, history)

    assert first is not None
    assert second is not None
    np.testing.assert_array_equal(first.joint.generators, second.joint.generators)
    np.testing.assert_array_equal(first.joint.generators, calibration.integrity_generators)
    np.testing.assert_array_equal(
        first.local_flow.gradient_lower_s,
        -first.local_flow.gradient_upper_s,
    )


def test_timestamped_delay_queue_propagates_to_governor_time(
    generated: GeneratedAircraft,
) -> None:
    history = _history()
    commands = history.commands_rad.copy()
    commands[np.argmin(np.abs(history.timestamps_s + 0.04)), 0] = 0.1
    commands[np.argmin(np.abs(history.timestamps_s)), 0] = -0.1
    command_history = IssuedCommandHistory(history.timestamps_s, commands)
    observer = _observer(generated)
    assert observer.update(generated.cells[0].anchor[:6], 0.0, command_history)
    estimate = observer.estimate(0.01, command_history)

    assert estimate is not None
    assert estimate.state.center[12] > 0.0


def test_pose_innovation_corrects_the_hidden_center_flow(
    generated: GeneratedAircraft,
) -> None:
    observer = _observer(generated)
    initial = generated.cells[0].anchor
    truth = _aircraft_step(generated, initial, np.zeros(3), 0.005)
    pose = truth[:6].copy()
    pose[0] += 1.0e-4
    history = _history()
    assert observer.update(pose, 0.005, history)
    estimate = observer.estimate(0.005, history)

    assert estimate is not None
    assert np.linalg.norm(estimate.center_flow_w.center) > 0.0


def test_uncontained_innovation_and_excessive_age_are_unavailable(
    generated: GeneratedAircraft,
) -> None:
    pose = generated.cells[0].anchor[:6].copy()
    pose[0] += 1.0
    history = _history()
    observer = _observer(generated)
    assert not observer.update(pose, 0.0, history)
    assert observer.estimate(0.0, history) is None

    observer = _observer(generated)
    assert observer.update(generated.cells[0].anchor[:6], 0.0, history)
    assert observer.estimate(0.021, history) is None


def test_initialization_window_precedes_integrity_output(
    generated: GeneratedAircraft,
) -> None:
    calibration = replace(_calibration(generated), initialization_samples=2)
    history = _history()
    observer = _observer(generated, calibration)
    initial = generated.cells[0].anchor

    assert observer.update(initial[:6], 0.0, history)
    assert observer.estimate(0.0, history) is None

    successor = _aircraft_step(generated, initial, np.zeros(3), 0.005)
    assert observer.update(successor[:6], 0.005, history)
    assert observer.estimate(0.005, history) is not None


def test_rejected_innovation_does_not_correct_the_filter(
    generated: GeneratedAircraft,
) -> None:
    history = _history()
    observer = _observer(generated)
    initial = generated.cells[0].anchor
    assert observer.update(initial[:6], 0.0, history)
    expected = observer._propagate_mean(
        observer._mean.copy(),
        0.0,
        0.005,
        history,
    )
    outlier = expected[:6].copy()
    outlier[0] += 1.0

    assert not observer.update(outlier, 0.005, history)
    np.testing.assert_allclose(observer._mean, expected, rtol=0.0, atol=1.0e-14)


def test_duplicate_or_dropped_pose_samples_cannot_restore_integrity(
    generated: GeneratedAircraft,
) -> None:
    history = _history()
    initial = generated.cells[0].anchor
    observer = _observer(generated)
    assert observer.update(initial[:6], 0.0, history)
    with pytest.raises(ValueError, match="strictly increasing"):
        observer.update(initial[:6], 0.0, history)

    observer = _observer(generated)
    assert observer.update(initial[:6], 0.0, history)
    assert not observer.update(initial[:6], 0.011, history)
    assert observer.estimate(0.011, history) is None
    assert not observer.update(initial[:6], 0.016, history)


def test_calibrated_pose_timestamp_jitter_is_accepted(
    generated: GeneratedAircraft,
) -> None:
    history = _history()
    observer = _observer(generated)
    initial = generated.cells[0].anchor
    assert observer.update(initial[:6], 0.0, history)

    timestamp = 0.0055
    successor = _aircraft_step(generated, initial, np.zeros(3), timestamp)
    assert observer.update(successor[:6], timestamp, history)


def test_body_flow_encloses_rotation_over_the_prediction_horizon(
    generated: GeneratedAircraft,
) -> None:
    observer = _observer(generated)
    history = _history()
    assert observer.update(generated.cells[0].anchor[:6], 0.0, history)
    observer._mean[15:] = (0.05, 0.0, 0.0)
    estimate = observer.estimate(0.0, history)

    assert estimate is not None
    center_flow = Interval(
        estimate.local_flow.center_lower_m_s,
        estimate.local_flow.center_upper_m_s,
    )
    for yaw in np.linspace(-np.pi, np.pi, 33):
        rotated = body_to_world((0.0, 0.0, yaw)).T @ observer._mean[15:]
        assert center_flow.contains(rotated)


def test_generated_transition_uses_the_five_millisecond_model(
    generated: GeneratedAircraft,
) -> None:
    cell = generated.cells[0]
    state_matrix = np.empty((15, 15))
    for index in range(15):
        delta = max(1.0e-7, 1.0e-5 * generated.state_scale[index])
        offset = np.zeros(15)
        offset[index] = delta
        state_matrix[:, index] = (
            _observer_step(
                generated,
                cell.anchor + offset,
                cell.control_anchor,
                0.005,
            )
            - _observer_step(
                generated,
                cell.anchor - offset,
                cell.control_anchor,
                0.005,
            )
        ) / (2.0 * delta)
    flow_matrix = np.empty((15, 3))
    for index in range(3):
        offset = np.zeros(3)
        offset[index] = 1.0e-5
        flow_matrix[:, index] = (
            _observer_step(
                generated,
                cell.anchor,
                cell.control_anchor,
                0.005,
                offset,
            )
            - _observer_step(
                generated,
                cell.anchor,
                cell.control_anchor,
                0.005,
                -offset,
            )
        ) / (2.0e-5)

    np.testing.assert_allclose(cell.observer_state_matrix, state_matrix)
    np.testing.assert_allclose(cell.observer_flow_matrix, flow_matrix)


def test_body_flow_projection_contains_attitude_and_world_flow_boxes() -> None:
    attitude = Interval((-0.2, -0.1, 1.3), (0.3, 0.2, 1.8))
    flow_w = Interval((-0.5, -0.4, -0.3), (0.6, 0.7, 0.8))
    body_flow = _body_flow_bounds(attitude, flow_w)
    random = np.random.default_rng(7)
    for _ in range(1000):
        angles = random.uniform(attitude.lower, attitude.upper)
        flow = random.uniform(flow_w.lower, flow_w.upper)
        assert body_flow.contains(body_to_world(angles).T @ flow)


def _aircraft_step(
    generated: GeneratedAircraft,
    state: np.ndarray,
    command: np.ndarray,
    step_s: float,
    center_b: np.ndarray | None = None,
) -> np.ndarray:
    aircraft = generated.aircraft
    density = 0.5 * sum(generated.bounds.density_kg_m3)

    flow = np.zeros(3) if center_b is None else np.asarray(center_b, dtype=float)

    def derivative(value: np.ndarray) -> np.ndarray:
        strips_b = np.broadcast_to(flow, aircraft.strip_table.r_b_m.shape)
        return aircraft.derivative_local_flow(
            value,
            command,
            flow,
            strips_b,
            density,
        )

    k1 = derivative(state)
    k2 = derivative(state + 0.5 * step_s * k1)
    k3 = derivative(state + 0.5 * step_s * k2)
    k4 = derivative(state + step_s * k3)
    return state + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _observer_step(
    generated: GeneratedAircraft,
    state: np.ndarray,
    command: np.ndarray,
    step_s: float,
    center_b: np.ndarray | None = None,
) -> np.ndarray:
    aircraft = generated.aircraft
    density = 0.5 * sum(generated.bounds.density_kg_m3)
    flow = np.zeros(3) if center_b is None else np.asarray(center_b, dtype=float)

    def derivative(value: np.ndarray) -> np.ndarray:
        strips_b = np.broadcast_to(flow, aircraft.strip_table.r_b_m.shape)
        return aircraft.derivative_local_flow(
            value,
            command,
            flow,
            strips_b,
            density,
        )

    initial = derivative(state)
    return state + step_s * derivative(state + 0.5 * step_s * initial)
