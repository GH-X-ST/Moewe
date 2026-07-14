"""Tests for generated affine prediction and delayed command propagation."""

from __future__ import annotations

import numpy as np
import pytest

from control.flow import FlowBounds
from control.interval import Zonotope
from control.predictor import (
    FastPredictor,
    _generate_aircraft,
    _rotation_derivatives,
)
from control.uncertainty import (
    Bounds,
    FAST_PERIOD_S,
    GOVERNOR_PERIOD_S,
    NEXT_UPDATE_STAGE,
    PREDICTION_PERIOD_S,
    PREDICTION_STAGES,
)
from models.aircraft import Aircraft
from models.geometry import (
    RigidBodyGeometry,
    body_to_world,
    point_velocities,
    world_points,
)


def _geometry() -> RigidBodyGeometry:
    body = (
        (0.1055, 0.382, 0.0),
        (0.1055, -0.382, 0.0),
        (-0.38875, 0.182, 0.014),
        (-0.38875, -0.182, 0.014),
        (-0.37175, 0.0, -0.116),
    )
    contact = body[2:4]
    return RigidBodyGeometry(body, contact, contact)


def _bounds(aircraft: Aircraft) -> Bounds:
    return Bounds(
        flow=FlowBounds(
            (-0.02, -0.02, -0.02),
            (0.02, 0.02, 0.02),
            np.full((3, 3), -0.004),
            np.full((3, 3), 0.004),
            (0.003, 0.003, 0.003),
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
            [1.0e-5] * 3 + [1.0e-6] * 3 + [1.0e-4] * 3 + [1.0e-5] * 3 + [1.0e-6] * 3
        ),
        command_delay_s=(0.0, 0.073),
        nonlinear_remainder_abs=np.array(
            [2.0e-5] * 3 + [2.0e-6] * 3 + [2.0e-4] * 3 + [2.0e-4] * 3 + [2.0e-5] * 3
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


def _state_belief(state: np.ndarray, bounds: Bounds) -> Zonotope:
    return Zonotope(
        np.asarray(state, dtype=float).reshape(15),
        np.diag(bounds.state_estimation_abs),
    )


def _generated() -> tuple[Aircraft, Bounds, object]:
    aircraft = Aircraft()
    bounds = _bounds(aircraft)
    state = np.zeros(15)
    state[:3] = (2.0, 2.0, 2.0)
    state[6] = 6.0
    generated = _generate_aircraft(
        aircraft,
        _geometry(),
        bounds,
        state,
        np.zeros(3),
    )
    return aircraft, bounds, generated


def test_nominal_model_dimensions_and_periods() -> None:

    aircraft, _, generated = _generated()
    cell = generated.cells[0]
    assert FAST_PERIOD_S == 0.020
    assert GOVERNOR_PERIOD_S == 0.100
    assert PREDICTION_PERIOD_S == 0.200
    assert PREDICTION_STAGES == 10
    assert NEXT_UPDATE_STAGE == 5
    assert cell.state_matrix.shape == (15, 15)
    assert cell.control_matrix.shape == (15, 3)
    assert cell.gain.shape == (3, 15)
    assert cell.flow_generators.shape == (
        15,
        12 + 3 * aircraft.strip_table.r_b_m.shape[0],
    )
    assert np.all(cell.gain[:, :3] == 0.0)
    assert np.all(cell.gain[:, 5] == 0.0)


def test_analytic_rotation_derivatives_match_central_differences() -> None:
    rng = np.random.default_rng(7041)
    result = np.empty((4, 3, 3))
    step = 1.0e-6
    for _ in range(64):
        attitude = rng.uniform(-0.8, 0.8, 3)
        _rotation_derivatives(attitude, result)
        np.testing.assert_allclose(result[0], body_to_world(attitude), atol=1.0e-14)
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = step
            numerical = (
                body_to_world(attitude + offset) - body_to_world(attitude - offset)
            ) / (2.0 * step)
            np.testing.assert_allclose(result[axis + 1], numerical, atol=2.0e-10)


def test_analytic_body_point_and_velocity_jacobians() -> None:
    _, _, generated = _generated()
    predictor = FastPredictor(generated)
    prediction = predictor.prediction
    state = generated.cells[0].anchor.copy()
    state[3:6] = (0.2, -0.15, 0.3)
    state[6:12] = (5.8, 0.2, -0.1, 0.3, -0.2, 0.4)
    prediction.state_center[0] = state
    prediction.state_reference[0].fill(0.0)
    prediction.generator_count[0] = 0
    _rotation_derivatives(state[3:6], predictor._stage_rotations[0])

    body = np.asarray(generated.geometry.body_b_m)
    predictor._point_affine_into(prediction, 0, body, 0, predictor._reference_abs)
    point_jacobian = predictor._point_jacobian[: body.shape[0]].copy()

    contact = np.asarray(generated.geometry.contact_b_m)
    predictor._velocity_affine_into(
        prediction,
        0,
        contact,
        0,
        predictor._reference_abs,
    )
    velocity_jacobian = predictor._point_jacobian[: contact.shape[0]].copy()

    step = 1.0e-6
    for index in range(15):
        offset = np.zeros(15)
        offset[index] = step
        numerical_points = (
            world_points(state + offset, body) - world_points(state - offset, body)
        ) / (2.0 * step)
        numerical_velocity = (
            point_velocities(state + offset, contact)
            - point_velocities(state - offset, contact)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            point_jacobian[:, :, index],
            numerical_points,
            atol=3.0e-9,
        )
        np.testing.assert_allclose(
            velocity_jacobian[:, :, index],
            numerical_velocity,
            atol=3.0e-9,
        )


def test_predictor_is_affine_in_one_three_value_reference() -> None:

    aircraft, bounds, generated = _generated()
    predictor = FastPredictor(generated)
    belief = _state_belief(generated.cells[0].anchor, bounds)
    queue = np.zeros((bounds.queue_length, 3))
    first = predictor.predict(belief, queue, 0)
    identity = id(first)
    second = predictor.predict(belief, queue, 0)
    assert id(second) == identity
    assert second.state_center.shape == (11, 15)
    assert second.state_reference.shape == (11, 15, 3)
    assert np.all(second.generator_count[1:] > second.generator_count[:-1])
    for stage, count in enumerate(second.generator_count):
        np.testing.assert_allclose(
            second.state_radius[stage],
            np.sum(np.abs(second.state_generators[stage, :, :count]), axis=1),
        )
    np.testing.assert_array_equal(second.issued_center, np.zeros((10, 3)))
    np.testing.assert_array_equal(
        second.issued_reference,
        np.broadcast_to(np.eye(3), (10, 3, 3)),
    )
    for reference in (
        np.zeros(3),
        0.2 * aircraft.control_lower_rad,
        0.2 * aircraft.control_upper_rad,
    ):
        affine = second.state_center[5] + second.state_reference[5] @ reference
        enclosure = second.state_interval(5, reference, reference)
        assert enclosure.contains(affine)


def test_measured_delay_uses_the_complete_command_queue() -> None:
    """Carry every issued command whose onset can intersect a fast stage."""

    _, bounds, generated = _generated()
    predictor = FastPredictor(generated)
    queue = np.array(
        [
            (-0.12, 0.01, 0.03),
            (-0.04, 0.02, -0.02),
            (0.05, -0.03, 0.01),
            (0.11, -0.04, -0.03),
        ]
    )
    queue_radius = np.full(queue.shape, 0.02)
    prediction = predictor.predict(
        _state_belief(generated.cells[0].anchor, bounds),
        queue,
        0,
        queue_radius,
    )
    assert bounds.queue_length == 4
    assert np.all(
        prediction.applied_center[0] - prediction.applied_radius[0]
        <= np.minimum(queue[0], queue[1]) - queue_radius[0]
    )
    assert np.all(
        prediction.applied_center[0] + prediction.applied_radius[0]
        >= np.maximum(queue[0], queue[1]) + queue_radius[0]
    )
    assert np.any(np.abs(prediction.applied_reference[4]) > 0.0)


def test_cg_airspeed_support_includes_joint_flow_uncertainty() -> None:
    _, bounds, generated = _generated()
    prediction = FastPredictor(generated).predict(
        _state_belief(generated.cells[0].anchor, bounds),
        np.zeros((bounds.queue_length, 3)),
        0,
    )
    upper, coefficient = prediction.airspeed_support(0, 1.0)
    lower, negative_coefficient = prediction.airspeed_support(0, -1.0)
    for signs in (
        np.array((-1.0, -1.0, -1.0)),
        np.array((1.0, -1.0, 1.0)),
        np.ones(3),
    ):
        flow = prediction.flow_center + signs * prediction.flow_radius
        speed = np.linalg.norm(generated.cells[0].anchor[6:9] - flow)
        assert speed <= upper + coefficient @ np.zeros(3)
        assert -speed <= lower + negative_coefficient @ np.zeros(3)


def test_all_uncertainty_groups_enter_one_joint_prediction() -> None:
    _, bounds, generated = _generated()
    cell = generated.cells[0]
    assert cell.model_generators.shape == (15, 18)
    assert np.all(np.linalg.norm(cell.model_generators, axis=0) > 0.0)
    flow_norm = np.linalg.norm(cell.flow_generators, axis=0)
    assert np.all(flow_norm[:12] > 0.0)
    strip_remainder = flow_norm[12:].reshape(-1, 3)
    assert np.all(np.any(strip_remainder > 0.0, axis=1))
    belief = _state_belief(cell.anchor, bounds)
    prediction = FastPredictor(generated).predict(
        belief,
        np.zeros((bounds.queue_length, 3)),
        0,
    )
    model_start = belief.generators.shape[1]
    model_stop = model_start + cell.model_generators.shape[1]
    np.testing.assert_array_equal(
        prediction.state_generators[1, :, model_start:model_stop],
        cell.model_generators,
    )
    count = prediction.generator_count[1]
    np.testing.assert_array_equal(
        prediction.state_generators[1, :, count - 15 : count],
        np.diag(cell.stage_remainder_abs),
    )
    assert np.all(prediction.issued_radius >= bounds.command_error_abs_rad)
    minimum_geometry_radius = (
        bounds.body_inflation_m + bounds.mission_position_error_abs_m
    )
    assert np.all(prediction.body_radius >= minimum_geometry_radius)


def test_predictor_is_equivariant_to_translation_and_global_yaw() -> None:
    _, bounds, generated = _generated()
    cell = generated.cells[0]
    queue = np.zeros((bounds.queue_length, 3))
    empty = np.empty((15, 0))
    base = FastPredictor(generated).predict(
        Zonotope(cell.anchor, empty),
        queue,
        0,
    )

    for position, yaw in (
        ((1.0, -2.0, 2.5), -2.4),
        ((4.0, 3.0, 1.5), -0.7),
        ((-3.0, 5.0, 3.0), 0.9),
        ((2.0, -4.0, 2.0), 2.6),
    ):
        state = cell.anchor.copy()
        state[:3] = position
        state[5] = yaw
        prediction = FastPredictor(generated).predict(
            Zonotope(state, empty),
            queue,
            0,
        )
        yaw_shift = yaw - cell.anchor[5]
        cosine = np.cos(yaw_shift)
        sine = np.sin(yaw_shift)
        rotation = np.array(
            (
                (cosine, sine, 0.0),
                (-sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        for stage in range(PREDICTION_STAGES + 1):
            expected = base.state_center[stage].copy()
            expected[:3] = position + rotation @ (expected[:3] - cell.anchor[:3])
            expected[5] += yaw_shift
            np.testing.assert_allclose(prediction.state_center[stage], expected)
            np.testing.assert_allclose(
                prediction.state_reference[stage, :3],
                rotation @ base.state_reference[stage, :3],
            )
            count = prediction.generator_count[stage]
            np.testing.assert_allclose(
                prediction.state_generators[stage, :3, :count],
                rotation @ base.state_generators[stage, :3, :count],
            )


def test_flight_prediction_skips_unused_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bounds, generated = _generated()
    predictor = FastPredictor(generated)

    def fail(*_: object) -> None:
        raise AssertionError("flight prediction populated geometry")

    monkeypatch.setattr(predictor, "_populate_geometry", fail)
    predictor.predict(
        _state_belief(generated.cells[0].anchor, bounds),
        np.zeros((bounds.queue_length, 3)),
        0,
        populate_geometry=False,
    )
