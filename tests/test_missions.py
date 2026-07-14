"""Tests for gate and landing mission geometry."""

from __future__ import annotations

from math import cos, radians, sin, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from control.missions import (
    CompilationDomain,
    FreeSpace,
    GateMission,
    Halfspaces,
    LandingMission,
    distance_support,
    error_support,
    preterminal_support_constraints,
)
from control.flow import FlowBounds
from control.predictor import Prediction
from models.geometry import (
    RigidBodyGeometry,
    point_velocities,
    world_points,
)
from simulation.gate import Gate
from simulation.platform import Platform

MOEWE_GEOMETRY = RigidBodyGeometry(
    body_b_m=(
        (0.2705, 0.0, 0.0),
        (-0.3420, 0.0, 0.0),
        (0.1055, 0.3820, 0.0),
        (0.1055, -0.3820, 0.0),
        (-0.2970, 0.1820, 0.0),
        (-0.2970, -0.1820, 0.0),
        (-0.3270, 0.0, -0.1190),
        (0.1000, 0.0800, 0.0400),
        (0.1000, -0.0800, 0.0400),
    ),
    contact_b_m=(
        (0.1000, 0.0800, 0.0400),
        (0.1000, -0.0800, 0.0400),
    ),
    footprint_b_m=(
        (0.2705, 0.0, 0.0),
        (-0.3420, 0.0, 0.0),
        (0.1055, 0.3820, 0.0),
        (0.1055, -0.3820, 0.0),
        (0.1000, 0.0800, 0.0400),
        (0.1000, -0.0800, 0.0400),
    ),
)


def _state(
    position: np.ndarray,
    yaw_rad: float = 0.0,
    velocity_b_m_s: tuple[float, float, float] = (8.0, 0.0, 0.0),
    rates_b_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> np.ndarray:
    value = np.zeros(15)
    value[:3] = position
    value[3:6] = roll_rad, pitch_rad, yaw_rad
    value[6:9] = velocity_b_m_s
    value[9:12] = rates_b_rad_s
    return value


def _axes(angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([cos(angle_rad), sin(angle_rad), 0.0])
    lateral = np.array([-sin(angle_rad), cos(angle_rad), 0.0])
    return forward, lateral


def _domain(
    mission: GateMission | LandingMission,
    attitude_radius: float | None = None,
) -> CompilationDomain:
    if attitude_radius is None:
        attitude_radius = 0.02 if isinstance(mission, GateMission) else 0.01
    if isinstance(mission, GateMission):
        yaw = -float(np.arctan2(mission.normal_w[1], mission.normal_w[0]))
        center = _state(mission.center_w_m - 0.4 * mission.normal_w, yaw)
    else:
        normal = np.cross(mission.length_axis_w, mission.width_axis_w)
        yaw = -float(np.arctan2(mission.length_axis_w[1], mission.length_axis_w[0]))
        center = _state(
            mission.center_w_m + 0.08 * normal,
            yaw,
            velocity_b_m_s=(0.0, 0.0, 0.4),
        )
    radius = np.array(
        [
            2.0,
            2.0,
            1.0,
            attitude_radius,
            attitude_radius,
            attitude_radius,
            0.25,
            0.25,
            0.25,
            0.2,
            0.2,
            0.2,
            0.5,
            0.5,
            0.5,
        ]
    )
    return CompilationDomain(center - radius, center + radius)


def _generated(
    geometry: RigidBodyGeometry = MOEWE_GEOMETRY,
    center_flow_b_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SimpleNamespace:
    flow = np.asarray(center_flow_b_m_s, dtype=float)
    bounds = SimpleNamespace(
        flow=FlowBounds(
            flow,
            flow,
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            np.zeros(3),
        ),
        body_inflation_m=0.005,
        mission_position_error_abs_m=0.005,
        mission_attitude_error_abs_rad=0.005,
    )
    return SimpleNamespace(geometry=geometry, bounds=bounds)


GENERATED = _generated()


def _gate_mission(
    center: tuple[float, float, float] = (2.0, -1.0, 1.4),
    angle_rad: float = 0.0,
    width_m: float = 1.2,
    height_m: float = 0.6,
) -> GateMission:
    normal, width = _axes(angle_rad)
    center_array = np.asarray(center, dtype=float)
    return GateMission(
        free_space=FreeSpace.box(center_array - 5.0, center_array + 5.0),
        center_w_m=center_array,
        normal_w=normal,
        width_axis_w=width,
        width_m=width_m,
        height_m=height_m,
        target_airspeed_m_s=8.0,
        heading_abs_max_rad=radians(10.0),
        roll_abs_max_rad=radians(15.0),
        pitch_bounds_rad=(radians(-10.0), radians(15.0)),
        airspeed_bounds_m_s=(7.0, 9.0),
        frame_clearance_m=0.005,
    )


def _gate_states(mission: GateMission) -> np.ndarray:
    angle = -float(np.arctan2(mission.normal_w[1], mission.normal_w[0]))
    distances = np.array([-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8])
    return np.stack(
        [
            _state(mission.center_w_m + distance * mission.normal_w, angle)
            for distance in distances
        ]
    )


def _landing_mission(
    center: tuple[float, float, float] = (3.0, 1.0, 0.8),
    angle_rad: float = 0.0,
    length_m: float = 1.2,
    width_m: float = 1.0,
) -> LandingMission:
    length, width = _axes(angle_rad)
    center_array = np.asarray(center, dtype=float)
    return LandingMission(
        free_space=FreeSpace.box(center_array - 5.0, center_array + 5.0),
        center_w_m=center_array,
        length_axis_w=length,
        width_axis_w=width,
        length_m=length_m,
        width_m=width_m,
        height_bounds_m=(0.02, 0.08),
        roll_abs_max_rad=radians(15.0),
        touchdown_pitch_rad=0.0,
        pitch_error_abs_max_rad=radians(10.0),
        normal_speed_max_m_s=0.8,
        tangential_speed_max_m_s=1.0,
        platform_clearance_m=0.005,
    )


def _landing_states(
    mission: LandingMission,
    velocity_b_m_s: tuple[float, float, float] = (0.0, 0.0, 0.4),
    rates_b_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> np.ndarray:
    normal = np.cross(mission.length_axis_w, mission.width_axis_w)
    angle = -float(np.arctan2(mission.length_axis_w[1], mission.length_axis_w[0]))
    return np.stack(
        [
            _state(
                mission.center_w_m + height * normal,
                angle,
                velocity_b_m_s,
                rates_b_rad_s,
                roll_rad,
                pitch_rad,
            )
            for height in (0.25, 0.14, 0.08, 0.04)
        ]
    )


def _prediction(
    mission: GateMission | LandingMission,
    generated: SimpleNamespace = GENERATED,
) -> Prediction:
    geometry = generated.geometry
    prediction = Prediction(
        1,
        body_count=geometry.body_b_m.shape[0],
        contact_count=geometry.contact_b_m.shape[0],
        footprint_count=geometry.footprint_b_m.shape[0],
    )
    prediction.generator_count.fill(0)
    prediction.reference_center.fill(0.0)
    prediction.reference_radius.fill(0.0)
    prediction.flow_center.fill(0.0)
    prediction.flow_radius.fill(0.0)
    prediction.state_generators.fill(0.0)
    prediction.state_reference.fill(0.0)
    prediction.state_reference[:, :3] = np.eye(3)

    if isinstance(mission, GateMission):
        angle = -float(np.arctan2(mission.normal_w[1], mission.normal_w[0]))
        positions = [
            mission.center_w_m + distance * mission.normal_w
            for distance in np.linspace(-0.8, 0.8, 11)
        ]
        states = np.stack([_state(position, angle) for position in positions])
    else:
        normal = np.cross(mission.length_axis_w, mission.width_axis_w)
        angle = -float(np.arctan2(mission.length_axis_w[1], mission.length_axis_w[0]))
        heights = np.concatenate((np.linspace(0.25, 0.08, 10), (0.035,)))
        states = np.stack(
            [
                _state(
                    mission.center_w_m + height * normal,
                    angle,
                    velocity_b_m_s=(0.0, 0.0, 0.4),
                )
                for height in heights
            ]
        )
    prediction.state_center[:] = states

    for stage in range(10):
        midpoint = 0.5 * (states[stage] + states[stage + 1])
        prediction.body_center[stage] = world_points(
            midpoint,
            geometry.body_b_m,
        )
        prediction.contact_center[stage] = world_points(
            midpoint,
            geometry.contact_b_m,
        )
        prediction.footprint_center[stage] = world_points(
            midpoint,
            geometry.footprint_b_m,
        )
        prediction.contact_velocity_center[stage] = point_velocities(
            midpoint,
            geometry.contact_b_m,
        )
    for reference in (
        prediction.body_reference,
        prediction.contact_reference,
        prediction.footprint_reference,
    ):
        reference[:] = np.eye(3)
    prediction.contact_velocity_reference.fill(0.0)
    for radius in (
        prediction.body_radius,
        prediction.contact_radius,
        prediction.footprint_radius,
        prediction.contact_velocity_radius,
    ):
        radius.fill(0.0)
    return prediction


def test_immutable_contract_data_and_mission_scale() -> None:
    """Keep explicit geometry and mission data immutable."""

    assert np.ptp(MOEWE_GEOMETRY.body_b_m[:, 1]) == pytest.approx(0.764)
    assert np.ptp(MOEWE_GEOMETRY.body_b_m[:, 0]) > 0.6
    mission = _gate_mission()
    domain = _domain(mission)
    terminal = mission.terminal_halfspaces(GENERATED)
    assert isinstance(domain, CompilationDomain)
    assert isinstance(terminal, Halfspaces)
    assert mission.free_space_halfspaces is mission.free_space.halfspaces
    for value in (
        MOEWE_GEOMETRY.body_b_m,
        domain.lower,
        domain.upper,
        domain.center,
        domain.radius,
        terminal.matrix,
        terminal.bounds,
    ):
        assert not value.flags.writeable


def test_geometry_must_be_explicit_and_contact_sets_belong_to_body() -> None:
    """Do not admit point-body defaults or external contact vertices."""

    with pytest.raises(TypeError):
        RigidBodyGeometry()
    with pytest.raises(ValueError, match="contact_b_m"):
        RigidBodyGeometry(
            body_b_m=((0.0, 0.0, 0.0),),
            contact_b_m=((1.0, 0.0, 0.0),),
            footprint_b_m=((0.0, 0.0, 0.0),),
        )


@pytest.mark.parametrize("angle_rad", (0.0, radians(37.0), radians(-63.0)))
def test_translated_rotated_gate_and_full_frame_rejection(angle_rad: float) -> None:
    """Pass a translated gate and reject a frame strike by the complete wing."""

    mission = _gate_mission(angle_rad=angle_rad)
    states = _gate_states(mission)
    assert mission.realized(states, GENERATED)
    assert np.allclose(mission.error(states[3], GENERATED), 0.0, atol=1.0e-12)
    assert mission.distance(states[0]) == pytest.approx(0.8)
    assert mission.terminal_halfspaces(GENERATED).contains(
        mission.error(states[3], GENERATED)
    )

    narrow = _gate_mission(angle_rad=angle_rad, width_m=0.75)
    assert not narrow.realized(_gate_states(narrow), GENERATED)


def test_gate_error_uses_body_relative_center_flow() -> None:
    """Use the same air-relative speed coordinate as robust support bounds."""

    mission = _gate_mission()
    generated = _generated(center_flow_b_m_s=(2.0, 0.0, 0.0))
    state = _gate_states(mission)[3]
    state[6] += 2.0
    assert mission.error(state, generated)[-1] == pytest.approx(0.0)

    prediction = _prediction(mission, generated)
    prediction.state_center[5] = state
    prediction.flow_center[:] = (2.0, 0.0, 0.0)
    for facet in mission.terminal_halfspaces(generated).matrix[-2:]:
        offset, reference = error_support(
            mission,
            prediction,
            5,
            facet,
            generated,
            _domain(mission),
        )
        assert offset == pytest.approx(float(facet @ mission.error(state, generated)))
        np.testing.assert_array_equal(reference, np.zeros(3))


def test_gate_monitor_matches_simulator() -> None:
    """Use one realized-event predicate in the controller and simulator."""

    mission = _gate_mission(angle_rad=radians(29.0))
    simulator = Gate(
        GENERATED.geometry,
        mission.center_w_m,
        mission.normal_w,
        mission.width_axis_w,
        mission.width_m,
        mission.height_m,
        mission.frame_clearance_m,
        GENERATED.bounds.body_inflation_m,
        GENERATED.bounds.mission_position_error_abs_m,
        GENERATED.bounds.mission_attitude_error_abs_rad,
    )
    states = _gate_states(mission)
    missed = states.copy()
    missed[:, :3] += 0.7 * mission.width_axis_w
    assert mission.realized(states, GENERATED) == simulator.passed(states)
    assert mission.realized(missed, GENERATED) == simulator.passed(missed)
    assert not mission.realized(missed, GENERATED)


@pytest.mark.parametrize("angle_rad", (0.0, radians(41.0)))
def test_translated_rotated_landing_first_contact(angle_rad: float) -> None:
    """Locate first contact and retain the full footprint on the platform."""

    mission = _landing_mission(angle_rad=angle_rad)
    states = _landing_states(mission)
    assert mission.realized(states, GENERATED)
    state = _state(
        mission.center_w_m + np.array([0.0, 0.0, 0.04]),
        -angle_rad,
        velocity_b_m_s=(0.0, 0.0, 0.4),
    )
    error = mission.error(state, GENERATED)
    normal = np.cross(mission.length_axis_w, mission.width_axis_w)
    velocity = point_velocities(state, GENERATED.geometry.contact_b_m)
    expected_velocity = np.column_stack(
        (
            -velocity @ normal,
            velocity @ mission.length_axis_w,
            velocity @ mission.width_axis_w,
        )
    ).reshape(-1)
    assert error.shape == (5 + 3 * velocity.shape[0],)
    np.testing.assert_allclose(error[5:], expected_velocity, atol=1.0e-12)
    assert mission.terminal_halfspaces(GENERATED).contains(error)

    narrow = _landing_mission(angle_rad=angle_rad, width_m=0.75)
    assert not narrow.realized(_landing_states(narrow), GENERATED)


def test_landing_rejects_unapproved_first_contact_and_angular_speed() -> None:
    """Reject earlier body contact and angular contact-point velocity."""

    body = tuple(MOEWE_GEOMETRY.body_b_m) + ((0.0, 0.0, 0.08),)
    unapproved = RigidBodyGeometry(
        body,
        MOEWE_GEOMETRY.contact_b_m,
        MOEWE_GEOMETRY.footprint_b_m,
    )
    collision = _landing_mission()
    collision_aircraft = _generated(unapproved)
    assert not collision.realized(_landing_states(collision), collision_aircraft)

    mission = _landing_mission()
    rotating = _landing_states(mission, rates_b_rad_s=(8.0, 0.0, 0.0))
    assert not mission.realized(rotating, GENERATED)
    contact_velocity = point_velocities(rotating[-1], GENERATED.geometry.contact_b_m)
    assert np.ptp(contact_velocity[:, 2]) > 1.0


def test_landing_rejects_tangential_speed_and_touchdown_attitude() -> None:
    """Enforce tangential velocity, roll, and touchdown pitch at contact."""

    mission = _landing_mission()
    fast = _landing_states(mission, velocity_b_m_s=(2.0, 0.0, 0.4))
    rolled = _landing_states(mission, roll_rad=radians(20.0))
    pitched = _landing_states(mission, pitch_rad=radians(15.0))
    assert not mission.realized(fast, GENERATED)
    assert not mission.realized(rolled, GENERATED)
    assert not mission.realized(pitched, GENERATED)


def test_landing_monitor_matches_simulator() -> None:
    """Share event-located platform contact with the simulator."""

    mission = _landing_mission(angle_rad=radians(-33.0))
    simulator = Platform(
        GENERATED.geometry,
        mission.center_w_m,
        mission.length_axis_w,
        mission.width_axis_w,
        mission.length_m,
        mission.width_m,
        mission.normal_speed_max_m_s,
        mission.tangential_speed_max_m_s,
        mission.roll_abs_max_rad,
        mission.touchdown_pitch_rad,
        mission.pitch_error_abs_max_rad,
        mission.platform_clearance_m,
        GENERATED.bounds.body_inflation_m,
        GENERATED.bounds.mission_position_error_abs_m,
        GENERATED.bounds.mission_attitude_error_abs_rad,
    )
    states = _landing_states(mission)
    missed = states.copy()
    missed[:, :3] += 0.7 * mission.width_axis_w
    assert mission.realized(states, GENERATED) == simulator.landed(states)
    assert mission.realized(missed, GENERATED) == simulator.landed(missed)
    assert not mission.realized(missed, GENERATED)


def test_terminal_support_rows_are_affine_and_deterministic() -> None:
    """Return fixed-order physical-reference rows over the ten-stage horizon."""

    gate = _gate_mission(angle_rad=radians(17.0))
    gate_prediction = _prediction(gate)
    domain = _domain(gate)
    first_a, first_b = gate.terminal_support_constraints(
        gate_prediction,
        GENERATED,
        domain,
    )
    second_a, second_b = gate.terminal_support_constraints(
        gate_prediction,
        GENERATED,
        domain,
    )
    body_count = GENERATED.geometry.body_b_m.shape[0]
    gate_rows = 42 * body_count + 11 * 8
    assert first_a.shape == (gate_rows, 3)
    assert first_b.shape == (gate_rows,)
    np.testing.assert_array_equal(first_a, second_a)
    np.testing.assert_array_equal(first_b, second_b)
    assert np.all(np.isfinite(first_a))
    assert np.all(np.isfinite(first_b))

    landing = _landing_mission(angle_rad=radians(-21.0))
    landing_prediction = _prediction(landing)
    domain = _domain(landing)
    first_a, first_b = landing.terminal_support_constraints(
        landing_prediction,
        GENERATED,
        domain,
    )
    second_a, second_b = landing.terminal_support_constraints(
        landing_prediction,
        GENERATED,
        domain,
    )
    body_count = GENERATED.geometry.body_b_m.shape[0]
    contact_count = GENERATED.geometry.contact_b_m.shape[0]
    footprint_count = GENERATED.geometry.footprint_b_m.shape[0]
    forbidden_count = body_count - contact_count
    rows = (
        9 * body_count
        + contact_count
        + forbidden_count
        + 4 * footprint_count
        + 6 * contact_count
        + 20
    )
    assert first_a.shape == (rows, 3)
    assert first_b.shape == (rows,)
    np.testing.assert_array_equal(first_a, second_a)
    np.testing.assert_array_equal(first_b, second_b)


def test_preterminal_rows_cover_every_swept_body_point() -> None:
    """Keep nonterminal gate and landing predictions before contact."""

    for mission in (_gate_mission(), _landing_mission()):
        prediction = _prediction(mission)
        matrix, bounds = preterminal_support_constraints(
            mission,
            prediction,
            GENERATED,
        )
        body_count = GENERATED.geometry.body_b_m.shape[0]
        stages = prediction.body_center.shape[0]
        assert matrix.shape == (stages * body_count, 3)
        assert bounds.shape == (stages * body_count,)

        if isinstance(mission, GateMission):
            axis = mission.normal_w
            clearance = mission.frame_clearance_m
        else:
            axis = -np.cross(mission.length_axis_w, mission.width_axis_w)
            clearance = mission.platform_clearance_m
        radius = float(np.max(np.linalg.norm(GENERATED.geometry.body_b_m, axis=1)))
        limit = -(clearance + radius * GENERATED.bounds.mission_attitude_error_abs_rad)
        for stage in range(stages):
            for point in range(body_count):
                offset, reference = prediction.body_support(stage, point, axis)
                row = stage * body_count + point
                np.testing.assert_allclose(matrix[row], reference, atol=1.0e-12)
                assert bounds[row] == pytest.approx(
                    limit + axis @ mission.center_w_m - offset
                )


def test_preterminal_rows_reject_gate_crossing_and_platform_contact() -> None:
    """Reject any swept body point that reaches the terminal surface early."""

    gate = _gate_mission()
    gate_prediction = _prediction(gate)
    gate_prediction.body_reference.fill(0.0)
    gate_prediction.body_radius.fill(0.0)
    gate_prediction.body_center[:] = gate.center_w_m - gate.normal_w
    gate_prediction.body_center[4, 2] = gate.center_w_m + gate.normal_w
    _, gate_bounds = preterminal_support_constraints(
        gate,
        gate_prediction,
        GENERATED,
    )
    assert gate_bounds[4 * gate_prediction.body_count + 2] < 0.0

    landing = _landing_mission()
    landing_prediction = _prediction(landing)
    normal = np.cross(landing.length_axis_w, landing.width_axis_w)
    landing_prediction.body_reference.fill(0.0)
    landing_prediction.body_radius.fill(0.0)
    landing_prediction.body_center[:] = landing.center_w_m + normal
    landing_prediction.body_center[7, 5] = landing.center_w_m - normal
    _, landing_bounds = preterminal_support_constraints(
        landing,
        landing_prediction,
        GENERATED,
    )
    assert landing_bounds[7 * landing_prediction.body_count + 5] < 0.0


def test_error_and_distance_support_contract() -> None:
    """Expose exact linear rows and local nonlinear support coefficients."""

    gate = _gate_mission(angle_rad=radians(23.0))
    prediction = _prediction(gate)
    lateral = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    offset, reference = error_support(
        gate,
        prediction,
        5,
        lateral,
        GENERATED,
        _domain(gate),
    )
    assert offset == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(reference, gate.width_axis_w, atol=1.0e-12)
    offset, reference = distance_support(gate, prediction, 0, 1.0)
    assert offset == pytest.approx(gate.distance(prediction.state_center[0]))
    np.testing.assert_allclose(reference, -gate.normal_w, atol=1.0e-12)

    heading = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    offset, reference = error_support(
        gate,
        prediction,
        5,
        heading,
        GENERATED,
        _domain(gate),
    )
    assert offset == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(reference, np.zeros(3), atol=1.0e-12)

    landing = _landing_mission()
    prediction = _prediction(landing)
    sink = np.zeros(5 + 3 * GENERATED.geometry.contact_b_m.shape[0])
    sink[5] = 1.0
    offset, reference = error_support(
        landing,
        prediction,
        9,
        sink,
        GENERATED,
        _domain(landing),
    )
    assert offset == pytest.approx(0.4)
    np.testing.assert_allclose(reference, np.zeros(3), atol=1.0e-12)


def test_landing_terminal_set_covers_every_contact_velocity_component() -> None:
    """Bind three velocity coordinates for every permitted contact point."""

    mission = _landing_mission()
    contact_count = GENERATED.geometry.contact_b_m.shape[0]
    dimension = 5 + 3 * contact_count
    terminal = mission.terminal_halfspaces(GENERATED)
    tangent_limit = mission.tangential_speed_max_m_s / sqrt(2.0)
    assert terminal.matrix.shape == (2 * dimension, dimension)
    lower = -terminal.bounds[dimension:]
    upper = terminal.bounds[:dimension]
    for point in range(contact_count):
        normal, length, width = 5 + 3 * point + np.arange(3)
        assert lower[normal] == pytest.approx(0.0)
        assert upper[normal] == pytest.approx(mission.normal_speed_max_m_s)
        assert lower[length] == pytest.approx(-tangent_limit)
        assert upper[length] == pytest.approx(tangent_limit)
        assert lower[width] == pytest.approx(-tangent_limit)
        assert upper[width] == pytest.approx(tangent_limit)


def test_landing_error_support_covers_positive_and_negative_velocity_facets() -> None:
    """Support every signed contact-velocity coordinate without aggregation."""

    mission = _landing_mission(angle_rad=radians(19.0))
    prediction = _prediction(mission)
    stage = 9
    prediction.state_center[stage, 6:12] = (0.3, -0.2, 0.4, 0.7, -0.5, 0.9)
    prediction.contact_velocity_center[stage] = point_velocities(
        prediction.state_center[stage],
        GENERATED.geometry.contact_b_m,
    )
    predicted_error = mission.error(prediction.state_center[stage], GENERATED)
    for facet in mission.terminal_halfspaces(GENERATED).matrix:
        offset, reference = error_support(
            mission,
            prediction,
            stage,
            facet,
            GENERATED,
            _domain(mission),
        )
        assert offset == pytest.approx(float(facet @ predicted_error))
        if np.any(facet[5:] != 0.0):
            np.testing.assert_allclose(reference, np.zeros(3), atol=1.0e-12)


@pytest.mark.parametrize(
    ("coordinate", "value"),
    (
        (5, radians(25.0)),
        (3, radians(25.0)),
        (4, radians(20.0)),
        (6, 10.0),
    ),
)
def test_gate_event_rows_cover_every_horizon_endpoint(
    coordinate: int,
    value: float,
) -> None:
    """Enforce heading, attitude, and speed at every crossing-horizon endpoint."""

    mission = _gate_mission()
    geometry_rows = 42 * GENERATED.geometry.body_b_m.shape[0]
    event_rows_per_stage = 8
    for stage in range(11):
        prediction = _prediction(mission)
        prediction.state_center[stage, coordinate] = value
        _, bounds = mission.terminal_support_constraints(
            prediction,
            GENERATED,
            _domain(mission),
        )
        first = geometry_rows + stage * event_rows_per_stage
        assert np.min(bounds[first : first + event_rows_per_stage]) < 0.0


@pytest.mark.parametrize(("stage", "coordinate"), ((9, 3), (9, 4), (10, 3), (10, 4)))
def test_landing_event_rows_cover_both_segment_endpoints(
    stage: int,
    coordinate: int,
) -> None:
    """Enforce roll and pitch on both endpoints bracketing first contact."""

    mission = _landing_mission()
    prediction = _prediction(mission)
    prediction.state_center[stage, coordinate] = radians(25.0)
    _, bounds = mission.terminal_support_constraints(
        prediction,
        GENERATED,
        _domain(mission),
    )
    block = 0 if stage == 9 else 1
    first = bounds.size - 20 + 10 * block
    attitude = bounds[first + np.array((3, 4, 8, 9))]
    assert np.min(attitude) < 0.0


def test_terminal_error_support_is_feasible_for_narrow_predicted_tubes() -> None:
    """Keep heading, airspeed, and contact-velocity facets locally usable."""

    cases = (
        (_gate_mission(angle_rad=radians(31.0)), 5),
        (_landing_mission(angle_rad=radians(-27.0)), 9),
    )
    for mission, stage in cases:
        prediction = _prediction(mission)
        terminal = mission.terminal_halfspaces(GENERATED)
        for facet, bound in zip(
            terminal.matrix,
            terminal.bounds,
            strict=True,
        ):
            offset, reference = error_support(
                mission,
                prediction,
                stage,
                facet,
                GENERATED,
                _domain(mission),
            )
            assert offset + reference @ np.zeros(3) <= bound + 1.0e-12


def test_mission_runtime_contract() -> None:
    mission = _gate_mission()
    required = (
        "error",
        "distance",
        "terminal_halfspaces",
        "free_space_halfspaces",
        "realized",
        "terminal_support_constraints",
    )
    assert all(hasattr(mission, name) for name in required)
