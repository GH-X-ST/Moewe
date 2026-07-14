"""Offline nonlinear propagation, generation, and falsification tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from control.flow import SHARED_FLOW_GENERATOR_COUNT, FlowBounds
from control.interval import Interval, Zonotope
from control.missions import FreeSpace, GateMission, LandingMission
from control.oracle import (
    GeometryEnclosure,
    NonlinearOracle,
    OraclePrediction,
    OracleStage,
)
from control.predictor import FastPredictor, Prediction, _generate_aircraft
from control.uncertainty import Bounds, FAST_PERIOD_S, PREDICTION_STAGES
from models.aircraft import Aircraft
from models.geometry import RigidBodyGeometry


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


def _bounds() -> Bounds:
    return Bounds(
        flow=FlowBounds(
            (-0.01, -0.01, -0.01),
            (0.01, 0.01, 0.01),
            np.full((3, 3), -0.002),
            np.full((3, 3), 0.002),
            (0.002, 0.002, 0.002),
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
        nonlinear_remainder_abs=np.zeros(15),
        numerical_remainder_abs=np.zeros(15),
        body_inflation_m=1.0e-4,
        mission_position_error_abs_m=1.0e-4,
        mission_attitude_error_abs_rad=1.0e-4,
        roll_abs_max_rad=np.deg2rad(60.0),
        pitch_abs_max_rad=np.deg2rad(60.0),
        airspeed_m_s=(1.0, 15.0),
        alpha_abs_max_rad=np.deg2rad(45.0),
        body_rate_abs_max_rad_s=10.0,
    )


def _setup() -> tuple[NonlinearOracle, Zonotope, np.ndarray]:
    aircraft = Aircraft()
    bounds = _bounds()
    state = np.zeros(15)
    state[2] = 2.0
    state[4] = -0.05616951
    state[6:9] = (5.96653821, 0.0, 0.63278890)
    control = np.array((0.0, 0.11181891, 0.0))
    state[12:15] = control
    generated = _generate_aircraft(
        aircraft,
        _geometry(),
        bounds,
        state,
        control,
    )
    belief = Zonotope(state, np.diag(bounds.state_estimation_abs))
    queue = np.tile(control, (bounds.queue_length, 1))
    return NonlinearOracle(generated), belief, queue


def _inner(value: Interval, fraction: float = 0.5) -> Interval:
    return Interval.from_midpoint(value.center, fraction * value.radius)


def _fast_geometry(
    prediction: Prediction,
    stage: int,
    reference: np.ndarray,
    fraction: float = 0.5,
) -> GeometryEnclosure:
    def enclosure(
        center: np.ndarray,
        coefficient: np.ndarray,
        radius: np.ndarray,
    ) -> Interval:
        value = Interval.from_midpoint(
            center[stage] + coefficient[stage] @ reference,
            radius[stage],
        )
        return _inner(value, fraction)

    return GeometryEnclosure(
        enclosure(
            prediction.body_center,
            prediction.body_reference,
            prediction.body_radius,
        ),
        enclosure(
            prediction.contact_center,
            prediction.contact_reference,
            prediction.contact_radius,
        ),
        enclosure(
            prediction.footprint_center,
            prediction.footprint_reference,
            prediction.footprint_radius,
        ),
        enclosure(
            prediction.contact_velocity_center,
            prediction.contact_velocity_reference,
            prediction.contact_velocity_radius,
        ),
    )


def test_picard_step_contains_simultaneous_corner_and_shared_flow() -> None:
    """Validate an aircraft step with every uncertainty group active."""

    oracle, belief, _ = _setup()
    cell = oracle.generated.cells[0]
    command = oracle._issued_command(
        belief.interval_hull(),
        Interval.point(belief.center),
        cell,
        Interval.point(cell.control_anchor),
    )
    flow = oracle._flow.affine_form()
    derivative = oracle._affine_derivative(
        belief.interval_hull(),
        command,
        flow[0],
        flow[1:],
    ).interval_hull()
    factors = np.ones(oracle.factor_count(belief))
    realization = oracle._realization(belief, factors)
    concrete_command = oracle.aircraft.clip_control(
        cell.control_anchor
        + cell.gain
        @ (
            np.asarray(realization["state"])
            + np.asarray(realization["measurement"])[0]
            - belief.center
        )
        + np.asarray(realization["command_error"])[0]
    )
    concrete = oracle._concrete_derivative(
        np.asarray(realization["state"]),
        concrete_command,
        np.asarray(realization["flows"])[0],
        realization,
    )
    assert derivative.contains(concrete)

    successor, continuous = oracle._step(belief, command, 1.0e-4)
    assert oracle._inside_domain(continuous, oracle.joint_flow.interval_hull()[:3])
    assert successor.interval_hull().lower.shape == (15,)
    geometry = oracle._geometry(continuous)
    assert geometry.occupied.lower.shape == (5, 3)
    joint = oracle.joint_flow
    assert joint.generators.shape[1] >= SHARED_FLOW_GENERATOR_COUNT
    strip_count = oracle.aircraft.strip_table.r_b_m.shape[0]
    for column in range(3):
        values = joint.generators[:, column].reshape(strip_count + 1, 3)
        np.testing.assert_allclose(values, np.tile(values[0], (strip_count + 1, 1)))


def test_hard_domain_uses_air_relative_velocity() -> None:
    oracle, belief, _ = _setup()
    lower = belief.interval_hull().lower.copy()
    upper = belief.interval_hull().upper.copy()
    lower[6] = 1.5
    upper[6] = 1.5
    state = Interval(lower, upper)
    assert not oracle._inside_domain(state, Interval.point((1.0, 0.0, 0.0)))


def test_ten_stage_queue_and_remainder_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete horizon and uncertain measured command queue."""

    oracle, belief, queue = _setup()
    cell = oracle.generated.cells[0]

    def stage(
        initial: Zonotope,
        issued: Interval,
        applied: Interval,
    ) -> OracleStage:
        state = initial.interval_hull()
        geometry = oracle._geometry(state)
        return OracleStage(
            initial,
            initial,
            state,
            geometry,
            geometry,
            issued,
            applied,
            (FAST_PERIOD_S,),
            oracle.joint_flow,
        )

    monkeypatch.setattr(oracle, "_propagate_stage", stage)
    queue_radius = np.full_like(queue, 2.0e-3)
    uncertain_queue = tuple(
        Interval.from_midpoint(center, radius)
        for center, radius in zip(queue, queue_radius, strict=True)
    )
    prediction = oracle.propagate(
        belief,
        uncertain_queue,
        cell,
        cell.control_anchor,
    )
    point_prediction = oracle.propagate(
        belief,
        queue,
        cell,
        cell.control_anchor,
    )

    assert len(prediction.stages) == PREDICTION_STAGES
    assert prediction.stages[0].applied_command.contains(queue[0] - queue_radius[0])
    assert prediction.stages[0].applied_command.contains(queue[1] + queue_radius[1])
    for uncertain, point in zip(
        prediction.stages,
        point_prediction.stages,
        strict=True,
    ):
        assert point.issued_command.subset(uncertain.issued_command)
    assert np.max(prediction.stages[5].initial.radius) > np.max(
        point_prediction.stages[5].initial.radius
    )
    remainder = oracle.remainder_bounds(
        belief,
        uncertain_queue,
        cell,
        Interval(cell.control_anchor - 1.0e-3, cell.control_anchor + 1.0e-3),
    )
    assert remainder.nonlinear_abs.shape == (PREDICTION_STAGES, 15)
    assert remainder.numerical_abs.shape == (PREDICTION_STAGES, 15)
    assert np.all(remainder.nonlinear_abs >= 0.0)
    assert np.all(remainder.numerical_abs > 0.0)


def test_compare_uses_swept_runtime_geometry() -> None:
    """Compare continuous oracle sets with predictor geometry arrays."""

    oracle, belief, queue = _setup()
    cell = oracle.generated.cells[0]
    reference = cell.control_anchor
    fast = FastPredictor(oracle.generated).predict(belief, queue, 0)
    stages = []
    for stage in range(PREDICTION_STAGES):
        initial_box = _inner(fast.state_interval(stage, reference, reference))
        successor_box = _inner(fast.state_interval(stage + 1, reference, reference))
        initial = Zonotope.from_interval(initial_box)
        successor = Zonotope.from_interval(successor_box)
        geometry = _fast_geometry(fast, stage, reference)
        issued = _inner(
            Interval.from_midpoint(
                fast.issued_center[stage] + fast.issued_reference[stage] @ reference,
                fast.issued_radius[stage],
            )
        )
        applied = _inner(
            Interval.from_midpoint(
                fast.applied_center[stage] + fast.applied_reference[stage] @ reference,
                fast.applied_radius[stage],
            )
        )
        stages.append(
            OracleStage(
                initial,
                successor,
                initial_box.hull(successor_box),
                oracle._geometry(successor_box),
                geometry,
                issued,
                applied,
                (FAST_PERIOD_S,),
                oracle.joint_flow,
            )
        )
    initial = Zonotope.from_interval(_inner(belief.interval_hull()))
    nonlinear = OraclePrediction(
        initial,
        oracle._geometry(initial.interval_hull()),
        tuple(stages),
    )
    comparison = oracle.compare(fast, nonlinear, reference)
    assert comparison.valid

    first = stages[0]
    occupied = first.continuous_geometry.occupied
    escaped = Interval(occupied.lower + 100.0, occupied.upper + 100.0)
    bad_geometry = replace(first.continuous_geometry, occupied=escaped)
    bad = replace(
        nonlinear,
        stages=(replace(first, continuous_geometry=bad_geometry), *stages[1:]),
    )
    failed = oracle.compare(fast, bad, reference)
    assert np.min(failed.state_margin) >= 0.0
    assert failed.occupied_margin[1] < 0.0
    assert not failed.valid


def test_deterministic_corners_rollout_and_adversarial_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run concrete full-horizon corners and deterministic group search."""

    oracle, belief, queue = _setup()
    cell = oracle.generated.cells[0]
    corners = oracle.deterministic_corners(belief)
    assert np.array_equal(corners, oracle.deterministic_corners(belief))
    assert np.all(np.abs(corners) == 1.0)
    for factors in corners[:2]:
        states = oracle.rollout(
            belief,
            queue,
            cell,
            cell.control_anchor,
            factors,
            integration_steps=2,
        )
        assert states.shape == (PREDICTION_STAGES + 1, 15)
        assert np.all(np.isfinite(states))

    fast = FastPredictor(oracle.generated).predict(belief, queue, 0)

    def escaped_rollout(
        _: Zonotope,
        __: np.ndarray,
        ___: object,
        reference: np.ndarray,
        factors: np.ndarray,
        integration_steps: int = 10,
    ) -> np.ndarray:
        del integration_steps
        states = np.stack(
            [
                fast.state_center[stage] + fast.state_reference[stage] @ reference
                for stage in range(PREDICTION_STAGES + 1)
            ]
        )
        del factors
        return states

    monkeypatch.setattr(oracle, "rollout", escaped_rollout)
    fast.body_center[:, :, 0] += 10.0
    result = oracle.falsify(
        fast,
        belief,
        queue,
        cell,
        cell.control_anchor,
        passes=1,
    )
    assert result.escaped
    assert np.all(np.abs(result.factors) <= 1.0)


def _landing_prediction(
    oracle: NonlinearOracle,
    geometry: RigidBodyGeometry,
    upward: bool = False,
    forbidden_penetration: bool = False,
) -> OraclePrediction:
    command = Interval.point(np.zeros(3))
    joint = oracle.joint_flow
    stages = []
    initial_state = np.zeros(15)
    initial_state[2] = 0.1
    initial_state[6] = 5.0
    initial = Zonotope(initial_state, np.zeros((15, 0)))

    def geometry_box(contact: Interval, velocity_z: float) -> GeometryEnclosure:
        forbidden = Interval.point(((0.0, 0.0, 0.3),))
        occupied = Interval(
            np.vstack((forbidden.lower, contact.lower)),
            np.vstack((forbidden.upper, contact.upper)),
        )
        velocity = Interval.point(np.tile((0.0, 0.0, velocity_z), (2, 1)))
        footprint = Interval(
            contact.lower.copy(),
            contact.upper.copy(),
        )
        return GeometryEnclosure(occupied, contact, footprint, velocity)

    initial_contact = Interval.point(((-0.2, -0.2, 0.1), (-0.2, 0.2, 0.1)))
    initial_geometry = geometry_box(initial_contact, -0.2)
    for index in range(PREDICTION_STAGES):
        state_lower = initial_state.copy()
        state_upper = initial_state.copy()
        state_lower[2] = 0.05
        state_upper[2] = 0.1
        state_lower[8] = 0.2
        state_upper[8] = 0.2
        if index == 3:
            state_lower[2] = -0.25 if forbidden_penetration else -0.01
            contact = Interval(
                ((-0.2, -0.2, -0.01), (-0.2, 0.2, 0.02)),
                ((-0.2, -0.2, 0.05), (-0.2, 0.2, 0.08)),
            )
            boundary_contact = Interval(
                ((-0.2, -0.2, -0.01), (-0.2, 0.2, 0.03)),
                ((-0.2, -0.2, 0.0), (-0.2, 0.2, 0.05)),
            )
        else:
            contact = Interval(
                ((-0.2, -0.2, 0.05), (-0.2, 0.2, 0.05)),
                ((-0.2, -0.2, 0.1), (-0.2, 0.2, 0.1)),
            )
            boundary_contact = Interval.point(((-0.2, -0.2, 0.08), (-0.2, 0.2, 0.08)))
        state = Interval(state_lower, state_upper)
        velocity_z = 0.2 if upward and index == 3 else -0.2
        continuous_geometry = geometry_box(contact, velocity_z)
        boundary_geometry = geometry_box(boundary_contact, velocity_z)
        stages.append(
            OracleStage(
                Zonotope.from_interval(state),
                Zonotope.from_interval(state),
                state,
                boundary_geometry,
                continuous_geometry,
                command,
                command,
                (FAST_PERIOD_S,),
                joint,
            )
        )
    return OraclePrediction(initial, initial_geometry, tuple(stages))


def _gate_prediction(
    oracle: NonlinearOracle,
    heading_rad: float = 0.0,
) -> OraclePrediction:
    command = Interval.point(np.zeros(3))
    states = []
    for index in range(PREDICTION_STAGES + 1):
        state = np.zeros(15)
        state[0] = -1.0 + 0.2 * index
        state[5] = heading_rad
        state[6] = 5.0
        states.append(state)
    initial = Zonotope(states[0], np.zeros((15, 0)))
    stages = []
    for index in range(PREDICTION_STAGES):
        continuous = Interval(
            np.minimum(states[index], states[index + 1]),
            np.maximum(states[index], states[index + 1]),
        )
        successor = Zonotope(states[index + 1], np.zeros((15, 0)))
        stages.append(
            OracleStage(
                Zonotope(states[index], np.zeros((15, 0))),
                successor,
                continuous,
                oracle._geometry(Interval.point(states[index + 1])),
                oracle._geometry(continuous),
                command,
                command,
                (FAST_PERIOD_S,),
                oracle.joint_flow,
            )
        )
    return OraclePrediction(
        initial,
        oracle._geometry(Interval.point(states[0])),
        tuple(stages),
    )


def test_gate_event_enforces_terminal_air_data_and_attitude() -> None:
    """Require the complete swept body and terminal gate quantities."""

    oracle, _, _ = _setup()
    mission = GateMission(
        FreeSpace.box((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0)),
        center_w_m=(0.0, 0.0, 0.0),
        normal_w=(1.0, 0.0, 0.0),
        width_axis_w=(0.0, 1.0, 0.0),
        width_m=1.0,
        height_m=1.0,
        target_airspeed_m_s=5.0,
        heading_abs_max_rad=0.2,
        roll_abs_max_rad=0.2,
        pitch_bounds_rad=(-0.2, 0.2),
        airspeed_bounds_m_s=(4.5, 5.5),
        frame_clearance_m=0.0,
    )
    assert oracle._gate_event(_gate_prediction(oracle), mission)
    assert not oracle._gate_event(_gate_prediction(oracle, heading_rad=0.5), mission)


def test_first_contact_signs_and_aircraft_geometry() -> None:
    """Permit realization-dependent contact only through approved points."""

    oracle, _, _ = _setup()
    body = ((0.0, 0.0, -0.2), (-0.2, -0.2, 0.0), (-0.2, 0.2, 0.0))
    geometry = RigidBodyGeometry(body, body[1:], body[1:])
    oracle = NonlinearOracle(replace(oracle.generated, geometry=geometry))
    mission = LandingMission(
        FreeSpace.box((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0)),
        center_w_m=(0.0, 0.0, 0.0),
        length_axis_w=(1.0, 0.0, 0.0),
        width_axis_w=(0.0, 1.0, 0.0),
        length_m=2.0,
        width_m=2.0,
        height_bounds_m=(-0.1, 0.2),
        roll_abs_max_rad=0.2,
        touchdown_pitch_rad=0.0,
        pitch_error_abs_max_rad=0.2,
        normal_speed_max_m_s=1.0,
        tangential_speed_max_m_s=1.0,
        platform_clearance_m=0.0,
    )
    valid = _landing_prediction(oracle, geometry)
    assert oracle._landing_event(valid, mission)
    assert not oracle._landing_event(
        _landing_prediction(oracle, geometry, upward=True),
        mission,
    )
    assert not oracle._landing_event(
        _landing_prediction(oracle, geometry, forbidden_penetration=True),
        mission,
    )
