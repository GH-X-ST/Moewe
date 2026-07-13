"""Deterministic tube unit tests and concrete falsification rollouts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import pi

import numpy as np
import pytest

from control.flow import AffineFlow, FlowBounds
from control.interval import AffineForm, Interval, Zonotope
from control.tube import (
    BodyTube,
    FeedbackSegment,
    FlightDomain,
    MEASURED_COMMAND_ONSET_DELAY_S,
    ModelUncertainty,
    SegmentTube,
    TubeCertificationError,
    TubePropagator,
)
from models.aircraft import Aircraft
from models.geometry import RigidBodyGeometry, point_velocities, world_points

DT_S = 0.02
TEST_GEOMETRY = RigidBodyGeometry(
    body_b_m=((0.0, 0.0, 0.0),),
    contact_b_m=((0.0, 0.0, 0.0),),
    footprint_b_m=((0.0, 0.0, 0.0),),
)


@dataclass(frozen=True)
class _Realization:
    state: np.ndarray
    flow: AffineFlow
    density_kg_m3: float
    mass_kg: float
    force_error_n: np.ndarray
    moment_error_n_m: np.ndarray
    angular_error_rad_s2: np.ndarray
    actuator_tau_s: float
    command_error_rad: np.ndarray


def _setup() -> tuple[
    Aircraft,
    ModelUncertainty,
    FeedbackSegment,
    Interval,
    TubePropagator,
]:
    aircraft = Aircraft()
    flow = FlowBounds(
        center_lower_m_s=(-0.01, -0.01, -0.01),
        center_upper_m_s=(0.01, 0.01, 0.01),
        gradient_lower_s=np.full((3, 3), -0.002),
        gradient_upper_s=np.full((3, 3), 0.002),
        remainder_abs_m_s=(0.002, 0.002, 0.002),
    )
    uncertainty = ModelUncertainty(
        flow=flow,
        density_kg_m3=(1.224, 1.226),
        mass_kg=(0.1429, 0.1431),
        coefficient_scale=(1.0, 1.0),
        force_error_abs_n=(1.0e-5, 1.0e-5, 1.0e-5),
        moment_error_abs_n_m=(1.0e-7, 1.0e-7, 1.0e-7),
        angular_accel_error_abs_rad_s2=(1.0e-4, 1.0e-4, 1.0e-4),
        moment_reference_offset_abs_m=(0.0, 0.0, 0.0),
        actuator_tau_s=(0.059, 0.061),
        command_error_abs_rad=(1.0e-5, 1.0e-5, 1.0e-5),
        command_delay_max_s=0.0,
    )
    state = np.zeros(15)
    state[:3] = (2.0, 2.0, 2.0)
    state[6] = 6.0
    radius = np.array(
        [
            1.0e-5,
            1.0e-5,
            1.0e-5,
            1.0e-6,
            1.0e-6,
            1.0e-6,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-5,
            1.0e-5,
            1.0e-5,
            1.0e-6,
            1.0e-6,
            1.0e-6,
        ]
    )
    initial = Interval.from_midpoint(state, radius)
    gain = np.zeros((3, 15))
    gain[0, 3] = -0.02
    gain[1, 4] = -0.02
    gain[2, 5] = -0.02
    segment = FeedbackSegment(state, np.zeros(3), gain)
    propagator = TubePropagator(
        aircraft=aircraft,
        uncertainty=uncertainty,
        domain=FlightDomain(
            roll_abs_max_rad=pi / 3.0,
            pitch_abs_max_rad=pi / 3.0,
            airspeed_bounds_m_s=(1.0, 15.0),
            alpha_abs_max_rad=pi / 4.0,
            body_rate_abs_max_rad_s=10.0,
        ),
        geometry=TEST_GEOMETRY,
        substeps=4,
        picard_iterations=24,
        max_subdivisions=2,
    )
    return aircraft, uncertainty, segment, initial, propagator


def test_interval_arithmetic_extrema() -> None:
    """Exercise algebraic, periodic, and restricted inverse extrema."""

    product = Interval(-2.0, 3.0) * Interval(-4.0, 5.0)
    assert product.contains(-12.0)
    assert product.contains(15.0)
    angles = Interval(-0.5 * pi, 0.5 * pi)
    assert angles.sin().contains(-1.0)
    assert angles.sin().contains(1.0)
    assert Interval(0.0, 2.0 * pi).cos().contains(-1.0)
    assert Interval(0.0, 2.0 * pi).cos().contains(1.0)
    assert Interval(-0.4, 0.6).tan().contains(np.tan(-0.4))
    with pytest.raises(ValueError, match="pole"):
        Interval(0.4 * pi, 0.6 * pi).tan()
    bearing = Interval(-1.0, 2.0).atan2(Interval(0.5, 3.0))
    assert bearing.contains(np.arctan2(-1.0, 0.5))
    assert bearing.contains(np.arctan2(2.0, 0.5))
    vector = Interval((-1.0, 2.0, -3.0), (4.0, 5.0, -1.0))
    assert vector.norm().contains(np.linalg.norm((0.0, 3.0, -2.0)))
    assert vector.affine_map(np.eye(3)).contains((0.0, 3.0, -2.0))


def test_zonotope_operations_preserve_affine_dependence() -> None:
    """Keep generator dependence through affine and Minkowski maps."""

    dependent = Zonotope((0.0, 0.0), ((1.0,), (1.0,)))
    cancelled = dependent.affine_map(((1.0, -1.0),))
    assert float(cancelled.interval_hull().radius[0]) < 1e-12

    summed = dependent.minkowski(dependent)
    assert summed.generators.shape[1] >= 2
    assert summed.contains((2.0, 2.0))


def test_shared_gradient_tightens_loads_and_contains_realizations() -> None:
    """Retain one gradient across strips through the load enclosure."""

    aircraft, uncertainty, _, initial, propagator = _setup()
    gradient_lower = np.zeros((3, 3))
    gradient_upper = np.zeros((3, 3))
    gradient_lower[2, 0] = -0.2
    gradient_upper[2, 0] = 0.2
    flow = FlowBounds(
        np.zeros(3),
        np.zeros(3),
        gradient_lower,
        gradient_upper,
        np.zeros(3),
    )
    exact = replace(
        uncertainty,
        flow=flow,
        density_kg_m3=(1.225, 1.225),
        mass_kg=(0.143, 0.143),
        coefficient_scale=(1.0, 1.0),
        force_error_abs_n=np.zeros(3),
        moment_error_abs_n_m=np.zeros(3),
        angular_accel_error_abs_rad_s2=np.zeros(3),
        actuator_tau_s=(0.06, 0.06),
        command_error_abs_rad=np.zeros(3),
    )
    propagator = replace(propagator, uncertainty=exact)
    state = Interval.point(initial.center)
    force, moment = propagator._aero_loads(state)
    strip_lower, strip_upper = flow.strip_bounds(aircraft.strip_table.r_b_m)
    independent_force, independent_moment = propagator._aero_loads_with_flow(
        state,
        AffineForm.from_interval(Interval.point(np.zeros(3))),
        AffineForm.from_interval(Interval(strip_lower, strip_upper)),
    )

    assert moment.radius[2] < independent_moment.radius[2]
    assert force.radius[2] < independent_force.radius[2]
    for gradient_value in np.linspace(-0.2, 0.2, 41):
        gradient = np.zeros((3, 3))
        gradient[2, 0] = gradient_value
        strip_flow = AffineFlow(
            np.zeros(3),
            gradient,
            np.zeros_like(aircraft.strip_table.r_b_m),
        ).strip_flow(aircraft.strip_table.r_b_m)
        realized_force, realized_moment = aircraft.aero_loads_local_flow(
            initial.center,
            np.zeros(3),
            strip_flow,
            rho=1.225,
        )
        assert force.contains(realized_force)
        assert moment.contains(realized_moment)


def test_moment_reference_uncertainty_shifts_aerodynamic_moment() -> None:
    """Enclose the moment-arm change from every declared reference offset."""

    aircraft, uncertainty, _, initial, propagator = _setup()
    zero_flow = FlowBounds(
        np.zeros(3),
        np.zeros(3),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        np.zeros(3),
    )
    cg_error = np.array((0.01, 0.008, 0.006))
    exact = replace(
        uncertainty,
        flow=zero_flow,
        density_kg_m3=(1.225, 1.225),
        coefficient_scale=(1.0, 1.0),
        force_error_abs_n=np.zeros(3),
        moment_error_abs_n_m=np.zeros(3),
        moment_reference_offset_abs_m=cg_error,
    )
    propagator = replace(propagator, uncertainty=exact)
    force, moment = propagator._aero_loads(Interval.point(initial.center))
    realized_force, realized_moment = aircraft.aero_loads_local_flow(
        initial.center,
        np.zeros(3),
        np.zeros_like(aircraft.strip_table.r_b_m),
        rho=1.225,
    )
    assert force.contains(realized_force)
    for corner in np.ndindex((2, 2, 2)):
        sign = 2.0 * np.asarray(corner, dtype=float) - 1.0
        shifted = realized_moment - np.cross(sign * cg_error, realized_force)
        assert moment.contains(shifted)


def test_simultaneous_flow_model_actuator_extrema() -> None:
    """Contain simultaneous extrema from every implemented uncertainty group."""

    aircraft, uncertainty, segment, initial, propagator = _setup()
    command = segment.command_box(
        initial,
        aircraft,
        uncertainty.command_error_abs_rad,
    )
    derivative = propagator._derivative(initial, command)
    for sign in (-1.0, 1.0):
        realization = _realization(
            sign,
            aircraft,
            uncertainty,
            initial,
        )
        concrete = _concrete_derivative(
            realization.state,
            realization,
            segment,
            aircraft,
        )
        assert derivative.contains(concrete)


def test_nominal_twenty_millisecond_tube_and_geometry_shapes() -> None:
    """Certify one segment and expose all state and body enclosures."""

    aircraft, _, segment, initial, propagator = _setup()
    tube = propagator.propagate(initial, segment, DT_S)
    geometry = propagator.geometry
    body_count = np.asarray(geometry.body_b_m).reshape(-1, 3).shape[0]
    contact_count = np.asarray(geometry.contact_b_m).reshape(-1, 3).shape[0]
    footprint_count = np.asarray(geometry.footprint_b_m).reshape(-1, 3).shape[0]
    assert tube.successor.lower.shape == (15,)
    assert len(tube.states) >= propagator.substeps
    assert sum(tube.durations_s) == pytest.approx(DT_S)
    assert len(tube.body.occupied) == len(tube.states)
    assert len(tube.joint_flow) == len(tube.states)
    expected_flow_dimension = 3 * (aircraft.strip_table.r_b_m.shape[0] + 1)
    for flow_slice in tube.joint_flow:
        assert flow_slice.center.size == expected_flow_dimension
    for state_box, occupied, contact, footprint, velocity in zip(
        tube.states,
        tube.body.occupied,
        tube.body.contact,
        tube.body.footprint,
        tube.body.contact_velocity,
        strict=True,
    ):
        assert propagator.domain.contains(state_box, aircraft)
        assert occupied.lower.shape == (body_count, 3)
        assert contact.lower.shape == (contact_count, 3)
        assert footprint.lower.shape == (footprint_count, 3)
        assert velocity.lower.shape == (contact_count, 3)


def test_shared_flow_generators_reach_successor_zonotope() -> None:
    """Carry correlated centre/gradient effects through the state update."""

    aircraft, uncertainty, segment, initial, propagator = _setup()
    point = Interval.point(initial.center)
    flow = uncertainty.flow.affine_form(aircraft.strip_table.r_b_m)
    command = segment.command_box(point, aircraft, np.zeros(3))
    derivative = propagator._affine_derivative(
        point,
        command,
        flow[0],
        flow[1:],
    )
    initial_set = Zonotope.from_interval(point)
    full_successor = propagator._validated_successor(
        initial_set,
        derivative,
        DT_S,
    )
    successor = replace(propagator, max_generators=27)._validated_successor(
        initial_set,
        derivative,
        DT_S,
    )
    coupled = np.count_nonzero(
        np.abs(successor.generators) > 1.0e-15,
        axis=0,
    )

    assert derivative.generator_count == 12 + 3 * len(aircraft.strip_table.r_b_m)
    assert full_successor.generators.shape[1] == (
        initial_set.generators.shape[1] + derivative.generator_count + 15
    )
    assert successor.generators.shape[1] == 27
    assert np.any(coupled >= 2)


def test_each_time_piece_uses_an_independent_flow_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset temporal flow factors while sharing them across strips."""

    _, _, segment, initial, propagator = _setup()
    original = propagator._affine_derivative
    observed = []

    def track(
        state: Interval,
        command: Interval,
        center_flow: AffineForm,
        strip_flow: AffineForm,
    ) -> AffineForm:
        observed.append(center_flow.basis)
        return original(state, command, center_flow, strip_flow)

    monkeypatch.setattr(propagator, "_affine_derivative", track)
    tube = propagator.propagate(initial, segment, DT_S)
    piece_bases = []
    for basis in observed:
        if not piece_bases or basis is not piece_bases[-1]:
            piece_bases.append(basis)

    assert len(piece_bases) == len(tube.states)
    assert len({id(basis) for basis in piece_bases}) == len(piece_bases)


def test_tube_verification_deadline() -> None:
    """Abort deterministic candidate verification at its deadline."""

    _, _, segment, initial, propagator = _setup()
    with pytest.raises(TimeoutError, match="verification deadline"):
        propagator.propagate(initial, segment, DT_S, deadline=0.0)


def test_command_delay_requires_complete_issued_history() -> None:
    """Reject delayed propagation without every required ZOH command box."""

    _, uncertainty, segment, initial, propagator = _setup()
    delayed = replace(
        propagator,
        uncertainty=replace(
            uncertainty,
            command_delay_max_s=DT_S,
        ),
    )
    with pytest.raises(ValueError, match="requires 1 issued command history"):
        delayed.propagate(initial, segment, DT_S)


def test_measured_command_delay_requires_four_command_boxes() -> None:
    """Cover the measured 73 ms delay at the 20 ms control period."""

    _, uncertainty, segment, initial, propagator = _setup()
    defaulted = ModelUncertainty(
        flow=uncertainty.flow,
        density_kg_m3=uncertainty.density_kg_m3,
        mass_kg=uncertainty.mass_kg,
        coefficient_scale=uncertainty.coefficient_scale,
        force_error_abs_n=uncertainty.force_error_abs_n,
        moment_error_abs_n_m=uncertainty.moment_error_abs_n_m,
        angular_accel_error_abs_rad_s2=(uncertainty.angular_accel_error_abs_rad_s2),
        moment_reference_offset_abs_m=(uncertainty.moment_reference_offset_abs_m),
        actuator_tau_s=uncertainty.actuator_tau_s,
        command_error_abs_rad=uncertainty.command_error_abs_rad,
    )
    assert defaulted.command_delay_max_s == MEASURED_COMMAND_ONSET_DELAY_S
    delayed = replace(propagator, uncertainty=defaulted)
    command = Interval.point((0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="requires 4 issued command history"):
        delayed.propagate(
            initial,
            segment,
            DT_S,
            command_history=(command,) * 3,
        )
    tube = delayed.propagate(
        initial,
        segment,
        DT_S,
        command_history=(command,) * 4,
    )
    assert len(tube.command_history) == 4


def test_prior_delayed_command_is_enclosed_with_residual_error() -> None:
    """Hull current and prior issued commands before residual uncertainty."""

    aircraft, uncertainty, _, _, propagator = _setup()
    error = np.full(3, 0.01)
    delayed = replace(
        propagator,
        uncertainty=replace(
            uncertainty,
            command_error_abs_rad=error,
            command_delay_max_s=DT_S,
        ),
    )
    issued = Interval.point((-0.03, 0.02, -0.01))
    prior = Interval.point((0.04, -0.05, 0.03))
    history = delayed._validate_command_history((prior,), DT_S)
    applied = delayed._applied_command_box(issued, history)
    expected = issued.hull(prior)
    assert applied.contains(expected.lower - error)
    assert applied.contains(expected.upper + error)
    assert applied.subset(
        Interval(aircraft.control_lower_rad, aircraft.control_upper_rad)
    )


@pytest.mark.parametrize(
    ("substeps", "force_split"),
    ((2, False), (1, True)),
    ids=("adjacent_substeps", "adaptive_halves"),
)
def test_delay_encloses_commands_issued_earlier_in_segment(
    substeps: int,
    force_split: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Carry issued-command dependence across every internal tube piece."""

    aircraft, uncertainty, _, initial, propagator = _setup()
    delayed = replace(
        propagator,
        uncertainty=replace(
            uncertainty,
            command_delay_max_s=0.5 * DT_S,
        ),
        substeps=substeps,
        max_subdivisions=1,
    )
    gain = np.zeros((3, 15))
    gain[0, 0] = 0.1
    segment = FeedbackSegment(initial.center, np.zeros(3), gain)
    prior = Interval.point((0.2, 0.0, 0.0))
    applied_boxes = []

    def step(
        initial_set: Zonotope,
        law: FeedbackSegment,
        duration_s: float,
        deadline: float | None,
        history: tuple[Interval, ...],
        issued_in_segment: Interval | None,
    ) -> tuple[Zonotope, Interval]:
        del deadline
        if force_split and duration_s > 0.5 * DT_S:
            raise TubeCertificationError("force adaptive split")
        initial_box = initial_set.interval_hull()
        issued = law.command_box(initial_box, aircraft, np.zeros(3))
        applied_boxes.append(
            delayed._applied_command_box(
                issued,
                history,
                issued_in_segment,
            )
        )
        center = np.array(initial_set.center)
        center[0] += 1.0
        successor = Zonotope(center, initial_set.generators)
        return successor, initial_box.hull(successor.interval_hull())

    monkeypatch.setattr(delayed, "_step", step)
    delayed.propagate(
        initial,
        segment,
        DT_S,
        command_history=(prior,),
    )
    assert len(applied_boxes) == 2
    assert applied_boxes[1].lower[0] < 0.01
    assert applied_boxes[1].upper[0] >= 0.2


def test_command_history_shifts_across_a_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace the oldest FIFO box with each segment's issued-command hull."""

    aircraft, uncertainty, segment, initial, propagator = _setup()
    delayed = replace(
        propagator,
        uncertainty=replace(
            uncertainty,
            command_delay_max_s=2.0 * DT_S,
        ),
    )
    first = Interval.point((0.01, -0.01, 0.0))
    second = Interval.point((0.02, 0.0, -0.01))

    def propagate(
        initial_set: Interval | Zonotope,
        _: FeedbackSegment,
        dt_s: float,
        stop: object = None,
        deadline: float | None = None,
        command_history: tuple[Interval, ...] = (),
    ) -> SegmentTube:
        del stop, deadline
        initial_zonotope = (
            initial_set
            if isinstance(initial_set, Zonotope)
            else Zonotope.from_interval(initial_set)
        )
        state_box = initial_zonotope.interval_hull()
        return SegmentTube(
            initial=initial_zonotope,
            successor=initial_zonotope,
            states=(state_box,),
            body=BodyTube((), (), (), ()),
            durations_s=(dt_s,),
            command_history=command_history,
        )

    monkeypatch.setattr(delayed, "propagate", propagate)
    tubes = delayed.propagate_plan(
        initial,
        (segment, segment),
        DT_S,
        initial_command_history=(first, second),
    )
    assert len(tubes) == 2
    np.testing.assert_array_equal(tubes[1].command_history[0].lower, second.lower)
    np.testing.assert_array_equal(tubes[1].command_history[0].upper, second.upper)
    expected = segment.command_box(tubes[0].states[0], aircraft, np.zeros(3))
    for state_box in tubes[0].states[1:]:
        expected = expected.hull(segment.command_box(state_box, aircraft, np.zeros(3)))
    np.testing.assert_array_equal(tubes[1].command_history[1].lower, expected.lower)
    np.testing.assert_array_equal(tubes[1].command_history[1].upper, expected.upper)


def test_hard_domain_rejection() -> None:
    """Reject a candidate whose initial attitude violates the hard domain."""

    _, _, segment, initial, propagator = _setup()
    lower = np.array(initial.lower)
    upper = np.array(initial.upper)
    lower[4] = 1.2
    upper[4] = 1.21
    with pytest.raises(TubeCertificationError, match="hard domain"):
        propagator.propagate(Interval(lower, upper), segment, DT_S)

    center = np.array(initial.center)
    center[9] = propagator.domain.body_rate_abs_max_rad_s - 1.0e-4
    uncertainty = replace(
        propagator.uncertainty,
        angular_accel_error_abs_rad_s2=(10.0, 0.0, 0.0),
    )
    escaping = replace(propagator, uncertainty=uncertainty)
    with pytest.raises(TubeCertificationError, match="hard domain"):
        escaping.propagate(
            Interval.from_midpoint(center, initial.radius),
            segment,
            DT_S,
        )

    reverse_flow = FlowBounds(
        center_lower_m_s=(7.5, 0.0, 0.0),
        center_upper_m_s=(7.5, 0.0, 0.0),
        gradient_lower_s=np.zeros((3, 3)),
        gradient_upper_s=np.zeros((3, 3)),
        remainder_abs_m_s=np.zeros(3),
    )
    invalid_flow = replace(
        propagator,
        uncertainty=replace(propagator.uncertainty, flow=reverse_flow),
    )
    with pytest.raises(TubeCertificationError, match="forward-flight"):
        invalid_flow.propagate(initial, segment, DT_S)


@pytest.mark.parametrize(
    "change",
    (
        {"roll_abs_max_rad": 0.0},
        {"pitch_abs_max_rad": 0.5 * pi},
        {"airspeed_bounds_m_s": (0.0, 15.0)},
        {"airspeed_bounds_m_s": (15.0, 1.0)},
        {"alpha_abs_max_rad": 0.5 * pi},
        {"body_rate_abs_max_rad_s": np.inf},
    ),
)
def test_invalid_flight_domain_limits_are_rejected(
    change: dict[str, object],
) -> None:
    """Reject invalid limits before they can support a certificate."""

    _, _, _, _, propagator = _setup()
    with pytest.raises(ValueError, match="flight-domain limits"):
        replace(propagator.domain, **change)


def test_corner_and_adversarial_rollouts_stay_in_tube() -> None:
    """Falsify the tube against deterministic high-resolution rollouts."""

    aircraft, uncertainty, segment, initial, propagator = _setup()
    tube = propagator.propagate(initial, segment, DT_S)
    assert len(tube.states) >= propagator.substeps
    boundaries = np.cumsum(tube.durations_s)
    for sign in (-1.0, 1.0, -0.5, 0.5):
        realization = _realization(
            sign,
            aircraft,
            uncertainty,
            initial,
        )
        states = _rollout(realization, segment, aircraft, steps=200)
        assert tube.successor.interval_hull().contains(states[-1])
        for sample_index, state in enumerate(states):
            time_s = DT_S * sample_index / (len(states) - 1)
            tube_index = min(
                int(np.searchsorted(boundaries, time_s, side="left")),
                len(tube.states) - 1,
            )
            assert tube.states[tube_index].contains(state)
            assert tube.body.occupied[tube_index].contains(
                world_points(state, propagator.geometry.body_b_m)
            )
            assert tube.body.contact[tube_index].contains(
                world_points(state, propagator.geometry.contact_b_m)
            )
            assert tube.body.contact_velocity[tube_index].contains(
                point_velocities(state, propagator.geometry.contact_b_m)
            )

    random = np.random.default_rng(4217)
    for _ in range(256):
        realization = _random_realization(
            random,
            aircraft,
            uncertainty,
            initial,
        )
        states = _rollout(realization, segment, aircraft, steps=40)
        assert tube.successor.interval_hull().contains(states[-1])
        for sample_index, state in enumerate(states):
            time_s = DT_S * sample_index / (len(states) - 1)
            tube_index = min(
                int(np.searchsorted(boundaries, time_s, side="left")),
                len(tube.states) - 1,
            )
            assert tube.states[tube_index].contains(state)


def test_switching_affine_flow_signal_stays_in_tube() -> None:
    """Contain arbitrary temporal switching inside the joint amplitude set."""

    aircraft, uncertainty, segment, initial, propagator = _setup()
    tube = propagator.propagate(initial, segment, DT_S)
    flow = uncertainty.flow
    remainder = np.broadcast_to(
        flow.remainder_abs_m_s,
        aircraft.strip_table.r_b_m.shape,
    )
    lower = AffineFlow(
        flow.center_lower_m_s,
        flow.gradient_lower_s,
        -remainder,
    )
    upper = AffineFlow(
        flow.center_upper_m_s,
        flow.gradient_upper_s,
        remainder,
    )
    realization = _realization(0.5, aircraft, uncertainty, initial)
    realization = replace(
        realization,
        state=np.array(initial.center),
        density_kg_m3=float(np.mean(uncertainty.density_kg_m3)),
        mass_kg=float(np.mean(uncertainty.mass_kg)),
        force_error_n=np.zeros(3),
        moment_error_n_m=np.zeros(3),
        angular_error_rad_s2=np.zeros(3),
        actuator_tau_s=float(np.mean(uncertainty.actuator_tau_s)),
        command_error_rad=np.zeros(3),
    )
    schedules = (
        lambda time_s: lower if time_s < 0.5 * DT_S else upper,
        lambda time_s: (
            lower
            if int(min(time_s, np.nextafter(DT_S, 0.0)) / (DT_S / 20)) % 2 == 0
            else upper
        ),
    )
    boundaries = np.cumsum(tube.durations_s)
    for schedule in schedules:
        states = _rollout_with_flow(
            realization,
            segment,
            aircraft,
            schedule,
            steps=200,
        )
        assert tube.successor.interval_hull().contains(states[-1])
        for sample_index, state in enumerate(states):
            time_s = DT_S * sample_index / (len(states) - 1)
            tube_index = min(
                int(np.searchsorted(boundaries, time_s, side="left")),
                len(tube.states) - 1,
            )
            assert tube.states[tube_index].contains(state)


def test_failed_coarse_enclosure_forces_temporal_subdivision() -> None:
    """Bisect a segment until Picard self-inclusion is established."""

    _, _, segment, initial, propagator = _setup()
    adaptive = replace(
        propagator,
        substeps=1,
        picard_iterations=2,
        max_subdivisions=6,
    )
    tube = adaptive.propagate(initial, segment, DT_S)
    assert len(tube.durations_s) > 1
    assert sum(tube.durations_s) == pytest.approx(DT_S)


def test_terminal_stop_prevents_post_event_subdivision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop after a certified first half without propagating fictitious flight."""

    _, _, segment, initial, propagator = _setup()
    adaptive = replace(propagator, substeps=1, max_subdivisions=1)
    calls = []

    def step(
        state: Zonotope,
        _: FeedbackSegment,
        dt_s: float,
        __: float | None,
        ___: tuple[Interval, ...],
        ____: Interval | None,
    ) -> tuple[Zonotope, Interval]:
        if dt_s == DT_S:
            raise TubeCertificationError("split")
        calls.append(dt_s)
        return state, state.interval_hull()

    monkeypatch.setattr(adaptive, "_step", step)
    tube = adaptive.propagate(initial, segment, DT_S, stop=lambda _: True)
    assert calls == [0.5 * DT_S]
    assert tube.durations_s == (0.5 * DT_S,)


def _realization(
    sign: float,
    aircraft: Aircraft,
    uncertainty: ModelUncertainty,
    initial: Interval,
) -> _Realization:
    lower_sign = sign < 0.0
    state = np.where(lower_sign, initial.lower, initial.upper)
    if abs(sign) < 1.0:
        state = initial.center + sign * (initial.upper - initial.lower) * 0.5
    flow = uncertainty.flow
    center = np.where(
        lower_sign,
        flow.center_lower_m_s,
        flow.center_upper_m_s,
    )
    gradient = np.where(
        lower_sign,
        flow.gradient_lower_s,
        flow.gradient_upper_s,
    )
    remainder_limit = np.broadcast_to(
        np.asarray(flow.remainder_abs_m_s, dtype=float),
        aircraft.strip_table.r_b_m.shape,
    )
    remainder_pattern = np.where(
        np.arange(remainder_limit.size).reshape(remainder_limit.shape) % 2,
        -sign,
        sign,
    )
    return _Realization(
        state=np.asarray(state, dtype=float),
        flow=AffineFlow(center, gradient, remainder_pattern * remainder_limit),
        density_kg_m3=(
            uncertainty.density_kg_m3[0] if lower_sign else uncertainty.density_kg_m3[1]
        ),
        mass_kg=(uncertainty.mass_kg[0] if lower_sign else uncertainty.mass_kg[1]),
        force_error_n=sign * np.asarray(uncertainty.force_error_abs_n, dtype=float),
        moment_error_n_m=sign
        * np.asarray(uncertainty.moment_error_abs_n_m, dtype=float),
        angular_error_rad_s2=sign
        * np.asarray(
            uncertainty.angular_accel_error_abs_rad_s2,
            dtype=float,
        ),
        actuator_tau_s=(
            uncertainty.actuator_tau_s[0]
            if lower_sign
            else uncertainty.actuator_tau_s[1]
        ),
        command_error_rad=sign
        * np.asarray(uncertainty.command_error_abs_rad, dtype=float),
    )


def _random_realization(
    random: np.random.Generator,
    aircraft: Aircraft,
    uncertainty: ModelUncertainty,
    initial: Interval,
) -> _Realization:
    flow = uncertainty.flow
    locations = aircraft.strip_table.r_b_m
    remainder_limit = np.broadcast_to(flow.remainder_abs_m_s, locations.shape)
    return _Realization(
        state=random.uniform(initial.lower, initial.upper),
        flow=AffineFlow(
            random.uniform(flow.center_lower_m_s, flow.center_upper_m_s),
            random.uniform(flow.gradient_lower_s, flow.gradient_upper_s),
            random.uniform(-remainder_limit, remainder_limit),
        ),
        density_kg_m3=random.uniform(*uncertainty.density_kg_m3),
        mass_kg=random.uniform(*uncertainty.mass_kg),
        force_error_n=random.uniform(
            -uncertainty.force_error_abs_n,
            uncertainty.force_error_abs_n,
        ),
        moment_error_n_m=random.uniform(
            -uncertainty.moment_error_abs_n_m,
            uncertainty.moment_error_abs_n_m,
        ),
        angular_error_rad_s2=random.uniform(
            -uncertainty.angular_accel_error_abs_rad_s2,
            uncertainty.angular_accel_error_abs_rad_s2,
        ),
        actuator_tau_s=random.uniform(*uncertainty.actuator_tau_s),
        command_error_rad=random.uniform(
            -uncertainty.command_error_abs_rad,
            uncertainty.command_error_abs_rad,
        ),
    )


def _concrete_derivative(
    state: np.ndarray,
    realization: _Realization,
    segment: FeedbackSegment,
    aircraft: Aircraft,
) -> np.ndarray:
    center = np.asarray(realization.flow.center_b_m_s, dtype=float)
    strips = realization.flow.strip_flow(aircraft.strip_table.r_b_m)
    command = aircraft.clip_control(
        segment.command(state, aircraft) + realization.command_error_rad
    )
    derivative = aircraft.derivative_local_flow(
        state,
        command,
        center,
        strips,
        realization.density_kg_m3,
    )
    force, _ = aircraft.aero_loads_local_flow(
        state,
        center,
        strips,
        realization.density_kg_m3,
    )
    derivative[6:9] += force * (1.0 / realization.mass_kg - 1.0 / aircraft.mass_kg)
    derivative[6:9] += realization.force_error_n / realization.mass_kg
    derivative[9:12] += (
        aircraft.inertia_inv_b @ realization.moment_error_n_m
        + realization.angular_error_rad_s2
    )
    derivative[12:15] = (command - state[12:15]) / realization.actuator_tau_s
    return derivative


def _rollout(
    realization: _Realization,
    segment: FeedbackSegment,
    aircraft: Aircraft,
    steps: int,
) -> tuple[np.ndarray, ...]:
    step_s = DT_S / steps
    state = np.array(realization.state)
    states = [state.copy()]

    def derivative(value: np.ndarray) -> np.ndarray:
        return _concrete_derivative(value, realization, segment, aircraft)

    for _ in range(steps):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * step_s * k1)
        k3 = derivative(state + 0.5 * step_s * k2)
        k4 = derivative(state + step_s * k3)
        state = state + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        states.append(state.copy())
    return tuple(states)


def _rollout_with_flow(
    realization: _Realization,
    segment: FeedbackSegment,
    aircraft: Aircraft,
    flow_at: Callable[[float], AffineFlow],
    steps: int,
) -> tuple[np.ndarray, ...]:
    step_s = DT_S / steps
    state = np.array(realization.state)
    states = [state.copy()]

    def derivative(value: np.ndarray, time_s: float) -> np.ndarray:
        current = replace(realization, flow=flow_at(time_s))
        return _concrete_derivative(value, current, segment, aircraft)

    for index in range(steps):
        time_s = index * step_s
        k1 = derivative(state, time_s)
        k2 = derivative(state + 0.5 * step_s * k1, time_s + 0.5 * step_s)
        k3 = derivative(state + 0.5 * step_s * k2, time_s + 0.5 * step_s)
        k4 = derivative(state + step_s * k3, time_s + step_s)
        state = state + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        states.append(state.copy())
    return tuple(states)
