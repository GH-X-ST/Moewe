"""Deterministic tube unit tests and concrete falsification rollouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi

import numpy as np
import pytest

from control.flow import AffineFlow, FlowBounds
from control.interval import Interval, Zonotope
from control.tube import (
    FeedbackSegment,
    FlightDomain,
    ModelUncertainty,
    TubeCertificationError,
    TubePropagator,
)
from models.aircraft import Aircraft
from models.geometry import RigidBodyGeometry, point_velocities, world_points

DT_S = 0.02


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
        rate_abs_m_s2=(1.0, 1.0, 1.0),
    )
    uncertainty = ModelUncertainty(
        flow=flow,
        density_kg_m3=(1.224, 1.226),
        mass_kg=(0.1429, 0.1431),
        coefficient_scale=(1.0, 1.0),
        force_error_abs_n=(1.0e-5, 1.0e-5, 1.0e-5),
        moment_error_abs_n_m=(1.0e-7, 1.0e-7, 1.0e-7),
        angular_accel_error_abs_rad_s2=(1.0e-4, 1.0e-4, 1.0e-4),
        actuator_tau_s=(0.059, 0.061),
        command_error_abs_rad=(1.0e-5, 1.0e-5, 1.0e-5),
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
        geometry=RigidBodyGeometry(),
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


def test_tube_verification_deadline() -> None:
    """Abort deterministic candidate verification at its deadline."""

    _, _, segment, initial, propagator = _setup()
    with pytest.raises(TimeoutError, match="verification deadline"):
        propagator.propagate(initial, segment, DT_S, deadline=0.0)


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
        rate_abs_m_s2=np.zeros(3),
    )
    invalid_flow = replace(
        propagator,
        uncertainty=replace(propagator.uncertainty, flow=reverse_flow),
    )
    with pytest.raises(TubeCertificationError, match="forward-flight"):
        invalid_flow.propagate(initial, segment, DT_S)


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
        assert tube.successor.contains(states[-1])
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
        assert tube.successor.contains(states[-1])
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
