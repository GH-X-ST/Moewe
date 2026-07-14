"""Tests for nominal aircraft linearization and normalized transfer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from control.flow import FlowBounds
from control.predictor import (
    _AircraftModel,
    GeneratedAircraft,
    LQR_CONTROL_WEIGHT,
    LQR_STATE_WEIGHT,
    _generate_aircraft,
)
from control.uncertainty import Bounds
from models.aircraft import Aircraft, default_aircraft_config
from models.geometry import RigidBodyGeometry


def _geometry() -> RigidBodyGeometry:
    body = np.array(
        [
            (0.11, 0.38, 0.0),
            (0.11, -0.38, 0.0),
            (-0.39, 0.18, 0.01),
            (-0.39, -0.18, 0.01),
        ]
    )
    return RigidBodyGeometry(body, body[2:], body[2:])


def _bounds(aircraft: Aircraft) -> Bounds:
    zero3 = np.zeros(3)
    tau = np.asarray(aircraft.config.actuator_tau_s)
    return Bounds(
        flow=FlowBounds(zero3, zero3, np.zeros((3, 3)), np.zeros((3, 3)), zero3),
        density_kg_m3=(1.225, 1.225),
        aerodynamic_scale=(1.0, 1.0),
        force_residual_abs_n=zero3,
        moment_residual_abs_n_m=zero3,
        mass_kg=(aircraft.mass_kg, aircraft.mass_kg),
        cg_residual_abs_m=zero3,
        inertia_residual_abs_kg_m2=np.zeros((3, 3)),
        actuator_tau_lower_s=tau,
        actuator_tau_upper_s=tau,
        command_error_abs_rad=np.full(3, 1.0e-6),
        state_estimation_abs=np.full(15, 1.0e-6),
        command_delay_s=(0.0, 0.073),
        nonlinear_remainder_abs=np.full(15, 1.0e-7),
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


def _anchor(speed_m_s: float) -> np.ndarray:
    state = np.zeros(15)
    state[6] = speed_m_s
    return state


def test_nominal_model_cannot_claim_oracle_verification() -> None:
    aircraft = Aircraft()
    generated = _generate_aircraft(
        aircraft,
        _geometry(),
        _bounds(aircraft),
        _anchor(6.0),
        np.zeros(3),
    )
    assert generated.cells
    assert isinstance(generated, _AircraftModel)
    assert not isinstance(generated, GeneratedAircraft)


def test_lqr_failure_leaves_no_accepted_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aircraft = Aircraft()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise ValueError("no stabilizing gain")

    monkeypatch.setattr("control.predictor._gain_cell", fail)
    generated = _generate_aircraft(
        aircraft,
        _geometry(),
        _bounds(aircraft),
        _anchor(6.0),
        np.zeros(3),
    )
    assert not generated.cells


def test_different_aircraft_relinearizes_with_fixed_lqr_weights() -> None:
    """Linearize a different physical model with the same normalized weights."""

    state_weight = LQR_STATE_WEIGHT.copy()
    control_weight = LQR_CONTROL_WEIGHT.copy()
    first_aircraft = Aircraft()
    first = _generate_aircraft(
        first_aircraft,
        _geometry(),
        _bounds(first_aircraft),
        _anchor(6.0),
        np.zeros(3),
    )

    base = default_aircraft_config()
    surfaces = tuple(
        replace(surface, span_m=1.12 * surface.span_m, chord_m=0.93 * surface.chord_m)
        for surface in base.surfaces
    )
    config = replace(
        base,
        mass_kg=0.205,
        inertia_b_kg_m2=1.75 * np.asarray(base.inertia_b_kg_m2),
        s_ref_m2=1.12 * 0.93 * base.s_ref_m2,
        b_ref_m=1.12 * base.b_ref_m,
        c_ref_m=0.93 * base.c_ref_m,
        drag_area_fuse_m2=1.6 * base.drag_area_fuse_m2,
        surfaces=surfaces,
        actuator_tau_s=(0.045, 0.072, 0.055),
    )
    second_aircraft = Aircraft(config)
    second = _generate_aircraft(
        second_aircraft,
        _geometry(),
        _bounds(second_aircraft),
        _anchor(5.2),
        np.zeros(3),
    )

    assert not isinstance(first, GeneratedAircraft)
    assert not isinstance(second, GeneratedAircraft)
    assert np.array_equal(LQR_STATE_WEIGHT, state_weight)
    assert np.array_equal(LQR_CONTROL_WEIGHT, control_weight)
    assert not np.allclose(
        first.cells[0].state_matrix,
        second.cells[0].state_matrix,
    )
    assert not np.allclose(first.cells[0].gain, second.cells[0].gain)


def test_normalization_contains_asymmetric_physical_limits() -> None:
    aircraft = Aircraft()
    anchor = _anchor(6.0)
    generated = _generate_aircraft(
        aircraft,
        _geometry(),
        _bounds(aircraft),
        anchor,
        np.zeros(3),
    )
    lower = anchor[12:15] - generated.state_scale[12:15]
    upper = anchor[12:15] + generated.state_scale[12:15]
    assert np.all(lower <= aircraft.control_lower_rad)
    assert np.all(upper >= aircraft.control_upper_rad)
    assert not isinstance(generated, GeneratedAircraft)
