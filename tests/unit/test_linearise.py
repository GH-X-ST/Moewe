from __future__ import annotations

import numpy as np

from moewe.control import (
    TrimSpec,
    finite_difference_linearisation,
    linearisation_step_check,
    pseudo_trim,
    to_euler_discrete,
)
from moewe.sim.actuator import ActuatorConfig
from moewe.sim.glider_model import nominal_glider


def _reference() -> tuple:
    trim = pseudo_trim(TrimSpec(airspeed_m_s=7.0, flight_path_angle_rad=0.0))
    return trim.state, trim.command_rad, nominal_glider()


def test_finite_difference_linearisation_dimensions_and_finiteness() -> None:
    state, command, model = _reference()

    lin = finite_difference_linearisation(state, command, model=model)

    assert lin.a.shape == (15, 15)
    assert lin.b.shape == (15, 3)
    assert lin.f_ref.shape == (15,)
    assert np.isfinite(lin.a).all()
    assert np.isfinite(lin.b).all()


def test_euler_discrete_linearisation_dimensions() -> None:
    state, command, model = _reference()
    lin = finite_difference_linearisation(state, command, model=model)

    discrete = to_euler_discrete(lin, dt_s=0.02)

    assert discrete.mode == "discrete_euler"
    assert discrete.dt_s == 0.02
    assert discrete.a.shape == (15, 15)
    assert discrete.b.shape == (15, 3)


def test_linearisation_step_check_is_small_for_small_perturbation() -> None:
    state, command, model = _reference()
    lin = finite_difference_linearisation(
        state,
        command,
        model=model,
        actuator_config=ActuatorConfig(time_constant_s=(0.06, 0.06, 0.06)),
    )
    dx = np.zeros(15)
    dx[3] = 1e-4
    dx[6] = 1e-3
    du = np.array([1e-4, -1e-4, 5e-5])

    check = linearisation_step_check(
        lin,
        dx,
        du,
        dt_s=0.01,
        model=model,
        actuator_config=ActuatorConfig(time_constant_s=(0.06, 0.06, 0.06)),
    )

    assert check.error_norm < 1e-5
    assert np.isfinite(check.nonlinear_next).all()
    assert np.isfinite(check.linear_next).all()
