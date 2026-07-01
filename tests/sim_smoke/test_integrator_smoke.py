from __future__ import annotations

import numpy as np

from moewe.sim.actuator import ActuatorConfig, ActuatorModel
from moewe.sim.glider_model import nominal_glider
from moewe.sim.integrator import IntegratorConfig, simulate_fixed_step
from moewe.sim.state import FlightState
from moewe.sim.updraft import AnnularUpdraft, FanUpdraft
from moewe.tasks import LAUNCH_GATE_NOMINAL_POSITION_W_M, SINGLE_FAN_CENTER_XY_M


def _initial_state() -> FlightState:
    return FlightState(
        position_w_m=np.array(LAUNCH_GATE_NOMINAL_POSITION_W_M),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([6.0, 0.0, 0.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def test_still_air_glide_smoke_runs_without_nans() -> None:
    model = nominal_glider()
    states = simulate_fixed_step(
        initial_state=_initial_state(),
        command=np.zeros(3),
        steps=80,
        model=model,
        config=IntegratorConfig(dt_s=0.01),
    )

    assert all(state.finite() for state in states)


def test_updraft_crossing_changes_energy_in_expected_direction() -> None:
    model = nominal_glider()
    config = IntegratorConfig(dt_s=0.01)
    updraft = AnnularUpdraft.from_fans(
        [FanUpdraft(SINGLE_FAN_CENTER_XY_M, strength_m_s=3.0, ring_radius_m=0.35, ring_thickness_m=0.5)]
    )

    still = simulate_fixed_step(_initial_state(), np.zeros(3), 50, model=model, config=config)
    rising = simulate_fixed_step(
        _initial_state(),
        np.zeros(3),
        50,
        model=model,
        wind_model=updraft,
        config=config,
    )

    assert rising[-1].mechanical_energy_j(model.mass_kg) > still[-1].mechanical_energy_j(model.mass_kg)


def test_actuator_states_remain_inside_limits() -> None:
    actuator = ActuatorModel(
        ActuatorConfig(
            lower_limits_rad=(-0.2, -0.2, -0.2),
            upper_limits_rad=(0.2, 0.2, 0.2),
            time_constant_s=(0.0, 0.0, 0.0),
        )
    )
    states = simulate_fixed_step(
        initial_state=_initial_state(),
        command=np.array([10.0, -10.0, 10.0]),
        steps=5,
        model=nominal_glider(),
        actuator=actuator,
        config=IntegratorConfig(dt_s=0.01),
    )

    assert np.all(np.abs(states[-1].surfaces_rad) <= 0.2)
