from __future__ import annotations

import numpy as np

from moewe.sim.state import FlightState
from moewe.tasks import FixedInitialState, Scenario, UniformInitialStateSampler


def _nominal_state() -> FlightState:
    return FlightState(
        position_w_m=np.array([0.0, 0.0, 1.0]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([6.0, 0.0, 0.1]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def test_fixed_initial_state_ignores_seed() -> None:
    sampler = FixedInitialState(_nominal_state())

    np.testing.assert_allclose(sampler.sample(seed=1).as_vector(), sampler.sample(seed=2).as_vector())


def test_uniform_initial_state_sampler_is_seeded_and_bounded() -> None:
    half_width = np.zeros(15)
    half_width[0:3] = [0.1, 0.2, 0.3]
    sampler = UniformInitialStateSampler(_nominal_state(), half_width=half_width)

    a = sampler.sample(seed=8)
    b = sampler.sample(seed=8)
    delta = a.as_vector() - _nominal_state().as_vector()

    np.testing.assert_allclose(a.as_vector(), b.as_vector())
    assert np.all(np.abs(delta) <= half_width + 1e-12)


def test_scenario_returns_seeded_initial_state() -> None:
    half_width = np.zeros(15)
    half_width[0] = 0.5
    scenario = Scenario(
        name="seeded_gate",
        initial_state_sampler=UniformInitialStateSampler(_nominal_state(), half_width=half_width),
        seed=42,
    )

    np.testing.assert_allclose(
        scenario.initial_state().as_vector(),
        scenario.initial_state().as_vector(),
    )
