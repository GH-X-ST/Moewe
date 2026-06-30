from __future__ import annotations

import numpy as np

from moewe.sim.state import FlightState, STATE_SIZE


def test_state_vector_round_trip() -> None:
    vector = np.linspace(-1.0, 1.0, STATE_SIZE)
    state = FlightState.from_vector(vector)

    np.testing.assert_allclose(state.as_vector(), vector)


def test_mechanical_energy_uses_world_z_up_height() -> None:
    state = FlightState(
        position_w_m=np.array([0.0, 0.0, 2.0]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([3.0, 0.0, 4.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )

    assert state.mechanical_energy_j(2.0) > 2.0 * 9.0 * 2.0
    assert state.finite()
