from __future__ import annotations

import numpy as np

from moewe.sim.glider_model import nominal_glider
from moewe.sim.state import FlightState
from moewe.tasks import (
    FOUR_FAN_CENTERS_XY_M,
    FRONT_EXIT_GATE_CENTRE_W_M,
    FRONT_EXIT_GATE_HEIGHT_M,
    FRONT_EXIT_GATE_NORMAL_W,
    FRONT_EXIT_GATE_WIDTH_M,
    LAUNCH_GATE_NOMINAL_POSITION_W_M,
    SINGLE_FAN_CENTER_XY_M,
    TRUE_SAFE_FLIGHT_VOLUME,
    TRUE_SAFE_X_W_M,
    TRUE_SAFE_Y_W_M,
    TRUE_SAFE_Z_W_M,
    front_exit_gate,
    state_is_launch_gate_compliant,
)


def _state(
    *,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (6.0, 0.0, 0.0),
) -> FlightState:
    return FlightState(
        position_w_m=np.array(position, dtype=float),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array(velocity, dtype=float),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def test_true_safe_volume_matches_measured_arena_contract() -> None:
    assert TRUE_SAFE_X_W_M == (1.2, 6.6)
    assert TRUE_SAFE_Y_W_M == (0.0, 4.4)
    assert TRUE_SAFE_Z_W_M == (0.4, 3.5)
    assert TRUE_SAFE_FLIGHT_VOLUME.x_min_m == TRUE_SAFE_X_W_M[0]
    assert TRUE_SAFE_FLIGHT_VOLUME.x_max_m == TRUE_SAFE_X_W_M[1]
    assert TRUE_SAFE_FLIGHT_VOLUME.y_min_m == TRUE_SAFE_Y_W_M[0]
    assert TRUE_SAFE_FLIGHT_VOLUME.y_max_m == TRUE_SAFE_Y_W_M[1]
    assert TRUE_SAFE_FLIGHT_VOLUME.z_min_m == TRUE_SAFE_Z_W_M[0]
    assert TRUE_SAFE_FLIGHT_VOLUME.z_max_m == TRUE_SAFE_Z_W_M[1]


def test_front_exit_gate_is_small_rectangle_on_true_safe_front_plane() -> None:
    gate = front_exit_gate()

    np.testing.assert_allclose(gate.centre_w_m, FRONT_EXIT_GATE_CENTRE_W_M)
    np.testing.assert_allclose(gate.normal_w, FRONT_EXIT_GATE_NORMAL_W)
    assert gate.width_m == FRONT_EXIT_GATE_WIDTH_M
    assert gate.height_m == FRONT_EXIT_GATE_HEIGHT_M
    assert gate.centre_w_m[0] == TRUE_SAFE_X_W_M[1]
    assert gate.centre_w_m[0] > LAUNCH_GATE_NOMINAL_POSITION_W_M[0] + 5.0
    assert gate.centre_w_m[1] == LAUNCH_GATE_NOMINAL_POSITION_W_M[1]
    assert gate.centre_w_m[2] == LAUNCH_GATE_NOMINAL_POSITION_W_M[2]
    assert gate.width_m < TRUE_SAFE_Y_W_M[1] - TRUE_SAFE_Y_W_M[0]
    assert gate.height_m < TRUE_SAFE_Z_W_M[1] - TRUE_SAFE_Z_W_M[0]
    assert gate.width_m > nominal_glider().b_ref_m
    assert np.isclose(gate.width_m, 1.0)
    assert np.isclose(gate.height_m, 0.8)


def test_launch_gate_compliance_rejects_old_origin_initial_state() -> None:
    assert state_is_launch_gate_compliant(_state(position=LAUNCH_GATE_NOMINAL_POSITION_W_M))
    assert not state_is_launch_gate_compliant(_state(position=(0.0, 0.0, 1.0)))
    assert not state_is_launch_gate_compliant(
        _state(position=LAUNCH_GATE_NOMINAL_POSITION_W_M, velocity=(3.0, 0.0, 0.0))
    )


def test_updraft_centres_are_in_measured_world_coordinates() -> None:
    assert SINGLE_FAN_CENTER_XY_M == (4.2, 2.4)
    assert FOUR_FAN_CENTERS_XY_M == ((3.0, 3.6), (5.4, 3.6), (3.0, 1.2), (5.4, 1.2))
