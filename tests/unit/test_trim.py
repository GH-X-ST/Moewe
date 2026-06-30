from __future__ import annotations

import numpy as np

from moewe.control import TrimSpec, pseudo_trim
from moewe.sim.frames import body_to_world
from moewe.sim.glider_model import nominal_glider


def test_pseudo_trim_returns_state_command_and_residual_report() -> None:
    result = pseudo_trim(TrimSpec(airspeed_m_s=7.0, flight_path_angle_rad=0.05), model=nominal_glider())

    assert result.method == "deterministic_pseudo_trim"
    assert not result.converged
    assert result.state.finite()
    assert result.command_rad.shape == (3,)
    assert result.residual.force_b_n.shape == (3,)
    assert result.residual.moment_b_n_m.shape == (3,)
    assert np.isfinite(result.residual.residual_norm)


def test_pseudo_trim_vertical_speed_target_is_constructed_exactly() -> None:
    result = pseudo_trim(TrimSpec(airspeed_m_s=8.0, vertical_speed_m_s=0.4))
    world_velocity = body_to_world(result.state.velocity_b_m_s, result.state.euler_rad)

    np.testing.assert_allclose(world_velocity[2], 0.4, atol=1e-12)
    assert abs(result.residual.vertical_velocity_error_m_s) < 1e-12


def test_pseudo_trim_is_deterministic() -> None:
    spec = TrimSpec(airspeed_m_s=8.0, flight_path_angle_rad=-0.03, command_rad=(0.01, -0.02, 0.0))

    a = pseudo_trim(spec)
    b = pseudo_trim(spec)

    np.testing.assert_allclose(a.state.as_vector(), b.state.as_vector())
    np.testing.assert_allclose(a.command_rad, b.command_rad)
    np.testing.assert_allclose(a.residual.force_b_n, b.residual.force_b_n)
