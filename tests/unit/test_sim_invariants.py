from __future__ import annotations

import numpy as np

from moewe.sim.actuator import ActuatorConfig, ActuatorModel
from moewe.sim.aero_strip import section_coefficients
from moewe.sim.frames import body_to_world, gravity_body, rotation_body_to_world
from moewe.sim.glider_model import nominal_glider
from moewe.sim.integrator import IntegratorConfig, simulate_fixed_step
from moewe.sim.state import FlightState
from moewe.sim.updraft import AnnularUpdraft, FanUpdraft
from moewe.tasks import LAUNCH_GATE_NOMINAL_POSITION_W_M, SINGLE_FAN_CENTER_XY_M


def _state(velocity_b_m_s: np.ndarray) -> FlightState:
    return FlightState(
        position_w_m=np.array(LAUNCH_GATE_NOMINAL_POSITION_W_M),
        euler_rad=np.zeros(3),
        velocity_b_m_s=velocity_b_m_s,
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def test_frame_axis_and_euler_sign_conventions() -> None:
    small = 1e-3

    level = rotation_body_to_world(0.0, 0.0, 0.0)
    np.testing.assert_allclose(level @ [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(level @ [0.0, 0.0, 1.0], [0.0, 0.0, -1.0])

    assert (rotation_body_to_world(small, 0.0, 0.0) @ [0.0, 1.0, 0.0])[2] < 0.0
    assert (rotation_body_to_world(0.0, small, 0.0) @ [1.0, 0.0, 0.0])[2] > 0.0
    assert (rotation_body_to_world(0.0, 0.0, small) @ [1.0, 0.0, 0.0])[1] > 0.0


def test_gravity_is_down_in_world_z_for_free_body() -> None:
    gravity_w = body_to_world(gravity_body(np.zeros(3)), np.zeros(3))

    assert gravity_w[2] < 0.0
    np.testing.assert_allclose(gravity_w[:2], [0.0, 0.0])


def test_canonical_state_vector_order_is_stable() -> None:
    state = FlightState(
        position_w_m=np.array([1.0, 2.0, 3.0]),
        euler_rad=np.array([4.0, 5.0, 6.0]),
        velocity_b_m_s=np.array([7.0, 8.0, 9.0]),
        rates_b_rad_s=np.array([10.0, 11.0, 12.0]),
        surfaces_rad=np.array([13.0, 14.0, 15.0]),
    )

    np.testing.assert_allclose(state.as_vector(), np.arange(1.0, 16.0))


def test_aero_drag_opposes_relative_airflow() -> None:
    model = nominal_glider(enable_corrections=False)
    result = model.evaluate_aero(_state(np.array([8.0, 0.0, 0.0])))

    assert float(np.dot(result.force_b_n, result.air_velocity_cg_b_m_s)) < 0.0


def test_stall_blend_is_finite_across_angle_sweep() -> None:
    alpha = np.linspace(-1.2, 1.2, 101)
    cl, cd = section_coefficients(
        alpha_rad=alpha,
        local_deflection_rad=np.zeros_like(alpha),
        aspect_ratio=np.full_like(alpha, 6.0),
        cd0=np.full_like(alpha, 0.02),
        alpha0_rad=np.zeros_like(alpha),
        efficiency=np.full_like(alpha, 0.8),
        flap_scale=np.zeros_like(alpha),
    )

    assert np.isfinite(cl).all()
    assert np.isfinite(cd).all()
    assert np.all(cd > 0.0)


def test_disabling_empirical_correction_preserves_nominal_strip_loads() -> None:
    state = _state(np.array([8.0, 1.0, 0.2]))
    enabled = nominal_glider(enable_corrections=True).evaluate_aero(state)
    disabled = nominal_glider(enable_corrections=False).evaluate_aero(state)

    np.testing.assert_allclose(enabled.strip_forces.force_b_n, disabled.strip_forces.force_b_n)
    np.testing.assert_allclose(enabled.strip_forces.moment_b_n_m, disabled.strip_forces.moment_b_n_m)
    assert np.linalg.norm(enabled.correction_coefficients) > 0.0
    np.testing.assert_allclose(disabled.correction_coefficients, np.zeros(3))


def test_annular_updraft_is_upward_in_world_z() -> None:
    updraft = AnnularUpdraft.from_fans([FanUpdraft((0.0, 0.0), 2.0, 0.0, 0.4)])
    velocity = updraft.velocity_at(np.array([[0.0, 0.0, 0.0]]))

    assert velocity[0, 2] > 0.0


def test_short_updraft_rollout_changes_specific_energy() -> None:
    model = nominal_glider()
    initial = _state(np.array([6.0, 0.0, 0.0]))
    updraft = AnnularUpdraft.from_fans([FanUpdraft(SINGLE_FAN_CENTER_XY_M, 3.0, 0.35, 0.5)])

    still = simulate_fixed_step(initial, np.zeros(3), 50, model=model, config=IntegratorConfig(dt_s=0.01))
    rising = simulate_fixed_step(
        initial,
        np.zeros(3),
        50,
        model=model,
        wind_model=updraft,
        config=IntegratorConfig(dt_s=0.01),
    )

    assert rising[-1].mechanical_energy_j(model.mass_kg) > still[-1].mechanical_energy_j(model.mass_kg)


def test_actuator_limits_and_lattice_remain_deterministic() -> None:
    config = ActuatorConfig(
        lower_limits_rad=(-0.2, -0.2, -0.2),
        upper_limits_rad=(0.2, 0.2, 0.2),
        time_constant_s=(0.0, 0.0, 0.0),
        lattice_step_rad=0.05,
    )
    model_a = ActuatorModel(config)
    model_b = ActuatorModel(config)

    a = model_a.step(np.zeros(3), np.array([0.24, -0.11, 0.02]), 0.01)
    b = model_b.step(np.zeros(3), np.array([0.24, -0.11, 0.02]), 0.01)

    np.testing.assert_allclose(a, b)
    assert np.all(a <= 0.2)
    assert np.all(a >= -0.2)


def test_fixed_step_integration_short_rollout_is_finite() -> None:
    states = simulate_fixed_step(
        _state(np.array([8.0, 0.0, 0.2])),
        np.zeros(3),
        10,
        model=nominal_glider(),
        config=IntegratorConfig(dt_s=0.01),
    )

    assert all(state.finite() for state in states)
