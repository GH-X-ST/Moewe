from __future__ import annotations

import numpy as np

from moewe.sim.glider_model import EmpiricalCorrectionConfig, nominal_glider
from moewe.sim.state import FlightState
from moewe.sim.updraft import AnnularUpdraft, FanUpdraft


def _state(velocity_b: np.ndarray) -> FlightState:
    return FlightState(
        position_w_m=np.array([0.0, 0.0, 1.0]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=velocity_b,
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def test_symmetric_still_air_model_has_near_zero_lateral_loads() -> None:
    model = nominal_glider(enable_corrections=True)

    result = model.evaluate_aero(_state(np.array([8.0, 0.0, 0.4])))

    assert abs(result.force_b_n[1]) < 1e-10
    assert abs(result.moment_b_n_m[0]) < 1e-10
    assert abs(result.moment_b_n_m[2]) < 1e-10


def test_positive_sideslip_gets_weathercock_correction_sign() -> None:
    model = nominal_glider(enable_corrections=True)

    result = model.evaluate_aero(_state(np.array([8.0, 1.0, 0.2])))

    assert result.beta_rad > 0.0
    assert result.correction_coefficients[2] < 0.0
    assert result.moment_b_n_m[2] < 0.0


def test_corrections_zero_when_disabled_and_bounded_when_enabled() -> None:
    disabled = EmpiricalCorrectionConfig(enabled=False)
    enabled = EmpiricalCorrectionConfig.conservative_lateral_stability()

    np.testing.assert_allclose(disabled.evaluate(1.0, 1.0, -1.0, 1.0), np.zeros(3))
    coeffs = enabled.evaluate(10.0, 10.0, -10.0, 1.0)

    assert np.all(np.abs(coeffs) <= enabled.coefficient_limit)


def test_correction_activation_is_continuous_near_transition() -> None:
    correction = EmpiricalCorrectionConfig.conservative_lateral_stability()

    before = correction.activation(correction.activation_start_rad - 1e-6)
    after = correction.activation(correction.activation_start_rad + 1e-6)

    assert abs(after - before) < 1e-4


def test_panel_and_cg_wind_modes_differ_in_spatially_varying_updraft() -> None:
    model = nominal_glider(enable_corrections=False)
    updraft = AnnularUpdraft.from_fans(
        [
            FanUpdraft(
                centre_xy_m=(0.35, 0.25),
                strength_m_s=3.0,
                ring_radius_m=0.25,
                ring_thickness_m=0.10,
            )
        ]
    )
    state = _state(np.array([8.0, 0.0, 0.4]))

    panel = model.evaluate_aero(state, wind_model=updraft, wind_mode="panel")
    cg = model.evaluate_aero(state, wind_model=updraft, wind_mode="cg")

    assert not np.allclose(panel.force_b_n, cg.force_b_n)
