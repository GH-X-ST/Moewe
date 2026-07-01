from __future__ import annotations

import numpy as np

from moewe.sim.glider_model import (
    EmpiricalCorrectionConfig,
    NAUSICAA_FUSELAGE_DRAG_AREA_M2,
    NAUSICAA_HTAIL_CHORD_M,
    NAUSICAA_HTAIL_EFFICIENCY,
    NAUSICAA_HTAIL_ROOT_LE_B_M,
    NAUSICAA_HTAIL_SPAN_M,
    NAUSICAA_INERTIA_B_KG_M2,
    NAUSICAA_INERTIA_DIAGONAL_KG_M2,
    NAUSICAA_MASS_KG,
    NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD,
    NAUSICAA_TAIL_CD0,
    NAUSICAA_TOTAL_WING_DIHEDRAL_RAD,
    NAUSICAA_VTAIL_CHORD_M,
    NAUSICAA_VTAIL_EFFICIENCY,
    NAUSICAA_VTAIL_ROOT_LE_B_M,
    NAUSICAA_VTAIL_HEIGHT_M,
    NAUSICAA_WING_ALPHA0_RAD,
    NAUSICAA_WING_AREA_M2,
    NAUSICAA_WING_CHORD_M,
    NAUSICAA_WING_CD0,
    NAUSICAA_WING_DIHEDRAL_RAD,
    NAUSICAA_WING_EFFICIENCY,
    NAUSICAA_WING_ROOT_LE_B_M,
    NAUSICAA_WING_SPAN_M,
    nominal_glider,
)
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


def test_nominal_glider_uses_nausicaa_design_constants() -> None:
    model = nominal_glider(enable_corrections=False)
    surface_names = np.asarray(model.strips.surface_name)

    assert model.mass_kg == NAUSICAA_MASS_KG
    np.testing.assert_allclose(model.inertia_b_kg_m2, NAUSICAA_INERTIA_B_KG_M2)
    np.testing.assert_allclose(np.diag(model.inertia_b_kg_m2), NAUSICAA_INERTIA_DIAGONAL_KG_M2)
    assert model.s_ref_m2 == NAUSICAA_WING_AREA_M2
    assert model.b_ref_m == NAUSICAA_WING_SPAN_M
    assert model.c_ref_m == NAUSICAA_WING_CHORD_M
    assert model.drag_area_m2 == NAUSICAA_FUSELAGE_DRAG_AREA_M2

    wing = surface_names == "wing"
    horizontal_tail = surface_names == "horizontal_tail"
    vertical_tail = surface_names == "vertical_tail"
    np.testing.assert_allclose(
        model.strips.r_strip_b_m[wing, 0],
        NAUSICAA_WING_ROOT_LE_B_M[0] - 0.25 * NAUSICAA_WING_CHORD_M,
    )
    np.testing.assert_allclose(
        model.strips.r_strip_b_m[horizontal_tail, 0],
        NAUSICAA_HTAIL_ROOT_LE_B_M[0] - 0.25 * NAUSICAA_HTAIL_CHORD_M,
    )
    np.testing.assert_allclose(
        model.strips.r_strip_b_m[vertical_tail, 0],
        NAUSICAA_VTAIL_ROOT_LE_B_M[0] - 0.25 * NAUSICAA_VTAIL_CHORD_M,
    )
    np.testing.assert_allclose(model.strips.area_m2[wing].sum(), NAUSICAA_WING_AREA_M2)
    np.testing.assert_allclose(model.strips.chord_m[wing], NAUSICAA_WING_CHORD_M)
    np.testing.assert_allclose(model.strips.cd0[wing], NAUSICAA_WING_CD0)
    np.testing.assert_allclose(model.strips.alpha0_rad[wing], NAUSICAA_WING_ALPHA0_RAD)
    np.testing.assert_allclose(model.strips.efficiency[wing], NAUSICAA_WING_EFFICIENCY)
    np.testing.assert_allclose(model.strips.chord_m[horizontal_tail], NAUSICAA_HTAIL_CHORD_M)
    np.testing.assert_allclose(model.strips.cd0[horizontal_tail], NAUSICAA_TAIL_CD0)
    np.testing.assert_allclose(model.strips.efficiency[horizontal_tail], NAUSICAA_HTAIL_EFFICIENCY)
    np.testing.assert_allclose(
        model.strips.area_m2[horizontal_tail].sum(),
        NAUSICAA_HTAIL_SPAN_M * NAUSICAA_HTAIL_CHORD_M,
    )
    np.testing.assert_allclose(model.strips.chord_m[vertical_tail], NAUSICAA_VTAIL_CHORD_M)
    np.testing.assert_allclose(model.strips.cd0[vertical_tail], NAUSICAA_TAIL_CD0)
    np.testing.assert_allclose(model.strips.efficiency[vertical_tail], NAUSICAA_VTAIL_EFFICIENCY)
    np.testing.assert_allclose(
        model.strips.area_m2[vertical_tail].sum(),
        NAUSICAA_VTAIL_HEIGHT_M * NAUSICAA_VTAIL_CHORD_M,
    )
    np.testing.assert_allclose(2.0 * NAUSICAA_WING_DIHEDRAL_RAD, NAUSICAA_TOTAL_WING_DIHEDRAL_RAD)
    np.testing.assert_allclose(
        np.abs(model.strips.normal_b[wing, 1]),
        np.sin(NAUSICAA_WING_DIHEDRAL_RAD),
    )
    np.testing.assert_allclose(
        model.strips.normal_b[wing, 2],
        -np.cos(NAUSICAA_WING_DIHEDRAL_RAD),
    )
    assert NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD == np.deg2rad(18.0)


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
