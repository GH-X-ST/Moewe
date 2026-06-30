from __future__ import annotations

import numpy as np

from moewe.sim.aero_strip import evaluate_strip_forces, section_coefficients
from moewe.sim.glider_model import nominal_glider


def test_section_coefficients_are_finite_near_zero_angle() -> None:
    cl, cd = section_coefficients(
        alpha_rad=np.array([0.0]),
        local_deflection_rad=np.array([0.0]),
        aspect_ratio=np.array([6.0]),
        cd0=np.array([0.02]),
        alpha0_rad=np.array([0.0]),
        efficiency=np.array([0.8]),
        flap_scale=np.array([0.0]),
    )

    assert np.isfinite(cl).all()
    assert np.isfinite(cd).all()
    assert cd[0] > 0.0


def test_strip_forces_have_no_nans_at_zero_airspeed() -> None:
    model = nominal_glider(enable_corrections=False)

    forces = evaluate_strip_forces(
        geometry=model.strips,
        velocity_b_m_s=np.zeros(3),
        rates_b_rad_s=np.zeros(3),
        wind_strip_b_m_s=np.zeros((model.strips.strip_count, 3)),
        surfaces_rad=np.zeros(3),
        rho_kg_m3=1.225,
    )

    assert np.isfinite(forces.force_b_n).all()
    assert np.isfinite(forces.moment_b_n_m).all()


def test_symmetric_still_air_strip_case_has_near_zero_lateral_loads() -> None:
    model = nominal_glider(enable_corrections=False)

    forces = evaluate_strip_forces(
        geometry=model.strips,
        velocity_b_m_s=np.array([8.0, 0.0, 0.4]),
        rates_b_rad_s=np.zeros(3),
        wind_strip_b_m_s=np.zeros((model.strips.strip_count, 3)),
        surfaces_rad=np.zeros(3),
        rho_kg_m3=1.225,
    )

    assert abs(forces.force_b_n[1]) < 1e-10
    assert abs(forces.moment_b_n_m[0]) < 1e-10
    assert abs(forces.moment_b_n_m[2]) < 1e-10
