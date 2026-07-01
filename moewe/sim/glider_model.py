"""Transparent nominal glider model with optional bounded corrections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aero_strip import (
    ControlSurfaceMapping,
    LiftingSurface,
    StallModel,
    StripForces,
    StripGeometry,
    build_strip_geometry,
    evaluate_strip_forces,
)
from .frames import body_to_world_rows, gravity_body, world_to_body, world_to_body_rows
from .state import FlightState

EPS = 1e-9

NAUSICAA_MASS_KG = 0.14856
NAUSICAA_INERTIA_B_KG_M2 = np.array(
    [
        [0.0027026951327610775, 0.0, 2.3150631199284152e-05],
        [0.0, 0.0029875937118918686, 0.0],
        [2.3150631199284152e-05, 0.0, 0.005606240152469395],
    ],
    dtype=float,
)
NAUSICAA_INERTIA_DIAGONAL_KG_M2 = tuple(float(value) for value in np.diag(NAUSICAA_INERTIA_B_KG_M2))
NAUSICAA_WING_ROOT_LE_B_M = (0.10550000000000002, 0.0, -0.003462554542618976)
NAUSICAA_HTAIL_ROOT_LE_B_M = (-0.29775, 0.0, 0.014037445457381024)
NAUSICAA_VTAIL_ROOT_LE_B_M = (-0.31275, 0.0, 0.0030374454573810234)
NAUSICAA_WING_SPAN_M = 0.764
NAUSICAA_WING_CHORD_M = 0.165
NAUSICAA_WING_AREA_M2 = NAUSICAA_WING_SPAN_M * NAUSICAA_WING_CHORD_M
NAUSICAA_TOTAL_WING_DIHEDRAL_RAD = float(np.deg2rad(9.28))
NAUSICAA_WING_DIHEDRAL_RAD = 0.5 * NAUSICAA_TOTAL_WING_DIHEDRAL_RAD
NAUSICAA_AILERON_ETA_START = 0.30
NAUSICAA_AILERON_ETA_END = 0.85
NAUSICAA_HTAIL_SPAN_M = 0.364
NAUSICAA_HTAIL_CHORD_M = 0.091
NAUSICAA_VTAIL_HEIGHT_M = 0.119
NAUSICAA_VTAIL_CHORD_M = 0.059
NAUSICAA_WING_CD0 = 0.018 * 3.0
NAUSICAA_TAIL_CD0 = 0.020 * 3.0
NAUSICAA_WING_ALPHA0_RAD = 0.001
NAUSICAA_HTAIL_ALPHA0_RAD = float(np.deg2rad(-3.0))
NAUSICAA_WING_EFFICIENCY = 0.82 * 0.31
NAUSICAA_HTAIL_EFFICIENCY = 0.78 * 0.31
NAUSICAA_VTAIL_EFFICIENCY = 0.75 * 0.31
NAUSICAA_FUSELAGE_DRAG_AREA_M2 = 4.89753346075483e-05 * 5.0
NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD = float(np.deg2rad(18.0))


@dataclass(frozen=True)
class LateralCoefficients:
    c_y0: float = 0.0
    c_y_beta: float = 0.0
    c_y_p: float = 0.0
    c_y_r: float = 0.0
    c_l0: float = 0.0
    c_l_beta: float = 0.0
    c_l_p: float = 0.0
    c_l_r: float = 0.0
    c_n0: float = 0.0
    c_n_beta: float = 0.0
    c_n_p: float = 0.0
    c_n_r: float = 0.0

    def evaluate(self, beta_rad: float, p_hat: float, r_hat: float) -> np.ndarray:
        features = np.array([1.0, beta_rad, p_hat, r_hat], dtype=float)
        rows = np.array(
            [
                [self.c_y0, self.c_y_beta, self.c_y_p, self.c_y_r],
                [self.c_l0, self.c_l_beta, self.c_l_p, self.c_l_r],
                [self.c_n0, self.c_n_beta, self.c_n_p, self.c_n_r],
            ],
            dtype=float,
        )
        return rows @ features


@dataclass(frozen=True)
class EmpiricalCorrectionConfig:
    """Optional low-order bounded correction for known lateral and stall gaps."""

    enabled: bool = False
    attached: LateralCoefficients = LateralCoefficients()
    transition: LateralCoefficients = LateralCoefficients()
    coefficient_limit: float = 0.5
    activation_start_rad: float = np.deg2rad(12.0)
    activation_width_rad: float = np.deg2rad(4.0)

    @classmethod
    def conservative_lateral_stability(cls) -> "EmpiricalCorrectionConfig":
        return cls(
            enabled=True,
            attached=LateralCoefficients(
                c_y_beta=-0.35,
                c_l_beta=-0.04,
                c_l_p=-0.12,
                c_n_beta=-0.20,
                c_n_r=-0.05,
            ),
            transition=LateralCoefficients(c_y_r=-0.20, c_l_r=-0.06, c_n_p=-0.04),
            coefficient_limit=0.35,
        )

    def activation(self, alpha_rad: float) -> float:
        return float(
            0.5
            * (
                1.0
                + np.tanh(
                    (abs(float(alpha_rad)) - float(self.activation_start_rad))
                    / max(float(self.activation_width_rad), EPS)
                )
            )
        )

    def evaluate(self, beta_rad: float, p_hat: float, r_hat: float, alpha_rad: float) -> np.ndarray:
        if not self.enabled:
            return np.zeros(3)
        coeffs = self.attached.evaluate(beta_rad, p_hat, r_hat)
        coeffs += self.activation(alpha_rad) * self.transition.evaluate(beta_rad, p_hat, r_hat)
        limit = abs(float(self.coefficient_limit))
        return np.clip(coeffs, -limit, limit)


@dataclass(frozen=True)
class AeroResult:
    force_b_n: np.ndarray
    moment_b_n_m: np.ndarray
    wind_cg_w_m_s: np.ndarray
    wind_cg_b_m_s: np.ndarray
    air_velocity_cg_b_m_s: np.ndarray
    speed_m_s: float
    alpha_rad: float
    beta_rad: float
    correction_coefficients: np.ndarray
    strip_forces: StripForces


@dataclass(frozen=True)
class LoadResult:
    force_b_n: np.ndarray
    moment_b_n_m: np.ndarray
    gravity_b_m_s2: np.ndarray
    aero: AeroResult


@dataclass(frozen=True)
class GliderModel:
    mass_kg: float
    inertia_b_kg_m2: np.ndarray
    s_ref_m2: float
    b_ref_m: float
    c_ref_m: float
    drag_area_m2: float
    strips: StripGeometry
    stall: StallModel = StallModel()
    correction: EmpiricalCorrectionConfig = EmpiricalCorrectionConfig()
    rho_kg_m3: float = 1.225

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia_b_kg_m2, dtype=float).reshape(3, 3)
        if self.mass_kg <= 0.0:
            raise ValueError("Glider mass must be positive.")
        object.__setattr__(self, "inertia_b_kg_m2", inertia)

    @property
    def inertia_inv_b_kg_m2(self) -> np.ndarray:
        return np.linalg.inv(self.inertia_b_kg_m2)

    def strip_positions_world(self, state: FlightState) -> np.ndarray:
        return state.position_w_m + body_to_world_rows(self.strips.r_strip_b_m, state.euler_rad)

    def _sample_wind(
        self,
        state: FlightState,
        wind_model: object | None,
        wind_mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        strip_points_w = self.strip_positions_world(state)
        cg_point = state.position_w_m.reshape(1, 3)
        if wind_model is None:
            wind_cg_w = np.zeros(3)
            wind_strip_w = np.zeros_like(strip_points_w)
        elif hasattr(wind_model, "velocity_at"):
            wind_cg_w = np.asarray(wind_model.velocity_at(cg_point), dtype=float).reshape(-1, 3)[0]
            wind_strip_w = (
                np.broadcast_to(wind_cg_w, strip_points_w.shape)
                if wind_mode == "cg"
                else np.asarray(wind_model.velocity_at(strip_points_w), dtype=float).reshape(-1, 3)
            )
        elif callable(wind_model):
            wind_cg_w = np.asarray(wind_model(cg_point), dtype=float).reshape(-1, 3)[0]
            wind_strip_w = (
                np.broadcast_to(wind_cg_w, strip_points_w.shape)
                if wind_mode == "cg"
                else np.asarray(wind_model(strip_points_w), dtype=float).reshape(-1, 3)
            )
        else:
            wind_cg_w = np.asarray(wind_model, dtype=float).reshape(3)
            wind_strip_w = np.broadcast_to(wind_cg_w, strip_points_w.shape)
        return wind_cg_w, wind_strip_w

    def evaluate_aero(
        self,
        state: FlightState,
        wind_model: object | None = None,
        wind_mode: str = "panel",
        rho_kg_m3: float | None = None,
    ) -> AeroResult:
        if wind_mode not in {"cg", "panel"}:
            raise ValueError("wind_mode must be 'cg' or 'panel'.")
        rho = self.rho_kg_m3 if rho_kg_m3 is None else float(rho_kg_m3)
        wind_cg_w, wind_strip_w = self._sample_wind(state, wind_model, wind_mode)
        wind_cg_b = world_to_body(wind_cg_w, state.euler_rad)
        wind_strip_b = world_to_body_rows(wind_strip_w, state.euler_rad)
        strip_forces = evaluate_strip_forces(
            geometry=self.strips,
            velocity_b_m_s=state.velocity_b_m_s,
            rates_b_rad_s=state.rates_b_rad_s,
            wind_strip_b_m_s=wind_strip_b,
            surfaces_rad=state.surfaces_rad,
            rho_kg_m3=rho,
            stall=self.stall,
        )
        v_air_b = state.velocity_b_m_s - wind_cg_b
        speed = float(np.linalg.norm(v_air_b))
        q_bar = 0.5 * rho * speed**2
        alpha = float(np.arctan2(v_air_b[2], v_air_b[0]))
        beta = float(np.arcsin(np.clip(v_air_b[1] / max(speed, EPS), -1.0, 1.0)))
        p_hat = 0.0 if speed <= EPS else float(state.rates_b_rad_s[0] * self.b_ref_m / (2.0 * speed))
        r_hat = 0.0 if speed <= EPS else float(state.rates_b_rad_s[2] * self.b_ref_m / (2.0 * speed))
        cy, cl, cn = self.correction.evaluate(beta, p_hat, r_hat, alpha)
        correction_force = q_bar * self.s_ref_m2 * np.array([0.0, cy, 0.0], dtype=float)
        correction_moment = q_bar * self.s_ref_m2 * np.array(
            [self.b_ref_m * cl, 0.0, self.b_ref_m * cn],
            dtype=float,
        )
        fuselage_drag = (
            -0.5 * rho * self.drag_area_m2 * speed * v_air_b
            if speed > EPS
            else np.zeros(3)
        )
        force = strip_forces.force_b_n + correction_force + fuselage_drag
        moment = strip_forces.moment_b_n_m + correction_moment
        return AeroResult(
            force_b_n=force,
            moment_b_n_m=moment,
            wind_cg_w_m_s=wind_cg_w,
            wind_cg_b_m_s=wind_cg_b,
            air_velocity_cg_b_m_s=v_air_b,
            speed_m_s=speed,
            alpha_rad=alpha,
            beta_rad=beta,
            correction_coefficients=np.array([cy, cl, cn], dtype=float),
            strip_forces=strip_forces,
        )

    def evaluate_loads(
        self,
        state: FlightState,
        wind_model: object | None = None,
        wind_mode: str = "panel",
        rho_kg_m3: float | None = None,
    ) -> LoadResult:
        aero = self.evaluate_aero(state, wind_model=wind_model, wind_mode=wind_mode, rho_kg_m3=rho_kg_m3)
        gravity_b = gravity_body(state.euler_rad)
        total_force = aero.force_b_n + self.mass_kg * gravity_b
        return LoadResult(
            force_b_n=total_force,
            moment_b_n_m=aero.moment_b_n_m,
            gravity_b_m_s2=gravity_b,
            aero=aero,
        )


def nominal_glider(enable_corrections: bool = True) -> GliderModel:
    """Return the Nausicaa nominal glider plant for simulation development.

    Mass, inertia, root locations, and primary dimensions trace to the active
    Nausicaa as-built runtime model with the measured nose-ballast case. The
    compact strip aerodynamics remain a reduced model boundary.
    """

    wing = LiftingSurface(
        name="wing",
        root_le_b_m=np.array(NAUSICAA_WING_ROOT_LE_B_M, dtype=float),
        chord_m=NAUSICAA_WING_CHORD_M,
        span_m=NAUSICAA_WING_SPAN_M,
        strip_count=6,
        symmetric=True,
        vertical=False,
        cd0=NAUSICAA_WING_CD0,
        alpha0_rad=NAUSICAA_WING_ALPHA0_RAD,
        efficiency=NAUSICAA_WING_EFFICIENCY,
        dihedral_rad=NAUSICAA_WING_DIHEDRAL_RAD,
        control=ControlSurfaceMapping(
            axis=0,
            sign=-1.0,
            eta_start=NAUSICAA_AILERON_ETA_START,
            eta_end=NAUSICAA_AILERON_ETA_END,
            chord_fraction=0.30,
        ),
    )
    horizontal_tail = LiftingSurface(
        name="horizontal_tail",
        root_le_b_m=np.array(NAUSICAA_HTAIL_ROOT_LE_B_M, dtype=float),
        chord_m=NAUSICAA_HTAIL_CHORD_M,
        span_m=NAUSICAA_HTAIL_SPAN_M,
        strip_count=4,
        symmetric=True,
        vertical=False,
        cd0=NAUSICAA_TAIL_CD0,
        alpha0_rad=NAUSICAA_HTAIL_ALPHA0_RAD,
        efficiency=NAUSICAA_HTAIL_EFFICIENCY,
        control=ControlSurfaceMapping(axis=1, sign=-1.0, eta_start=0.0, eta_end=1.0, chord_fraction=0.30),
    )
    vertical_tail = LiftingSurface(
        name="vertical_tail",
        root_le_b_m=np.array(NAUSICAA_VTAIL_ROOT_LE_B_M, dtype=float),
        chord_m=NAUSICAA_VTAIL_CHORD_M,
        span_m=NAUSICAA_VTAIL_HEIGHT_M,
        strip_count=4,
        symmetric=False,
        vertical=True,
        cd0=NAUSICAA_TAIL_CD0,
        alpha0_rad=0.0,
        efficiency=NAUSICAA_VTAIL_EFFICIENCY,
        control=ControlSurfaceMapping(axis=2, sign=1.0, eta_start=0.0, eta_end=1.0, chord_fraction=0.35),
    )
    strips = build_strip_geometry((wing, horizontal_tail, vertical_tail))
    correction = (
        EmpiricalCorrectionConfig.conservative_lateral_stability()
        if enable_corrections
        else EmpiricalCorrectionConfig(enabled=False)
    )
    return GliderModel(
        mass_kg=NAUSICAA_MASS_KG,
        inertia_b_kg_m2=NAUSICAA_INERTIA_B_KG_M2.copy(),
        s_ref_m2=NAUSICAA_WING_AREA_M2,
        b_ref_m=NAUSICAA_WING_SPAN_M,
        c_ref_m=NAUSICAA_WING_CHORD_M,
        drag_area_m2=NAUSICAA_FUSELAGE_DRAG_AREA_M2,
        strips=strips,
        correction=correction,
    )
