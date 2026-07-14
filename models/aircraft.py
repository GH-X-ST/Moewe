"""Configurable stripwise aircraft aerodynamic model."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np
import numpy.typing as npt

from models.state import G_M_S2, as_state

DEFAULT_RHO_KG_M3 = 1.225
EPS = 1.0e-9
SMOOTH_ABS_EPS = 1.0e-6
AILERON = 0
ELEVATOR = 1
RUDDER = 2

_WORLD_AXIS_SIGNS = np.array([1.0, -1.0, -1.0])
_FLAT_PLATE_AR = np.array(
    [0.167, 0.333, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 6.0],
    dtype=float,
)
_FLAT_PLATE_A_LE = np.array(
    [
        3.0,
        3.64,
        4.48,
        7.18,
        10.2,
        13.38,
        14.84,
        14.49,
        9.95,
        12.93,
        15.0,
        15.0,
    ],
    dtype=float,
)
_FLAT_PLATE_A_TE = np.array(
    [
        5.9,
        15.51,
        32.57,
        39.44,
        48.22,
        59.29,
        21.55,
        7.74,
        7.05,
        5.26,
        6.5,
        6.5,
    ],
    dtype=float,
)
_FLAT_PLATE_ALPHA_LE_DEG = np.array(
    [
        59.0,
        58.6,
        58.2,
        50.0,
        41.53,
        26.7,
        23.44,
        21.0,
        18.63,
        14.28,
        11.6,
        10.0,
    ],
    dtype=float,
)
_FLAT_PLATE_ALPHA_TE_DEG = np.array(
    [
        59.0,
        58.6,
        58.2,
        51.85,
        41.46,
        28.09,
        39.4,
        35.86,
        26.76,
        19.76,
        16.43,
        14.0,
    ],
    dtype=float,
)
_FLAT_PLATE_ALPHA_HIGH_DEG = np.array(
    [49.0, 54.0, 56.0, 48.0, 40.0, 29.0, 27.0, 25.0, 24.0, 22.0, 22.0, 20.0],
    dtype=float,
)


@dataclass(frozen=True)
class SurfaceLimit:
    """Asymmetric aggregate control-surface limit in radians."""

    negative_rad: float
    positive_rad: float


@dataclass(frozen=True)
class ControlSurface:
    """Control-surface geometry and aggregate-command mixing."""

    chord_fraction: float
    eta_start: float
    eta_end: float
    input_axis: int
    input_sign: float = 1.0


@dataclass(frozen=True)
class LiftingSurface:
    """Rectangular lifting-surface definition in CG-centred body axes."""

    root_le_b_m: tuple[float, float, float]
    chord_m: float
    span_m: float
    dihedral_rad: float
    strip_count: int
    symmetric: bool
    vertical: bool
    cd0: float
    alpha0_rad: float
    control_surface: ControlSurface | None = None


@dataclass(frozen=True)
class AircraftConfig:
    """Physical constants and geometry for one aircraft model."""

    mass_kg: float
    inertia_b_kg_m2: npt.ArrayLike
    s_ref_m2: float
    b_ref_m: float
    c_ref_m: float
    drag_area_fuse_m2: float
    surfaces: tuple[LiftingSurface, ...]
    control_limits_rad: tuple[SurfaceLimit, SurfaceLimit, SurfaceLimit]
    actuator_tau_s: tuple[float, float, float] = (0.06, 0.06, 0.06)


@dataclass(frozen=True)
class _StripTable:
    r_b_m: np.ndarray
    area_m2: np.ndarray
    chord_m: np.ndarray
    aspect_ratio: np.ndarray
    span_axis_b: np.ndarray
    normal_b: np.ndarray
    moment_axis_b: np.ndarray
    control_mix: np.ndarray
    flap_chord_fraction: np.ndarray
    cd0: np.ndarray
    alpha0_rad: np.ndarray


def flat_plate_coefficients(
    aspect_ratio: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flat-plate transition coefficients by aspect ratio."""

    ar = np.clip(
        np.asarray(aspect_ratio, dtype=float),
        _FLAT_PLATE_AR[0],
        _FLAT_PLATE_AR[-1],
    )
    return (
        np.interp(ar, _FLAT_PLATE_AR, _FLAT_PLATE_A_LE),
        np.interp(ar, _FLAT_PLATE_AR, _FLAT_PLATE_A_TE),
        np.deg2rad(np.interp(ar, _FLAT_PLATE_AR, _FLAT_PLATE_ALPHA_LE_DEG)),
        np.deg2rad(np.interp(ar, _FLAT_PLATE_AR, _FLAT_PLATE_ALPHA_TE_DEG)),
        np.deg2rad(np.interp(ar, _FLAT_PLATE_AR, _FLAT_PLATE_ALPHA_HIGH_DEG)),
    )


def default_aircraft_config() -> AircraftConfig:
    """Return the default aircraft geometry with a 143 g mass."""

    x_cg = 0.10550000000000002
    z_cg = -0.008037445457381024
    wing_span = 0.764
    wing_chord = 0.165
    htail_span = 0.364
    htail_chord = 0.091
    vtail_span = 0.119
    vtail_chord = 0.059

    wing_root_le = (
        x_cg,
        0.0,
        -0.0085 - 0.5 * 0.0060 - z_cg,
    )
    htail_root_le = (
        x_cg - (0.426 - 0.25 * htail_chord),
        0.0,
        0.0045 + 0.5 * 0.0030 - z_cg,
    )
    vtail_root_le = (
        x_cg - (0.433 - 0.25 * vtail_chord),
        0.0,
        -0.0050 - z_cg,
    )
    surfaces = (
        LiftingSurface(
            root_le_b_m=wing_root_le,
            chord_m=wing_chord,
            span_m=wing_span,
            dihedral_rad=np.deg2rad(0.5 * 9.28),
            strip_count=6,
            symmetric=True,
            vertical=False,
            cd0=0.018,
            alpha0_rad=0.001,
            control_surface=ControlSurface(
                chord_fraction=0.30,
                eta_start=0.30,
                eta_end=0.85,
                input_axis=AILERON,
                input_sign=-1.0,
            ),
        ),
        LiftingSurface(
            root_le_b_m=htail_root_le,
            chord_m=htail_chord,
            span_m=htail_span,
            dihedral_rad=0.0,
            strip_count=4,
            symmetric=True,
            vertical=False,
            cd0=0.020,
            alpha0_rad=np.deg2rad(-3.0),
            control_surface=ControlSurface(
                chord_fraction=0.30,
                eta_start=0.0,
                eta_end=1.0,
                input_axis=ELEVATOR,
                input_sign=-1.0,
            ),
        ),
        LiftingSurface(
            root_le_b_m=vtail_root_le,
            chord_m=vtail_chord,
            span_m=vtail_span,
            dihedral_rad=0.0,
            strip_count=4,
            symmetric=False,
            vertical=True,
            cd0=0.020,
            alpha0_rad=0.0,
            control_surface=ControlSurface(
                chord_fraction=0.35,
                eta_start=0.0,
                eta_end=1.0,
                input_axis=RUDDER,
                input_sign=1.0,
            ),
        ),
    )
    return AircraftConfig(
        mass_kg=0.143,
        inertia_b_kg_m2=np.array(
            [
                [0.0027026951327610775, 0.0, 2.3150631199284152e-05],
                [0.0, 0.0029875937118918686, 0.0],
                [2.3150631199284152e-05, 0.0, 0.005606240152469395],
            ],
            dtype=float,
        ),
        s_ref_m2=wing_span * wing_chord,
        b_ref_m=wing_span,
        c_ref_m=wing_chord,
        drag_area_fuse_m2=4.89753346075483e-05,
        surfaces=surfaces,
        control_limits_rad=(
            SurfaceLimit(np.deg2rad(-21.5), np.deg2rad(19.3)),
            SurfaceLimit(np.deg2rad(-32.0), np.deg2rad(23.7)),
            SurfaceLimit(np.deg2rad(-33.0), np.deg2rad(33.0)),
        ),
    )


@dataclass
class Aircraft:
    """Stripwise aerodynamic and rigid-body aircraft model."""

    config: AircraftConfig = field(default_factory=default_aircraft_config)

    def __post_init__(self) -> None:
        self.mass_kg = float(self.config.mass_kg)
        self.inertia_b_kg_m2 = np.asarray(
            self.config.inertia_b_kg_m2,
            dtype=float,
        )
        self.inertia_inv_b = np.linalg.inv(self.inertia_b_kg_m2)
        self.strip_table = _build_strip_table(self.config.surfaces)
        self.control_lower_rad = np.array(
            [limit.negative_rad for limit in self.config.control_limits_rad],
            dtype=float,
        )
        self.control_upper_rad = np.array(
            [limit.positive_rad for limit in self.config.control_limits_rad],
            dtype=float,
        )
        arrays = (
            self.inertia_b_kg_m2,
            self.inertia_inv_b,
            self.control_lower_rad,
            self.control_upper_rad,
            *(
                value
                for value in vars(self.strip_table).values()
                if isinstance(value, np.ndarray)
            ),
        )
        for array in arrays:
            array.flags.writeable = False

    def __call__(
        self,
        state: npt.ArrayLike,
        control: npt.ArrayLike,
        wind_model: object = None,
        rho: float = DEFAULT_RHO_KG_M3,
    ) -> np.ndarray:
        """Return the derivative of the 15-state aircraft vector."""

        state_array = as_state(state)
        center_b, strips_b = self.sample_local_flow(state_array, wind_model)
        return self.derivative_local_flow(
            state_array,
            control,
            center_b,
            strips_b,
            rho,
        )

    def derivative_local_flow(
        self,
        state: npt.ArrayLike,
        control: npt.ArrayLike,
        center_b_m_s: npt.ArrayLike,
        strip_b_m_s: npt.ArrayLike,
        rho: float = DEFAULT_RHO_KG_M3,
    ) -> np.ndarray:
        """Return state dynamics from body-relative local flow."""

        state_array = as_state(state)
        target = self.clip_control(control)
        f_aero_b, m_aero_b = self.aero_loads_local_flow(
            state_array,
            center_b_m_s,
            strip_b_m_s,
            float(rho),
        )
        phi, theta = state_array[3], state_array[4]
        c_wb = _c_wb_numpy(phi, theta, state_array[5])
        v_b = state_array[6:9]
        omega_b = state_array[9:12]
        surface = state_array[12:15]
        gravity_b = c_wb.T @ np.array([0.0, 0.0, G_M_S2], dtype=float)
        f_total_b = f_aero_b + self.mass_kg * gravity_b
        v_dot_b = f_total_b / self.mass_kg - np.cross(omega_b, v_b)
        omega_dot_b = self.inertia_inv_b @ (
            m_aero_b - np.cross(omega_b, self.inertia_b_kg_m2 @ omega_b)
        )
        euler_dot = _t_euler_numpy(phi, theta) @ omega_b
        tau = np.maximum(
            np.asarray(self.config.actuator_tau_s, dtype=float),
            EPS,
        )
        surface_dot = (target - surface) / tau
        position_dot = _WORLD_AXIS_SIGNS * (c_wb @ v_b)
        return np.concatenate(
            [position_dot, euler_dot, v_dot_b, omega_dot_b, surface_dot]
        )

    def sample_local_flow(
        self,
        state: npt.ArrayLike,
        wind_model: object = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a world flow field at the CG and aerodynamic strips."""

        x = as_state(state)
        c_wb = _c_wb_numpy(x[3], x[4], x[5])
        c_bw = c_wb.T
        r_cg_w = _WORLD_AXIS_SIGNS * x[:3]
        r_strip_w = r_cg_w + self.strip_table.r_b_m @ c_wb.T
        wind_cg_w, wind_strip_w = _sample_wind(
            wind_model,
            r_cg_w,
            r_strip_w,
        )
        return c_bw @ wind_cg_w, wind_strip_w @ c_bw.T

    def clip_control(self, control: npt.ArrayLike) -> np.ndarray:
        """Clip aggregate control targets to surface limits."""

        values = np.asarray(control, dtype=float).reshape(3)
        return np.clip(values, self.control_lower_rad, self.control_upper_rad)

    def aero_loads_local_flow(
        self,
        state: npt.ArrayLike,
        center_b_m_s: npt.ArrayLike,
        strip_b_m_s: npt.ArrayLike,
        rho: float = DEFAULT_RHO_KG_M3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return aerodynamic force and moment from local strip flow."""

        state_array = as_state(state)
        center_b = np.asarray(center_b_m_s, dtype=float).reshape(3)
        strip_b = np.asarray(strip_b_m_s, dtype=float).reshape(
            self.strip_table.r_b_m.shape
        )
        v_b = state_array[6:9]
        omega_b = state_array[9:12]
        surface = self.clip_control(state_array[12:15])
        v_air_cg_b = v_b - center_b
        v_air_strip_b = (
            v_b
            + np.cross(
                np.broadcast_to(omega_b, self.strip_table.r_b_m.shape),
                self.strip_table.r_b_m,
            )
            - strip_b
        )
        span_speed = np.sum(
            v_air_strip_b * self.strip_table.span_axis_b,
            axis=1,
        )
        v_plane_b = v_air_strip_b - self.strip_table.span_axis_b * span_speed[:, None]
        speed_plane = np.linalg.norm(v_plane_b, axis=1)
        speed_safe = np.maximum(speed_plane, EPS)
        drag_dir_b = -v_plane_b / speed_safe[:, None]
        lift_vec_b = (
            self.strip_table.normal_b
            - np.sum(
                self.strip_table.normal_b * drag_dir_b,
                axis=1,
                keepdims=True,
            )
            * drag_dir_b
        )
        lift_dir_b = _normalize_rows(lift_vec_b)
        alpha_strip = np.arctan2(
            -np.sum(v_plane_b * self.strip_table.normal_b, axis=1),
            v_plane_b[:, 0],
        )
        delta_local = self.strip_table.control_mix @ surface
        cl, cd, cm = _flat_plate_section(
            alpha_rad=alpha_strip,
            alpha_dot_rad_s=np.zeros_like(alpha_strip),
            aspect_ratio=self.strip_table.aspect_ratio,
            chord_m=self.strip_table.chord_m,
            flap_chord_fraction=self.strip_table.flap_chord_fraction,
            speed_m_s=speed_safe,
            delta_f_rad=delta_local,
            alpha0_rad=self.strip_table.alpha0_rad,
            cd0=self.strip_table.cd0,
        )
        q_bar_strip = 0.5 * rho * speed_plane**2
        force_scale = (q_bar_strip * self.strip_table.area_m2)[:, None]
        f_strip_b = force_scale * (cl[:, None] * lift_dir_b + cd[:, None] * drag_dir_b)
        m_strip_b = np.cross(self.strip_table.r_b_m, f_strip_b)
        m_strip_b += (
            q_bar_strip * self.strip_table.area_m2 * self.strip_table.chord_m * cm
        )[:, None] * self.strip_table.moment_axis_b
        f_aero_b = np.sum(f_strip_b, axis=0)
        m_aero_b = np.sum(m_strip_b, axis=0)
        speed_cg = float(np.linalg.norm(v_air_cg_b))
        if speed_cg > EPS:
            f_aero_b += (
                -0.5 * rho * self.config.drag_area_fuse_m2 * speed_cg * v_air_cg_b
            )
        return f_aero_b, m_aero_b


def _build_strip_table(surfaces: tuple[LiftingSurface, ...]) -> _StripTable:
    rows = []
    for surface in surfaces:
        rows.extend(
            _vertical_rows(surface) if surface.vertical else _horizontal_rows(surface)
        )
    return _StripTable(
        r_b_m=np.asarray([row["r_b_m"] for row in rows], dtype=float),
        area_m2=np.asarray([row["area_m2"] for row in rows], dtype=float),
        chord_m=np.asarray([row["chord_m"] for row in rows], dtype=float),
        aspect_ratio=np.asarray(
            [row["aspect_ratio"] for row in rows],
            dtype=float,
        ),
        span_axis_b=np.asarray(
            [row["span_axis_b"] for row in rows],
            dtype=float,
        ),
        normal_b=np.asarray([row["normal_b"] for row in rows], dtype=float),
        moment_axis_b=np.asarray(
            [row["moment_axis_b"] for row in rows],
            dtype=float,
        ),
        control_mix=np.asarray(
            [row["control_mix"] for row in rows],
            dtype=float,
        ),
        flap_chord_fraction=np.asarray(
            [row["flap_chord_fraction"] for row in rows],
            dtype=float,
        ),
        cd0=np.asarray([row["cd0"] for row in rows], dtype=float),
        alpha0_rad=np.asarray(
            [row["alpha0_rad"] for row in rows],
            dtype=float,
        ),
    )


def _horizontal_rows(surface: LiftingSurface) -> list[dict[str, object]]:
    root = np.asarray(surface.root_le_b_m, dtype=float)
    half_span = 0.5 * surface.span_m
    projected_semispan = half_span * np.cos(surface.dihedral_rad)
    strip_span = half_span / surface.strip_count
    centres = (np.arange(surface.strip_count) + 0.5) * strip_span
    rows = []
    side_signs = (1.0, -1.0) if surface.symmetric else (1.0,)
    for side_sign in side_signs:
        span_axis = _unit(
            np.array(
                [
                    0.0,
                    side_sign * np.cos(surface.dihedral_rad),
                    -np.sin(surface.dihedral_rad),
                ],
                dtype=float,
            )
        )
        normal = _unit(
            np.array(
                [
                    0.0,
                    -side_sign * np.sin(surface.dihedral_rad),
                    -np.cos(surface.dihedral_rad),
                ],
                dtype=float,
            )
        )
        for centre in centres:
            y_b = side_sign * centre * np.cos(surface.dihedral_rad)
            z_b = root[2] - centre * np.sin(surface.dihedral_rad)
            eta = abs(y_b) / max(projected_semispan, EPS)
            rows.append(
                _strip_row(
                    surface,
                    strip_span,
                    np.array([root[0] - 0.25 * surface.chord_m, y_b, z_b]),
                    span_axis,
                    normal,
                    np.array([0.0, 1.0, 0.0]),
                    eta,
                    side_sign,
                    surface.span_m / surface.chord_m,
                )
            )
    return rows


def _vertical_rows(surface: LiftingSurface) -> list[dict[str, object]]:
    root = np.asarray(surface.root_le_b_m, dtype=float)
    strip_span = surface.span_m / surface.strip_count
    centres = (np.arange(surface.strip_count) + 0.5) * strip_span
    rows = []
    for centre in centres:
        rows.append(
            _strip_row(
                surface,
                strip_span,
                np.array(
                    [
                        root[0] - 0.25 * surface.chord_m,
                        0.0,
                        root[2] - centre,
                    ],
                    dtype=float,
                ),
                np.array([0.0, 0.0, -1.0]),
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                0.0,
                1.0,
                2.0 * surface.span_m / surface.chord_m,
            )
        )
    return rows


def _strip_row(
    surface: LiftingSurface,
    strip_span: float,
    r_b_m: np.ndarray,
    span_axis_b: np.ndarray,
    normal_b: np.ndarray,
    moment_axis_b: np.ndarray,
    eta: float,
    side_sign: float,
    aspect_ratio: float,
) -> dict[str, object]:
    control_mix = np.zeros(3)
    flap_chord_fraction = 0.0
    control = surface.control_surface
    if control is not None and control.eta_start <= eta <= control.eta_end:
        control_sign = control.input_sign
        if control.input_axis == AILERON:
            control_sign *= -1.0 if side_sign > 0.0 else 1.0
        control_mix[control.input_axis] = control_sign
        flap_chord_fraction = control.chord_fraction
    return {
        "r_b_m": r_b_m,
        "area_m2": surface.chord_m * strip_span,
        "chord_m": surface.chord_m,
        "aspect_ratio": aspect_ratio,
        "span_axis_b": span_axis_b,
        "normal_b": normal_b,
        "moment_axis_b": moment_axis_b,
        "control_mix": control_mix,
        "flap_chord_fraction": flap_chord_fraction,
        "cd0": surface.cd0,
        "alpha0_rad": surface.alpha0_rad,
    }


def _flat_plate_section(
    alpha_rad: np.ndarray,
    alpha_dot_rad_s: np.ndarray,
    aspect_ratio: np.ndarray,
    chord_m: np.ndarray,
    flap_chord_fraction: np.ndarray,
    speed_m_s: np.ndarray,
    delta_f_rad: np.ndarray,
    alpha0_rad: np.ndarray,
    cd0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_le, a_te, alpha_le, alpha_te, alpha_high = flat_plate_coefficients(aspect_ratio)
    speed = np.maximum(speed_m_s, EPS)
    cf_c = np.clip(flap_chord_fraction, 0.0, 1.0)
    cl_alpha = (
        2.0
        * pi
        * (
            aspect_ratio
            / (aspect_ratio + 2.0 * (aspect_ratio + 4.0) / (aspect_ratio + 2.0))
        )
    )
    theta_f = np.arccos(2.0 * cf_c - 1.0)
    tau_f = 1.0 - (theta_f - np.sin(theta_f)) / pi
    delta_cl = cl_alpha * tau_f * delta_f_rad
    tau_te = 4.5 * chord_m / speed
    tau_le = 0.5 * chord_m / speed
    alpha_lift = alpha_rad - alpha0_rad
    abs_alpha_lift = _smooth_abs(alpha_lift)
    f_te = 0.5 * (
        1.0 - np.tanh(a_te * (abs_alpha_lift - tau_te * alpha_dot_rad_s - alpha_te))
    )
    f_le = 0.5 * (
        1.0 - np.tanh(a_le * (abs_alpha_lift - tau_le * alpha_dot_rad_s - alpha_le))
    )
    sqrt_f_te = np.sqrt(np.maximum(f_te, 0.0))
    sin_alpha = np.sin(alpha_lift)
    cos_alpha = np.cos(alpha_lift)
    cl_attached = (
        0.25
        * (1.0 + sqrt_f_te) ** 2
        * (
            cl_alpha * sin_alpha * cos_alpha**2
            + f_le**2 * pi * _smooth_abs(sin_alpha) * sin_alpha * cos_alpha
        )
        + delta_cl
    )
    cd_attached = cd0 + cl_attached * _safe_tan(alpha_lift)
    cm_attached = (
        -0.25
        * (1.0 + sqrt_f_te) ** 2
        * (
            0.0625
            * (-1.0 + 6.0 * sqrt_f_te - 5.0 * f_te)
            * cl_alpha
            * sin_alpha
            * cos_alpha
            + 0.17 * f_le**2 * pi * _smooth_abs(sin_alpha) * sin_alpha
        )
    )
    cf = cf_c * chord_m
    c_prime = np.sqrt(
        (chord_m - cf) ** 2 + cf**2 + 2.0 * cf * (chord_m - cf) * np.cos(delta_f_rad)
    )
    alpha_f = np.arcsin(
        np.clip(cf * np.sin(delta_f_rad) / np.maximum(c_prime, EPS), -1.0, 1.0)
    )
    alpha_ps = alpha_rad - alpha0_rad + alpha_f
    cd_90 = 1.98 - 4.26e-2 * delta_f_rad**2 + 2.1e-1 * delta_f_rad
    sin_ps = np.sin(alpha_ps)
    cos_ps = np.cos(alpha_ps)
    normal_gain = 1.0 / (0.56 + 0.44 * _smooth_abs(sin_ps)) - 0.41 * (
        1.0 - np.exp(-17.0 / aspect_ratio)
    )
    cn = cd_90 * sin_ps * normal_gain
    ca = 0.5 * 0.03 * cos_ps
    cl_post = cn * cos_ps - ca * sin_ps
    cd_post = cn * sin_ps + ca * cos_ps
    cm_post = -cn * (0.25 - 0.175 * (1.0 - 2.0 * alpha_ps / pi))
    sigma = 0.5 * (1.0 + np.tanh(20.0 * (alpha_high - _smooth_abs(alpha_rad))))
    cl = sigma * cl_attached + (1.0 - sigma) * cl_post
    cd = sigma * cd_attached + (1.0 - sigma) * cd_post
    cm = sigma * cm_attached + (1.0 - sigma) * cm_post
    return cl, cd, cm


def _sample_wind(
    wind_model: object,
    r_cg_w: np.ndarray,
    r_strip_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if wind_model is None:
        return np.zeros(3), np.zeros_like(r_strip_w)
    if callable(wind_model):
        points_cg = r_cg_w.reshape(1, 3) * _WORLD_AXIS_SIGNS
        wind_cg = np.asarray(
            wind_model(points_cg),
            dtype=float,
        ).reshape(-1, 3)[0]
        wind_strip = np.asarray(
            wind_model(r_strip_w * _WORLD_AXIS_SIGNS),
            dtype=float,
        )
    else:
        wind_cg = np.asarray(wind_model, dtype=float).reshape(3)
        wind_strip = np.broadcast_to(wind_cg, r_strip_w.shape)
    return wind_cg * _WORLD_AXIS_SIGNS, wind_strip * _WORLD_AXIS_SIGNS


def _c_wb_numpy(phi: float, theta: float, psi: float) -> np.ndarray:
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    return np.array(
        [
            [
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ],
            [
                c_theta * s_psi,
                s_phi * s_theta * s_psi + c_phi * c_psi,
                c_phi * s_theta * s_psi - s_phi * c_psi,
            ],
            [-s_theta, s_phi * c_theta, c_phi * c_theta],
        ],
        dtype=float,
    )


def _t_euler_numpy(phi: float, theta: float) -> np.ndarray:
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    t_theta = np.tan(theta)
    return np.array(
        [
            [1.0, s_phi * t_theta, c_phi * t_theta],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta],
        ],
        dtype=float,
    )


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, EPS)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _smooth_abs(value: np.ndarray) -> np.ndarray:
    return np.sqrt(value * value + SMOOTH_ABS_EPS**2)


def _safe_tan(angle_rad: np.ndarray) -> np.ndarray:
    return np.tan(np.clip(angle_rad, -0.5 * pi + 1.0e-6, 0.5 * pi - 1.0e-6))
