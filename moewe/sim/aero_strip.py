"""Strip-based aerodynamic loading for small fixed-wing gliders."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

EPS = 1e-9


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise ValueError("Cannot normalise a zero vector.")
    return np.asarray(vector, dtype=float) / norm


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, EPS)


def flap_lift_scale(chord_fraction: float) -> float:
    """Return a thin-airfoil flap effectiveness proxy for a chord fraction."""

    chord_fraction = float(np.clip(chord_fraction, EPS, 1.0 - EPS))
    theta_f = np.arccos(2.0 * chord_fraction - 1.0)
    return float(1.0 - (theta_f - np.sin(theta_f)) / pi)


@dataclass(frozen=True)
class ControlSurfaceMapping:
    axis: int
    sign: float
    eta_start: float
    eta_end: float
    chord_fraction: float


@dataclass(frozen=True)
class LiftingSurface:
    name: str
    root_le_b_m: np.ndarray
    chord_m: float
    span_m: float
    strip_count: int
    symmetric: bool
    vertical: bool
    cd0: float
    alpha0_rad: float
    efficiency: float
    dihedral_rad: float = 0.0
    control: ControlSurfaceMapping | None = None


@dataclass(frozen=True)
class StallModel:
    attached_stall_rad: float = np.deg2rad(12.0)
    blend_width_rad: float = np.deg2rad(3.0)
    post_stall_drag_gain: float = 1.8


@dataclass(frozen=True)
class StripGeometry:
    r_strip_b_m: np.ndarray
    area_m2: np.ndarray
    chord_m: np.ndarray
    aspect_ratio: np.ndarray
    span_axis_b: np.ndarray
    normal_b: np.ndarray
    control_mix: np.ndarray
    cd0: np.ndarray
    alpha0_rad: np.ndarray
    efficiency: np.ndarray
    flap_scale: np.ndarray
    surface_name: tuple[str, ...]

    @property
    def strip_count(self) -> int:
        return int(self.area_m2.size)


@dataclass(frozen=True)
class StripForces:
    force_b_n: np.ndarray
    moment_b_n_m: np.ndarray
    alpha_rad: np.ndarray
    beta_rad: np.ndarray
    dynamic_pressure_pa: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    local_deflection_rad: np.ndarray
    strip_force_b_n: np.ndarray


def smooth_attached_weight(alpha_rad: np.ndarray, stall: StallModel) -> np.ndarray:
    alpha = np.asarray(alpha_rad, dtype=float)
    return 0.5 * (
        1.0
        + np.tanh(
            (float(stall.attached_stall_rad) - np.abs(alpha))
            / max(float(stall.blend_width_rad), EPS)
        )
    )


def section_coefficients(
    alpha_rad: np.ndarray,
    local_deflection_rad: np.ndarray,
    aspect_ratio: np.ndarray,
    cd0: np.ndarray,
    alpha0_rad: np.ndarray,
    efficiency: np.ndarray,
    flap_scale: np.ndarray,
    stall: StallModel | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strip lift and drag coefficients with smooth post-stall blending."""

    stall_model = StallModel() if stall is None else stall
    alpha_eff = (
        np.asarray(alpha_rad, dtype=float)
        - np.asarray(alpha0_rad, dtype=float)
        + np.asarray(flap_scale, dtype=float) * np.asarray(local_deflection_rad, dtype=float)
    )
    ar = np.maximum(np.asarray(aspect_ratio, dtype=float), EPS)
    eff = np.maximum(np.asarray(efficiency, dtype=float), EPS)
    lift_slope = 2.0 * pi * ar / (ar + 2.0)
    cl_attached = lift_slope * alpha_eff
    cd_attached = np.asarray(cd0, dtype=float) + cl_attached**2 / (pi * eff * ar)
    cl_post = 2.0 * np.sin(alpha_eff) * np.cos(alpha_eff)
    cd_post = np.asarray(cd0, dtype=float) + float(stall_model.post_stall_drag_gain) * np.sin(alpha_eff) ** 2
    attached = smooth_attached_weight(alpha_eff, stall_model)
    return attached * cl_attached + (1.0 - attached) * cl_post, attached * cd_attached + (1.0 - attached) * cd_post


def _horizontal_rows(surface: LiftingSurface) -> list[dict[str, object]]:
    root = np.asarray(surface.root_le_b_m, dtype=float).reshape(3)
    half_span = 0.5 * float(surface.span_m)
    strip_span = half_span / int(surface.strip_count)
    centres = (np.arange(surface.strip_count) + 0.5) * strip_span
    projected_semispan = max(half_span * np.cos(surface.dihedral_rad), EPS)
    rows: list[dict[str, object]] = []
    sides = (1.0, -1.0) if surface.symmetric else (1.0,)
    for side in sides:
        span_axis_b = _unit(np.array([0.0, side * np.cos(surface.dihedral_rad), -np.sin(surface.dihedral_rad)]))
        normal_b = _unit(np.array([0.0, -side * np.sin(surface.dihedral_rad), -np.cos(surface.dihedral_rad)]))
        for distance in centres:
            y_b = side * distance * np.cos(surface.dihedral_rad)
            z_b = root[2] - distance * np.sin(surface.dihedral_rad)
            eta = abs(y_b) / projected_semispan
            control_mix = np.zeros(3)
            flap_scale = 0.0
            if surface.control is not None and surface.control.eta_start <= eta <= surface.control.eta_end:
                control_sign = float(surface.control.sign)
                if surface.control.axis == 0:
                    control_sign *= -1.0 if side > 0.0 else 1.0
                control_mix[int(surface.control.axis)] = control_sign
                flap_scale = flap_lift_scale(surface.control.chord_fraction)
            rows.append(
                {
                    "r": np.array([root[0] - 0.25 * surface.chord_m, y_b, z_b]),
                    "area": float(surface.chord_m) * strip_span,
                    "chord": float(surface.chord_m),
                    "aspect_ratio": float(surface.span_m) / float(surface.chord_m),
                    "span_axis": span_axis_b,
                    "normal": normal_b,
                    "control_mix": control_mix,
                    "cd0": float(surface.cd0),
                    "alpha0": float(surface.alpha0_rad),
                    "efficiency": float(surface.efficiency),
                    "flap_scale": flap_scale,
                    "name": surface.name,
                }
            )
    return rows


def _vertical_rows(surface: LiftingSurface) -> list[dict[str, object]]:
    root = np.asarray(surface.root_le_b_m, dtype=float).reshape(3)
    strip_span = float(surface.span_m) / int(surface.strip_count)
    centres = (np.arange(surface.strip_count) + 0.5) * strip_span
    span_axis_b = np.array([0.0, 0.0, -1.0])
    normal_b = np.array([0.0, -1.0, 0.0])
    rows: list[dict[str, object]] = []
    for distance in centres:
        control_mix = np.zeros(3)
        flap_scale = 0.0
        if surface.control is not None:
            control_mix[int(surface.control.axis)] = float(surface.control.sign)
            flap_scale = flap_lift_scale(surface.control.chord_fraction)
        rows.append(
            {
                "r": np.array([root[0] - 0.25 * surface.chord_m, 0.0, root[2] - distance]),
                "area": float(surface.chord_m) * strip_span,
                "chord": float(surface.chord_m),
                "aspect_ratio": 2.0 * float(surface.span_m) / float(surface.chord_m),
                "span_axis": span_axis_b,
                "normal": normal_b,
                "control_mix": control_mix,
                "cd0": float(surface.cd0),
                "alpha0": float(surface.alpha0_rad),
                "efficiency": float(surface.efficiency),
                "flap_scale": flap_scale,
                "name": surface.name,
            }
        )
    return rows


def build_strip_geometry(surfaces: tuple[LiftingSurface, ...]) -> StripGeometry:
    rows: list[dict[str, object]] = []
    for surface in surfaces:
        rows.extend(_vertical_rows(surface) if surface.vertical else _horizontal_rows(surface))
    if not rows:
        raise ValueError("At least one lifting-surface strip is required.")
    return StripGeometry(
        r_strip_b_m=np.asarray([row["r"] for row in rows], dtype=float),
        area_m2=np.asarray([row["area"] for row in rows], dtype=float),
        chord_m=np.asarray([row["chord"] for row in rows], dtype=float),
        aspect_ratio=np.asarray([row["aspect_ratio"] for row in rows], dtype=float),
        span_axis_b=np.asarray([row["span_axis"] for row in rows], dtype=float),
        normal_b=np.asarray([row["normal"] for row in rows], dtype=float),
        control_mix=np.asarray([row["control_mix"] for row in rows], dtype=float),
        cd0=np.asarray([row["cd0"] for row in rows], dtype=float),
        alpha0_rad=np.asarray([row["alpha0"] for row in rows], dtype=float),
        efficiency=np.asarray([row["efficiency"] for row in rows], dtype=float),
        flap_scale=np.asarray([row["flap_scale"] for row in rows], dtype=float),
        surface_name=tuple(str(row["name"]) for row in rows),
    )


def evaluate_strip_forces(
    geometry: StripGeometry,
    velocity_b_m_s: np.ndarray,
    rates_b_rad_s: np.ndarray,
    wind_strip_b_m_s: np.ndarray,
    surfaces_rad: np.ndarray,
    rho_kg_m3: float,
    stall: StallModel | None = None,
    control_effectiveness: np.ndarray | None = None,
) -> StripForces:
    """Evaluate panel loads from local relative airflow at each strip."""

    velocity_b = np.asarray(velocity_b_m_s, dtype=float).reshape(3)
    rates_b = np.asarray(rates_b_rad_s, dtype=float).reshape(3)
    wind_strip_b = np.asarray(wind_strip_b_m_s, dtype=float).reshape(geometry.strip_count, 3)
    surfaces = np.asarray(surfaces_rad, dtype=float).reshape(3)
    effectiveness = np.ones(3) if control_effectiveness is None else np.asarray(control_effectiveness, dtype=float).reshape(3)

    v_rot_b = np.cross(np.broadcast_to(rates_b, geometry.r_strip_b_m.shape), geometry.r_strip_b_m)
    v_air_strip_b = velocity_b + v_rot_b - wind_strip_b
    span_speed = np.sum(v_air_strip_b * geometry.span_axis_b, axis=1)
    v_plane_b = v_air_strip_b - geometry.span_axis_b * span_speed[:, None]
    speed_plane = np.linalg.norm(v_plane_b, axis=1)
    speed_total = np.linalg.norm(v_air_strip_b, axis=1)
    drag_dir_b = -v_plane_b / np.maximum(speed_plane, EPS)[:, None]
    lift_seed = geometry.normal_b - np.sum(geometry.normal_b * drag_dir_b, axis=1, keepdims=True) * drag_dir_b
    lift_dir_b = _normalise_rows(lift_seed)
    alpha = np.arctan2(-np.sum(v_plane_b * geometry.normal_b, axis=1), v_plane_b[:, 0])
    beta = np.arcsin(np.clip(span_speed / np.maximum(speed_total, EPS), -1.0, 1.0))
    q_bar = 0.5 * float(rho_kg_m3) * speed_plane**2
    local_deflection = (geometry.control_mix * effectiveness.reshape(1, 3)) @ surfaces
    cl, cd = section_coefficients(
        alpha_rad=alpha,
        local_deflection_rad=local_deflection,
        aspect_ratio=geometry.aspect_ratio,
        cd0=geometry.cd0,
        alpha0_rad=geometry.alpha0_rad,
        efficiency=geometry.efficiency,
        flap_scale=geometry.flap_scale,
        stall=stall,
    )
    force_scale = (q_bar * geometry.area_m2)[:, None]
    strip_force = force_scale * (cl[:, None] * lift_dir_b + cd[:, None] * drag_dir_b)
    strip_moment = np.cross(geometry.r_strip_b_m, strip_force)
    return StripForces(
        force_b_n=np.sum(strip_force, axis=0),
        moment_b_n_m=np.sum(strip_moment, axis=0),
        alpha_rad=alpha,
        beta_rad=beta,
        dynamic_pressure_pa=q_bar,
        cl=cl,
        cd=cd,
        local_deflection_rad=local_deflection,
        strip_force_b_n=strip_force,
    )
