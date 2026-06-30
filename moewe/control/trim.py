"""Pseudo-trim utilities for local-control development.

The first Moewe trim implementation is deliberately named pseudo-trim. It
constructs a deterministic operating point and reports force, moment, speed,
vertical-velocity, and angular-rate residuals without claiming a solved
nonlinear aerodynamic trim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.frames import body_to_world
from moewe.sim.glider_model import GliderModel, nominal_glider
from moewe.sim.rigid_body import rigid_body_derivative
from moewe.sim.state import FlightState


@dataclass(frozen=True)
class TrimSpec:
    """Requested local operating point in SI units and radians."""

    airspeed_m_s: float
    flight_path_angle_rad: float = 0.0
    vertical_speed_m_s: float | None = None
    bank_angle_rad: float = 0.0
    heading_rad: float = 0.0
    turn_rate_rad_s: float = 0.0
    altitude_m: float = 1.0
    command_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wind_mode: str = "panel"

    def resolved_flight_path_angle_rad(self) -> float:
        if self.airspeed_m_s <= 0.0:
            raise ValueError("Trim airspeed_m_s must be positive.")
        if self.vertical_speed_m_s is None:
            return float(self.flight_path_angle_rad)
        ratio = np.clip(float(self.vertical_speed_m_s) / float(self.airspeed_m_s), -1.0, 1.0)
        return float(np.arcsin(ratio))


@dataclass(frozen=True)
class TrimResidual:
    """Residual report for pseudo-trim construction."""

    force_b_n: np.ndarray
    moment_b_n_m: np.ndarray
    acceleration_b_m_s2: np.ndarray
    speed_derivative_m_s2: float
    vertical_velocity_error_m_s: float
    angular_rate_error_rad_s: np.ndarray
    residual_norm: float


@dataclass(frozen=True)
class TrimResult:
    """Pseudo-trim state, command, and residual report."""

    state: FlightState
    command_rad: np.ndarray
    residual: TrimResidual
    converged: bool
    method: str


def pseudo_trim(
    spec: TrimSpec,
    model: GliderModel | None = None,
    wind_model: object | None = None,
) -> TrimResult:
    """Construct a deterministic pseudo-trim and report residuals.

    The state uses body-axis airspeed aligned with body x and attitude pitch set
    to the requested flight-path angle. This makes vertical velocity explicit
    while leaving aerodynamic residuals visible for later true-trim work.
    """

    glider = nominal_glider() if model is None else model
    gamma = spec.resolved_flight_path_angle_rad()
    command = np.asarray(spec.command_rad, dtype=float).reshape(3)
    state = FlightState(
        position_w_m=np.array([0.0, 0.0, float(spec.altitude_m)], dtype=float),
        euler_rad=np.array([float(spec.bank_angle_rad), gamma, float(spec.heading_rad)], dtype=float),
        velocity_b_m_s=np.array([float(spec.airspeed_m_s), 0.0, 0.0], dtype=float),
        rates_b_rad_s=np.array([0.0, 0.0, float(spec.turn_rate_rad_s)], dtype=float),
        surfaces_rad=command.copy(),
    )
    loads = glider.evaluate_loads(state, wind_model=wind_model, wind_mode=spec.wind_mode)
    derivative = rigid_body_derivative(
        state=state,
        force_b_n=loads.force_b_n,
        moment_b_n_m=loads.moment_b_n_m,
        mass_kg=glider.mass_kg,
        inertia_b_kg_m2=glider.inertia_b_kg_m2,
    )
    velocity_dot = derivative[6:9]
    speed = max(float(np.linalg.norm(state.velocity_b_m_s)), 1e-12)
    speed_derivative = float(np.dot(state.velocity_b_m_s, velocity_dot) / speed)
    world_velocity = body_to_world(state.velocity_b_m_s, state.euler_rad)
    target_vertical = (
        float(spec.vertical_speed_m_s)
        if spec.vertical_speed_m_s is not None
        else float(spec.airspeed_m_s) * np.sin(gamma)
    )
    vertical_error = float(world_velocity[2] - target_vertical)
    angular_rate_error = state.rates_b_rad_s - np.array([0.0, 0.0, float(spec.turn_rate_rad_s)], dtype=float)
    residual_vector = np.concatenate(
        [
            loads.force_b_n / glider.mass_kg,
            loads.moment_b_n_m,
            np.array([speed_derivative, vertical_error], dtype=float),
            angular_rate_error,
        ]
    )
    residual = TrimResidual(
        force_b_n=loads.force_b_n,
        moment_b_n_m=loads.moment_b_n_m,
        acceleration_b_m_s2=velocity_dot,
        speed_derivative_m_s2=speed_derivative,
        vertical_velocity_error_m_s=vertical_error,
        angular_rate_error_rad_s=angular_rate_error,
        residual_norm=float(np.linalg.norm(residual_vector)),
    )
    return TrimResult(
        state=state,
        command_rad=command,
        residual=residual,
        converged=False,
        method="deterministic_pseudo_trim",
    )
