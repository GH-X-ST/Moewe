"""Rigid-body equations for the 15-state Moewe glider model."""

from __future__ import annotations

import numpy as np

from .frames import body_to_world, euler_rate_matrix
from .state import FlightState


def rigid_body_derivative(
    state: FlightState,
    force_b_n: np.ndarray,
    moment_b_n_m: np.ndarray,
    mass_kg: float,
    inertia_b_kg_m2: np.ndarray,
    surface_rates_rad_s: np.ndarray | None = None,
) -> np.ndarray:
    """Return the 15-state derivative for body-axis force and moment."""

    force_b = np.asarray(force_b_n, dtype=float).reshape(3)
    moment_b = np.asarray(moment_b_n_m, dtype=float).reshape(3)
    inertia = np.asarray(inertia_b_kg_m2, dtype=float).reshape(3, 3)
    omega = state.rates_b_rad_s
    velocity = state.velocity_b_m_s
    position_dot = body_to_world(velocity, state.euler_rad)
    euler_dot = euler_rate_matrix(state.euler_rad[0], state.euler_rad[1]) @ omega
    velocity_dot = force_b / float(mass_kg) - np.cross(omega, velocity)
    rates_dot = np.linalg.solve(inertia, moment_b - np.cross(omega, inertia @ omega))
    surface_dot = np.zeros(3) if surface_rates_rad_s is None else np.asarray(surface_rates_rad_s, dtype=float).reshape(3)
    return np.concatenate([position_dot, euler_dot, velocity_dot, rates_dot, surface_dot])
