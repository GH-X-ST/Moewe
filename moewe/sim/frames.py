"""Frame utilities for the Moewe simulator.

Conventions:

- World frame is z-up: x is the arena mission axis, y is lateral, z is up.
- Body frame is x-forward, y-starboard, z-down.
- Euler angles use a 3-2-1 yaw-pitch-roll sequence. Positive roll is right
  wing down, positive pitch is nose up, and positive yaw rotates the nose
  towards positive world y when level.
- All angles are radians and all distances use SI units.
"""

from __future__ import annotations

import numpy as np

GRAVITY_M_S2 = 9.80665
EPS = 1e-12

_WORLD_UP_FROM_WORLD_DOWN = np.diag([1.0, 1.0, -1.0])


def _body_to_world_down_rotation(phi: float, theta: float, psi: float) -> np.ndarray:
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


def rotation_body_to_world(phi: float, theta: float, psi: float) -> np.ndarray:
    """Return the matrix mapping body vectors to world z-up vectors."""

    return _WORLD_UP_FROM_WORLD_DOWN @ _body_to_world_down_rotation(
        float(phi),
        float(theta),
        float(psi),
    )


def rotation_world_to_body(phi: float, theta: float, psi: float) -> np.ndarray:
    """Return the matrix mapping world z-up vectors to body vectors."""

    return rotation_body_to_world(phi, theta, psi).T


def body_to_world(vector_b: np.ndarray, euler_rad: np.ndarray) -> np.ndarray:
    """Map one body-frame vector into the world z-up frame."""

    phi, theta, psi = np.asarray(euler_rad, dtype=float).reshape(3)
    return rotation_body_to_world(phi, theta, psi) @ np.asarray(vector_b, dtype=float)


def world_to_body(vector_w: np.ndarray, euler_rad: np.ndarray) -> np.ndarray:
    """Map one world z-up vector into the body frame."""

    phi, theta, psi = np.asarray(euler_rad, dtype=float).reshape(3)
    return rotation_world_to_body(phi, theta, psi) @ np.asarray(vector_w, dtype=float)


def body_to_world_rows(rows_b: np.ndarray, euler_rad: np.ndarray) -> np.ndarray:
    """Map an ``(n, 3)`` array of body vectors into world z-up rows."""

    phi, theta, psi = np.asarray(euler_rad, dtype=float).reshape(3)
    return np.asarray(rows_b, dtype=float) @ rotation_body_to_world(phi, theta, psi).T


def world_to_body_rows(rows_w: np.ndarray, euler_rad: np.ndarray) -> np.ndarray:
    """Map an ``(n, 3)`` array of world z-up vectors into body rows."""

    phi, theta, psi = np.asarray(euler_rad, dtype=float).reshape(3)
    return np.asarray(rows_w, dtype=float) @ rotation_world_to_body(phi, theta, psi).T


def gravity_world(g_m_s2: float = GRAVITY_M_S2) -> np.ndarray:
    """Return gravitational acceleration in the world z-up frame."""

    return np.array([0.0, 0.0, -float(g_m_s2)], dtype=float)


def gravity_body(euler_rad: np.ndarray, g_m_s2: float = GRAVITY_M_S2) -> np.ndarray:
    """Return gravitational acceleration in body axes."""

    return world_to_body(gravity_world(g_m_s2), euler_rad)


def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
    """Return the matrix mapping body rates ``[p, q, r]`` to Euler rates."""

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    if abs(c_theta) < EPS:
        raise ValueError("Euler pitch is too close to +/- pi/2 for rate mapping.")
    t_theta = np.tan(theta)
    return np.array(
        [
            [1.0, s_phi * t_theta, c_phi * t_theta],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta],
        ],
        dtype=float,
    )
