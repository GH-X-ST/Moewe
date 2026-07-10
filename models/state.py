"""State layout and kinematics for the 15-state glider model."""

from __future__ import annotations

from math import cos, sin

import numpy as np
import numpy.typing as npt


G_M_S2 = 9.81


def as_state(state: npt.ArrayLike) -> np.ndarray:
    """Return a 15-vector state array."""

    return np.asarray(state, dtype=float).reshape(15)


def sink_rate_down(state: npt.ArrayLike) -> float:
    """Return vertical speed with positive value downward."""

    x = as_state(state)
    phi, theta = x[3], x[4]
    u_b, v_b, w_b = x[6:9]
    c_theta = cos(theta)
    return (
        -sin(theta) * u_b
        + sin(phi) * c_theta * v_b
        + cos(phi) * c_theta * w_b
    )


def mechanical_energy_rate(
    state: npt.ArrayLike,
    state_derivative: npt.ArrayLike,
) -> float:
    """Return the derivative of specific mechanical energy."""

    x = as_state(state)
    derivative = as_state(state_derivative)
    return float(G_M_S2 * derivative[2] + x[6:9] @ derivative[6:9])
