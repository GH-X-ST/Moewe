"""State layout and kinematics for the 15-state glider model."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

G_M_S2 = 9.81


def as_state(state: npt.ArrayLike) -> np.ndarray:
    """Return a 15-vector state array."""

    return np.asarray(state, dtype=float).reshape(15)
