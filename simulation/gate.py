"""Mission gate crossing condition for simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


State = Sequence[float]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class Gate:
    """Rectangular mission gate in public z-up world axes."""

    center_w_m: Vector3 = (6.6, 2.2, 1.4)
    normal_w: Vector3 = (1.0, 0.0, 0.0)
    width_axis_w: Vector3 = (0.0, 1.0, 0.0)
    width_m: float = 1.2
    height_m: float = 0.5

    def passed(self, previous_state: State, state: State) -> bool:
        """Return whether the state segment crosses the gate aperture."""

        center = np.asarray(self.center_w_m, dtype=float)
        normal = _unit(np.asarray(self.normal_w, dtype=float))
        width = np.asarray(self.width_axis_w, dtype=float)
        width_axis = _unit(width - normal * float(width @ normal))
        height_axis = np.cross(normal, width_axis)
        previous = np.asarray(previous_state[:3], dtype=float)
        current = np.asarray(state[:3], dtype=float)

        previous_distance = float((previous - center) @ normal)
        current_distance = float((current - center) @ normal)
        denominator = previous_distance - current_distance
        if denominator == 0.0:
            return False

        ratio = previous_distance / denominator
        if ratio < 0.0 or ratio > 1.0:
            return False

        offset = previous + ratio * (current - previous) - center
        return (
            abs(float(offset @ width_axis)) <= 0.5 * self.width_m
            and abs(float(offset @ height_axis)) <= 0.5 * self.height_m
        )


def build_gate() -> Gate:
    """Build a reusable mission gate plug-in instance."""

    return Gate()


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)
