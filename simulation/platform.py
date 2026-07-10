"""Platform landing terminal condition for simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import radians

import numpy as np

from models.state import as_state, sink_rate_down


State = Sequence[float]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class Platform:
    """Finite horizontal landing platform in public z-up world axes."""

    center_w_m: Vector3 = (6.0, 2.2, 1.0)
    length_m: float = 1.0
    width_m: float = 1.0
    sink_rate_max_m_s: float = 1.0
    speed_max_m_s: float = 5.0
    roll_max_rad: float = radians(20)
    pitch_bounds_rad: tuple[float, float] = (radians(-10), radians(25))
    body_radius_m: float = 0.0

    def landed(self, previous_state: State, state: State) -> bool:
        """Return whether the state segment touches down on the platform."""

        previous = as_state(previous_state)
        current = as_state(state)
        platform_z = self.center_w_m[2]
        previous_z = previous[2]
        current_z = current[2]
        if previous_z <= platform_z or current_z > platform_z:
            return False

        ratio = (previous_z - platform_z) / (previous_z - current_z)
        touchdown = previous + ratio * (current - previous)
        x_w, y_w = touchdown[:2]
        phi, theta = touchdown[3:5]
        sink_rate = sink_rate_down(touchdown)
        speed = float(np.linalg.norm(touchdown[6:9]))

        return (
            abs(x_w - self.center_w_m[0])
            <= 0.5 * self.length_m - self.body_radius_m
            and abs(y_w - self.center_w_m[1])
            <= 0.5 * self.width_m - self.body_radius_m
            and 0.0 <= sink_rate <= self.sink_rate_max_m_s
            and speed <= self.speed_max_m_s
            and abs(phi) <= self.roll_max_rad
            and self.pitch_bounds_rad[0] <= theta <= self.pitch_bounds_rad[1]
        )
