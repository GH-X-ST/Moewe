"""Launch gate definition for simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import radians

Bounds3D = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]
State = Sequence[float]

LAUNCH_POSITION_BOUNDS_M: Bounds3D = (
    (1.20, 1.40),
    (2.10, 2.30),
    (1.30, 1.70),
)
LAUNCH_ATTITUDE_BOUNDS_RAD: Bounds3D = (
    (radians(-4), radians(16)),
    (radians(-10), radians(12)),
    (radians(-10), radians(7)),
)
LAUNCH_BODY_VELOCITY_BOUNDS_M_S: Bounds3D = (
    (4.5, 6.8),
    (-0.3, 1.5),
    (-0.3, 0.9),
)
LAUNCH_BODY_RATE_BOUNDS_RAD_S: Bounds3D = (
    (-0.6, 1.0),
    (-0.8, 1.2),
    (-0.9, 0.7),
)


@dataclass(frozen=True)
class LaunchGate:
    """Release-state checks at the launch gate."""

    position_bounds_m: Bounds3D = LAUNCH_POSITION_BOUNDS_M
    attitude_bounds_rad: Bounds3D = LAUNCH_ATTITUDE_BOUNDS_RAD
    body_velocity_bounds_m_s: Bounds3D = LAUNCH_BODY_VELOCITY_BOUNDS_M_S
    body_rate_bounds_rad_s: Bounds3D = LAUNCH_BODY_RATE_BOUNDS_RAD_S

    def contains(self, state: State) -> bool:
        """Return whether the state lies inside the launch gate."""

        return (
            _inside(state[0:3], self.position_bounds_m)
            and _inside(state[3:6], self.attitude_bounds_rad)
            and _inside(state[6:9], self.body_velocity_bounds_m_s)
            and _inside(state[9:12], self.body_rate_bounds_rad_s)
        )

    def passed(self, previous_state: State, state: State) -> bool:
        """Return whether the state enters the launch gate."""

        if self.contains(state):
            return True

        plane_x = 0.5 * sum(self.position_bounds_m[0])
        previous_x = previous_state[0]
        current_x = state[0]
        if current_x <= previous_x or not previous_x <= plane_x <= current_x:
            return False

        ratio = (plane_x - previous_x) / (current_x - previous_x)
        crossing = [
            previous + ratio * (current - previous)
            for previous, current in zip(previous_state[:12], state[:12], strict=True)
        ]
        return self.contains(crossing)


def _inside(values: State, bounds: Bounds3D) -> bool:
    return (
        bounds[0][0] <= values[0] <= bounds[0][1]
        and bounds[1][0] <= values[1] <= bounds[1][1]
        and bounds[2][0] <= values[2] <= bounds[2][1]
    )
