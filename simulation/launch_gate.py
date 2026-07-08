"""Launch gate definition for simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import radians


Bounds3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
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
    """Launch gate plug-in for release-state checks."""

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


def build_launch_gate() -> LaunchGate:
    """Build a reusable launch gate plug-in instance."""

    return LaunchGate()


def _inside(values: State, bounds: Bounds3D) -> bool:
    return (
        bounds[0][0] <= values[0] <= bounds[0][1]
        and bounds[1][0] <= values[1] <= bounds[1][1]
        and bounds[2][0] <= values[2] <= bounds[2][1]
    )
