"""Arena volume definitions for simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Bounds3D = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]
Point3D = Sequence[float]

TRACKER_LIMIT_BOUNDS_M: Bounds3D = (
    (0.0, 8.0),
    (0.0, 4.8),
    (0.0, 3.5),
)
TRUE_SAFE_BOUNDS_M: Bounds3D = (
    (1.2, 6.6),
    (0.0, 4.4),
    (0.4, 3.5),
)


@dataclass(frozen=True)
class ArenaConfig:
    """Tracker and true safety bounds in public z-up world axes."""

    tracker_limit_bounds_m: Bounds3D = TRACKER_LIMIT_BOUNDS_M
    true_safe_bounds_m: Bounds3D = TRUE_SAFE_BOUNDS_M


DEFAULT_ARENA_CONFIG = ArenaConfig()


@dataclass
class Arena:
    """Tracker and safety-volume checks for the flight arena."""

    config: ArenaConfig = DEFAULT_ARENA_CONFIG

    def contains_tracker_limit(self, point_w_up_m: Point3D) -> bool:
        """Return whether the point or state lies inside tracker limits."""

        return _contains(point_w_up_m, self.config.tracker_limit_bounds_m)

    def contains_safe_volume(self, point_w_up_m: Point3D) -> bool:
        """Return whether the point lies inside the true safe volume."""

        return _contains(point_w_up_m, self.config.true_safe_bounds_m)


def _contains(point_w_up_m: Point3D, bounds_m: Bounds3D) -> bool:
    x_w, y_w, z_w = point_w_up_m[:3]
    return (
        bounds_m[0][0] <= x_w <= bounds_m[0][1]
        and bounds_m[1][0] <= y_w <= bounds_m[1][1]
        and bounds_m[2][0] <= z_w <= bounds_m[2][1]
    )
