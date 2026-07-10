"""Platform landing terminal condition for simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians

import numpy.typing as npt

from models.geometry import RigidBodyGeometry, Vector3, platform_landing


@dataclass(frozen=True)
class Platform:
    """Finite planar landing platform in public z-up world axes."""

    center_w_m: Vector3 = (6.0, 2.2, 1.0)
    length_axis_w: Vector3 = (1.0, 0.0, 0.0)
    width_axis_w: Vector3 = (0.0, 1.0, 0.0)
    length_m: float = 1.0
    width_m: float = 1.0
    normal_speed_max_m_s: float = 1.0
    contact_speed_max_m_s: float = 5.0
    roll_max_rad: float = radians(20)
    pitch_bounds_rad: tuple[float, float] = (radians(-10), radians(25))
    margin_m: float = 0.0
    geometry: RigidBodyGeometry = RigidBodyGeometry()

    def landed(
        self,
        previous_state: npt.ArrayLike,
        state: npt.ArrayLike,
    ) -> bool:
        """Return whether the state segment touches down on the platform."""

        return platform_landing(
            previous_state,
            state,
            self.geometry,
            self.center_w_m,
            self.length_axis_w,
            self.width_axis_w,
            self.length_m,
            self.width_m,
            self.normal_speed_max_m_s,
            self.contact_speed_max_m_s,
            self.roll_max_rad,
            self.pitch_bounds_rad,
            self.margin_m,
        )
