"""Platform first-contact event for simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy.typing as npt

from models.geometry import RigidBodyGeometry, orthogonal_axes, platform_landing


@dataclass(frozen=True)
class Platform:
    """Finite platform using the controller's realized-event geometry."""

    geometry: RigidBodyGeometry
    center_w_m: npt.ArrayLike
    length_axis_w: npt.ArrayLike
    width_axis_w: npt.ArrayLike
    length_m: float
    width_m: float
    normal_speed_max_m_s: float
    tangential_speed_max_m_s: float
    roll_abs_max_rad: float
    touchdown_pitch_rad: float
    pitch_error_abs_max_rad: float
    platform_clearance_m: float
    body_inflation_m: float
    position_error_m: float
    attitude_error_rad: float

    def __post_init__(self) -> None:
        length, width, _ = orthogonal_axes(
            self.length_axis_w,
            self.width_axis_w,
        )
        object.__setattr__(self, "length_axis_w", length)
        object.__setattr__(self, "width_axis_w", width)

    def landed(self, states: npt.ArrayLike) -> bool:
        """Return whether event-located first contact is admissible."""

        return platform_landing(
            states,
            self.geometry,
            self.center_w_m,
            self.length_axis_w,
            self.width_axis_w,
            self.length_m,
            self.width_m,
            self.normal_speed_max_m_s,
            self.tangential_speed_max_m_s,
            self.roll_abs_max_rad,
            self.pitch_error_abs_max_rad,
            self.touchdown_pitch_rad,
            self.platform_clearance_m,
            self.body_inflation_m,
            self.position_error_m,
            self.attitude_error_rad,
        )
