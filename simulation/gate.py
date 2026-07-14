"""Gate terminal event for simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy.typing as npt

from models.geometry import RigidBodyGeometry, gate_crossing, orthogonal_axes


@dataclass(frozen=True)
class Gate:
    """Rectangular gate using the controller's realized-event geometry."""

    geometry: RigidBodyGeometry
    center_w_m: npt.ArrayLike
    normal_w: npt.ArrayLike
    width_axis_w: npt.ArrayLike
    width_m: float
    height_m: float
    frame_clearance_m: float
    body_inflation_m: float
    position_error_m: float
    attitude_error_rad: float

    def __post_init__(self) -> None:
        normal, width, _ = orthogonal_axes(self.normal_w, self.width_axis_w)
        object.__setattr__(self, "normal_w", normal)
        object.__setattr__(self, "width_axis_w", width)

    def passed(self, states: npt.ArrayLike) -> bool:
        """Return whether a dense realized trajectory passes the gate."""

        return gate_crossing(
            states,
            self.geometry,
            self.center_w_m,
            self.normal_w,
            self.width_axis_w,
            self.width_m,
            self.height_m,
            self.frame_clearance_m,
            self.body_inflation_m,
            self.position_error_m,
            self.attitude_error_rad,
        )
