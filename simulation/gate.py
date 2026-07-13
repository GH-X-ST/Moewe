"""Gate terminal condition for simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy.typing as npt

from models.geometry import (
    RigidBodyGeometry,
    Vector3,
    gate_crossing,
    orthogonal_axes,
)


@dataclass(frozen=True)
class Gate:
    """Rectangular mission gate in public z-up world axes."""

    center_w_m: Vector3 = (6.6, 2.2, 1.4)
    normal_w: Vector3 = (1.0, 0.0, 0.0)
    width_axis_w: Vector3 = (0.0, 1.0, 0.0)
    width_m: float = 1.2
    height_m: float = 0.5
    margin_m: float = 0.0
    geometry: RigidBodyGeometry = RigidBodyGeometry()

    def __post_init__(self) -> None:
        normal, width, _ = orthogonal_axes(self.normal_w, self.width_axis_w)
        object.__setattr__(self, "normal_w", tuple(normal))
        object.__setattr__(self, "width_axis_w", tuple(width))

    def passed(
        self,
        states: npt.ArrayLike,
    ) -> bool:
        """Return whether a realized dense trajectory passes the gate."""

        return gate_crossing(
            states,
            self.geometry,
            self.center_w_m,
            self.normal_w,
            self.width_axis_w,
            self.width_m,
            self.height_m,
            self.margin_m,
        )
