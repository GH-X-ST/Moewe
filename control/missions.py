"""Mission descriptors for gate crossing and platform landing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, pi, radians
from typing import Protocol

import numpy as np
import numpy.typing as npt

from models.geometry import (
    RigidBodyGeometry,
    gate_crossing,
    platform_landing,
    world_points,
)
from models.state import as_state


Bounds3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
StatePredicate = Callable[[npt.ArrayLike], bool]
OccupiedDistance = Callable[[np.ndarray], float]


class Mission(Protocol):
    """Mission descriptor consumed by the recoverability controller."""

    @property
    def safe(self) -> StatePredicate:
        """Return the mission safe-set predicate."""

        ...

    @property
    def delta(self) -> float:
        """Return the required one-step progress decrease."""

        ...

    def event(
        self,
        previous_state: npt.ArrayLike,
        state: npt.ArrayLike,
    ) -> bool:
        """Return whether a state segment completes the mission."""

        ...

    def progress(self, state: npt.ArrayLike) -> float:
        """Return the mission progress value."""

        ...

    def running_cost(
        self,
        state: npt.ArrayLike,
        control: npt.ArrayLike,
    ) -> float:
        """Return the mission running cost."""

        ...


@dataclass(frozen=True)
class GateMission:
    """Rectangular gate crossing mission."""

    safe: StatePredicate
    center_w_m: tuple[float, float, float] = (6.6, 2.2, 1.4)
    normal_w: tuple[float, float, float] = (1.0, 0.0, 0.0)
    width_axis_w: tuple[float, float, float] = (0.0, 1.0, 0.0)
    width_m: float = 1.2
    height_m: float = 0.5
    margin_m: float = 0.0
    geometry: RigidBodyGeometry = RigidBodyGeometry()
    weights: tuple[float, float, float, float] = (1.0, 2.0, 2.0, 0.2)
    control_weight: float = 0.01
    delta: float = 0.01

    def event(
        self,
        previous_state: npt.ArrayLike,
        state: npt.ArrayLike,
    ) -> bool:
        """Return whether a segment crosses the certified aperture."""

        return gate_crossing(
            previous_state,
            state,
            self.geometry,
            self.center_w_m,
            self.normal_w,
            self.width_axis_w,
            self.width_m,
            self.height_m,
            self.margin_m,
        )

    def progress(self, state: npt.ArrayLike) -> float:
        """Return the gate progress value."""

        x = as_state(state)
        center = np.asarray(self.center_w_m, dtype=float)
        normal = np.asarray(self.normal_w, dtype=float)
        width_axis = np.asarray(self.width_axis_w, dtype=float)
        height_axis = np.cross(normal, width_axis)
        offset = x[:3] - center
        heading = atan2(-float(normal[1]), float(normal[0]))
        heading_error = _wrap_pi(float(x[5]) - heading)
        w_plane, w_width, w_height, w_heading = self.weights
        plane_gap = max(-float(offset @ normal), 0.0)
        return (
            w_plane * plane_gap * plane_gap
            + w_width * float(offset @ width_axis) ** 2
            + w_height * float(offset @ height_axis) ** 2
            + w_heading * heading_error * heading_error
        )

    def running_cost(
        self,
        state: npt.ArrayLike,
        control: npt.ArrayLike,
    ) -> float:
        """Return the gate running cost."""

        u = np.asarray(control, dtype=float)
        return self.progress(state) + self.control_weight * float(u @ u)


@dataclass(frozen=True)
class LandingMission:
    """Constrained planar platform landing mission."""

    safe: StatePredicate
    center_w_m: tuple[float, float, float] = (6.0, 2.2, 1.0)
    length_axis_w: tuple[float, float, float] = (1.0, 0.0, 0.0)
    width_axis_w: tuple[float, float, float] = (0.0, 1.0, 0.0)
    length_m: float = 1.0
    width_m: float = 1.0
    normal_speed_max_m_s: float = 1.0
    contact_speed_max_m_s: float = 5.0
    roll_max_rad: float = radians(20.0)
    pitch_bounds_rad: tuple[float, float] = (
        radians(-10.0),
        radians(25.0),
    )
    margin_m: float = 0.0
    geometry: RigidBodyGeometry = RigidBodyGeometry()
    touchdown_pitch_rad: float = radians(4.0)
    weights: tuple[float, float, float, float, float] = (
        1.0,
        1.0,
        2.0,
        0.1,
        0.5,
    )
    control_weight: float = 0.01
    delta: float = 0.01

    def event(
        self,
        previous_state: npt.ArrayLike,
        state: npt.ArrayLike,
    ) -> bool:
        """Return whether a segment satisfies touchdown constraints."""

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

    def progress(self, state: npt.ArrayLike) -> float:
        """Return the landing progress value."""

        x = as_state(state)
        w_x, w_y, w_z, w_v, w_theta = self.weights
        return (
            w_x * (x[0] - self.center_w_m[0]) ** 2
            + w_y * (x[1] - self.center_w_m[1]) ** 2
            + w_z * (x[2] - self.center_w_m[2]) ** 2
            + w_v * float(x[6:9] @ x[6:9])
            + w_theta * (x[4] - self.touchdown_pitch_rad) ** 2
        )

    def running_cost(
        self,
        state: npt.ArrayLike,
        control: npt.ArrayLike,
    ) -> float:
        """Return the landing running cost."""

        u = np.asarray(control, dtype=float)
        return self.progress(state) + self.control_weight * float(u @ u)


def box_safe(bounds_m: Bounds3D) -> StatePredicate:
    """Return a position-box safe-set predicate."""

    def contains(state: npt.ArrayLike) -> bool:
        point = as_state(state)[:3]
        return (
            bounds_m[0][0] <= point[0] <= bounds_m[0][1]
            and bounds_m[1][0] <= point[1] <= bounds_m[1][1]
            and bounds_m[2][0] <= point[2] <= bounds_m[2][1]
        )

    return contains


def obstacle_safe(
    distance: OccupiedDistance,
    geometry: RigidBodyGeometry,
    clearance_m: float,
) -> StatePredicate:
    """Return an obstacle-clearance safe-set predicate."""

    def contains(state: npt.ArrayLike) -> bool:
        return distance(world_points(state, geometry.body_b_m)) >= clearance_m

    return contains


def combine_safe(*predicates: StatePredicate) -> StatePredicate:
    """Return the intersection of safe-set predicates."""

    def contains(state: npt.ArrayLike) -> bool:
        return all(predicate(state) for predicate in predicates)

    return contains


def _wrap_pi(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi
