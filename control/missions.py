"""Tube-certified gate and landing missions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import radians
from typing import Protocol

import numpy as np
import numpy.typing as npt

from control.interval import Interval
from control.tube import BodyTube, SegmentTube, body_points
from models.geometry import (
    RigidBodyGeometry,
    gate_crossing,
    orthogonal_axes,
    platform_landing,
    point_velocities,
    world_points,
)
from models.state import as_state

Vector3 = tuple[float, float, float]
Bounds3D = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


@dataclass(frozen=True)
class FreeSpace:
    """Axis-aligned arena with optional forbidden boxes."""

    arena_m: Bounds3D
    forbidden_m: tuple[Bounds3D, ...] = ()

    def contains(self, tube: BodyTube | SegmentTube) -> bool:
        """Return whether every continuous occupied-body box is free."""

        body = tube.body if isinstance(tube, SegmentTube) else tube
        return all(self.contains_occupancy(item) for item in body.occupied)

    def contains_occupancy(self, occupied: Interval) -> bool:
        """Return whether one continuous occupied-body enclosure is free."""

        lower, upper = _occupied_bounds(occupied)
        arena_lower, arena_upper = _box_arrays(self.arena_m)
        if np.any(lower <= arena_lower) or np.any(upper >= arena_upper):
            return False
        return all(
            not _boxes_intersect(lower, upper, *(_box_arrays(box)))
            for box in self.forbidden_m
        )

    def nominal_constraints(
        self,
        states: npt.ArrayLike,
        geometry: RigidBodyGeometry,
    ) -> np.ndarray:
        """Return full-body free-space inequalities for nominal states."""

        arena_lower, arena_upper = _box_arrays(self.arena_m)
        values = []
        for state in np.asarray(states, dtype=float).reshape(-1, 15):
            points = world_points(state, geometry.body_b_m)
            values.extend((points - arena_lower).reshape(-1))
            values.extend((arena_upper - points).reshape(-1))
            body_lower = np.min(points, axis=0)
            body_upper = np.max(points, axis=0)
            for box in self.forbidden_m:
                box_lower, box_upper = _box_arrays(box)
                separation = np.concatenate(
                    (box_lower - body_upper, body_lower - box_upper)
                )
                values.append(float(np.max(separation)))
        return np.asarray(values, dtype=float)


class Mission(Protocol):
    """Planning objective and robust terminal contract."""

    @property
    def identity(self) -> str:
        """Return the immutable mission identity."""

        ...

    def terminal(self, tubes: tuple[SegmentTube, ...]) -> bool:
        """Return whether a complete tube sequence is safe and terminal."""

        ...

    def running_cost(
        self,
        state: npt.ArrayLike,
        control: npt.ArrayLike,
    ) -> float:
        """Return one nominal performance cost."""

        ...

    def nominal_constraints(self, states: npt.ArrayLike) -> npt.ArrayLike:
        """Return nominal inequalities that must be nonnegative."""

        ...


@dataclass(frozen=True)
class GateMission:
    """Robust full-body passage through a rectangular aperture."""

    free_space: FreeSpace
    center_w_m: Vector3 = (6.6, 2.2, 1.4)
    normal_w: Vector3 = (1.0, 0.0, 0.0)
    width_axis_w: Vector3 = (0.0, 1.0, 0.0)
    width_m: float = 1.2
    height_m: float = 0.5
    margin_m: float = 0.0
    geometry: RigidBodyGeometry = RigidBodyGeometry()
    control_weight: float = 0.01

    def __post_init__(self) -> None:
        normal, width, _ = orthogonal_axes(self.normal_w, self.width_axis_w)
        object.__setattr__(self, "normal_w", _vector(normal))
        object.__setattr__(self, "width_axis_w", _vector(width))

    @property
    def identity(self) -> str:
        """Return a deterministic gate-contract identity."""

        return _identity(
            "gate",
            self.free_space,
            self.center_w_m,
            self.normal_w,
            self.width_axis_w,
            self.width_m,
            self.height_m,
            self.margin_m,
            self.geometry,
        )

    def terminal(self, tubes: tuple[SegmentTube, ...]) -> bool:
        """Certify full-body passage without gate-frame contact."""

        if not tubes:
            return False
        center = np.asarray(self.center_w_m, dtype=float)
        normal = np.asarray(self.normal_w, dtype=float)
        width = np.asarray(self.width_axis_w, dtype=float)
        height = np.cross(normal, width)
        start = body_points(tubes[0].initial, self.geometry.body_b_m)
        end = body_points(tubes[-1].successor, self.geometry.body_b_m)
        start_distance = _project(start, normal, center)
        end_distance = _project(end, normal, center)
        if np.max(start_distance.upper) >= 0.0:
            return False
        if np.min(end_distance.lower) <= 0.0:
            return False

        half_width = 0.5 * self.width_m - self.margin_m
        half_height = 0.5 * self.height_m - self.margin_m
        for tube in tubes[:-1]:
            trailing = _project(
                body_points(tube.successor, self.geometry.body_b_m),
                normal,
                center,
            )
            if not np.any(trailing.upper <= 0.0):
                return False
        pieces = tuple(occupied for tube in tubes for occupied in tube.body.occupied)
        crosses_plane = False
        for occupied in pieces:
            distance = _project(occupied, normal, center)
            if np.min(distance.lower) > 0.0 or np.max(distance.upper) < 0.0:
                continue
            crosses_plane = True
            if not _inside_aperture(
                occupied,
                center,
                width,
                height,
                half_width,
                half_height,
            ):
                return False
        if not crosses_plane:
            return False
        if not self.free_space.contains_occupancy(start):
            return False
        return all(self.free_space.contains_occupancy(occupied) for occupied in pieces)

    def realized(
        self,
        states: npt.ArrayLike,
    ) -> bool:
        """Evaluate passage on a realized dense trajectory."""

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

    def running_cost(
        self,
        state: npt.ArrayLike,
        control: npt.ArrayLike,
    ) -> float:
        """Penalize nominal gate-centre error and control effort."""

        offset = as_state(state)[:3] - np.asarray(self.center_w_m)
        command = np.asarray(control, dtype=float).reshape(3)
        return float(offset @ offset + self.control_weight * (command @ command))

    def nominal_constraints(self, states: npt.ArrayLike) -> np.ndarray:
        """Return nominal free-space and gate-passage inequalities."""

        values = np.asarray(states, dtype=float).reshape(-1, 15)
        center = np.asarray(self.center_w_m, dtype=float)
        normal = np.asarray(self.normal_w, dtype=float)
        width = np.asarray(self.width_axis_w, dtype=float)
        height = np.cross(normal, width)
        start = world_points(values[0], self.geometry.body_b_m)
        end = world_points(values[-1], self.geometry.body_b_m)
        constraints = list(
            self.free_space.nominal_constraints(values[:-1], self.geometry)
        )
        constraints.append(float(-np.max((start - center) @ normal)))
        constraints.append(float(np.min((end - center) @ normal)))
        half_width = 0.5 * self.width_m - self.margin_m
        half_height = 0.5 * self.height_m - self.margin_m
        constraints.extend(half_width - np.abs((end - center) @ width))
        constraints.extend(half_height - np.abs((end - center) @ height))
        return np.asarray(constraints, dtype=float)


@dataclass(frozen=True)
class LandingMission:
    """Robust first-contact landing on a finite planar platform."""

    free_space: FreeSpace
    center_w_m: Vector3 = (6.0, 2.2, 1.0)
    length_axis_w: Vector3 = (1.0, 0.0, 0.0)
    width_axis_w: Vector3 = (0.0, 1.0, 0.0)
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
    control_weight: float = 0.01

    def __post_init__(self) -> None:
        length, width, _ = orthogonal_axes(
            self.length_axis_w,
            self.width_axis_w,
        )
        object.__setattr__(self, "length_axis_w", _vector(length))
        object.__setattr__(self, "width_axis_w", _vector(width))

    @property
    def identity(self) -> str:
        """Return a deterministic landing-contract identity."""

        return _identity(
            "landing",
            self.free_space,
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
            self.geometry,
        )

    def terminal(self, tubes: tuple[SegmentTube, ...]) -> bool:
        """Certify realization-dependent first contact with the platform."""

        if not tubes:
            return False
        preterminal = _landing_pieces(tubes[:-1])
        pieces = _landing_pieces((tubes[-1],))
        if not pieces:
            return False
        center = np.asarray(self.center_w_m, dtype=float)
        length = np.asarray(self.length_axis_w, dtype=float)
        width = np.asarray(self.width_axis_w, dtype=float)
        normal = np.cross(length, width)
        start_body = body_points(tubes[0].initial, self.geometry.body_b_m)
        start_height = _project(start_body, normal, center)
        if np.min(start_height.lower) <= 0.0:
            return False
        end_contact = body_points(
            tubes[-1].successor,
            self.geometry.contact_b_m,
        )
        if np.min(_project(end_contact, normal, center).upper) > 0.0:
            return False

        candidate_indices = []
        for index, (_, _, contact, _, _) in enumerate(pieces):
            height = _project(contact, normal, center)
            if np.any((height.lower <= 0.0) & (height.upper >= 0.0)):
                candidate_indices.append(index)
        if not candidate_indices:
            return False

        first_candidate = candidate_indices[0]
        last_candidate = candidate_indices[-1]
        if not self.free_space.contains_occupancy(start_body):
            return False
        if not all(
            self.free_space.contains_occupancy(occupied)
            for _, occupied, _, _, _ in (preterminal + pieces[: last_candidate + 1])
        ):
            return False
        for _, occupied, _, _, _ in preterminal + pieces[:first_candidate]:
            if np.min(_project(occupied, normal, center).lower) <= 0.0:
                return False

        half_length = 0.5 * self.length_m - self.margin_m
        half_width = 0.5 * self.width_m - self.margin_m
        body_vertices = np.asarray(self.geometry.body_b_m, dtype=float).reshape(
            -1,
            3,
        )
        contact_points = np.asarray(
            self.geometry.contact_b_m,
            dtype=float,
        ).reshape(-1, 3)
        allowed_contact = np.any(
            np.all(
                np.isclose(
                    body_vertices[:, None, :],
                    contact_points[None, :, :],
                ),
                axis=2,
            ),
            axis=1,
        )
        for index in candidate_indices:
            state, occupied, contact, footprint, velocity = pieces[index]
            if np.any(~allowed_contact):
                other = Interval(
                    occupied.lower[~allowed_contact],
                    occupied.upper[~allowed_contact],
                )
                if np.min(_project(other, normal, center).lower) <= 0.0:
                    return False
            if not _inside_aperture(
                footprint,
                center,
                length,
                width,
                half_length,
                half_width,
            ):
                return False
            contact_height = _project(contact, normal, center)
            meeting = (contact_height.lower <= 0.0) & (contact_height.upper >= 0.0)
            if not _contact_velocity_valid(
                Interval(velocity.lower[meeting], velocity.upper[meeting]),
                normal,
                self.normal_speed_max_m_s,
                self.contact_speed_max_m_s,
            ):
                return False
            if (
                state.lower[3] < -self.roll_max_rad
                or state.upper[3] > self.roll_max_rad
                or state.lower[4] < self.pitch_bounds_rad[0]
                or state.upper[4] > self.pitch_bounds_rad[1]
            ):
                return False
        return True

    def realized(
        self,
        states: npt.ArrayLike,
    ) -> bool:
        """Evaluate first contact on a realized dense trajectory."""

        return platform_landing(
            states,
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

    def running_cost(
        self,
        state: npt.ArrayLike,
        control: npt.ArrayLike,
    ) -> float:
        """Penalize nominal touchdown error and control effort."""

        value = as_state(state)
        offset = value[:3] - np.asarray(self.center_w_m)
        pitch_error = value[4] - self.touchdown_pitch_rad
        command = np.asarray(control, dtype=float).reshape(3)
        return float(
            offset @ offset
            + 0.1 * (value[6:9] @ value[6:9])
            + pitch_error * pitch_error
            + self.control_weight * (command @ command)
        )

    def nominal_constraints(self, states: npt.ArrayLike) -> np.ndarray:
        """Return nominal precontact and touchdown inequalities."""

        values = np.asarray(states, dtype=float).reshape(-1, 15)
        center = np.asarray(self.center_w_m, dtype=float)
        length = np.asarray(self.length_axis_w, dtype=float)
        width = np.asarray(self.width_axis_w, dtype=float)
        normal = np.cross(length, width)
        contact = world_points(values[-1], self.geometry.contact_b_m)
        footprint = world_points(values[-1], self.geometry.footprint_b_m)
        velocities = point_velocities(values[-1], self.geometry.contact_b_m)
        height = (contact - center) @ normal
        sink = -velocities @ normal
        constraints = list(
            self.free_space.nominal_constraints(values[:-1], self.geometry)
        )
        start = world_points(values[0], self.geometry.body_b_m)
        constraints.append(float(np.min((start - center) @ normal)))
        constraints.append(float(-np.min(height)))
        constraints.extend(
            0.5 * self.length_m - self.margin_m - np.abs((footprint - center) @ length)
        )
        constraints.extend(
            0.5 * self.width_m - self.margin_m - np.abs((footprint - center) @ width)
        )
        constraints.extend(sink)
        constraints.extend(self.normal_speed_max_m_s - sink)
        constraints.extend(
            self.contact_speed_max_m_s - np.linalg.norm(velocities, axis=1)
        )
        constraints.extend(
            (
                self.roll_max_rad - abs(values[-1, 3]),
                values[-1, 4] - self.pitch_bounds_rad[0],
                self.pitch_bounds_rad[1] - values[-1, 4],
            )
        )
        return np.asarray(constraints, dtype=float)


def _landing_pieces(
    tubes: tuple[SegmentTube, ...],
) -> tuple[tuple[Interval, Interval, Interval, Interval, Interval], ...]:
    return tuple(
        item
        for tube in tubes
        for item in zip(
            tube.states,
            tube.body.occupied,
            tube.body.contact,
            tube.body.footprint,
            tube.body.contact_velocity,
            strict=True,
        )
    )


def _project(points: Interval, axis: np.ndarray, origin: np.ndarray) -> Interval:
    return ((points - origin) * axis).sum(axis=-1)


def _inside_aperture(
    points: Interval,
    center: np.ndarray,
    first_axis: np.ndarray,
    second_axis: np.ndarray,
    first_half_extent: float,
    second_half_extent: float,
) -> bool:
    first = _project(points, first_axis, center)
    second = _project(points, second_axis, center)
    return bool(
        np.min(first.lower) > -first_half_extent
        and np.max(first.upper) < first_half_extent
        and np.min(second.lower) > -second_half_extent
        and np.max(second.upper) < second_half_extent
    )


def _contact_velocity_valid(
    velocity: Interval,
    normal: np.ndarray,
    normal_speed_max_m_s: float,
    contact_speed_max_m_s: float,
) -> bool:
    normal_velocity = _project(velocity, normal, np.zeros(3))
    speed_upper = velocity.square().sum(axis=1).sqrt().upper
    return bool(
        np.min(normal_velocity.lower) >= -normal_speed_max_m_s
        and np.max(normal_velocity.upper) <= 0.0
        and np.max(speed_upper) <= contact_speed_max_m_s
    )


def _occupied_bounds(occupied: Interval) -> tuple[np.ndarray, np.ndarray]:
    lower = occupied.lower.reshape(-1, 3)
    upper = occupied.upper.reshape(-1, 3)
    return np.min(lower, axis=0), np.max(upper, axis=0)


def _box_arrays(bounds: Bounds3D) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(bounds, dtype=float)
    return values[:, 0], values[:, 1]


def _boxes_intersect(
    first_lower: np.ndarray,
    first_upper: np.ndarray,
    second_lower: np.ndarray,
    second_upper: np.ndarray,
) -> bool:
    return bool(
        np.all(first_upper >= second_lower) and np.all(first_lower <= second_upper)
    )


def _vector(value: np.ndarray) -> Vector3:
    return tuple(float(component) for component in value)


def _identity(name: str, *values: object) -> str:
    digest = sha256(name.encode("ascii"))
    for value in values:
        if isinstance(value, RigidBodyGeometry):
            arrays = (
                value.body_b_m,
                value.contact_b_m,
                value.footprint_b_m,
            )
            for array in arrays:
                digest.update(np.asarray(array, dtype=float).tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()
