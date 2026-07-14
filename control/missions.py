"""Gate and landing descriptors for terminal capture."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import pi, sqrt
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import numpy.typing as npt

from models.geometry import (
    RigidBodyGeometry,
    body_to_world,
    gate_crossing,
    orthogonal_axes,
    platform_landing,
    point_velocities,
)
from models.state import as_state

if TYPE_CHECKING:
    from control.predictor import Prediction


@dataclass(frozen=True)
class ApproachDomain:
    """Complete bounded state fiber on which a mission is compiled."""

    lower: npt.ArrayLike
    upper: npt.ArrayLike
    center: np.ndarray = field(init=False)
    radius: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        lower = _immutable(self.lower, (15,))
        upper = _immutable(self.upper, (15,))
        center = _immutable(0.5 * (lower + upper), (15,))
        radius = _immutable(0.5 * (upper - lower), (15,))
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)

    def contains(self, state: npt.ArrayLike) -> bool:
        """Return whether a state belongs to the complete approach box."""

        value = as_state(state)
        return bool(np.all(value >= self.lower) and np.all(value <= self.upper))


@dataclass(frozen=True)
class Halfspaces:
    """Immutable rows defining a convex set as ``matrix @ x <= bounds``."""

    matrix: npt.ArrayLike
    bounds: npt.ArrayLike

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        bounds = np.asarray(self.bounds, dtype=float).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[0] != bounds.size:
            raise ValueError("half-space rows and bounds must match")
        matrix = matrix.copy()
        bounds = bounds.copy()
        matrix.flags.writeable = False
        bounds.flags.writeable = False
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "bounds", bounds)

    @classmethod
    def box(cls, lower: npt.ArrayLike, upper: npt.ArrayLike) -> Halfspaces:
        """Return the half-spaces of an axis-aligned box."""

        low = np.asarray(lower, dtype=float).reshape(-1)
        high = np.asarray(upper, dtype=float).reshape(low.shape)
        eye = np.eye(low.size)
        return cls(
            np.vstack((eye, -eye)),
            np.concatenate((high, -low)),
        )

    def contains(self, value: npt.ArrayLike) -> bool:
        """Return whether a point satisfies every half-space."""

        point = np.asarray(value, dtype=float).reshape(self.matrix.shape[1])
        return bool(np.all(self.matrix @ point <= self.bounds))


@dataclass(frozen=True)
class FreeSpace:
    """Convex planar free space for the complete occupied body."""

    halfspaces: Halfspaces

    @classmethod
    def box(cls, lower_w_m: npt.ArrayLike, upper_w_m: npt.ArrayLike) -> FreeSpace:
        """Return an axis-aligned world-space flight volume."""

        return cls(Halfspaces.box(lower_w_m, upper_w_m))


class Mission(Protocol):
    """Manuscript terminal-mission contract."""

    @property
    def approach_domain(self) -> ApproachDomain:
        """Return the compiled terminal-approach domain."""

        ...

    def error(self, state: npt.ArrayLike) -> np.ndarray:
        """Return terminal-error coordinates."""

        ...

    def distance(self, state: npt.ArrayLike) -> float:
        """Return remaining distance in the declared approach direction."""

        ...

    @property
    def terminal_halfspaces(self) -> Halfspaces:
        """Return the exact terminal error set."""

        ...

    @property
    def free_space_halfspaces(self) -> Halfspaces:
        """Return planar complete-body free-space constraints."""

        ...

    def realized(self, states: npt.ArrayLike) -> bool:
        """Return whether measured states realize the terminal event."""

        ...

    def terminal_support_constraints(
        self,
        prediction: Prediction,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return affine reference rows for the robust terminal event."""

        ...


@dataclass(frozen=True)
class GateMission:
    """Full-body passage through a rectangular aperture."""

    approach_domain: ApproachDomain
    free_space: FreeSpace
    geometry: RigidBodyGeometry
    center_w_m: npt.ArrayLike
    normal_w: npt.ArrayLike
    width_axis_w: npt.ArrayLike
    width_m: float
    height_m: float
    center_flow_b_m_s: npt.ArrayLike
    target_airspeed_m_s: float
    heading_abs_max_rad: float
    roll_abs_max_rad: float
    pitch_bounds_rad: tuple[float, float]
    airspeed_bounds_m_s: tuple[float, float]
    frame_clearance_m: float
    body_inflation_m: float
    position_error_m: float
    attitude_error_rad: float
    _terminal_halfspaces: Halfspaces = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normal, width, _ = orthogonal_axes(self.normal_w, self.width_axis_w)
        object.__setattr__(self, "center_w_m", _immutable(self.center_w_m, (3,)))
        object.__setattr__(self, "normal_w", _immutable(normal, (3,)))
        object.__setattr__(self, "width_axis_w", _immutable(width, (3,)))
        object.__setattr__(
            self,
            "center_flow_b_m_s",
            _immutable(self.center_flow_b_m_s, (3,)),
        )

        clearance = self._clearance
        lower = np.array(
            [
                -0.5 * self.width_m + clearance,
                -0.5 * self.height_m + clearance,
                -self.heading_abs_max_rad,
                -self.roll_abs_max_rad,
                self.pitch_bounds_rad[0],
                self.airspeed_bounds_m_s[0] - self.target_airspeed_m_s,
            ]
        )
        upper = np.array(
            [
                0.5 * self.width_m - clearance,
                0.5 * self.height_m - clearance,
                self.heading_abs_max_rad,
                self.roll_abs_max_rad,
                self.pitch_bounds_rad[1],
                self.airspeed_bounds_m_s[1] - self.target_airspeed_m_s,
            ]
        )
        object.__setattr__(self, "_terminal_halfspaces", Halfspaces.box(lower, upper))

    @property
    def terminal_halfspaces(self) -> Halfspaces:
        """Return aperture, heading, attitude, and airspeed limits."""

        return self._terminal_halfspaces

    @property
    def free_space_halfspaces(self) -> Halfspaces:
        """Return planar complete-body free-space constraints."""

        return self.free_space.halfspaces

    def error(self, state: npt.ArrayLike) -> np.ndarray:
        """Return aperture, heading, roll, pitch, and airspeed error."""

        value = as_state(state)
        height_axis = np.cross(self.normal_w, self.width_axis_w)
        offset = value[:3] - self.center_w_m
        heading = _wrap(_gate_yaw(self) - value[5])
        airspeed = float(np.linalg.norm(value[6:9] - self.center_flow_b_m_s))
        return np.array(
            [
                offset @ self.width_axis_w,
                offset @ height_axis,
                heading,
                value[3],
                value[4],
                airspeed - self.target_airspeed_m_s,
            ]
        )

    def distance(self, state: npt.ArrayLike) -> float:
        """Return entrance-side distance along the crossing normal."""

        position = as_state(state)[:3]
        return float(-self.normal_w @ (position - self.center_w_m))

    def realized(self, states: npt.ArrayLike) -> bool:
        """Evaluate full-body gate passage from measured states."""

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

    def terminal_support_constraints(
        self,
        prediction: Prediction,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative affine rows for robust gate crossing."""

        stages = prediction.body_center.shape[0]
        body_count = self.geometry.body_b_m.shape[0]
        height = np.cross(self.normal_w, self.width_axis_w)
        axes = (
            self.width_axis_w,
            -self.width_axis_w,
            height,
            -height,
        )
        limits = (
            0.5 * self.width_m - self._support_clearance,
            0.5 * self.width_m - self._support_clearance,
            0.5 * self.height_m - self._support_clearance,
            0.5 * self.height_m - self._support_clearance,
        )
        terminal = self.terminal_halfspaces
        event_rows = np.flatnonzero(np.all(terminal.matrix[:, :2] == 0.0, axis=1))
        row_count = body_count * (2 + 4 * stages) + (stages + 1) * event_rows.size
        matrix = np.empty((row_count, 3))
        bounds = np.empty(row_count)
        row = 0
        for point in self.geometry.body_b_m:
            row = _body_state_row(
                prediction,
                self.approach_domain,
                point.reshape(1, 3),
                0,
                self.normal_w,
                self.center_w_m,
                -self._clearance,
                matrix,
                bounds,
                row,
            )
        for point in self.geometry.body_b_m:
            row = _body_state_row(
                prediction,
                self.approach_domain,
                point.reshape(1, 3),
                stages,
                -self.normal_w,
                self.center_w_m,
                -self._clearance,
                matrix,
                bounds,
                row,
            )
        for stage in range(stages):
            for axis, limit in zip(axes, limits, strict=True):
                for point in range(body_count):
                    row = _point_row(
                        prediction.body_support,
                        stage,
                        point,
                        axis,
                        self.center_w_m,
                        limit,
                        matrix,
                        bounds,
                        row,
                    )
        for stage in range(stages + 1):
            for terminal_row in event_rows:
                offset, reference = error_support(
                    self,
                    prediction,
                    stage,
                    terminal.matrix[terminal_row],
                )
                matrix[row] = reference
                bounds[row] = terminal.bounds[terminal_row] - offset
                row += 1
        return matrix, bounds

    @property
    def _clearance(self) -> float:
        radius = float(np.max(np.linalg.norm(self.geometry.body_b_m, axis=1)))
        return (
            self.frame_clearance_m
            + self.body_inflation_m
            + self.position_error_m
            + radius * self.attitude_error_rad
        )

    @property
    def _support_clearance(self) -> float:
        radius = float(np.max(np.linalg.norm(self.geometry.body_b_m, axis=1)))
        return self.frame_clearance_m + radius * self.attitude_error_rad


@dataclass(frozen=True)
class LandingMission:
    """First-contact landing on a finite planar platform."""

    approach_domain: ApproachDomain
    free_space: FreeSpace
    geometry: RigidBodyGeometry
    center_w_m: npt.ArrayLike
    length_axis_w: npt.ArrayLike
    width_axis_w: npt.ArrayLike
    length_m: float
    width_m: float
    height_bounds_m: tuple[float, float]
    roll_abs_max_rad: float
    touchdown_pitch_rad: float
    pitch_error_abs_max_rad: float
    normal_speed_max_m_s: float
    tangential_speed_max_m_s: float
    platform_clearance_m: float
    body_inflation_m: float
    position_error_m: float
    attitude_error_rad: float
    _terminal_halfspaces: Halfspaces = field(init=False, repr=False)

    def __post_init__(self) -> None:
        length, width, _ = orthogonal_axes(
            self.length_axis_w,
            self.width_axis_w,
        )
        object.__setattr__(self, "center_w_m", _immutable(self.center_w_m, (3,)))
        object.__setattr__(self, "length_axis_w", _immutable(length, (3,)))
        object.__setattr__(self, "width_axis_w", _immutable(width, (3,)))

        clearance = self._clearance
        tangent_limit = self.tangential_speed_max_m_s / sqrt(2.0)
        velocity_lower = np.tile(
            (0.0, -tangent_limit, -tangent_limit),
            self.geometry.contact_b_m.shape[0],
        )
        velocity_upper = np.tile(
            (self.normal_speed_max_m_s, tangent_limit, tangent_limit),
            self.geometry.contact_b_m.shape[0],
        )
        lower = np.concatenate(
            (
                (
                    -0.5 * self.length_m + clearance,
                    -0.5 * self.width_m + clearance,
                    self.height_bounds_m[0],
                    -self.roll_abs_max_rad,
                    -self.pitch_error_abs_max_rad,
                ),
                velocity_lower,
            )
        )
        upper = np.concatenate(
            (
                (
                    0.5 * self.length_m - clearance,
                    0.5 * self.width_m - clearance,
                    self.height_bounds_m[1],
                    self.roll_abs_max_rad,
                    self.pitch_error_abs_max_rad,
                ),
                velocity_upper,
            )
        )
        object.__setattr__(self, "_terminal_halfspaces", Halfspaces.box(lower, upper))

    @property
    def terminal_halfspaces(self) -> Halfspaces:
        """Return platform, attitude, and contact-velocity limits."""

        return self._terminal_halfspaces

    @property
    def free_space_halfspaces(self) -> Halfspaces:
        """Return planar complete-body free-space constraints."""

        return self.free_space.halfspaces

    def error(self, state: npt.ArrayLike) -> np.ndarray:
        """Return platform, attitude, and every contact-point velocity."""

        value = as_state(state)
        normal = np.cross(self.length_axis_w, self.width_axis_w)
        offset = value[:3] - self.center_w_m
        velocity = point_velocities(value, self.geometry.contact_b_m)
        contact_velocity = np.column_stack(
            (
                -velocity @ normal,
                velocity @ self.length_axis_w,
                velocity @ self.width_axis_w,
            )
        ).reshape(-1)
        return np.concatenate(
            (
                (
                    offset @ self.length_axis_w,
                    offset @ self.width_axis_w,
                    offset @ normal,
                    value[3],
                    value[4] - self.touchdown_pitch_rad,
                ),
                contact_velocity,
            )
        )

    def distance(self, state: npt.ArrayLike) -> float:
        """Return distance along the declared platform approach axis."""

        position = as_state(state)[:3]
        return float(-self.length_axis_w @ (position - self.center_w_m))

    def realized(self, states: npt.ArrayLike) -> bool:
        """Evaluate admissible first contact from measured states."""

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

    def terminal_support_constraints(
        self,
        prediction: Prediction,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative affine rows for robust first contact."""

        stages = prediction.body_center.shape[0]
        last = stages - 1
        normal = np.cross(self.length_axis_w, self.width_axis_w)
        forbidden = _noncontact_indices(
            self.geometry.body_b_m,
            self.geometry.contact_b_m,
        )
        body_count = self.geometry.body_b_m.shape[0]
        contact_count = self.geometry.contact_b_m.shape[0]
        footprint_count = self.geometry.footprint_b_m.shape[0]
        terminal = self.terminal_halfspaces
        state_rows = np.flatnonzero(np.any(terminal.matrix[:, :5] != 0.0, axis=1))
        row_count = (
            last * body_count
            + contact_count
            + forbidden.size
            + 4 * footprint_count
            + 6 * contact_count
            + 2 * state_rows.size
        )
        matrix = np.empty((row_count, 3))
        bounds = np.empty(row_count)
        row = 0

        for stage in range(last):
            for point in range(body_count):
                row = _point_row(
                    prediction.body_support,
                    stage,
                    point,
                    -normal,
                    self.center_w_m,
                    -self._support_clearance,
                    matrix,
                    bounds,
                    row,
                )
        for point in self.geometry.contact_b_m:
            row = _body_state_row(
                prediction,
                self.approach_domain,
                point.reshape(1, 3),
                stages,
                normal,
                self.center_w_m,
                self._clearance,
                matrix,
                bounds,
                row,
            )
        for point in forbidden:
            row = _point_row(
                prediction.body_support,
                last,
                int(point),
                -normal,
                self.center_w_m,
                -self._support_clearance,
                matrix,
                bounds,
                row,
            )

        footprint_axes = (
            self.length_axis_w,
            -self.length_axis_w,
            self.width_axis_w,
            -self.width_axis_w,
        )
        footprint_limits = (
            0.5 * self.length_m - self._support_clearance,
            0.5 * self.length_m - self._support_clearance,
            0.5 * self.width_m - self._support_clearance,
            0.5 * self.width_m - self._support_clearance,
        )
        for axis, limit in zip(footprint_axes, footprint_limits, strict=True):
            for point in range(footprint_count):
                row = _point_row(
                    prediction.footprint_support,
                    last,
                    point,
                    axis,
                    self.center_w_m,
                    limit,
                    matrix,
                    bounds,
                    row,
                )

        tangent_limit = self.tangential_speed_max_m_s / sqrt(2.0)
        velocity_axes = (
            (normal, 0.0),
            (-normal, self.normal_speed_max_m_s),
            (self.length_axis_w, tangent_limit),
            (-self.length_axis_w, tangent_limit),
            (self.width_axis_w, tangent_limit),
            (-self.width_axis_w, tangent_limit),
        )
        for point in range(contact_count):
            for axis, limit in velocity_axes:
                row = _point_row(
                    prediction.contact_velocity_support,
                    last,
                    point,
                    axis,
                    np.zeros(3),
                    limit,
                    matrix,
                    bounds,
                    row,
                )
        for stage in (last, stages):
            for terminal_row in state_rows:
                offset, reference = error_support(
                    self,
                    prediction,
                    stage,
                    terminal.matrix[terminal_row],
                )
                matrix[row] = reference
                bounds[row] = terminal.bounds[terminal_row] - offset
                row += 1
        return matrix, bounds

    @property
    def _clearance(self) -> float:
        radius = float(np.max(np.linalg.norm(self.geometry.body_b_m, axis=1)))
        return (
            self.platform_clearance_m
            + self.body_inflation_m
            + self.position_error_m
            + radius * self.attitude_error_rad
        )

    @property
    def _support_clearance(self) -> float:
        radius = float(np.max(np.linalg.norm(self.geometry.body_b_m, axis=1)))
        return self.platform_clearance_m + radius * self.attitude_error_rad


def error_support(
    mission: Mission,
    prediction: Prediction,
    stage: int,
    facet: npt.ArrayLike,
) -> tuple[float, np.ndarray]:
    """Return an affine upper support bound for one terminal-error facet."""

    direction = np.zeros(15)
    if isinstance(mission, GateMission):
        row = np.asarray(facet, dtype=float).reshape(6)
        height = np.cross(mission.normal_w, mission.width_axis_w)
        position_axis = row[0] * mission.width_axis_w + row[1] * height
        direction[:3] = position_axis
        direction[3] = row[3]
        direction[4] = row[4]
        direction[5] = -row[2]
        offset, reference = prediction.state_support(stage, direction)
        offset -= float(position_axis @ mission.center_w_m)
        offset += row[2] * _gate_yaw(mission)
        offset -= row[5] * mission.target_airspeed_m_s
        offset += abs(row[2]) * _heading_residual(mission)
        speed_offset, speed_reference = prediction.airspeed_support(
            stage,
            row[5],
        )
        offset += speed_offset
        reference += speed_reference
        return offset, reference

    landing = cast(LandingMission, mission)
    contact_count = landing.geometry.contact_b_m.shape[0]
    row = np.asarray(facet, dtype=float).reshape(5 + 3 * contact_count)
    normal = np.cross(landing.length_axis_w, landing.width_axis_w)
    position_axis = (
        row[0] * landing.length_axis_w + row[1] * landing.width_axis_w + row[2] * normal
    )
    direction[:3] = position_axis
    direction[3] = row[3]
    direction[4] = row[4]
    offset, reference = prediction.state_support(stage, direction)
    offset -= float(position_axis @ landing.center_w_m)
    offset -= row[4] * landing.touchdown_pitch_rad
    for point, velocity_row in enumerate(row[5:].reshape(contact_count, 3)):
        if not np.any(velocity_row):
            continue
        velocity_direction = (
            -velocity_row[0] * normal
            + velocity_row[1] * landing.length_axis_w
            + velocity_row[2] * landing.width_axis_w
        )
        velocity_offset, velocity_reference = prediction.contact_velocity_support(
            stage,
            point,
            velocity_direction,
        )
        offset += velocity_offset
        reference += velocity_reference
    return offset, reference


def preterminal_support_constraints(
    mission: Mission,
    prediction: Prediction,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the swept body strictly before the mission contact surface."""

    if isinstance(mission, GateMission):
        axis = mission.normal_w
        center = mission.center_w_m
        clearance = mission._support_clearance
    else:
        landing = cast(LandingMission, mission)
        axis = -np.cross(landing.length_axis_w, landing.width_axis_w)
        center = landing.center_w_m
        clearance = landing._support_clearance
    stages = prediction.body_center.shape[0]
    body_count = prediction.body_count
    matrix = np.empty((stages * body_count, 3))
    bounds = np.empty(stages * body_count)
    row = 0
    for stage in range(stages):
        for point in range(body_count):
            row = _point_row(
                prediction.body_support,
                stage,
                point,
                axis,
                center,
                -clearance,
                matrix,
                bounds,
                row,
            )
    return matrix, bounds


def distance_support(
    mission: Mission,
    prediction: Prediction,
    stage: int,
    sign: float,
) -> tuple[float, np.ndarray]:
    """Return the exact affine support of signed approach distance."""

    if isinstance(mission, GateMission):
        axis = mission.normal_w
    else:
        axis = cast(LandingMission, mission).length_axis_w
    direction = np.zeros(15)
    direction[:3] = -sign * axis
    offset, reference = prediction.state_support(stage, direction)
    if isinstance(mission, GateMission):
        center = mission.center_w_m
    else:
        center = cast(LandingMission, mission).center_w_m
    return offset + sign * float(axis @ center), reference


def _body_state_row(
    prediction: Prediction,
    domain: ApproachDomain,
    points_b_m: np.ndarray,
    stage: int,
    axis_w: np.ndarray,
    origin_w_m: np.ndarray,
    limit: float,
    matrix: np.ndarray,
    bounds: np.ndarray,
    row: int,
) -> int:
    direction = np.zeros(15)
    direction[:3] = axis_w
    offset, reference = prediction.state_support(stage, direction)
    support = _orientation_support(domain, points_b_m, axis_w)
    matrix[row] = reference
    bounds[row] = limit + axis_w @ origin_w_m - support - offset
    return row + 1


def _point_row(
    support: Callable[[int, int, npt.ArrayLike], tuple[float, np.ndarray]],
    stage: int,
    point: int,
    axis_w: np.ndarray,
    origin_w_m: np.ndarray,
    limit: float,
    matrix: np.ndarray,
    bounds: np.ndarray,
    row: int,
) -> int:
    offset, reference = support(stage, point, axis_w)
    matrix[row] = reference
    bounds[row] = limit + axis_w @ origin_w_m - offset
    return row + 1


def _orientation_support(
    domain: ApproachDomain,
    points_b_m: np.ndarray,
    axis_w: np.ndarray,
) -> float:
    rotation = body_to_world(domain.center[3:6])
    nominal = points_b_m @ rotation.T @ axis_w
    angle = min(float(np.sum(domain.radius[3:6])), 2.0)
    residual = np.linalg.norm(points_b_m, axis=1) * angle
    return float(np.max(nominal + residual))


def _noncontact_indices(body: np.ndarray, contact: np.ndarray) -> np.ndarray:
    permitted = np.any(
        np.all(body[:, None] == contact[None, :], axis=2),
        axis=1,
    )
    return np.flatnonzero(~permitted)


def _gate_yaw(mission: GateMission) -> float:
    return float(np.arctan2(-mission.normal_w[1], mission.normal_w[0]))


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _heading_residual(mission: GateMission) -> float:
    desired = _gate_yaw(mission)
    lower = desired - mission.approach_domain.upper[5]
    upper = desired - mission.approach_domain.lower[5]
    return 0.0 if lower >= -pi and upper <= pi else 2.0 * pi


def _immutable(value: npt.ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(shape).copy()
    array.flags.writeable = False
    return array
