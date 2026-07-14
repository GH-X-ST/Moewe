"""Rigid-body occupancy, contact, and terminal-event geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, sin

import numpy as np
import numpy.typing as npt

from models.state import as_state


@dataclass(frozen=True)
class RigidBodyGeometry:
    """Explicit full-body, permitted-contact, and footprint vertices."""

    body_b_m: npt.ArrayLike
    contact_b_m: npt.ArrayLike
    footprint_b_m: npt.ArrayLike

    def __post_init__(self) -> None:
        arrays = {}
        for name in ("body_b_m", "contact_b_m", "footprint_b_m"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(-1, 3)
            if value.size == 0 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite body points")
            value = value.copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
            arrays[name] = value

        body = arrays["body_b_m"]
        for name in ("contact_b_m", "footprint_b_m"):
            members = np.any(
                np.all(arrays[name][:, None] == body[None, :], axis=2),
                axis=1,
            )
            if not np.all(members):
                raise ValueError(f"{name} must be a subset of body_b_m")


def world_points(state: npt.ArrayLike, points_b_m: npt.ArrayLike) -> np.ndarray:
    """Return body-fixed points in the public world frame."""

    value = as_state(state)
    points = np.asarray(points_b_m, dtype=float).reshape(-1, 3)
    return value[:3] + points @ body_to_world(value[3:6]).T


def point_velocities(
    state: npt.ArrayLike,
    points_b_m: npt.ArrayLike,
) -> np.ndarray:
    """Return world velocities of body-fixed points."""

    value = as_state(state)
    points = np.asarray(points_b_m, dtype=float).reshape(-1, 3)
    velocity_b = value[6:9] + np.cross(value[9:12], points)
    return velocity_b @ body_to_world(value[3:6]).T


def gate_crossing(
    states: npt.ArrayLike,
    geometry: RigidBodyGeometry,
    center_w_m: npt.ArrayLike,
    normal_w: npt.ArrayLike,
    width_axis_w: npt.ArrayLike,
    width_m: float,
    height_m: float,
    frame_clearance_m: float,
    body_inflation_m: float,
    position_error_m: float,
    attitude_error_rad: float,
) -> bool:
    """Return whether a dense realized trajectory clears a gate."""

    trajectory = _trajectory(states)
    body = geometry.body_b_m
    center = np.asarray(center_w_m, dtype=float).reshape(3)
    normal, width_axis, height_axis = orthogonal_axes(normal_w, width_axis_w)
    radius = float(np.max(np.linalg.norm(body, axis=1)))
    clearance = (
        frame_clearance_m
        + body_inflation_m
        + position_error_m
        + radius * attitude_error_rad
    )
    half_width = 0.5 * width_m - clearance
    half_height = 0.5 * height_m - clearance
    swept = np.stack([world_points(state, body) for state in trajectory])
    plane_distance = (swept - center) @ normal

    if np.max(plane_distance[0]) + clearance >= 0.0:
        return False
    if np.min(plane_distance[-1]) - clearance <= 0.0:
        return False
    progress = (trajectory[-1, :3] - trajectory[0, :3]) @ normal
    if progress <= 0.0:
        return False

    crossing = False
    for index in range(trajectory.shape[0] - 1):
        distance = plane_distance[index : index + 2]
        if np.min(distance) > clearance or np.max(distance) < -clearance:
            continue
        crossing = True
        offsets = swept[index : index + 2] - center
        if np.max(np.abs(offsets @ width_axis)) > half_width:
            return False
        if np.max(np.abs(offsets @ height_axis)) > half_height:
            return False
    return crossing


def platform_landing(
    states: npt.ArrayLike,
    geometry: RigidBodyGeometry,
    center_w_m: npt.ArrayLike,
    length_axis_w: npt.ArrayLike,
    width_axis_w: npt.ArrayLike,
    length_m: float,
    width_m: float,
    normal_speed_max_m_s: float,
    tangential_speed_max_m_s: float,
    roll_abs_max_rad: float,
    pitch_error_abs_max_rad: float,
    touchdown_pitch_rad: float,
    platform_clearance_m: float,
    body_inflation_m: float,
    position_error_m: float,
    attitude_error_rad: float,
) -> bool:
    """Return whether event-located first contact is admissible."""

    trajectory = _trajectory(states)
    body = geometry.body_b_m
    contact = geometry.contact_b_m
    footprint = geometry.footprint_b_m
    center = np.asarray(center_w_m, dtype=float).reshape(3)
    length_axis, width_axis, normal = orthogonal_axes(
        length_axis_w,
        width_axis_w,
    )
    radius = float(np.max(np.linalg.norm(body, axis=1)))
    clearance = (
        platform_clearance_m
        + body_inflation_m
        + position_error_m
        + radius * attitude_error_rad
    )

    heights = np.stack(
        [(world_points(state, contact) - center) @ normal for state in trajectory]
    )
    minimum = np.min(heights, axis=1)
    if minimum[0] <= clearance:
        return False
    candidates = np.flatnonzero((minimum[:-1] > clearance) & (minimum[1:] <= clearance))
    if candidates.size == 0:
        return False
    segment = int(candidates[0])

    forbidden = _noncontact_points(body, contact)
    for state in trajectory[: segment + 1]:
        if forbidden.size:
            forbidden_height = (world_points(state, forbidden) - center) @ normal
            if np.min(forbidden_height) <= clearance:
                return False

    touchdown = _contact_state(
        trajectory[segment],
        trajectory[segment + 1],
        contact,
        center,
        normal,
        clearance,
    )
    contact_height = (world_points(touchdown, contact) - center) @ normal
    first_contact = np.isclose(
        contact_height,
        np.min(contact_height),
        atol=1.0e-9,
        rtol=0.0,
    )
    if forbidden.size:
        forbidden_height = (world_points(touchdown, forbidden) - center) @ normal
        if np.min(forbidden_height) <= clearance:
            return False

    contact_velocity = point_velocities(touchdown, contact[first_contact])
    normal_speed = -(contact_velocity @ normal)
    tangent = contact_velocity + normal_speed[:, None] * normal
    footprint_offset = world_points(touchdown, footprint) - center
    roll, pitch, _ = relative_attitude(
        touchdown[3:6],
        length_axis,
        width_axis,
    )
    return bool(
        np.max(np.abs(footprint_offset @ length_axis)) <= 0.5 * length_m - clearance
        and np.max(np.abs(footprint_offset @ width_axis)) <= 0.5 * width_m - clearance
        and np.min(normal_speed) >= 0.0
        and np.max(normal_speed) <= normal_speed_max_m_s
        and np.max(np.linalg.norm(tangent, axis=1)) <= tangential_speed_max_m_s
        and abs(roll) + attitude_error_rad <= roll_abs_max_rad
        and abs(pitch - touchdown_pitch_rad) + attitude_error_rad
        <= pitch_error_abs_max_rad
    )


def body_to_world(attitude_rad: npt.ArrayLike) -> np.ndarray:
    """Return the body-to-world rotation for the right-handed z-up frame."""

    phi, theta, psi = np.asarray(attitude_rad, dtype=float).reshape(3)
    c_phi, s_phi = cos(phi), sin(phi)
    c_theta, s_theta = cos(theta), sin(theta)
    c_psi, s_psi = cos(psi), sin(psi)
    return np.array(
        [
            [
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ],
            [
                -c_theta * s_psi,
                -s_phi * s_theta * s_psi - c_phi * c_psi,
                -c_phi * s_theta * s_psi + s_phi * c_psi,
            ],
            [s_theta, -s_phi * c_theta, -c_phi * c_theta],
        ]
    )


def relative_attitude(
    attitude_rad: npt.ArrayLike,
    forward_axis_w: npt.ArrayLike,
    lateral_axis_w: npt.ArrayLike,
) -> tuple[float, float, float]:
    """Return roll, pitch, and heading relative to a mission frame."""

    forward, lateral, normal = orthogonal_axes(
        forward_axis_w,
        lateral_axis_w,
    )
    desired = np.column_stack((forward, -lateral, -normal))
    relative = desired.T @ body_to_world(attitude_rad)
    pitch = asin(float(np.clip(-relative[2, 0], -1.0, 1.0)))
    roll = atan2(float(relative[2, 1]), float(relative[2, 2]))
    heading = atan2(float(relative[1, 0]), float(relative[0, 0]))
    return roll, pitch, heading


def orthogonal_axes(
    first_axis: npt.ArrayLike,
    second_axis: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized right-handed axes from two input directions."""

    first = np.asarray(first_axis, dtype=float).reshape(3)
    first = first / np.linalg.norm(first)
    second = np.asarray(second_axis, dtype=float).reshape(3)
    second = second - float(second @ first) * first
    second = second / np.linalg.norm(second)
    return first, second, np.cross(first, second)


def _trajectory(states: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 15:
        raise ValueError("realized trajectory must contain at least two states")
    return values


def _noncontact_points(body: np.ndarray, contact: np.ndarray) -> np.ndarray:
    permitted = np.any(
        np.all(body[:, None] == contact[None, :], axis=2),
        axis=1,
    )
    return body[~permitted]


def _contact_state(
    first: np.ndarray,
    second: np.ndarray,
    contact: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    clearance: float,
) -> np.ndarray:
    lower = 0.0
    upper = 1.0
    for _ in range(48):
        fraction = 0.5 * (lower + upper)
        state = first + fraction * (second - first)
        height: float = float(np.min((world_points(state, contact) - center) @ normal))
        if height > clearance:
            lower = fraction
        else:
            upper = fraction
    return first + upper * (second - first)
