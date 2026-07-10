"""Rigid-body occupancy and contact geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np
import numpy.typing as npt

from models.state import as_state


Vector3 = tuple[float, float, float]
_ORIGIN = ((0.0, 0.0, 0.0),)
_INTERPOLATION_RATIOS = np.linspace(0.0, 1.0, 65)


@dataclass(frozen=True)
class RigidBodyGeometry:
    """Finite-vertex body, contact, and landing-footprint sets."""

    body_b_m: npt.ArrayLike = _ORIGIN
    contact_b_m: npt.ArrayLike = _ORIGIN
    footprint_b_m: npt.ArrayLike = _ORIGIN


def world_points(state: npt.ArrayLike, points_b_m: npt.ArrayLike) -> np.ndarray:
    """Return body-fixed points in the public world frame."""

    x = as_state(state)
    points = np.asarray(points_b_m, dtype=float).reshape(-1, 3)
    return x[:3] + points @ body_to_world(x[3:6]).T


def point_velocities(
    state: npt.ArrayLike,
    points_b_m: npt.ArrayLike,
) -> np.ndarray:
    """Return world velocities of body-fixed points."""

    x = as_state(state)
    points = np.asarray(points_b_m, dtype=float).reshape(-1, 3)
    velocity_b = x[6:9] + np.cross(x[9:12], points)
    return velocity_b @ body_to_world(x[3:6]).T


def gate_crossing(
    previous_state: npt.ArrayLike,
    state: npt.ArrayLike,
    geometry: RigidBodyGeometry,
    center_w_m: Vector3,
    normal_w: Vector3,
    width_axis_w: Vector3,
    width_m: float,
    height_m: float,
    margin_m: float,
) -> bool:
    """Return whether the swept occupied body crosses a gate aperture."""

    previous = as_state(previous_state)
    current = as_state(state)
    state_delta = current - previous
    body = np.asarray(geometry.body_b_m, dtype=float).reshape(-1, 3)
    center = np.asarray(center_w_m, dtype=float)
    normal = np.asarray(normal_w, dtype=float)
    width_axis = np.asarray(width_axis_w, dtype=float)
    height_axis = np.cross(normal, width_axis)
    previous_distance = float((previous[:3] - center) @ normal)
    current_distance = float((current[:3] - center) @ normal)
    if (
        previous_distance > 0.0
        or current_distance < 0.0
        or current_distance <= previous_distance
    ):
        return False

    crossing_ratio = -previous_distance / (
        current_distance - previous_distance
    )
    ratios = np.unique(np.append(_INTERPOLATION_RATIOS, crossing_ratio))
    swept = np.stack(
        [
            world_points(
                previous + ratio * state_delta,
                body,
            )
            for ratio in ratios
        ]
    )
    plane_distance = (swept - center) @ normal
    intersects = (plane_distance.min(axis=1) <= 0.0) & (
        plane_distance.max(axis=1) >= 0.0
    )
    if not np.any(intersects):
        return False

    offsets = swept[intersects] - center
    return bool(
        np.max(np.abs(offsets @ width_axis))
        <= 0.5 * width_m - margin_m
        and np.max(np.abs(offsets @ height_axis))
        <= 0.5 * height_m - margin_m
    )


def platform_landing(
    previous_state: npt.ArrayLike,
    state: npt.ArrayLike,
    geometry: RigidBodyGeometry,
    center_w_m: Vector3,
    length_axis_w: Vector3,
    width_axis_w: Vector3,
    length_m: float,
    width_m: float,
    normal_speed_max_m_s: float,
    contact_speed_max_m_s: float,
    roll_max_rad: float,
    pitch_bounds_rad: tuple[float, float],
    margin_m: float,
) -> bool:
    """Return whether first contact satisfies the platform constraints."""

    previous = as_state(previous_state)
    current = as_state(state)
    state_delta = current - previous
    contact = np.asarray(geometry.contact_b_m, dtype=float).reshape(-1, 3)
    footprint_b = np.asarray(geometry.footprint_b_m, dtype=float).reshape(-1, 3)
    center = np.asarray(center_w_m, dtype=float)
    length_axis = np.asarray(length_axis_w, dtype=float)
    width_axis = np.asarray(width_axis_w, dtype=float)
    normal = np.cross(length_axis, width_axis)

    def contact_height(ratio: float) -> float:
        points = world_points(
            previous + ratio * state_delta,
            contact,
        )
        return float(np.min((points - center) @ normal))

    heights = np.array(
        [contact_height(ratio) for ratio in _INTERPOLATION_RATIOS]
    )
    if heights[0] <= 0.0 or heights[-1] > 0.0:
        return False
    upper_index = int(np.flatnonzero(heights <= 0.0)[0])
    lower = float(_INTERPOLATION_RATIOS[upper_index - 1])
    upper = float(_INTERPOLATION_RATIOS[upper_index])
    for _ in range(40):
        middle = 0.5 * (lower + upper)
        if contact_height(middle) > 0.0:
            lower = middle
        else:
            upper = middle

    touchdown = previous + upper * state_delta
    contact_heights = (world_points(touchdown, contact) - center) @ normal
    first_contact = contact_heights == np.min(contact_heights)
    contact_velocities = point_velocities(
        touchdown,
        contact[first_contact],
    )
    normal_speed = float(np.max(-contact_velocities @ normal))
    contact_speed = float(
        np.max(np.linalg.norm(contact_velocities, axis=1))
    )
    footprint = world_points(touchdown, footprint_b) - center
    phi, theta = touchdown[3:5]
    return bool(
        np.max(np.abs(footprint @ length_axis))
        <= 0.5 * length_m - margin_m
        and np.max(np.abs(footprint @ width_axis))
        <= 0.5 * width_m - margin_m
        and 0.0 <= normal_speed <= normal_speed_max_m_s
        and contact_speed <= contact_speed_max_m_s
        and abs(float(phi)) <= roll_max_rad
        and pitch_bounds_rad[0] <= float(theta) <= pitch_bounds_rad[1]
    )


def body_to_world(attitude_rad: npt.ArrayLike) -> np.ndarray:
    """Return the body-to-world rotation for the right-handed z-up frame."""

    phi, theta, psi = np.asarray(attitude_rad, dtype=float)
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
        ],
        dtype=float,
    )
