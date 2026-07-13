"""Rigid-body occupancy and contact geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np
import numpy.typing as npt

from models.aircraft import LiftingSurface, default_aircraft_config
from models.state import as_state

Vector3 = tuple[float, float, float]


def _surface_vertices(surface: LiftingSurface) -> np.ndarray:
    root = np.asarray(surface.root_le_b_m, dtype=float)
    chord = np.array([-surface.chord_m, 0.0, 0.0])
    if surface.vertical:
        tip = np.array([0.0, 0.0, -surface.span_m])
        return np.array([root, root + chord, root + tip, root + tip + chord])

    semispan = 0.5 * surface.span_m if surface.symmetric else surface.span_m
    vertices = [root, root + chord]
    side_signs = (1.0, -1.0) if surface.symmetric else (1.0,)
    for side_sign in side_signs:
        tip = np.array(
            [
                0.0,
                side_sign * semispan * cos(surface.dihedral_rad),
                -semispan * sin(surface.dihedral_rad),
            ]
        )
        vertices.extend((root + tip, root + tip + chord))
    return np.asarray(vertices)


def _surface_corners() -> np.ndarray:
    surfaces = default_aircraft_config().surfaces
    return np.vstack([_surface_vertices(surface) for surface in surfaces])


_SURFACE_CORNERS_B_M = _surface_corners()
_DEFAULT_BODY_B_M = tuple(
    tuple(float(value) for value in point) for point in _SURFACE_CORNERS_B_M
)
_LOWEST_SURFACE_Z_B_M = float(np.max(_SURFACE_CORNERS_B_M[:, 2]))
_DEFAULT_CONTACT_B_M = tuple(
    tuple(float(value) for value in point)
    for point in _SURFACE_CORNERS_B_M[
        np.isclose(_SURFACE_CORNERS_B_M[:, 2], _LOWEST_SURFACE_Z_B_M)
    ]
)


@dataclass(frozen=True)
class RigidBodyGeometry:
    """Finite body, first-contact, and landing-footprint vertices."""

    body_b_m: npt.ArrayLike = _DEFAULT_BODY_B_M
    contact_b_m: npt.ArrayLike = _DEFAULT_CONTACT_B_M
    footprint_b_m: npt.ArrayLike = _DEFAULT_BODY_B_M

    def __post_init__(self) -> None:
        for name in ("body_b_m", "contact_b_m", "footprint_b_m"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(-1, 3)
            if value.size == 0 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite body points")
            value = value.copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)


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
    states: npt.ArrayLike,
    geometry: RigidBodyGeometry,
    center_w_m: Vector3,
    normal_w: Vector3,
    width_axis_w: Vector3,
    width_m: float,
    height_m: float,
    margin_m: float,
) -> bool:
    """Return whether a realized dense trajectory passes through a gate."""

    trajectory = _trajectory(states)
    body = np.asarray(geometry.body_b_m, dtype=float).reshape(-1, 3)
    center = np.asarray(center_w_m, dtype=float)
    normal = np.asarray(normal_w, dtype=float)
    width_axis = np.asarray(width_axis_w, dtype=float)
    height_axis = np.cross(normal, width_axis)
    swept = np.stack([world_points(state, body) for state in trajectory])
    plane_distance = (swept - center) @ normal
    if np.max(plane_distance[0]) >= 0.0 or np.min(plane_distance[-1]) <= 0.0:
        return False
    if (trajectory[-1, :3] - center) @ normal <= (trajectory[0, :3] - center) @ normal:
        return False
    intersects = (plane_distance.min(axis=1) <= 0.0) & (
        plane_distance.max(axis=1) >= 0.0
    )
    if not np.any(intersects):
        return False

    offsets = swept[intersects] - center
    return bool(
        np.max(np.abs(offsets @ width_axis)) <= 0.5 * width_m - margin_m
        and np.max(np.abs(offsets @ height_axis)) <= 0.5 * height_m - margin_m
    )


def platform_landing(
    states: npt.ArrayLike,
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
    """Return whether realized, event-located first contact is admissible."""

    trajectory = _trajectory(states)
    body = np.asarray(geometry.body_b_m, dtype=float).reshape(-1, 3)
    contact = np.asarray(geometry.contact_b_m, dtype=float).reshape(-1, 3)
    footprint_b = np.asarray(geometry.footprint_b_m, dtype=float).reshape(-1, 3)
    center = np.asarray(center_w_m, dtype=float)
    length_axis = np.asarray(length_axis_w, dtype=float)
    width_axis = np.asarray(width_axis_w, dtype=float)
    normal = np.cross(length_axis, width_axis)

    contact_heights = np.stack(
        [(world_points(state, contact) - center) @ normal for state in trajectory]
    )
    if np.min(contact_heights[0]) <= 0.0:
        return False
    contacts = np.flatnonzero(np.min(contact_heights, axis=1) <= 0.0)
    if contacts.size == 0:
        return False
    contact_index = int(contacts[0])
    touchdown = trajectory[contact_index]
    if any(
        np.min((world_points(state, body) - center) @ normal) <= 0.0
        for state in trajectory[:contact_index]
    ):
        return False
    touchdown_heights = contact_heights[contact_index]
    if np.min(touchdown_heights) < -1.0e-9:
        return False
    first_contact = np.isclose(
        touchdown_heights,
        np.min(touchdown_heights),
        atol=1.0e-9,
        rtol=0.0,
    )
    noncontact = np.array(
        [point for point in body if not np.any(np.all(point == contact, axis=1))]
    )
    if (
        noncontact.size
        and np.min((world_points(touchdown, noncontact) - center) @ normal) <= 0.0
    ):
        return False
    contact_velocities = point_velocities(
        touchdown,
        contact[first_contact],
    )
    normal_speed = float(np.max(-contact_velocities @ normal))
    contact_speed = float(np.max(np.linalg.norm(contact_velocities, axis=1)))
    footprint = world_points(touchdown, footprint_b) - center
    phi, theta = touchdown[3:5]
    return bool(
        np.max(np.abs(footprint @ length_axis)) <= 0.5 * length_m - margin_m
        and np.max(np.abs(footprint @ width_axis)) <= 0.5 * width_m - margin_m
        and 0.0 <= normal_speed <= normal_speed_max_m_s
        and contact_speed <= contact_speed_max_m_s
        and abs(float(phi)) <= roll_max_rad
        and pitch_bounds_rad[0] <= float(theta) <= pitch_bounds_rad[1]
    )


def _trajectory(states: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 15:
        raise ValueError("realized trajectory must contain at least two states")
    return values


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


def orthogonal_axes(
    first_axis: npt.ArrayLike,
    second_axis: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized right-handed axes from two input directions."""

    first = np.asarray(first_axis, dtype=float).reshape(3)
    first_norm = float(np.linalg.norm(first))
    if first_norm == 0.0:
        raise ValueError("first axis must be nonzero")
    first = first / first_norm

    second = np.asarray(second_axis, dtype=float).reshape(3)
    second = second - float(second @ first) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm == 0.0:
        raise ValueError("axes must not be parallel")
    second = second / second_norm
    return first, second, np.cross(first, second)
