from __future__ import annotations

import numpy as np

from moewe.sim.frames import (
    GRAVITY_M_S2,
    body_to_world,
    gravity_body,
    rotation_body_to_world,
    world_to_body,
)


def test_level_axis_alignment() -> None:
    rotation = rotation_body_to_world(0.0, 0.0, 0.0)

    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotation @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(rotation @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0])


def test_body_world_round_trip() -> None:
    euler = np.array([0.2, -0.1, 0.3])
    vector_b = np.array([3.0, -2.0, 1.0])

    vector_w = body_to_world(vector_b, euler)

    np.testing.assert_allclose(world_to_body(vector_w, euler), vector_b, atol=1e-12)


def test_gravity_sign_in_level_body_frame() -> None:
    np.testing.assert_allclose(gravity_body(np.zeros(3)), [0.0, 0.0, GRAVITY_M_S2])
