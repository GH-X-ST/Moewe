from __future__ import annotations

import numpy as np

from moewe.sim.updraft import AnnularUpdraft, FanUpdraft, HeightProfile


def test_single_fan_peak_is_near_annular_radius() -> None:
    fan = FanUpdraft(
        centre_xy_m=(0.0, 0.0),
        strength_m_s=2.0,
        ring_radius_m=1.0,
        ring_thickness_m=0.2,
    )
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    values = fan.vertical_speed(points)

    assert np.argmax(values) == 1


def test_multi_fan_equals_sum_of_single_fans() -> None:
    fan_a = FanUpdraft((0.0, 0.0), 2.0, 0.5, 0.2)
    fan_b = FanUpdraft((1.0, 0.0), 1.0, 0.3, 0.2)
    model = AnnularUpdraft.from_fans([fan_a, fan_b])
    points = np.array([[0.2, 0.1, 0.0], [0.8, 0.0, 0.0]])

    np.testing.assert_allclose(
        model.vertical_speed_at(points),
        fan_a.vertical_speed(points) + fan_b.vertical_speed(points),
    )


def test_vertical_decay_is_monotonic_for_positive_height() -> None:
    fan = FanUpdraft(
        centre_xy_m=(0.0, 0.0),
        strength_m_s=2.0,
        ring_radius_m=1.0,
        ring_thickness_m=0.2,
        vertical_decay_m=1.0,
    )
    points = np.array([[1.0, 0.0, z] for z in [0.0, 1.0, 2.0]])

    values = fan.vertical_speed(points)

    assert values[0] > values[1] > values[2]


def test_height_profile_interpolation_and_vector_shapes() -> None:
    fan = FanUpdraft(
        centre_xy_m=(0.0, 0.0),
        strength_m_s=HeightProfile((0.0, 1.0), (2.0, 1.0)),
        ring_radius_m=1.0,
        ring_thickness_m=0.2,
    )
    model = AnnularUpdraft.from_fans([fan])
    points = np.zeros((2, 2, 3))
    points[..., 0] = 1.0
    points[..., 2] = [[0.0, 0.5], [1.0, 0.25]]

    vertical = model.vertical_speed_at(points)
    velocity = model.velocity_at(points)

    assert vertical.shape == (2, 2)
    assert velocity.shape == (2, 2, 3)
    assert vertical[0, 0] > vertical[0, 1] > vertical[1, 0]


def test_seeded_perturbations_are_reproducible() -> None:
    model = AnnularUpdraft.from_fans(
        [FanUpdraft((0.0, 0.0), 2.0, 1.0, 0.2, fluctuation_std_m_s=0.1)]
    )
    points = np.array([[1.0, 0.0, 0.0], [0.8, 0.0, 0.0]])

    a = model.vertical_speed_at(points, seed=123, include_perturbation=True)
    b = model.vertical_speed_at(points, seed=123, include_perturbation=True)

    np.testing.assert_allclose(a, b)
