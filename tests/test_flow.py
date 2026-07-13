"""Tests for body-relative affine flow and stripwise aircraft loads."""

from __future__ import annotations

from unittest import TestCase

import numpy as np

from control.flow import AffineFlow, FlowBounds
from models.aircraft import Aircraft
from models.geometry import body_to_world


class FlowTests(TestCase):
    """Verify affine-flow evaluation and aircraft flow interfaces."""

    def test_affine_strip_flow(self) -> None:
        """Evaluate value, gradient, and strip remainder together."""

        locations = np.array(
            [[0.2, -0.3, 0.1], [-0.4, 0.5, -0.2]],
            dtype=float,
        )
        center = np.array([1.0, -2.0, 0.5])
        gradient = np.array([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2], [-0.3, 0.1, 0.6]])
        remainder = np.array([[0.01, 0.02, -0.03], [-0.04, 0.05, 0.06]])
        flow = AffineFlow(center, gradient, remainder)

        np.testing.assert_allclose(
            flow.strip_flow(locations),
            center + locations @ gradient.T + remainder,
        )

    def test_flow_bounds_include_simultaneous_extrema(self) -> None:
        """Include joint value, gradient, and remainder extremes."""

        locations = np.array([[1.0, 2.0, 3.0], [2.0, 0.5, 1.0]])
        center_lower = np.array([-2.0, -1.0, 0.0])
        center_upper = np.array([1.0, 2.0, 3.0])
        gradient_lower = np.array(
            [[-0.3, -0.2, -0.1], [-0.4, -0.3, -0.2], [-0.5, -0.4, -0.3]]
        )
        gradient_upper = np.array([[0.2, 0.3, 0.4], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        remainder_abs = np.array([0.1, 0.2, 0.3])
        bounds = FlowBounds(
            center_lower,
            center_upper,
            gradient_lower,
            gradient_upper,
            remainder_abs,
            np.ones(3),
        )
        remainder = np.broadcast_to(remainder_abs, locations.shape)
        lower_flow = AffineFlow(center_lower, gradient_lower, -remainder)
        upper_flow = AffineFlow(center_upper, gradient_upper, remainder)

        lower, upper = bounds.strip_bounds(locations)

        np.testing.assert_allclose(lower, lower_flow.strip_flow(locations))
        np.testing.assert_allclose(upper, upper_flow.strip_flow(locations))
        self.assertTrue(bounds.contains(lower_flow))
        self.assertTrue(bounds.contains(upper_flow))

    def test_flow_rate_checks_every_affine_component(self) -> None:
        """Bound temporal changes in value, gradient, and remainder."""

        remainder = np.zeros((2, 3))
        previous = AffineFlow(np.zeros(3), np.zeros((3, 3)), remainder)
        bounds = FlowBounds(
            (-1.0,) * 3,
            (1.0,) * 3,
            np.full((3, 3), -1.0),
            np.full((3, 3), 1.0),
            (1.0,) * 3,
            (2.0,) * 3,
            gradient_rate_abs_s2=np.full((3, 3), 3.0),
            remainder_rate_abs_m_s2=(4.0,) * 3,
        )
        current = AffineFlow(
            np.full(3, 0.2),
            np.full((3, 3), 0.3),
            np.full_like(remainder, 0.4),
        )
        self.assertTrue(bounds.rate_contains(previous, current, 0.1))
        self.assertFalse(bounds.rate_contains(previous, current, 0.05))

    def test_local_flow_intrinsic_equivalence(self) -> None:
        """Remove absolute position and global yaw from intrinsic dynamics."""

        aircraft = Aircraft()
        first = np.zeros(15)
        first[:3] = (1.0, 1.5, 2.0)
        first[3:6] = (0.12, -0.08, 0.35)
        first[6:9] = (5.8, 0.2, 0.4)
        first[9:12] = (0.1, -0.15, 0.08)
        first[12:15] = (0.02, -0.03, 0.01)
        second = first.copy()
        second[:3] = (5.5, 3.8, 1.1)
        second[5] = -1.1
        center = np.array([0.4, -0.2, 0.3])
        gradient = np.zeros((3, 3))
        gradient[2, 1] = 0.5
        strips = AffineFlow(
            center,
            gradient,
            np.zeros_like(aircraft.strip_table.r_b_m),
        ).strip_flow(aircraft.strip_table.r_b_m)
        control = np.array([0.08, -0.12, 0.04])

        first_dot = aircraft.derivative_local_flow(first, control, center, strips)
        second_dot = aircraft.derivative_local_flow(second, control, center, strips)

        np.testing.assert_allclose(first_dot[3:], second_dot[3:])
        self.assertFalse(np.allclose(first_dot[:3], second_dot[:3]))

    def test_world_field_adapter_matches_direct_local_loads(self) -> None:
        """Match world-field sampling to direct body-relative loads."""

        aircraft = Aircraft()
        state = np.zeros(15)
        state[:3] = (2.0, 1.2, 1.8)
        state[3:6] = (0.15, -0.1, 0.4)
        state[6:9] = (6.0, -0.1, 0.3)
        state[9:12] = (0.05, -0.08, 0.03)
        state[12:15] = (0.01, -0.02, 0.015)
        world_flow = np.array([1.2, -0.4, 0.7])
        world_gradient = np.array(
            ((0.1, -0.2, 0.0), (0.05, 0.0, 0.1), (0.0, 0.3, -0.1))
        )
        rotation = body_to_world(state[3:6])
        expected_body = rotation.T @ world_flow

        def field(points_w_m: np.ndarray) -> np.ndarray:
            return world_flow + (points_w_m - state[:3]) @ world_gradient.T

        sampled_center, sampled_strips = aircraft.sample_local_flow(state, field)
        strip_positions = state[:3] + aircraft.strip_table.r_b_m @ rotation.T
        direct_strips = field(strip_positions) @ rotation
        sampled_loads = aircraft.aero_loads_local_flow(
            state,
            sampled_center,
            sampled_strips,
        )
        direct_loads = aircraft.aero_loads_local_flow(
            state,
            expected_body,
            direct_strips,
        )
        control = np.array([0.05, -0.1, 0.02])

        np.testing.assert_allclose(sampled_center, expected_body)
        np.testing.assert_allclose(sampled_strips, direct_strips)
        np.testing.assert_allclose(sampled_loads[0], direct_loads[0])
        np.testing.assert_allclose(sampled_loads[1], direct_loads[1])
        np.testing.assert_allclose(
            aircraft(state, control, field),
            aircraft.derivative_local_flow(
                state,
                control,
                expected_body,
                direct_strips,
            ),
        )

    def test_lateral_vertical_gradient_creates_asymmetric_loads(self) -> None:
        """Produce lateral force and roll/yaw moments from distributed flow."""

        aircraft = Aircraft()
        state = np.zeros(15)
        state[6] = 6.0
        locations = aircraft.strip_table.r_b_m
        gradient = np.zeros((3, 3))
        gradient[2, 1] = 3.0
        positive_flow = AffineFlow(
            np.zeros(3),
            gradient,
            np.zeros_like(locations),
        ).strip_flow(locations)
        negative_flow = AffineFlow(
            np.zeros(3),
            -gradient,
            np.zeros_like(locations),
        ).strip_flow(locations)
        positive_force, positive_moment = aircraft.aero_loads_local_flow(
            state,
            np.zeros(3),
            positive_flow,
        )
        negative_force, negative_moment = aircraft.aero_loads_local_flow(
            state,
            np.zeros(3),
            negative_flow,
        )
        center_force, center_moment = aircraft.aero_loads_local_flow(
            state,
            np.zeros(3),
            np.zeros_like(locations),
        )
        half_wing = aircraft.config.surfaces[0].strip_count

        self.assertTrue(np.all(positive_flow[:half_wing, 2] > 0.0))
        self.assertTrue(np.all(positive_flow[half_wing : 2 * half_wing, 2] < 0.0))
        self.assertGreater(abs(float(positive_force[1])), 0.01)
        self.assertGreater(abs(float(positive_moment[0])), 0.01)
        self.assertGreater(abs(float(positive_moment[2])), 0.001)
        self.assertAlmostEqual(float(center_force[1]), 0.0, places=12)
        self.assertAlmostEqual(float(center_moment[0]), 0.0, places=12)
        self.assertAlmostEqual(float(center_moment[2]), 0.0, places=12)
        np.testing.assert_allclose(positive_force[1], -negative_force[1])
        np.testing.assert_allclose(positive_moment[0], -negative_moment[0])
        np.testing.assert_allclose(positive_moment[2], -negative_moment[2])
