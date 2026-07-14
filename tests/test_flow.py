"""Tests for joint local flow and stripwise aircraft loads."""

from __future__ import annotations

from unittest import TestCase

import numpy as np

from control.flow import (
    CENTER_FLOW_GENERATOR_COUNT,
    FLOW_COMPONENT_ORDER,
    GRADIENT_COMPONENT_ORDER,
    GRADIENT_FLOW_GENERATOR_COUNT,
    SHARED_FLOW_GENERATOR_COUNT,
    AffineFlow,
    FlowBounds,
    JointFlow,
)
from control.interval import Interval
from models.aircraft import Aircraft
from models.geometry import body_to_world


class FlowTests(TestCase):
    """Verify the joint-flow factorization and aircraft interfaces."""

    def test_affine_strip_flow(self) -> None:
        """Evaluate value, gradient, and strip remainder together."""

        locations = np.array(((0.2, -0.3, 0.1), (-0.4, 0.5, -0.2)))
        center = np.array((1.0, -2.0, 0.5))
        gradient = np.array(((0.2, -0.1, 0.3), (0.4, 0.5, -0.2), (-0.3, 0.1, 0.6)))
        remainder = np.array(((0.01, 0.02, -0.03), (-0.04, 0.05, 0.06)))

        flow = AffineFlow(center, gradient, remainder)

        np.testing.assert_allclose(
            flow.strip_flow(locations),
            center + locations @ gradient.T + remainder,
        )

    def test_exact_joint_matrix_construction(self) -> None:
        """Construct Lambda centre and [S Hc, Q Hg, H epsilon] exactly."""

        locations = np.array(((0.4, -0.7, 0.2), (-0.3, 0.6, -0.1)))
        center_midpoint = np.array((0.2, -0.4, 0.7))
        center_generators = np.array(((0.3, -0.1), (0.2, 0.4), (-0.5, 0.6)))
        gradient_midpoint = np.array(
            ((0.1, -0.2, 0.3), (0.4, 0.5, -0.6), (-0.7, 0.8, 0.9))
        )
        gradient_generators = np.arange(27, dtype=float).reshape(9, 3) / 50.0
        remainder_midpoint = np.array(((0.02, -0.03, 0.04), (-0.05, 0.06, 0.01)))
        remainder_generators = np.array(
            (
                (0.10, 0.00, 0.02, 0.00),
                (0.00, 0.11, 0.00, 0.01),
                (0.03, 0.00, 0.12, 0.00),
                (0.00, 0.04, 0.00, 0.13),
                (0.14, 0.00, 0.05, 0.00),
                (0.00, 0.15, 0.00, 0.06),
            )
        )
        joint = JointFlow(
            locations,
            center_midpoint,
            center_generators,
            gradient_midpoint,
            gradient_generators,
            remainder_midpoint,
            remainder_generators,
        )
        sample_locations = np.vstack((np.zeros(3), locations))
        center_lift = np.tile(np.eye(3), (locations.shape[0] + 1, 1))
        gradient_lift = np.vstack(
            [np.kron(np.eye(3), location) for location in sample_locations]
        )
        epsilon_midpoint = np.concatenate((np.zeros(3), remainder_midpoint.reshape(-1)))
        epsilon_generators = np.vstack(
            (np.zeros((3, remainder_generators.shape[1])), remainder_generators)
        )
        expected_center = (
            center_lift @ center_midpoint
            + gradient_lift @ gradient_midpoint.reshape(-1)
            + epsilon_midpoint
        )
        expected_generators = np.column_stack(
            (
                center_lift @ center_generators,
                gradient_lift @ gradient_generators,
                epsilon_generators,
            )
        )

        self.assertEqual(FLOW_COMPONENT_ORDER, ("x", "y", "z"))
        self.assertEqual(
            GRADIENT_COMPONENT_ORDER,
            tuple((row, column) for row in range(3) for column in range(3)),
        )
        self.assertEqual(joint.center_factor_slice, slice(0, 2))
        self.assertEqual(joint.gradient_factor_slice, slice(2, 5))
        self.assertEqual(joint.remainder_factor_slice, slice(5, 9))
        np.testing.assert_allclose(joint.center, expected_center)
        np.testing.assert_allclose(joint.generators, expected_generators)
        np.testing.assert_allclose(joint.center[:3], center_midpoint)
        np.testing.assert_allclose(joint.generators[:3, :2], center_generators)
        np.testing.assert_array_equal(joint.generators[:3, 2:], 0.0)
        self.assertFalse(joint.center.flags.writeable)
        self.assertFalse(joint.generators.flags.writeable)

        factors = np.array((1.0, -1.0, -0.6, 0.25, 0.8, -1.0, 1.0, 1.0, -1.0))
        realization = joint.realization(factors)
        stacked = np.vstack(
            (realization.center_b_m_s, realization.strip_flow(locations))
        )
        np.testing.assert_allclose(joint.evaluate(factors), stacked)
        np.testing.assert_allclose(joint.strip_flow(factors), stacked[1:])

    def test_box_bounds_compile_exact_factor_layout(self) -> None:
        """Use three centre, nine row-major gradient, and three local columns."""

        locations, bounds = _general_flow_bounds()
        joint = bounds.joint_flow(locations)
        sample_locations = np.vstack((np.zeros(3), locations))
        center_lift = np.tile(np.eye(3), (locations.shape[0] + 1, 1))
        gradient_lift = np.vstack(
            [np.kron(np.eye(3), location) for location in sample_locations]
        )
        center_box = Interval(bounds.center_lower_m_s, bounds.center_upper_m_s)
        gradient_box = Interval(bounds.gradient_lower_s, bounds.gradient_upper_s)
        remainder_radius = np.broadcast_to(bounds.remainder_abs_m_s, locations.shape)
        epsilon_generators = np.vstack(
            (
                np.zeros((3, 3 * locations.shape[0])),
                np.diag(remainder_radius.reshape(-1)),
            )
        )
        expected_center = (
            center_lift @ center_box.center
            + gradient_lift @ gradient_box.center.reshape(-1)
        )
        expected_generators = np.column_stack(
            (
                center_lift @ np.diag(center_box.radius),
                gradient_lift @ np.diag(gradient_box.radius.reshape(-1)),
                epsilon_generators,
            )
        )

        self.assertEqual(CENTER_FLOW_GENERATOR_COUNT, 3)
        self.assertEqual(GRADIENT_FLOW_GENERATOR_COUNT, 9)
        self.assertEqual(SHARED_FLOW_GENERATOR_COUNT, 12)
        self.assertEqual(joint.center_factor_slice, slice(0, 3))
        self.assertEqual(joint.gradient_factor_slice, slice(3, 12))
        self.assertEqual(
            joint.remainder_factor_slice,
            slice(12, 12 + 3 * locations.shape[0]),
        )
        self.assertEqual(
            joint.generators.shape,
            (3 * (locations.shape[0] + 1), 12 + 3 * locations.shape[0]),
        )
        np.testing.assert_allclose(joint.center, expected_center)
        np.testing.assert_allclose(joint.generators, expected_generators)
        np.testing.assert_allclose(
            joint.affine_form().generators.reshape(joint.generators.shape),
            joint.generators,
        )

        lower, upper = bounds.strip_bounds(locations)
        hull = joint.independent_hull()
        np.testing.assert_allclose(
            lower,
            hull.lower.reshape(locations.shape[0] + 1, 3)[1:],
        )
        np.testing.assert_allclose(
            upper,
            hull.upper.reshape(locations.shape[0] + 1, 3)[1:],
        )

    def test_deterministic_corners_and_adversarial_support(self) -> None:
        """Contain deterministic vertices and attain analytic joint support."""

        locations, bounds = _general_flow_bounds()
        joint = bounds.joint_flow(locations)
        factor_count = joint.generators.shape[1]
        corners = (
            np.ones(factor_count),
            -np.ones(factor_count),
            np.where(np.arange(factor_count) % 2 == 0, 1.0, -1.0),
            np.where(np.arange(factor_count) % 3 == 0, -1.0, 1.0),
        )
        hull = joint.independent_hull()
        for factors in corners:
            realization = joint.realization(factors)
            stacked = joint.evaluate(factors)
            np.testing.assert_array_equal(
                stacked.reshape(-1),
                joint.center + joint.generators @ factors,
            )
            self.assertTrue(hull.contains(stacked.reshape(-1)))
            np.testing.assert_allclose(
                stacked[1:],
                realization.strip_flow(locations),
            )

        directions = (
            np.linspace(-1.1, 1.3, joint.center.size),
            np.array((0.2, -0.7, 0.5) * (locations.shape[0] + 1)),
            np.where(np.arange(joint.center.size) % 2 == 0, 0.8, -0.3),
        )
        for direction in directions:
            coefficients = direction @ joint.generators
            factors = np.where(coefficients >= 0.0, 1.0, -1.0)
            adversarial = joint.evaluate(factors).reshape(-1)
            self.assertAlmostEqual(
                float(direction @ adversarial),
                joint.support(direction),
                places=12,
            )
            self.assertLessEqual(
                joint.support(direction),
                joint.independent_support(direction) + 1.0e-12,
            )

    def test_joint_width_is_strictly_below_independent_strip_boxes(self) -> None:
        """Retain gradient cancellation that independent strip boxes lose."""

        locations = np.array(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        gradient_lower = np.zeros((3, 3))
        gradient_upper = np.zeros((3, 3))
        gradient_lower[2, 0] = -0.4
        gradient_upper[2, 0] = 0.4
        joint = FlowBounds(
            np.zeros(3),
            np.zeros(3),
            gradient_lower,
            gradient_upper,
            np.zeros(3),
        ).joint_flow(locations)
        direction = np.zeros((locations.shape[0] + 1, 3))
        direction[1:, 2] = 1.0

        joint_width = joint.support(direction) + joint.support(-direction)
        independent_width = joint.independent_support(
            direction
        ) + joint.independent_support(-direction)

        self.assertAlmostEqual(joint_width, 0.0, places=12)
        self.assertAlmostEqual(independent_width, 1.6, places=12)
        self.assertLess(joint_width, independent_width)

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
        center = np.array((0.4, -0.2, 0.3))
        gradient = np.zeros((3, 3))
        gradient[2, 1] = 0.5
        strips = AffineFlow(
            center,
            gradient,
            np.zeros_like(aircraft.strip_table.r_b_m),
        ).strip_flow(aircraft.strip_table.r_b_m)
        control = np.array((0.08, -0.12, 0.04))

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
        world_flow = np.array((1.2, -0.4, 0.7))
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
        control = np.array((0.05, -0.1, 0.02))

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

    def test_lateral_updraft_gradient_creates_asymmetric_loads(self) -> None:
        """Produce lateral force and roll/yaw moments from distributed flow."""

        aircraft = Aircraft()
        state = np.zeros(15)
        state[6] = 6.0
        locations = aircraft.strip_table.r_b_m
        gradient_lower = np.zeros((3, 3))
        gradient_upper = np.zeros((3, 3))
        gradient_lower[2, 1] = -3.0
        gradient_upper[2, 1] = 3.0
        joint = FlowBounds(
            np.zeros(3),
            np.zeros(3),
            gradient_lower,
            gradient_upper,
            np.zeros(3),
        ).joint_flow(locations)
        positive_factors = np.zeros(joint.generators.shape[1])
        gradient_column = joint.gradient_factor_slice.start + 3 * 2 + 1
        positive_factors[gradient_column] = 1.0
        positive_flow = joint.strip_flow(positive_factors)
        negative_flow = joint.strip_flow(-positive_factors)
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


def _general_flow_bounds() -> tuple[np.ndarray, FlowBounds]:
    locations = np.array(((0.4, -0.7, 0.2), (-0.3, 0.6, -0.1), (0.9, 0.1, 0.5)))
    bounds = FlowBounds(
        (-1.2, 0.3, -0.8),
        (0.7, 1.5, 0.4),
        np.array(((-0.4, 0.1, -0.2), (-0.3, -0.5, 0.2), (0.0, -0.6, -0.1))),
        np.array(((0.2, 0.5, 0.3), (0.4, 0.1, 0.7), (0.8, 0.2, 0.6))),
        (0.07, 0.11, 0.05),
    )
    return locations, bounds
