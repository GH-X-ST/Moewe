"""Tests for the deterministic three-input reference governor solver."""

from __future__ import annotations

from time import get_clock_info, perf_counter
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from control.governor import (
    SMOOTHING_WEIGHT,
    TRACKING_WEIGHT,
    GovernorSolver,
    _INFEASIBLE,
    _SOLVED,
    _TIMED_OUT,
)


def _solve(
    solver: GovernorSolver,
    nominal: np.ndarray | tuple[float, float, float],
    previous: np.ndarray | tuple[float, float, float],
    matrix: np.ndarray,
    bounds: np.ndarray,
    backup: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    result = np.empty(3)
    if isinstance(backup, tuple) and backup == (0.0, 0.0, 0.0):
        backup = solver._center
    assert (
        solver.solve_into(
            nominal,
            previous,
            matrix,
            bounds,
            result,
            backup,
        )
        == _SOLVED
    )
    return result


class GovernorSolverTests(TestCase):
    """Verify online active-set solutions against independent optimisation."""

    def test_fixed_weights_and_unmodified_nominal_reference(self) -> None:
        """Keep an admissible nominal reference when it is also previous."""

        self.assertEqual(TRACKING_WEIGHT, 1.0)
        self.assertEqual(SMOOTHING_WEIGHT, 0.1)
        solver = GovernorSolver((-2.0, -1.0, 0.0), (2.0, 3.0, 4.0), 1)
        nominal = np.array([0.4, 1.2, 2.5])
        result = _solve(
            solver,
            nominal,
            nominal,
            np.array([[1.0, 0.0, 0.0]]),
            np.array([1.0]),
        )

        np.testing.assert_allclose(result, nominal, atol=1.0e-14)

    def test_active_constraint_gives_minimum_modification(self) -> None:
        """Project onto an active oblique half-space in normalized coordinates."""

        solver = GovernorSolver((-1.0,) * 3, (1.0,) * 3, 1)
        nominal = np.array([0.8, 0.2, -0.1])
        result = _solve(
            solver,
            nominal,
            nominal,
            np.array([[1.0, 1.0, 0.0]]),
            np.array([0.5]),
        )

        np.testing.assert_allclose(result, (0.55, -0.05, -0.1), atol=1.0e-12)

    def test_normalized_margin_excludes_a_positive_stored_row_residual(self) -> None:
        solver = GovernorSolver((-1.0,) * 3, (1.0,) * 3, 1)
        matrix = np.array(((1.0, 0.0, 0.0),))
        bounds = np.array((0.0,))
        nominal = np.array((5.0e-11, 0.0, 0.0))

        result = _solve(solver, nominal, nominal, matrix, bounds, (-0.5, 0.0, 0.0))

        assert result[0] <= 0.0
        assert np.all(matrix @ result <= bounds)

    def test_physical_reference_bounds_are_enforced(self) -> None:
        """Activate all three supplied physical bounds when required."""

        solver = GovernorSolver((-2.0, -1.0, 0.0), (1.0, 3.0, 5.0), 0)
        first = _solve(
            solver,
            (5.0, -4.0, 8.0),
            (5.0, -4.0, 8.0),
            np.empty((0, 3)),
            np.empty(0),
        )
        saved = first.copy()
        second = _solve(
            solver,
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            np.empty((0, 3)),
            np.empty(0),
        )

        np.testing.assert_allclose(first, (1.0, -1.0, 5.0), atol=1.0e-12)
        np.testing.assert_array_equal(first, saved)
        self.assertFalse(np.shares_memory(first, second))

    def test_infeasible_inequalities_are_reported(self) -> None:
        """Distinguish infeasibility from a feasible constrained result."""

        solver = GovernorSolver((-1.0,) * 3, (1.0,) * 3, 2)
        result = np.empty(3)
        self.assertEqual(
            solver.solve_into(
                np.zeros(3),
                np.zeros(3),
                np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
                np.array([-0.5, -0.5]),
                result,
                np.zeros(3),
            ),
            _INFEASIBLE,
        )

    def test_redundant_rank_deficient_rows_are_skipped(self) -> None:
        """Find the optimum without solving singular active KKT systems."""

        solver = GovernorSolver((-1.0,) * 3, (1.0,) * 3, 4)
        nominal = np.array([0.8, 0.8, 0.7])
        result = _solve(
            solver,
            nominal,
            nominal,
            np.array(
                [
                    [1.0, 1.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            np.array([0.2, 0.4, 0.1, 1.0]),
        )

        np.testing.assert_allclose(result, (0.1, 0.1, 0.1), atol=1.0e-12)

    def test_expired_deadline_is_distinct(self) -> None:
        """Report timeout before returning a solver result."""

        solver = GovernorSolver((-1.0,) * 3, (1.0,) * 3, 0)
        result = np.empty(3)
        self.assertEqual(
            solver.solve_into(
                np.zeros(3),
                np.zeros(3),
                np.empty((0, 3)),
                np.empty(0),
                result,
                np.zeros(3),
                perf_counter(),
            ),
            _TIMED_OUT,
        )

    def test_deadline_interrupts_constraint_preparation(self) -> None:
        solver = GovernorSolver((-1.0,) * 3, (1.0,) * 3, 1)
        result = np.empty(3)
        with patch(
            "control.governor._deadline_reached",
            side_effect=(False, True),
        ):
            status = solver.solve_into(
                np.zeros(3),
                np.zeros(3),
                np.zeros((1, 3)),
                np.ones(1),
                result,
                np.zeros(3),
                perf_counter() + 1.0,
            )
        self.assertEqual(status, _TIMED_OUT)

    def test_deadline_clock_is_high_resolution_and_monotonic(self) -> None:
        clock = get_clock_info("perf_counter")
        self.assertTrue(clock.monotonic)
        self.assertLessEqual(clock.resolution, 1.0e-3)

    def test_large_deterministic_set_matches_scipy(self) -> None:
        """Match a trusted convex solver over publication-scale QP variation."""

        rng = np.random.default_rng(20260713)
        lower = np.array([-0.4, -2.0, 10.0])
        upper = np.array([0.8, 3.0, 22.0])
        center = 0.5 * (lower + upper)
        scale = 0.5 * (upper - lower)
        solver = GovernorSolver(lower, upper, 8)

        for case in range(1024):
            with self.subTest(case=case):
                constraints = rng.normal(size=(8, 3))
                constraints /= np.linalg.norm(constraints, axis=1)[:, None]
                witness = rng.uniform(-0.7, 0.7, 3)
                limits = constraints @ witness + rng.uniform(0.02, 0.8, 8)
                physical_a = constraints / scale
                physical_b = limits + physical_a @ center
                nominal_y = rng.uniform(-1.4, 1.4, 3)
                previous_y = rng.uniform(-1.0, 1.0, 3)
                nominal = center + scale * nominal_y
                previous = center + scale * previous_y

                actual = _solve(
                    solver,
                    nominal,
                    previous,
                    physical_a,
                    physical_b,
                    center + scale * witness,
                )
                expected = self._trusted_solution(
                    constraints,
                    limits,
                    witness,
                    nominal_y,
                    previous_y,
                )
                actual_y = (actual - center) / scale

                np.testing.assert_allclose(actual_y, expected, atol=3.0e-7)
                self.assertTrue(np.all(constraints @ actual_y <= limits + 1.0e-9))
                self.assertTrue(np.all(np.abs(actual_y) <= 1.0 + 1.0e-9))

    def test_online_active_set_scales_to_many_facets(self) -> None:
        rng = np.random.default_rng(9137)
        constraints = rng.normal(size=(96, 3))
        constraints /= np.linalg.norm(constraints, axis=1)[:, None]
        limits = np.full(96, 0.8)
        lower = -np.ones(3)
        upper = np.ones(3)
        solver = GovernorSolver(lower, upper, constraints.shape[0])
        result = np.empty(3)
        for case in range(32):
            with self.subTest(case=case):
                nominal = rng.uniform(-1.2, 1.2, 3)
                previous = rng.uniform(-1.0, 1.0, 3)
                status = solver.solve_into(
                    nominal,
                    previous,
                    constraints,
                    limits,
                    result,
                    backup=np.zeros(3),
                )
                self.assertEqual(status, 1)
                expected = self._trusted_solution(
                    constraints,
                    limits,
                    np.zeros(3),
                    nominal,
                    previous,
                )
                np.testing.assert_allclose(result, expected, atol=3.0e-7)

    def test_online_active_set_handles_a_degenerate_vertex(self) -> None:
        facet_count = 24
        angles = 2.0 * np.pi * np.arange(facet_count) / facet_count
        sides = np.column_stack((np.cos(angles), np.sin(angles), np.ones(facet_count)))
        constraints = np.vstack((sides, (0.0, 0.0, -1.0)))
        limits = np.ones(facet_count + 1)
        solver = GovernorSolver(
            -3.0 * np.ones(3),
            3.0 * np.ones(3),
            constraints.shape[0],
        )
        result = np.empty(3)
        status = solver.solve_into(
            np.array((0.0, 0.0, 2.2)),
            np.zeros(3),
            constraints,
            limits,
            result,
            backup=np.zeros(3),
        )
        self.assertEqual(
            status,
            1,
            (solver._iterate.copy(), solver._active[: solver._active_count].copy()),
        )
        np.testing.assert_allclose(result, (0.0, 0.0, 1.0), atol=1.0e-12)

    def _trusted_solution(
        self,
        constraints: np.ndarray,
        limits: np.ndarray,
        witness: np.ndarray,
        nominal: np.ndarray,
        previous: np.ndarray,
    ) -> np.ndarray:
        tracking = TRACKING_WEIGHT
        smoothing = SMOOTHING_WEIGHT

        def objective(reference: np.ndarray) -> float:
            nominal_error = reference - nominal
            previous_error = reference - previous
            return 0.5 * float(
                np.sum(tracking * nominal_error * nominal_error)
                + np.sum(smoothing * previous_error * previous_error)
            )

        def gradient(reference: np.ndarray) -> np.ndarray:
            return tracking * (reference - nominal) + smoothing * (reference - previous)

        result = minimize(
            objective,
            witness,
            jac=gradient,
            method="SLSQP",
            bounds=Bounds((-1.0,) * 3, (1.0,) * 3),
            constraints=LinearConstraint(constraints, -np.inf, limits),
            options={"ftol": 1.0e-14, "maxiter": 200, "disp": False},
        )
        self.assertTrue(result.success, result.message)
        return np.asarray(result.x)
