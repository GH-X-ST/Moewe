"""Tests for continuous nominal planning and local feedback design."""

from __future__ import annotations

from unittest import TestCase

import numpy as np

from control.flow import AffineFlow
from control.planner import NominalPlanner, PlanningTimeout
from models.aircraft import Aircraft


class _SurfaceMission:
    def __init__(self, target_rad: float) -> None:
        self.target_rad = target_rad

    def running_cost(self, _state: np.ndarray, control: np.ndarray) -> float:
        return float(control @ control)

    def nominal_constraints(self, states: np.ndarray) -> np.ndarray:
        return np.array([states[-2, 12] - self.target_rad])


class _InfeasibleMission(_SurfaceMission):
    def nominal_constraints(self, _states: np.ndarray) -> np.ndarray:
        return np.array([-1.0])


class PlannerTests(TestCase):
    """Verify constrained continuous planning with real aircraft dynamics."""

    def setUp(self) -> None:
        self.aircraft = Aircraft()
        flow = AffineFlow(
            np.zeros(3),
            np.zeros((3, 3)),
            np.zeros_like(self.aircraft.strip_table.r_b_m),
        )
        self.planner = NominalPlanner(
            self.aircraft,
            flow,
            horizons=(2, 3),
            max_iterations=50,
        )
        self.state = np.zeros(15)
        self.state[6] = 5.0

    def test_bounded_continuous_plan_and_feedback(self) -> None:
        """Enforce nominal inequalities with non-lattice bounded commands."""

        target = 0.015
        segments = self.planner.plan(
            self.state,
            _SurfaceMission(target),
        )
        controls = np.stack([segment.control_rad for segment in segments])
        states = np.stack([segment.state for segment in segments])
        gains = np.stack([segment.gain for segment in segments])

        self.assertEqual(controls.shape, (3, 3))
        self.assertEqual(states.shape, (3, 15))
        self.assertEqual(gains.shape, (3, 3, 15))
        self.assertGreaterEqual(states[-1, 12], target - 1.0e-7)
        self.assertTrue(np.all(controls >= self.aircraft.control_lower_rad - 1.0e-12))
        self.assertTrue(np.all(controls <= self.aircraft.control_upper_rad + 1.0e-12))

        levels = np.linspace(-1.0, 1.0, 11)
        lattice = np.where(
            levels >= 0.0,
            levels * self.aircraft.control_upper_rad[0],
            -levels * self.aircraft.control_lower_rad[0],
        )
        active_aileron = controls[np.abs(controls[:, 0]) > 1.0e-6, 0]
        self.assertTrue(
            any(
                np.min(np.abs(lattice - command)) > 1.0e-4 for command in active_aileron
            )
        )
        self.assertFalse(np.allclose(gains[0], gains[-1]))

    def test_infeasible_nominal_constraints_fail(self) -> None:
        """Reject a horizon whose hard inequalities cannot be satisfied."""

        with self.assertRaises(RuntimeError):
            self.planner.plan(
                self.state,
                _InfeasibleMission(0.0),
                horizons=2,
            )

    def test_wall_clock_deadline_aborts_candidate_generation(self) -> None:
        """Return control when the planning budget is exhausted."""

        with self.assertRaises(PlanningTimeout):
            self.planner.plan(
                self.state,
                _SurfaceMission(0.0),
                time_limit_s=0.0,
            )
