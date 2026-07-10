"""Regression tests for recoverability control."""

from __future__ import annotations

from unittest import TestCase

import numpy as np

from control.abstraction import (
    RecoverabilityObject,
    calibration_error,
    sample_abstraction,
)
from control.design import BoxGrid, control_lattice
from control.missions import (
    GateMission,
    LandingMission,
    box_safe,
    combine_safe,
    obstacle_safe,
)
from control.policy import (
    MissionRecoverabilityController,
    NoSafeControlError,
    PreparedMission,
)
from control.recoverability import RecoverabilityMap
from models.aircraft import Aircraft
from simulation.gate import Gate
from simulation.platform import Platform


STATE_BOUNDS = (
    (0.0, 2.0),
    *((-1.0, 1.0),) * 14,
)
MISSION_BOUNDS = ((0.0, 8.0), (0.0, 4.8), (0.4, 3.5))


class MissionTests(TestCase):
    """Verify gate, landing, and mission safety semantics."""

    def test_gate_and_platform_clearance(self) -> None:
        """Match controller and simulator clearance events."""

        safe = box_safe(MISSION_BOUNDS)
        gate = GateMission(safe, body_radius_m=0.1)
        gate_sim = Gate(body_radius_m=0.1)
        previous = np.zeros(15)
        current = np.zeros(15)
        previous[:3] = (6.0, 2.2, 1.4)
        current[:3] = (7.0, 2.2, 1.4)
        self.assertTrue(gate.event(previous, current))
        self.assertTrue(gate_sim.passed(previous, current))

        landing = LandingMission(safe, body_radius_m=0.1)
        platform = Platform(body_radius_m=0.1)
        previous[:3] = (6.0, 2.2, 1.1)
        current[:3] = (6.0, 2.2, 0.9)
        current[8] = 1.0
        self.assertTrue(landing.event(previous, current))
        self.assertTrue(platform.landed(previous, current))

    def test_safe_set_composition(self) -> None:
        """Combine arena and obstacle-clearance predicates."""

        arena = box_safe(MISSION_BOUNDS)
        obstacle = obstacle_safe(
            lambda point: float(np.linalg.norm(point - np.ones(3))),
            0.5,
        )
        safe = combine_safe(arena, obstacle)
        state = np.zeros(15)
        state[:3] = (2.0, 2.0, 2.0)
        self.assertTrue(safe(state))
        state[:3] = (1.0, 1.0, 1.0)
        self.assertFalse(safe(state))


class RecoverabilityTests(TestCase):
    """Verify aerodynamic recoverability coordinates."""

    def test_wind_corrected_air_data_and_surface_margin(self) -> None:
        """Use relative air velocity and conservative surface limits."""

        aircraft = Aircraft()
        wind = np.array([3.0, 0.0, 0.0])
        mapping = RecoverabilityMap(
            lambda state: aircraft.air_data(state, wind),
            lambda state: aircraft.energy_rate(state, wind),
        )
        state = np.zeros(15)
        state[2] = 1.5
        state[6] = 5.0
        coordinates = mapping(state)
        self.assertAlmostEqual(coordinates[1], -0.6)
        state[12] = aircraft.control_upper_rad[0]
        self.assertAlmostEqual(mapping(state)[8], 0.0)


class AbstractionTests(TestCase):
    """Verify design grids, disturbances, calibration, and persistence."""

    def test_grid_lattice_and_time_varying_disturbance(self) -> None:
        """Construct an empirical abstraction from concrete design data."""

        grid = BoxGrid(STATE_BOUNDS, (2,) + (1,) * 14)
        controls = control_lattice(((-1.0, 1.0),) * 3, (3, 1, 1))

        def dynamics(
            _state: np.ndarray,
            _control: np.ndarray,
            disturbance: object,
        ) -> np.ndarray:
            derivative = np.zeros(15)
            derivative[0] = float(disturbance)
            return derivative

        abstraction = sample_abstraction(
            controls[:1],
            grid.center_samples(),
            grid.cell,
            dynamics,
            lambda state: 1.0,
            1.0,
            disturbance_signals=(lambda time: time,),
        )
        self.assertEqual(grid.n_cells, 2)
        self.assertEqual(controls.shape, (3, 3))
        self.assertEqual(abstraction.sampled_successors[0][0], (1,))
        self.assertLess(calibration_error([True] * 1000), 0.05)


class PolicyTests(TestCase):
    """Verify mission selection, fallback, and failure behavior."""

    def test_policy_branches(self) -> None:
        """Select mission, fallback, and empty-safe-set outcomes."""

        controls = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        transitions = (((0,), (0,)),)
        abstraction = RecoverabilityObject(
            controls,
            transitions,
            transitions,
            np.ones(1),
            lambda state: 0,
        )
        mission = GateMission(box_safe(MISSION_BOUNDS))
        progress = PreparedMission(
            mission,
            ((0, 1),),
            np.zeros((1, 2)),
            np.array([[2.0, 1.0]]),
            lambda state, cell, index: False,
        )
        state = np.zeros(15)
        controller = MissionRecoverabilityController(
            abstraction,
            lambda_u=0.0,
        )
        self.assertTrue(
            np.array_equal(
                controller.control(state, np.zeros(3), progress),
                controls[1],
            )
        )

        empty = PreparedMission(
            mission,
            ((),),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            lambda state, cell, index: False,
        )
        with self.assertRaises(NoSafeControlError):
            controller.control(state, np.zeros(3), empty)
