"""Regression tests for recoverability control."""

from __future__ import annotations

from unittest import TestCase

import numpy as np

from control.abstraction import RecoverabilityObject, sample_abstraction
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
from models.geometry import RigidBodyGeometry, body_to_world
from simulation.gate import Gate
from simulation.platform import Platform


STATE_BOUNDS = (
    (0.0, 2.0),
    *((-1.0, 1.0),) * 14,
)
MISSION_BOUNDS = ((0.0, 8.0), (0.0, 4.8), (0.4, 3.5))
TEST_GEOMETRY = RigidBodyGeometry(
    body_b_m=(
        (-0.2, -0.4, -0.1),
        (-0.2, -0.4, 0.1),
        (-0.2, 0.4, -0.1),
        (-0.2, 0.4, 0.1),
        (0.2, -0.4, -0.1),
        (0.2, -0.4, 0.1),
        (0.2, 0.4, -0.1),
        (0.2, 0.4, 0.1),
    ),
    contact_b_m=((0.0, -0.1, 0.1), (0.0, 0.1, 0.1)),
    footprint_b_m=(
        (-0.2, -0.2, 0.1),
        (-0.2, 0.2, 0.1),
        (0.2, -0.2, 0.1),
        (0.2, 0.2, 0.1),
    ),
)


class GeometryTests(TestCase):
    """Verify body-to-world geometry kinematics."""

    def test_rotation_matrix(self) -> None:
        """Use a proper body-to-world rotation."""

        rotation = body_to_world((0.2, -0.1, 0.3))
        np.testing.assert_allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1.0e-15,
        )
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_aircraft_position_kinematics(self) -> None:
        """Use the same world frame in dynamics and geometry."""

        aircraft = Aircraft()
        state = np.zeros(15)
        state[3:6] = (0.2, -0.1, 0.3)
        state[6:9] = (5.0, 0.4, 0.2)
        derivative = aircraft(state, np.zeros(3))
        expected = body_to_world(state[3:6]) @ state[6:9]
        np.testing.assert_allclose(derivative[:3], expected)


class MissionTests(TestCase):
    """Verify gate, landing, and mission safety semantics."""

    def test_gate_occupancy(self) -> None:
        """Check the swept body against the gate aperture."""

        safe = box_safe(MISSION_BOUNDS)
        gate = GateMission(safe, geometry=TEST_GEOMETRY, margin_m=0.05)
        gate_sim = Gate(geometry=TEST_GEOMETRY, margin_m=0.05)
        previous = np.zeros(15)
        current = np.zeros(15)
        previous[:3] = (6.0, 2.2, 1.4)
        current[:3] = (7.0, 2.2, 1.4)
        self.assertTrue(gate.event(previous, current))
        self.assertTrue(gate_sim.passed(previous, current))
        previous[1] = 2.4
        current[1] = 2.4
        self.assertFalse(gate.event(previous, current))
        self.assertFalse(gate_sim.passed(previous, current))

    def test_landing_footprint(self) -> None:
        """Check the touchdown footprint against the platform."""

        safe = box_safe(MISSION_BOUNDS)
        landing = LandingMission(
            safe,
            geometry=TEST_GEOMETRY,
            margin_m=0.05,
        )
        platform = Platform(geometry=TEST_GEOMETRY, margin_m=0.05)
        previous = np.zeros(15)
        current = np.zeros(15)
        previous[:3] = (6.0, 2.2, 1.2)
        current[:3] = (6.0, 2.2, 1.0)
        current[8] = 1.0
        self.assertTrue(landing.event(previous, current))
        self.assertTrue(platform.landed(previous, current))

        current[:3] = (6.0, 2.7, 1.0)
        self.assertFalse(landing.event(previous, current))

    def test_landing_contact_speed(self) -> None:
        """Apply touchdown speed limits to first-contact points."""

        safe = box_safe(MISSION_BOUNDS)
        previous = np.zeros(15)
        current = np.zeros(15)
        previous[:3] = (6.0, 2.2, 1.2)
        current[:3] = (6.0, 2.2, 1.0)
        previous[[8, 9]] = (0.5, 1.0)
        current[[8, 9]] = (0.5, 1.0)
        landing = LandingMission(
            safe,
            normal_speed_max_m_s=0.55,
            geometry=TEST_GEOMETRY,
        )
        platform = Platform(
            normal_speed_max_m_s=0.55,
            geometry=TEST_GEOMETRY,
        )
        self.assertFalse(landing.event(previous, current))
        self.assertFalse(platform.landed(previous, current))

    def test_safe_set_composition(self) -> None:
        """Combine arena and obstacle-clearance predicates."""

        arena = box_safe(MISSION_BOUNDS)
        geometry = RigidBodyGeometry(
            body_b_m=((0.0, 0.0, 0.0), (0.4, 0.0, 0.0))
        )
        obstacle = obstacle_safe(
            lambda points: float(
                np.min(np.linalg.norm(points - np.ones(3), axis=1))
            ),
            geometry,
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
    """Verify empirical abstraction construction."""

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
