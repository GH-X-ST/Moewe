"""Tests for tube-certified free space and terminal missions."""

from __future__ import annotations

from math import radians
from unittest import TestCase

import numpy as np

from control.flow import FlowBounds
from control.interval import Interval
from control.missions import FreeSpace, GateMission, LandingMission
from control.tube import BodyTube, ModelUncertainty, SegmentTube
from models.aircraft import Aircraft
from models.geometry import RigidBodyGeometry, point_velocities
from simulation.gate import Gate
from simulation.platform import Platform

POINT_GEOMETRY = RigidBodyGeometry(
    body_b_m=((0.0, 0.0, 0.0),),
    contact_b_m=((0.0, 0.0, 0.0),),
    footprint_b_m=((0.0, 0.0, 0.0),),
)


def _state(
    position: tuple[float, float, float],
    attitude: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rates_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    value = np.zeros(15)
    value[:3] = position
    value[3:6] = attitude
    value[6:9] = velocity_b
    value[9:12] = rates_b
    return value


def _point_box(
    center: tuple[float, float, float],
    radius: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Interval:
    return Interval.from_midpoint(
        np.asarray(center, dtype=float).reshape(1, 3),
        np.asarray(radius, dtype=float).reshape(1, 3),
    )


def _body_tube(occupied: tuple[Interval, ...]) -> BodyTube:
    zero_velocity = tuple(_point_box((0.0, 0.0, 0.0)) for _ in occupied)
    return BodyTube(occupied, occupied, occupied, zero_velocity)


def _segment(
    initial: np.ndarray,
    successor: np.ndarray,
    states: tuple[Interval, ...],
    occupied: tuple[Interval, ...],
    contact: tuple[Interval, ...] | None = None,
    footprint: tuple[Interval, ...] | None = None,
    velocity: tuple[Interval, ...] | None = None,
) -> SegmentTube:
    contact = occupied if contact is None else contact
    footprint = occupied if footprint is None else footprint
    if velocity is None:
        velocity = tuple(_point_box((0.0, 0.0, -0.4)) for _ in occupied)
    return SegmentTube(
        Interval.point(initial),
        Interval.point(successor),
        states,
        BodyTube(occupied, contact, footprint, velocity),
    )


def _gate_tube(
    initial_x: float = -1.0,
    successor_x: float = 1.0,
    crossing_y: float = 0.0,
) -> SegmentTube:
    occupied = (
        _point_box((-0.6, 0.0, 0.0), (0.25, 0.1, 0.1)),
        _point_box((0.0, crossing_y, 0.0), (0.2, 0.0, 0.1)),
        _point_box((0.6, 0.0, 0.0), (0.25, 0.1, 0.1)),
    )
    states = tuple(Interval.point(_state((x, 0.0, 0.0))) for x in (-0.6, 0.0, 0.6))
    return _segment(
        _state((initial_x, 0.0, 0.0)),
        _state((successor_x, 0.0, 0.0)),
        states,
        occupied,
    )


def _landing_tube(
    footprint_x: float = 0.0,
    candidate_attitudes: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
    geometry: RigidBodyGeometry = POINT_GEOMETRY,
    candidate_velocities: tuple[Interval, Interval] | None = None,
) -> SegmentTube:
    contact_y = float(np.asarray(geometry.contact_b_m).reshape(-1, 3)[0, 1])
    occupied = (
        _point_box((0.0, contact_y, 0.8), (0.1, 0.1, 0.1)),
        _point_box((0.0, contact_y, 0.05), (0.1, 0.1, 0.08)),
        _point_box((0.0, contact_y, -0.05), (0.1, 0.1, 0.08)),
    )
    contact = (
        _point_box((0.0, contact_y, 0.8), (0.0, 0.0, 0.1)),
        _point_box((0.0, contact_y, 0.05), (0.0, 0.0, 0.08)),
        _point_box((0.0, contact_y, -0.05), (0.0, 0.0, 0.08)),
    )
    footprint = (
        _point_box((0.0, 0.0, 0.8), (0.2, 0.2, 0.0)),
        _point_box((footprint_x, 0.0, 0.0), (0.2, 0.2, 0.0)),
        _point_box((0.0, 0.0, 0.0), (0.2, 0.2, 0.0)),
    )
    velocities = (
        _point_box((0.0, 0.0, -0.4)),
        _point_box((0.0, 0.0, -0.4)),
        _point_box((0.0, 0.0, -0.4)),
    )
    if candidate_velocities is not None:
        velocities = (velocities[0], *candidate_velocities)
    states = (
        Interval.point(_state((0.0, 0.0, 0.8))),
        Interval.point(_state((0.0, 0.0, 0.05), candidate_attitudes[0])),
        Interval.point(_state((0.0, 0.0, -0.05), candidate_attitudes[1])),
    )
    return _segment(
        _state((0.0, 0.0, 1.0)),
        _state((0.0, contact_y, -0.1)),
        states,
        occupied,
        contact,
        footprint,
        velocities,
    )


class FreeSpaceTests(TestCase):
    """Verify conservative occupied-body box checks."""

    def test_arena_faces_and_forbidden_box(self) -> None:
        """Reject all six arena faces and any forbidden-box intersection."""

        free = FreeSpace(
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            (((0.4, 0.6), (0.4, 0.6), (0.4, 0.6)),),
        )
        self.assertTrue(free.contains(_body_tube((_point_box((0.2,) * 3),))))
        self.assertFalse(free.contains(_body_tube((_point_box((0.5,) * 3),))))
        outside = (
            _point_box((0.0, 0.2, 0.2)),
            _point_box((-0.01, 0.2, 0.2)),
            _point_box((1.0, 0.2, 0.2)),
            _point_box((1.01, 0.2, 0.2)),
            _point_box((0.2, -0.01, 0.2)),
            _point_box((0.2, 1.01, 0.2)),
            _point_box((0.2, 0.2, -0.01)),
            _point_box((0.2, 0.2, 1.01)),
        )
        for occupied in outside:
            with self.subTest(occupied=occupied.center.tolist()):
                self.assertFalse(free.contains(_body_tube((occupied,))))

    def test_default_contact_points_belong_to_the_body(self) -> None:
        """Keep every default first-contact point on the body model."""

        geometry = RigidBodyGeometry()
        body = np.asarray(geometry.body_b_m)
        for contact in np.asarray(geometry.contact_b_m):
            self.assertTrue(np.any(np.all(body == contact, axis=1)))


class GateMissionTests(TestCase):
    """Verify full-body robust gate passage."""

    def setUp(self) -> None:
        self.free = FreeSpace(((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)))
        self.mission = GateMission(
            self.free,
            center_w_m=(0.0, 0.0, 0.0),
            width_m=2.0,
            height_m=2.0,
            geometry=POINT_GEOMETRY,
        )

    def test_gate_terminal_and_full_tube_free_space(self) -> None:
        """Require the complete verified segment to remain in free space."""

        self.assertTrue(self.mission.terminal((_gate_tube(),)))
        limited = GateMission(
            FreeSpace(((-2.0, 0.0), (-2.0, 2.0), (-2.0, 2.0))),
            center_w_m=(0.0, 0.0, 0.0),
            width_m=2.0,
            height_m=2.0,
            geometry=POINT_GEOMETRY,
        )
        self.assertFalse(limited.terminal((_gate_tube(),)))

    def test_aperture_frame_and_plane_side_rejections(self) -> None:
        """Reject aperture escape, frame contact, and invalid plane sides."""

        counterexamples = (
            _gate_tube(crossing_y=1.1),
            _gate_tube(crossing_y=1.0),
            _gate_tube(initial_x=0.0),
            _gate_tube(initial_x=0.1),
            _gate_tube(successor_x=0.0),
            _gate_tube(successor_x=-0.1),
        )
        for tube in counterexamples:
            with self.subTest(tube=tube):
                self.assertFalse(self.mission.terminal((tube,)))

    def test_partial_passage_is_safe_before_final_exit(self) -> None:
        """Allow aperture entry while reserving complete exit for the deadline."""

        occupied = (_point_box((-0.6, 0.0, 0.0), (0.2, 0.1, 0.1)),)
        preterminal = _segment(
            _state((-1.0, 0.0, 0.0)),
            _state((-0.5, 0.0, 0.0)),
            (Interval.point(_state((-0.6, 0.0, 0.0))),),
            occupied,
        )
        entering = _segment(
            _state((-1.0, 0.0, 0.0)),
            _state((-0.2, 0.0, 0.0)),
            (Interval.point(_state((-0.2, 0.0, 0.0))),),
            (_point_box((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)),),
        )
        self.assertTrue(self.mission.terminal((preterminal, _gate_tube())))
        self.assertTrue(self.mission.terminal((entering, _gate_tube())))
        self.assertFalse(self.mission.terminal((_gate_tube(), _gate_tube())))
        frame_contact = _segment(
            _state((-1.0, 0.0, 0.0)),
            _state((-0.1, 0.0, 0.0)),
            (Interval.point(_state((-0.1, 0.0, 0.0))),),
            (_point_box((-0.1, 1.1, 0.0), (0.1, 0.1, 0.1)),),
        )
        self.assertFalse(self.mission.terminal((frame_contact, _gate_tube())))

    def test_normalized_axes_and_nominal_constraints(self) -> None:
        """Normalize mission axes and expose nonnegative planner constraints."""

        mission = GateMission(
            self.free,
            center_w_m=(0.0, 0.0, 0.0),
            normal_w=(2.0, 0.0, 0.0),
            width_axis_w=(1.0, 3.0, 0.0),
            width_m=2.0,
            height_m=2.0,
            geometry=POINT_GEOMETRY,
        )
        normal = np.asarray(mission.normal_w)
        width = np.asarray(mission.width_axis_w)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(width)), 1.0)
        self.assertAlmostEqual(float(normal @ width), 0.0)
        states = np.stack(
            (
                _state((-1.0, 0.0, 0.0)),
                _state((-0.4, 0.0, 0.0)),
                _state((0.5, 0.0, 0.0)),
            )
        )
        self.assertTrue(np.all(mission.nominal_constraints(states) >= 0.0))
        states[-1, 1] = 1.1
        self.assertTrue(np.any(mission.nominal_constraints(states) < 0.0))

    def test_realized_event_matches_simulator(self) -> None:
        """Use identical shared geometry for controller and simulator events."""

        mission = GateMission(
            self.free,
            center_w_m=(0.0, 0.0, 0.0),
            width_m=2.0,
            height_m=2.0,
            geometry=POINT_GEOMETRY,
        )
        simulator = Gate(
            center_w_m=mission.center_w_m,
            normal_w=mission.normal_w,
            width_axis_w=mission.width_axis_w,
            width_m=mission.width_m,
            height_m=mission.height_m,
            margin_m=mission.margin_m,
            geometry=mission.geometry,
        )
        successful = np.stack(
            (
                _state((-1.0, 0.0, 0.0)),
                _state((0.0, 0.0, 0.0)),
                _state((1.0, 0.0, 0.0)),
            )
        )
        missed = np.stack(
            (
                _state((-1.0, 1.1, 0.0)),
                _state((0.0, 1.1, 0.0)),
                _state((1.0, 1.1, 0.0)),
            )
        )

        self.assertTrue(mission.realized(successful))
        self.assertFalse(mission.realized(successful[[0, 2]]))
        self.assertEqual(
            mission.realized(successful),
            simulator.passed(successful),
        )
        self.assertEqual(
            mission.realized(missed),
            simulator.passed(missed),
        )

        full_body = GateMission(
            self.free,
            center_w_m=(0.0, 0.0, 0.0),
            width_m=2.0,
            height_m=2.0,
        )
        self.assertFalse(
            full_body.realized(
                np.stack(
                    (
                        _state((0.2, 0.0, 0.0)),
                        _state((0.35, 0.0, 0.0)),
                        _state((0.5, 0.0, 0.0)),
                    )
                )
            )
        )


class LandingMissionTests(TestCase):
    """Verify robust realization-dependent first contact."""

    def setUp(self) -> None:
        self.free = FreeSpace(((-2.0, 2.0), (-2.0, 2.0), (-0.2, 2.0)))
        self.mission = LandingMission(
            self.free,
            center_w_m=(0.0, 0.0, 0.0),
            length_m=2.0,
            width_m=2.0,
            geometry=POINT_GEOMETRY,
        )

    def test_realization_dependent_contact_and_full_tube_free_space(self) -> None:
        """Accept multiple possible contact pieces inside free space."""

        self.assertTrue(self.mission.terminal((_landing_tube(),)))
        limited = LandingMission(
            FreeSpace(((-2.0, 2.0), (-2.0, 2.0), (0.0, 2.0))),
            center_w_m=(0.0, 0.0, 0.0),
            length_m=2.0,
            width_m=2.0,
            geometry=POINT_GEOMETRY,
        )
        self.assertFalse(limited.terminal((_landing_tube(),)))

    def test_precontact_collision_and_footprint_rejections(self) -> None:
        """Reject an obstacle before contact and an overhanging footprint."""

        blocked = LandingMission(
            FreeSpace(
                self.free.arena_m,
                (((-0.2, 0.2), (-0.2, 0.2), (0.7, 0.9)),),
            ),
            center_w_m=(0.0, 0.0, 0.0),
            length_m=2.0,
            width_m=2.0,
            geometry=POINT_GEOMETRY,
        )
        self.assertFalse(blocked.terminal((_landing_tube(),)))
        self.assertFalse(self.mission.terminal((_landing_tube(footprint_x=0.9),)))

    def test_contact_is_confined_to_final_interval(self) -> None:
        """Keep the complete body above the platform before the deadline."""

        occupied = (_point_box((0.0, 0.0, 1.2), (0.1, 0.1, 0.1)),)
        preterminal = _segment(
            _state((0.0, 0.0, 1.5)),
            _state((0.0, 0.0, 1.0)),
            (Interval.point(_state((0.0, 0.0, 1.2))),),
            occupied,
        )
        self.assertTrue(self.mission.terminal((preterminal, _landing_tube())))
        self.assertFalse(self.mission.terminal((_landing_tube(), _landing_tube())))

    def test_angular_contact_velocity_rejection(self) -> None:
        """Reject angular-rate contact speed even with a stationary CG."""

        geometry = RigidBodyGeometry(
            body_b_m=((0.0, 0.5, 0.0),),
            contact_b_m=((0.0, 0.5, 0.0),),
            footprint_b_m=((0.0, 0.0, 0.0),),
        )
        contact_state = _state(
            (0.0, 0.0, 0.0),
            rates_b=(2.0, 0.0, 0.0),
        )
        angular_velocity = Interval.point(
            point_velocities(contact_state, geometry.contact_b_m)
        )
        mission = LandingMission(
            self.free,
            center_w_m=(0.0, 0.0, 0.0),
            length_m=2.0,
            width_m=2.0,
            normal_speed_max_m_s=0.5,
            geometry=geometry,
        )
        tube = _landing_tube(
            geometry=geometry,
            candidate_velocities=(angular_velocity, angular_velocity),
        )
        self.assertAlmostEqual(float(angular_velocity.center[0, 2]), -1.0)
        self.assertFalse(mission.terminal((tube,)))

    def test_sink_direction_and_total_contact_speed_rejections(self) -> None:
        """Reject upward contact and excessive tangential velocity."""

        invalid_velocities = (
            _point_box((0.0, 0.0, 0.1)),
            _point_box((6.0, 0.0, -0.4)),
        )
        for velocity in invalid_velocities:
            with self.subTest(velocity=velocity.center.tolist()):
                tube = _landing_tube(
                    candidate_velocities=(velocity, velocity),
                )
                self.assertFalse(self.mission.terminal((tube,)))

    def test_attitude_rejection(self) -> None:
        """Reject uncertain first-contact pieces outside attitude bounds."""

        attitudes = (
            ((radians(25.0), 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, radians(30.0), 0.0), (0.0, 0.0, 0.0)),
        )
        for candidate_attitudes in attitudes:
            with self.subTest(attitudes=candidate_attitudes):
                self.assertFalse(
                    self.mission.terminal(
                        (_landing_tube(candidate_attitudes=candidate_attitudes),)
                    )
                )

    def test_normalized_axes_and_nominal_constraints(self) -> None:
        """Normalize platform axes and expose touchdown constraints."""

        mission = LandingMission(
            self.free,
            center_w_m=(0.0, 0.0, 0.0),
            length_axis_w=(3.0, 0.0, 0.0),
            width_axis_w=(1.0, 2.0, 0.0),
            length_m=2.0,
            width_m=2.0,
            geometry=POINT_GEOMETRY,
        )
        length = np.asarray(mission.length_axis_w)
        width = np.asarray(mission.width_axis_w)
        self.assertAlmostEqual(float(np.linalg.norm(length)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(width)), 1.0)
        self.assertAlmostEqual(float(length @ width), 0.0)
        states = np.stack(
            (
                _state((0.0, 0.0, 1.0)),
                _state((0.0, 0.0, 0.5)),
                _state((0.0, 0.0, 0.0), velocity_b=(0.0, 0.0, 0.5)),
            )
        )
        self.assertTrue(np.all(mission.nominal_constraints(states) >= 0.0))
        states[-1, 0] = 1.1
        self.assertTrue(np.any(mission.nominal_constraints(states) < 0.0))

    def test_realized_event_matches_simulator(self) -> None:
        """Use identical shared geometry for landing evaluation."""

        mission = LandingMission(
            self.free,
            center_w_m=(0.0, 0.0, 0.0),
            length_m=2.0,
            width_m=2.0,
            geometry=POINT_GEOMETRY,
        )
        simulator = Platform(
            center_w_m=mission.center_w_m,
            length_axis_w=mission.length_axis_w,
            width_axis_w=mission.width_axis_w,
            length_m=mission.length_m,
            width_m=mission.width_m,
            normal_speed_max_m_s=mission.normal_speed_max_m_s,
            contact_speed_max_m_s=mission.contact_speed_max_m_s,
            roll_max_rad=mission.roll_max_rad,
            pitch_bounds_rad=mission.pitch_bounds_rad,
            margin_m=mission.margin_m,
            geometry=mission.geometry,
        )
        successful = np.stack(
            tuple(
                _state(position, velocity_b=(0.0, 0.0, 0.5))
                for position in (
                    (0.0, 0.0, 0.2),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, -0.1),
                )
            )
        )
        missed = np.stack(
            tuple(
                _state(position, velocity_b=(0.0, 0.0, 0.5))
                for position in (
                    (1.1, 0.0, 0.2),
                    (1.1, 0.0, 0.0),
                    (1.1, 0.0, -0.1),
                )
            )
        )

        self.assertTrue(mission.realized(successful))
        self.assertFalse(mission.realized(successful[[0, 2]]))
        self.assertEqual(
            mission.realized(successful),
            simulator.landed(successful),
        )
        self.assertEqual(
            mission.realized(missed),
            simulator.landed(missed),
        )


class MissionReuseTests(TestCase):
    """Verify mission descriptors remain separate from the frozen model."""

    def test_descriptor_changes_preserve_aircraft_and_uncertainty(self) -> None:
        """Reuse identical aircraft and uncertainty objects across missions."""

        aircraft = Aircraft()
        flow = FlowBounds(
            center_lower_m_s=(-2.0, -2.0, -5.0),
            center_upper_m_s=(2.0, 2.0, 5.0),
            gradient_lower_s=np.full((3, 3), -2.0),
            gradient_upper_s=np.full((3, 3), 2.0),
            remainder_abs_m_s=(0.25, 0.25, 0.5),
            rate_abs_m_s2=(5.0, 5.0, 10.0),
        )
        uncertainty = _model_uncertainty(flow, aircraft)
        model_ids = (id(aircraft), id(flow), id(uncertainty))
        strip_locations = aircraft.strip_table.r_b_m.copy()
        center_bounds = (
            np.asarray(flow.center_lower_m_s).copy(),
            np.asarray(flow.center_upper_m_s).copy(),
        )

        first_space = FreeSpace(((-2.0, 0.0), (-2.0, 2.0), (0.0, 2.0)))
        second_space = FreeSpace(((-4.0, 3.0), (-3.0, 3.0), (-1.0, 3.0)))
        missions = (
            GateMission(
                first_space,
                center_w_m=(0.0, 0.0, 1.0),
                width_m=1.0,
                height_m=0.5,
            ),
            GateMission(
                second_space,
                center_w_m=(2.0, -1.0, 1.5),
                width_m=1.8,
                height_m=0.9,
            ),
            LandingMission(
                first_space,
                center_w_m=(-0.5, 0.0, 0.5),
                length_m=1.0,
                width_m=0.8,
            ),
            LandingMission(
                second_space,
                center_w_m=(1.5, 1.0, 0.8),
                length_m=1.6,
                width_m=1.2,
            ),
        )

        self.assertEqual(len(missions), 4)
        self.assertEqual(len({mission.identity for mission in missions}), 4)
        self.assertEqual(model_ids, (id(aircraft), id(flow), id(uncertainty)))
        self.assertIs(uncertainty.flow, flow)
        np.testing.assert_array_equal(
            aircraft.strip_table.r_b_m,
            strip_locations,
        )
        np.testing.assert_array_equal(flow.center_lower_m_s, center_bounds[0])
        np.testing.assert_array_equal(flow.center_upper_m_s, center_bounds[1])


def _model_uncertainty(
    flow: FlowBounds,
    aircraft: Aircraft,
) -> ModelUncertainty:
    return ModelUncertainty(
        flow=flow,
        density_kg_m3=(1.2, 1.25),
        mass_kg=(0.95 * aircraft.mass_kg, 1.05 * aircraft.mass_kg),
        coefficient_scale=(0.9, 1.1),
        force_error_abs_n=np.full(3, 0.01),
        moment_error_abs_n_m=np.full(3, 0.001),
        angular_accel_error_abs_rad_s2=np.full(3, 0.1),
        actuator_tau_s=(0.05, 0.07),
        command_error_abs_rad=np.full(3, radians(0.5)),
    )
