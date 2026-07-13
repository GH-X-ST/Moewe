"""Committed terminal-controller state-machine tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from control.controller import CertifiedPlan, PlanCertifier, TerminalController
from control.interval import Interval
from control.missions import FreeSpace, GateMission
from control.tube import BodyTube, FeedbackSegment, SegmentTube
from models.aircraft import Aircraft
from models.geometry import RigidBodyGeometry


def test_initial_activation_and_continuous_feedback_command() -> None:
    """Activate a covered plan and evaluate its continuous feedback law."""

    aircraft = Aircraft()
    plan = _plan(3)
    controller = TerminalController(aircraft)
    state = _state(0.1)
    mission = _Mission(True)
    controller.activate(plan, _state_box(0.1, 0.1), mission)
    np.testing.assert_allclose(
        controller.command(state, _state_box(0.1, 0.1)),
        (0.012, 0.0, 0.0),
    )
    assert controller.plan is plan
    assert controller.status == "active"

    inactive = TerminalController(aircraft)
    with pytest.raises(ValueError, match="outside"):
        inactive.activate(plan, _state_box(2.0), mission)
    assert inactive.plan is None
    assert inactive.status == "inactive"


def test_delayed_activation_requires_covered_hardware_history() -> None:
    """Match the physical command FIFO to the initial plan certificate."""

    envelope = Interval((-0.03, -0.02, -0.01), (0.03, 0.02, 0.01))
    actual = Interval.point((0.01, 0.0, 0.0))
    plan = _plan(2, command_history=(envelope,))
    mission = _Mission(True)

    missing = TerminalController(Aircraft())
    with pytest.raises(ValueError, match="initial command history"):
        missing.activate(plan, _state_box(0.0), mission)

    covered = TerminalController(Aircraft())
    covered.activate(
        plan,
        _state_box(0.0),
        mission,
        command_history=(actual,),
    )
    assert covered.plan is plan


def test_certified_feedback_arrays_are_immutable_copies() -> None:
    """Prevent a certified law from changing after tube generation."""

    state = _state(0.0)
    control = np.zeros(3)
    gain = np.zeros((3, 15))
    segment = FeedbackSegment(state, control, gain)
    state[0] = 2.0
    control[0] = 0.1
    gain[0, 0] = 1.0
    assert segment.state[0] == 0.0
    assert segment.control_rad[0] == 0.0
    assert segment.gain[0, 0] == 0.0
    with pytest.raises(ValueError):
        segment.control_rad[0] = 0.1


def test_shorter_covered_same_mission_candidate_is_accepted() -> None:
    """Replace the commitment only with a covered shorter certificate."""

    aircraft = Aircraft()
    current = _plan(3)
    state = _state(0.1)
    candidate = _plan(
        2,
        start=0.1,
        mission_identity="gate",
        control_offset=0.04,
    )
    controller = TerminalController(aircraft)
    controller.activate(current, _state_box(0.1, 0.1), _Mission(True))
    np.testing.assert_allclose(
        controller.command(state, _state_box(0.1, 0.1), candidate),
        (0.05, 0.0, 0.0),
    )
    assert controller.plan is candidate
    assert controller.plan.remaining_horizon == 2


def test_candidate_must_cover_the_active_command_history() -> None:
    """Reject replanning against a different delayed-command FIFO."""

    active_history = Interval(
        (-0.02, -0.01, -0.01),
        (0.02, 0.01, 0.01),
    )
    actual_history = Interval.point((0.0, 0.0, 0.0))
    current = _plan(3, command_history=(active_history,))
    controller = TerminalController(Aircraft())
    controller.activate(
        current,
        _state_box(0.1, 0.1),
        _Mission(True),
        command_history=(actual_history,),
    )

    too_small = _plan(
        2,
        start=0.1,
        control_offset=0.04,
        command_history=(Interval.point((0.0, 0.0, 0.0)),),
    )
    controller.command(_state(0.1), _state_box(0.1, 0.1), too_small)
    assert controller.plan is current

    covering = _plan(
        2,
        start=0.1,
        control_offset=0.04,
        command_history=(
            Interval((-0.03, -0.02, -0.02), (0.03, 0.02, 0.02)),
        ),
    )
    controller.command(_state(0.1), _state_box(0.1, 0.1), covering)
    assert controller.plan is covering


@pytest.mark.parametrize(
    ("horizon", "start", "mission_identity"),
    (
        (2, 0.1, "landing"),
        (4, 0.1, "gate"),
        (2, 4.0, "gate"),
    ),
    ids=("wrong_mission", "longer_horizon", "uncovered_state"),
)
def test_invalid_candidates_retain_current_plan(
    horizon: int,
    start: float,
    mission_identity: str,
) -> None:
    """Reject mission, horizon, and initial-coverage mismatches."""

    aircraft = Aircraft()
    current = _plan(3)
    state = _state(0.1)
    candidate = _plan(horizon, start, mission_identity)
    controller = TerminalController(aircraft)
    controller.activate(current, _state_box(0.1, 0.1), _Mission(True))
    np.testing.assert_allclose(
        controller.command(state, _state_box(0.1, 0.1), candidate),
        (0.012, 0.0, 0.0),
    )
    assert controller.plan is current
    assert controller.plan.remaining_horizon == 3


def test_missing_candidate_retains_committed_plan() -> None:
    """Treat planner failure or timeout as a rejected candidate."""

    aircraft = Aircraft()
    current = _plan(3)
    state = _state(0.1)
    controller = TerminalController(aircraft)
    controller.activate(current, _state_box(0.1, 0.1), _Mission(True))
    np.testing.assert_allclose(
        controller.command(state, _state_box(0.1, 0.1), None),
        (0.012, 0.0, 0.0),
    )
    assert controller.plan is current
    assert controller.status == "active"


def test_expanded_belief_cannot_emit_an_uncertified_command() -> None:
    """Reject an observer set that no longer fits the plan inlet."""

    controller = TerminalController(Aircraft())
    controller.activate(_plan(2), _state_box(0.0, 0.1), _Mission(True))
    with pytest.raises(ValueError, match="active certificate"):
        controller.command(_state(0.0), _state_box(0.0, 1.0))
    assert controller.status == "out_of_envelope"
    assert controller.plan is None


def test_successor_coverage_countdown_and_terminal_absorption() -> None:
    """Carry the covered tail to its terminal segment."""

    aircraft = Aircraft()
    original = _plan(3)
    controller = TerminalController(aircraft)
    mission = _Mission(True)
    controller.activate(
        original,
        _state_box(0.0),
        mission,
    )

    controller.advance(_trajectory(0.0, 1.0), _state_box(1.0))
    assert controller.status == "active"
    assert controller.plan is not None
    assert controller.plan.remaining_horizon == 2
    assert controller.plan.feedback_segments[0] is original.feedback_segments[1]
    assert controller.plan.successor_sets[0] is original.successor_sets[1]
    assert controller.plan.state_tubes[0] is original.state_tubes[1]
    assert controller.plan.body_tubes[0] is original.body_tubes[1]

    controller.advance(_trajectory(1.0, 2.0), _state_box(2.0))
    assert controller.plan is not None
    assert controller.plan.remaining_horizon == 1
    mission.realized_result = True
    controller.advance(_trajectory(2.0, 3.0), _state_box(3.0))
    assert controller.status == "terminal"
    assert controller.plan is None
    assert mission.checked_states is not None
    np.testing.assert_array_equal(
        mission.checked_states,
        _trajectory(2.0, 3.0),
    )
    with pytest.raises(RuntimeError, match="active certificate"):
        controller.command(_state(3.0), _state_box(3.0))


def test_terminal_certificate_has_no_flying_tail() -> None:
    """Keep the final event interval as an absorbing certificate."""

    with pytest.raises(ValueError, match="no nonterminal tail"):
        _plan(1).tail()


def test_successor_escape_and_missing_final_event() -> None:
    """Reject escaped beliefs and a final interval without its event."""

    aircraft = Aircraft()
    escaped = TerminalController(aircraft)
    escaped.activate(_plan(2), _state_box(0.0), _Mission(True))
    escaped.advance(_trajectory(0.0, 5.0), _state_box(5.0))
    assert escaped.status == "out_of_envelope"
    assert escaped.plan is None

    missed = TerminalController(aircraft)
    missed.activate(_plan(1), _state_box(0.0), _Mission(True))
    missed.advance(_trajectory(0.0, 1.0), _state_box(1.0))
    assert missed.status == "out_of_envelope"
    assert missed.plan is None


def test_verified_terminal_event_requires_certified_successor() -> None:
    """Accept an event only when its endpoint remains certified."""

    controller = TerminalController(Aircraft())
    mission = _Mission(True, realized_result=True)
    controller.activate(_plan(1), _state_box(0.0), mission)
    controller.advance(_trajectory(0.0, 1.0), _state_box(1.0))
    assert controller.status == "terminal"
    assert controller.plan is None


def test_terminal_event_outside_certified_successor_is_rejected() -> None:
    """Reject geometrically valid terminal telemetry outside its tube."""

    controller = TerminalController(Aircraft())
    mission = _Mission(True, realized_result=True)
    controller.activate(_plan(1), _state_box(0.0), mission)
    controller.advance(_trajectory(0.0, 100.0), _state_box(100.0))
    assert controller.status == "out_of_envelope"
    assert controller.plan is None
    assert mission.checked_states is None


def test_terminal_trajectory_must_start_inside_active_inlet() -> None:
    """Reject terminal telemetry disconnected from the active certificate."""

    controller = TerminalController(Aircraft())
    mission = _Mission(True, realized_result=True)
    controller.activate(_plan(1), _state_box(0.0), mission)
    controller.advance(_trajectory(100.0, 1.0), _state_box(1.0))
    assert controller.status == "out_of_envelope"
    assert controller.plan is None
    assert mission.checked_states is None


def test_terminal_boolean_is_not_part_of_the_public_api() -> None:
    """Prevent a caller-provided flag from bypassing mission evaluation."""

    controller = TerminalController(Aircraft())
    plan = _plan(1)
    controller.activate(plan, _state_box(0.0), _Mission(True))
    with pytest.raises(TypeError, match="terminal_event"):
        controller.advance(  # type: ignore[call-arg]
            _trajectory(0.0, 100.0),
            _state_box(100.0),
            terminal_event=True,
        )
    assert controller.status == "active"
    assert controller.plan is plan


def test_unverified_terminal_event_cannot_absorb() -> None:
    """Reject a final interval when the mission monitor sees no event."""

    controller = TerminalController(Aircraft())
    mission = _Mission(True, realized_result=False)
    controller.activate(_plan(1), _state_box(0.0), mission)
    controller.advance(_trajectory(0.0, 1.0), _state_box(1.0))
    assert controller.status == "out_of_envelope"
    assert controller.plan is None


def test_terminal_monitor_requires_a_dense_state_trajectory() -> None:
    """Reject a single state before mission-event evaluation."""

    controller = TerminalController(Aircraft())
    mission = _Mission(True, realized_result=True)
    controller.activate(_plan(1), _state_box(0.0), mission)
    with pytest.raises(ValueError, match="at least two states"):
        controller.advance(_state(1.0), _state_box(1.0))
    assert controller.status == "active"
    assert mission.checked_states is None


def test_controller_uses_gate_geometry_to_verify_terminal_event() -> None:
    """Absorb only when the active gate contract accepts the trajectory."""

    geometry = RigidBodyGeometry(
        body_b_m=((0.0, 0.0, 0.0),),
        contact_b_m=((0.0, 0.0, 0.0),),
        footprint_b_m=((0.0, 0.0, 0.0),),
    )
    mission = GateMission(
        FreeSpace(((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))),
        center_w_m=(0.0, 0.0, 0.0),
        width_m=2.0,
        height_m=2.0,
        geometry=geometry,
    )
    controller = TerminalController(Aircraft())
    controller.activate(
        _plan(1, start=-1.0, mission_identity=mission.identity, step=2.0),
        _state_box(-1.0),
        mission,
    )
    controller.advance(
        np.stack((_state(-1.0), _state(0.0), _state(1.0))),
        _state_box(1.0),
    )
    assert controller.status == "terminal"
    assert controller.plan is None

    missed = TerminalController(Aircraft())
    missed.activate(
        _plan(1, start=-1.0, mission_identity=mission.identity, step=2.0),
        _state_box(-1.0),
        mission,
    )
    missed.advance(_trajectory(-1.0, 1.0), _state_box(1.0))
    assert missed.status == "out_of_envelope"
    assert missed.plan is None


def test_plan_certifier_builds_coherent_covered_tails() -> None:
    """Compile a terminal witness whose stored tails cover each successor."""

    segments = _plan(3).feedback_segments
    mission = _Mission(terminal_result=True)
    propagator = _StubPropagator()
    certifier = PlanCertifier(propagator)
    command_history = (Interval.point((0.01, 0.0, 0.0)),)
    plan = certifier.certify(
        _state_box(0.0),
        segments,
        mission,
        initial_command_history=command_history,
    )
    assert plan.feedback_segments == segments
    assert plan.remaining_horizon == 3
    assert plan.mission_identity == "gate"
    assert mission.checked_tubes is not None
    assert len(mission.checked_tubes) == 3
    assert propagator.initial_command_history is command_history
    tail = plan.tail()
    np.testing.assert_array_equal(
        tail.command_histories[0][0].lower,
        propagator.command_histories_used[1][0].lower,
    )
    np.testing.assert_array_equal(
        tail.command_histories[0][0].upper,
        propagator.command_histories_used[1][0].upper,
    )

    tail = plan
    while tail.remaining_horizon > 1:
        next_tail = tail.tail()
        assert tail.successor_sets[0].subset(next_tail.state_tubes[0][0])
        tail = next_tail
    assert tail.remaining_horizon == 1


def test_plan_certifier_rejects_failed_terminal_contract() -> None:
    """Reject an otherwise coherent tube sequence without termination."""

    segments = _plan(2).feedback_segments
    mission = _Mission(terminal_result=False)
    certifier = PlanCertifier(_StubPropagator())
    with pytest.raises(ValueError, match="mission certificate"):
        certifier.certify(
            _state_box(0.0),
            segments,
            mission,
        )
    assert mission.checked_tubes is not None


def _plan(
    horizon: int,
    start: float = 0.0,
    mission_identity: str = "gate",
    control_offset: float = 0.0,
    command_history: tuple[Interval, ...] = (),
    step: float = 1.0,
) -> CertifiedPlan:
    segments = []
    successors = []
    states = []
    bodies = []
    for index in range(horizon):
        position = start + index * step
        reference = _state(position)
        gain = np.zeros((3, 15))
        gain[0, 0] = 0.02
        segments.append(
            FeedbackSegment(
                reference,
                (control_offset + 0.01 * (index + 1), 0.0, 0.0),
                gain,
            )
        )
        states.append((_state_box(position),))
        successors.append(_state_box(position + step))
        bodies.append(_body_tube(position))
    return CertifiedPlan(
        feedback_segments=tuple(segments),
        successor_sets=tuple(successors),
        state_tubes=tuple(states),
        body_tubes=tuple(bodies),
        command_histories=tuple(command_history for _ in range(horizon)),
        remaining_horizon=horizon,
        mission_identity=mission_identity,
    )


def _state(position: float) -> np.ndarray:
    state = np.zeros(15)
    state[0] = position
    return state


def _trajectory(start: float, end: float) -> np.ndarray:
    return np.stack((_state(start), _state(end)))


def _state_box(position: float, position_radius: float = 0.25) -> Interval:
    state = _state(position)
    radius = np.zeros(15)
    radius[0] = position_radius
    return Interval.from_midpoint(state, radius)


def _body_tube(position: float) -> BodyTube:
    point = Interval.point(((position, 0.0, 0.0),))
    velocity = Interval.point(((1.0, 0.0, 0.0),))
    return BodyTube(
        occupied=(point,),
        contact=(point,),
        footprint=(point,),
        contact_velocity=(velocity,),
    )


class _Mission:
    def __init__(
        self,
        terminal_result: bool,
        identity: str = "gate",
        realized_result: bool = False,
    ) -> None:
        self.terminal_result = terminal_result
        self.identity = identity
        self.realized_result = realized_result
        self.checked_tubes: tuple[SegmentTube, ...] | None = None
        self.checked_states: np.ndarray | None = None

    def terminal(self, tubes: tuple[SegmentTube, ...]) -> bool:
        self.checked_tubes = tubes
        return self.terminal_result

    def realized(self, states: npt.ArrayLike) -> bool:
        self.checked_states = np.asarray(states, dtype=float)
        return self.realized_result


class _StubPropagator:
    def propagate_plan(
        self,
        initial: Interval,
        segments: tuple[FeedbackSegment, ...],
        dt_s: float,
        terminal: object = None,
        deadline: float | None = None,
        initial_command_history: tuple[Interval, ...] = (),
    ) -> tuple[SegmentTube, ...]:
        assert dt_s == 0.02
        self.initial_command_history = initial_command_history
        self.command_histories_used = []
        tubes = []
        state = initial
        command_history = initial_command_history
        position = float(initial.center[0])
        for index, segment in enumerate(segments):
            self.command_histories_used.append(command_history)
            successor = _state_box(position + index + 1.0)
            continuous = state.hull(successor)
            tubes.append(
                SegmentTube(
                    initial=state,
                    successor=successor,
                    states=(continuous,),
                    body=_body_tube(position + index),
                    command_history=command_history,
                )
            )
            state = successor
            if command_history:
                command_history = command_history[1:] + (
                    Interval.point(segment.control_rad),
                )
        return tuple(tubes)
