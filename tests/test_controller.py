"""Committed terminal-controller state-machine tests."""

from __future__ import annotations

import numpy as np
import pytest

from control.controller import CertifiedPlan, PlanCertifier, TerminalController
from control.interval import Interval
from control.tube import BodyTube, FeedbackSegment, SegmentTube
from models.aircraft import Aircraft


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
    controller.activate(
        original,
        _state_box(0.0),
        _Mission(True),
    )

    controller.advance(_state(1.0), _state_box(1.0))
    assert controller.status == "active"
    assert controller.plan is not None
    assert controller.plan.remaining_horizon == 2
    assert controller.plan.feedback_segments[0] is original.feedback_segments[1]
    assert controller.plan.successor_sets[0] is original.successor_sets[1]
    assert controller.plan.state_tubes[0] is original.state_tubes[1]
    assert controller.plan.body_tubes[0] is original.body_tubes[1]

    controller.advance(_state(2.0), _state_box(2.0))
    assert controller.plan is not None
    assert controller.plan.remaining_horizon == 1
    controller.advance(_state(3.0), _state_box(3.0), terminal_event=True)
    assert controller.status == "terminal"
    assert controller.plan is None
    with pytest.raises(RuntimeError, match="active certificate"):
        controller.command(_state(3.0), _state_box(3.0))


def test_successor_escape_and_missing_final_event() -> None:
    """Reject escaped beliefs and a final interval without its event."""

    aircraft = Aircraft()
    escaped = TerminalController(aircraft)
    escaped.activate(_plan(2), _state_box(0.0), _Mission(True))
    escaped.advance(_state(5.0), _state_box(5.0))
    assert escaped.status == "out_of_envelope"
    assert escaped.plan is None

    missed = TerminalController(aircraft)
    missed.activate(_plan(1), _state_box(0.0), _Mission(True))
    missed.advance(_state(1.0), _state_box(1.0))
    assert missed.status == "out_of_envelope"
    assert missed.plan is None


def test_observed_terminal_event_absorbs_before_successor_check() -> None:
    """Accept physical contact without requiring a flying successor state."""

    controller = TerminalController(Aircraft())
    controller.activate(_plan(1), _state_box(0.0), _Mission(True))
    controller.advance(
        _state(100.0),
        _state_box(100.0),
        terminal_event=True,
    )
    assert controller.status == "terminal"
    assert controller.plan is None


def test_plan_certifier_builds_coherent_covered_tails() -> None:
    """Compile a terminal witness whose stored tails cover each successor."""

    segments = _plan(3).feedback_segments
    mission = _Mission(terminal_result=True)
    certifier = PlanCertifier(_StubPropagator())
    plan = certifier.certify(
        _state_box(0.0),
        segments,
        mission,
    )
    assert plan.feedback_segments == segments
    assert plan.remaining_horizon == 3
    assert plan.mission_identity == "gate"
    assert mission.checked_tubes is not None
    assert len(mission.checked_tubes) == 3

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
) -> CertifiedPlan:
    segments = []
    successors = []
    states = []
    bodies = []
    for index in range(horizon):
        position = start + index
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
        successors.append(_state_box(position + 1.0))
        bodies.append(_body_tube(position))
    return CertifiedPlan(
        feedback_segments=tuple(segments),
        successor_sets=tuple(successors),
        state_tubes=tuple(states),
        body_tubes=tuple(bodies),
        remaining_horizon=horizon,
        mission_identity=mission_identity,
    )


def _state(position: float) -> np.ndarray:
    state = np.zeros(15)
    state[0] = position
    return state


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
    ) -> None:
        self.terminal_result = terminal_result
        self.identity = identity
        self.checked_tubes: tuple[SegmentTube, ...] | None = None

    def terminal(self, tubes: tuple[SegmentTube, ...]) -> bool:
        self.checked_tubes = tubes
        return self.terminal_result


class _StubPropagator:
    def propagate_plan(
        self,
        initial: Interval,
        segments: tuple[FeedbackSegment, ...],
        dt_s: float,
        terminal: object = None,
        deadline: float | None = None,
    ) -> tuple[SegmentTube, ...]:
        assert dt_s == 0.02
        tubes = []
        state = initial
        position = float(initial.center[0])
        for index, _ in enumerate(segments):
            successor = _state_box(position + index + 1.0)
            continuous = state.hull(successor)
            tubes.append(
                SegmentTube(
                    initial=state,
                    successor=successor,
                    states=(continuous,),
                    body=_body_tube(position + index),
                )
            )
            state = successor
        return tuple(tubes)
