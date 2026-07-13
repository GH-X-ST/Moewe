"""Committed terminal-plan certificate and runtime controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

import numpy as np
import numpy.typing as npt

from control.interval import Interval, Zonotope
from control.missions import Mission
from control.tube import BodyTube, FeedbackSegment, TubePropagator
from models.aircraft import Aircraft


@dataclass(frozen=True)
class CertifiedPlan:
    """Complete robust witness from the current set to termination."""

    feedback_segments: tuple[FeedbackSegment, ...]
    successor_sets: tuple[Interval, ...]
    state_tubes: tuple[tuple[Interval, ...], ...]
    body_tubes: tuple[BodyTube, ...]
    command_histories: tuple[tuple[Interval, ...], ...]
    remaining_horizon: int
    mission_identity: str

    def __post_init__(self) -> None:
        horizon = len(self.feedback_segments)
        lengths = (
            len(self.successor_sets),
            len(self.state_tubes),
            len(self.body_tubes),
            len(self.command_histories),
        )
        if horizon == 0 or self.remaining_horizon != horizon or any(
            length != horizon for length in lengths
        ):
            raise ValueError("certified plan components must match its horizon")

    def tail(self) -> CertifiedPlan:
        """Return the certified plan after its first segment."""

        if self.remaining_horizon == 1:
            raise ValueError("terminal plan has no nonterminal tail")

        return CertifiedPlan(
            self.feedback_segments[1:],
            self.successor_sets[1:],
            self.state_tubes[1:],
            self.body_tubes[1:],
            self.command_histories[1:],
            self.remaining_horizon - 1,
            self.mission_identity,
        )


@dataclass
class PlanCertifier:
    """Compile feedback segments into one terminal certificate."""

    propagator: TubePropagator
    dt_s: float = 0.02

    def certify(
        self,
        initial: Interval,
        segments: tuple[FeedbackSegment, ...],
        mission: Mission,
        time_limit_s: float | None = None,
        initial_command_history: tuple[Interval, ...] = (),
    ) -> CertifiedPlan:
        """Return a certified plan or raise when a contract is not met."""

        deadline = None if time_limit_s is None else monotonic() + time_limit_s
        tubes = self.propagator.propagate_plan(
            initial,
            segments,
            self.dt_s,
            terminal=mission.terminal,
            deadline=deadline,
            initial_command_history=initial_command_history,
        )
        if not mission.terminal(tubes):
            raise ValueError("plan violates the mission certificate")
        return CertifiedPlan(
            segments,
            tuple(_interval_hull(tube.successor) for tube in tubes),
            tuple((_interval_hull(tube.initial),) + tube.states for tube in tubes),
            tuple(tube.body for tube in tubes),
            tuple(tube.command_history for tube in tubes),
            len(segments),
            mission.identity,
        )


@dataclass
class TerminalController:
    """Execute and shorten a committed certified terminal plan."""

    aircraft: Aircraft
    plan: CertifiedPlan | None = None
    status: str = "inactive"
    _mission: Mission | None = field(default=None, init=False, repr=False)

    def activate(
        self,
        plan: CertifiedPlan,
        belief: Interval,
        mission: Mission,
        command_history: tuple[Interval, ...] = (),
    ) -> None:
        """Activate an initially feasible complete terminal plan."""

        if not belief.subset(_initial_set(plan)):
            raise ValueError("initial belief is outside the certificate")
        if plan.mission_identity != mission.identity:
            raise ValueError("plan and mission identities differ")
        if not _command_history_subset(
            command_history,
            plan.command_histories[0],
        ):
            raise ValueError("initial command history is outside the certificate")
        self.plan = plan
        self._mission = mission
        self.status = "active"

    def command(
        self,
        state: npt.ArrayLike,
        belief: Interval,
        candidate: CertifiedPlan | None = None,
    ) -> np.ndarray:
        """Return the current command, accepting only a covered candidate."""

        if self.plan is None or self.status != "active":
            raise RuntimeError("controller has no active certificate")
        if not belief.contains(state):
            self.status = "out_of_envelope"
            self.plan = None
            self._mission = None
            raise ValueError("state estimate is outside the current belief")
        if not belief.subset(_initial_set(self.plan)):
            self.status = "out_of_envelope"
            self.plan = None
            self._mission = None
            raise ValueError("belief is outside the active certificate")
        if candidate is not None and self._accepts(candidate, belief):
            self.plan = candidate
        return self.plan.feedback_segments[0].command(state, self.aircraft)

    def advance(
        self,
        states: npt.ArrayLike,
        belief: Interval,
    ) -> None:
        """Advance after checking the realized trajectory against the mission."""

        if self.plan is None or self._mission is None or self.status != "active":
            raise RuntimeError("controller has no active certificate")
        trajectory = np.asarray(states, dtype=float)
        if (
            trajectory.ndim != 2
            or trajectory.shape[0] < 2
            or trajectory.shape[1] != 15
        ):
            raise ValueError("realized trajectory must contain at least two states")
        if not _initial_set(self.plan).contains(trajectory[0]):
            self.status = "out_of_envelope"
            self.plan = None
            self._mission = None
            return
        state = trajectory[-1]
        if not belief.contains(state):
            self.status = "out_of_envelope"
            self.plan = None
            self._mission = None
            return
        successor = self.plan.successor_sets[0]
        if not belief.subset(successor):
            self.status = "out_of_envelope"
            self.plan = None
            self._mission = None
            return
        terminal_event = self._mission.realized(trajectory)
        if terminal_event:
            self.status = "terminal"
            self.plan = None
            self._mission = None
            return
        if self.plan.remaining_horizon == 1:
            self.status = "out_of_envelope"
            self.plan = None
            self._mission = None
            return
        self.plan = self.plan.tail()

    def _accepts(
        self,
        candidate: CertifiedPlan,
        belief: Interval,
    ) -> bool:
        if self.plan is None:
            return False
        return bool(
            candidate.mission_identity == self.plan.mission_identity
            and candidate.remaining_horizon <= self.plan.remaining_horizon
            and belief.subset(_initial_set(candidate))
            and _command_history_subset(
                self.plan.command_histories[0],
                candidate.command_histories[0],
            )
        )


def _initial_set(plan: CertifiedPlan) -> Interval:
    return plan.state_tubes[0][0]


def _command_history_subset(
    history: tuple[Interval, ...],
    certificate: tuple[Interval, ...],
) -> bool:
    return len(history) == len(certificate) and all(
        value.subset(bound)
        for value, bound in zip(history, certificate, strict=True)
    )


def _interval_hull(value: Interval | Zonotope) -> Interval:
    return value if isinstance(value, Interval) else value.interval_hull()
