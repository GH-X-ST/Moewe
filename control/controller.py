"""Committed terminal-plan certificate and runtime controller."""

from __future__ import annotations

from dataclasses import dataclass
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
    remaining_horizon: int
    mission_identity: str

    def tail(self) -> CertifiedPlan:
        """Return the certified plan after its first segment."""

        return CertifiedPlan(
            self.feedback_segments[1:],
            self.successor_sets[1:],
            self.state_tubes[1:],
            self.body_tubes[1:],
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
    ) -> CertifiedPlan:
        """Return a certified plan or raise when a contract is not met."""

        deadline = None if time_limit_s is None else monotonic() + time_limit_s
        tubes = self.propagator.propagate_plan(
            initial,
            segments,
            self.dt_s,
            mission.terminal,
            deadline,
        )
        if not mission.terminal(tubes):
            raise ValueError("plan violates the mission certificate")
        return CertifiedPlan(
            segments,
            tuple(_interval_hull(tube.successor) for tube in tubes),
            tuple((_interval_hull(tube.initial),) + tube.states for tube in tubes),
            tuple(tube.body for tube in tubes),
            len(segments),
            mission.identity,
        )


@dataclass
class TerminalController:
    """Execute and shorten a committed certified terminal plan."""

    aircraft: Aircraft
    plan: CertifiedPlan | None = None
    status: str = "inactive"

    def activate(
        self,
        plan: CertifiedPlan,
        belief: Interval,
        mission: Mission,
    ) -> None:
        """Activate an initially feasible complete terminal plan."""

        if not belief.subset(_initial_set(plan)):
            raise ValueError("initial belief is outside the certificate")
        if plan.mission_identity != mission.identity:
            raise ValueError("plan and mission identities differ")
        self.plan = plan
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
            raise ValueError("state estimate is outside the current belief")
        if not belief.subset(_initial_set(self.plan)):
            self.status = "out_of_envelope"
            self.plan = None
            raise ValueError("belief is outside the active certificate")
        if candidate is not None and self._accepts(candidate, belief):
            self.plan = candidate
        return self.plan.feedback_segments[0].command(state, self.aircraft)

    def advance(
        self,
        state: npt.ArrayLike,
        belief: Interval,
        terminal_event: bool = False,
    ) -> None:
        """Advance after one interval using the observed terminal event."""

        if self.plan is None or self.status != "active":
            raise RuntimeError("controller has no active certificate")
        if terminal_event:
            self.status = "terminal"
            self.plan = None
            return
        if not belief.contains(state):
            self.status = "out_of_envelope"
            self.plan = None
            return
        successor = self.plan.successor_sets[0]
        if not belief.subset(successor):
            self.status = "out_of_envelope"
            self.plan = None
            return
        if self.plan.remaining_horizon == 1:
            self.status = "out_of_envelope"
            self.plan = None
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
        )


def _initial_set(plan: CertifiedPlan) -> Interval:
    return plan.state_tubes[0][0]


def _interval_hull(value: Interval | Zonotope) -> Interval:
    return value if isinstance(value, Interval) else value.interval_hull()
