"""Runtime checks for online governor decisions."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from moewe.sim.state import FlightState

from .policy import OnlineGovernor


@dataclass(frozen=True)
class GovernorTimingCheck:
    """Timing summary for repeated online governor decisions."""

    tier: str
    decision_count: int
    mean_ms: float
    p95_ms: float
    p99_ms: float
    selected_count: int
    no_selection_count: int
    fallback_count: int
    admissible_candidate_count_min: int
    admissible_candidate_count_max: int

    def to_record(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "decision_count": self.decision_count,
            "mean_ms": self.mean_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "selected_count": self.selected_count,
            "no_selection_count": self.no_selection_count,
            "fallback_count": self.fallback_count,
            "admissible_candidate_count_min": self.admissible_candidate_count_min,
            "admissible_candidate_count_max": self.admissible_candidate_count_max,
        }


def check_governor_timing(
    governor: OnlineGovernor,
    states: Iterable[FlightState],
    *,
    tier: str = "balanced",
    repeat: int = 10,
) -> GovernorTimingCheck:
    """Run repeated governor decisions and return finite timing statistics."""

    if repeat <= 0:
        raise ValueError("repeat must be positive.")
    state_list = tuple(states)
    if not state_list:
        raise ValueError("At least one state is required.")

    durations_ms: list[float] = []
    selected_count = 0
    fallback_count = 0
    admissible_counts: list[int] = []
    for _ in range(repeat):
        for state in state_list:
            start = perf_counter()
            decision = governor.decide(state, tier=tier)
            durations_ms.append((perf_counter() - start) * 1000.0)
            selected_count += int(decision.selected_transition_id is not None)
            fallback_count += int(decision.fallback_used)
            admissible_counts.append(decision.admissible_candidate_count)

    values = np.asarray(durations_ms, dtype=float)
    decision_count = len(durations_ms)
    return GovernorTimingCheck(
        tier=tier,
        decision_count=decision_count,
        mean_ms=float(np.mean(values)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        selected_count=selected_count,
        no_selection_count=decision_count - selected_count,
        fallback_count=fallback_count,
        admissible_candidate_count_min=int(min(admissible_counts)),
        admissible_candidate_count_max=int(max(admissible_counts)),
    )
