"""Runtime retrieval timing report helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from moewe.sim.state import FlightState

from .retrieve import PrimitiveLibrary


@dataclass(frozen=True)
class RetrievalTimingCheck:
    """Finite timing summary for repeated primitive retrieval queries."""

    tier: str
    query_count: int
    mean_ms: float
    p95_ms: float
    p99_ms: float
    fallback_count: int
    candidate_count_min: int
    candidate_count_max: int

    def to_record(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "query_count": self.query_count,
            "mean_ms": self.mean_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "fallback_count": self.fallback_count,
            "candidate_count_min": self.candidate_count_min,
            "candidate_count_max": self.candidate_count_max,
        }


def check_retrieval_timing(
    library: PrimitiveLibrary,
    states: tuple[FlightState, ...] | list[FlightState],
    tier: str = "balanced",
    repeat: int = 10,
) -> RetrievalTimingCheck:
    """Measure retrieval latency without running controller or governor logic."""

    state_list = tuple(states)
    if not state_list:
        raise ValueError("Retrieval timing report requires at least one state.")
    if int(repeat) <= 0:
        raise ValueError("Retrieval timing repeat must be positive.")
    durations_ms: list[float] = []
    fallback_count = 0
    candidate_counts: list[int] = []
    for _ in range(int(repeat)):
        for state in state_list:
            start = perf_counter_ns()
            query = library.query(state, tier=tier)
            stop = perf_counter_ns()
            durations_ms.append((stop - start) / 1_000_000.0)
            fallback_count += int(query.fallback_used)
            candidate_counts.append(len(query.candidates))
    values = np.asarray(durations_ms, dtype=float)
    return RetrievalTimingCheck(
        tier=tier,
        query_count=int(values.size),
        mean_ms=float(np.mean(values)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        fallback_count=int(fallback_count),
        candidate_count_min=int(min(candidate_counts)),
        candidate_count_max=int(max(candidate_counts)),
    )
