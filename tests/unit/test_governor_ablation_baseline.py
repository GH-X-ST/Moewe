from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from moewe.baselines import (
    UnfilteredPrimitiveSelector,
    UnfilteredPrimitiveSelectorConfig,
    compare_governor_to_unfiltered,
)
from moewe.governor import OnlineGovernor
from moewe.primitives import PrimitiveLibraryCandidate, PrimitiveLibraryQuery
from moewe.returnability import PrimitiveTransition, ReturnabilityGraph
from moewe.sim.state import FlightState


@dataclass(frozen=True)
class _FixedLibrary:
    query_result: PrimitiveLibraryQuery

    def query(self, state: FlightState, tier: str = "balanced") -> PrimitiveLibraryQuery:
        assert state.finite()
        assert tier == self.query_result.tier
        return self.query_result


def _state() -> FlightState:
    return FlightState(
        position_w_m=np.array([0.0, 0.0, 4.0]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([12.0, 0.0, 0.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def _feature(
    *,
    energy: float = 0.0,
    safety: float = 0.0,
    gate_miss: float = 0.0,
    lift: float = 0.0,
) -> dict[str, object]:
    return {
        "terminal_specific_energy_change_j_kg": energy,
        "min_safety_margin_m": safety,
        "gate_miss_distance_m": gate_miss,
        "mean_positive_vertical_wind_m_s": lift,
    }


def _candidate(
    primitive_id: str,
    *,
    entry_class: str = "entry",
    exit_class: str = "terminal",
    feature: dict[str, object] | None = None,
) -> PrimitiveLibraryCandidate:
    return PrimitiveLibraryCandidate(
        primitive_id=primitive_id,
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class=entry_class,
        exit_class=exit_class,
        represented_primitive_ids=(primitive_id,),
        feature_record=_feature() if feature is None else feature,
    )


def _query(
    candidates: tuple[PrimitiveLibraryCandidate, ...],
    *,
    entry_class: str = "entry",
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> PrimitiveLibraryQuery:
    return PrimitiveLibraryQuery(
        tier="balanced",
        entry_class=entry_class,
        candidates=candidates,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        available_entry_classes=tuple(sorted({candidate.entry_class for candidate in candidates})),
    )


def _transition(
    primitive_id: str,
    entry_class: str,
    exit_class: str,
    *,
    retained: bool = True,
    safety_margin: float = 1.0,
    energy_change: float = 1.0,
) -> PrimitiveTransition:
    return PrimitiveTransition(
        primitive_id=primitive_id,
        design_case_id="design_still_air_trim_recovery",
        case_set="library_design",
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class=entry_class,
        exit_class=exit_class,
        retained=retained,
        rollout_success=retained,
        min_safety_margin_m=safety_margin,
        terminal_specific_energy_change_j_kg=energy_change,
        terminal_specific_energy_margin_j_kg=None,
        max_angle_of_attack_rad=0.1,
        max_command_abs_rad=0.1,
        failure_reason=None if retained else "floor",
        retention_reason="retained" if retained else "rollout_failure:floor",
        scenario_id="design_still_air_trim_recovery",
    )


def _graph(
    transitions: tuple[PrimitiveTransition, ...],
    *,
    safe_classes: tuple[str, ...] = ("entry", "terminal"),
    recoverable_classes: tuple[str, ...] = ("entry", "terminal"),
    terminal_success_classes: tuple[str, ...] = ("terminal",),
    forbidden_classes: tuple[str, ...] = (),
) -> ReturnabilityGraph:
    return ReturnabilityGraph(
        transitions=transitions,
        safe_classes=frozenset(safe_classes),
        forbidden_classes=frozenset(forbidden_classes),
        terminal_success_classes=frozenset(terminal_success_classes),
        recoverable_classes=frozenset(recoverable_classes),
        dead_end_classes=frozenset(),
        entry_supported_classes=frozenset(transition.entry_class for transition in transitions),
        exit_observed_classes=frozenset(transition.exit_class for transition in transitions),
    )


def test_unfiltered_selector_ranks_feature_records_deterministically() -> None:
    candidates = (
        _candidate("prim_b", feature=_feature(energy=2.0, safety=0.0)),
        _candidate("prim_a", feature=_feature(energy=2.0, safety=0.0)),
        _candidate("prim_c", feature=_feature(energy=1.0)),
    )
    selector = UnfilteredPrimitiveSelector(_FixedLibrary(_query(candidates)))

    decision = selector.decide(_state())

    assert decision.selected_candidate_id == "prim_a"
    assert [score.candidate_id for score in decision.candidate_scores] == ["prim_a", "prim_b", "prim_c"]
    assert decision.selected_score == 2.0


def test_retrieval_fallback_is_allowed_by_default_and_recorded() -> None:
    candidate = _candidate("prim_available", entry_class="available")
    query = _query(
        (candidate,),
        entry_class="requested",
        fallback_used=True,
        fallback_reason="no_exact_entry_class",
    )
    selector = UnfilteredPrimitiveSelector(_FixedLibrary(query))

    decision = selector.decide(_state())

    assert decision.fallback_used
    assert decision.fallback_reason == "no_exact_entry_class"
    assert decision.selected_candidate_id == "prim_available"
    assert decision.rejection_reasons == ()


def test_retrieval_fallback_can_be_rejected_explicitly() -> None:
    candidate = _candidate("prim_available", entry_class="available")
    query = _query(
        (candidate,),
        entry_class="requested",
        fallback_used=True,
        fallback_reason="no_exact_entry_class",
    )
    selector = UnfilteredPrimitiveSelector(
        _FixedLibrary(query),
        UnfilteredPrimitiveSelectorConfig(allow_retrieval_fallback=False),
    )

    decision = selector.decide(_state())

    assert decision.selected_candidate_id is None
    assert "retrieval_fallback_not_allowed" in decision.rejection_reasons


def test_comparison_marks_governor_blocking_unfiltered_selection() -> None:
    unsafe = _candidate("prim_unsafe", exit_class="forbidden", feature=_feature(energy=10.0))
    safe = _candidate("prim_safe", exit_class="terminal", feature=_feature(energy=1.0))
    query = _query((unsafe, safe))
    library = _FixedLibrary(query)
    graph = _graph(
        (
            _transition("prim_unsafe", "entry", "forbidden"),
            _transition("prim_safe", "entry", "terminal"),
        ),
        safe_classes=("entry", "terminal"),
        recoverable_classes=("entry", "terminal"),
        forbidden_classes=("forbidden",),
    )
    governor = OnlineGovernor(library, graph)
    selector = UnfilteredPrimitiveSelector(library)

    comparison = compare_governor_to_unfiltered(_state(), governor, selector)

    assert comparison.unfiltered_selected_candidate_id == "prim_unsafe"
    assert comparison.governor_selected_candidate_id == "prim_safe"
    assert comparison.same_selected_candidate is False
    assert comparison.governor_blocked_unfiltered_selection is True
    assert "forbidden_exit" in comparison.governor_rejection_reasons


def test_decision_and_comparison_records_are_deterministic_and_json_serialisable() -> None:
    unsafe = _candidate("prim_unsafe", exit_class="forbidden", feature=_feature(energy=10.0))
    safe = _candidate("prim_safe", exit_class="terminal", feature=_feature(energy=1.0))
    query = _query((unsafe, safe))
    library = _FixedLibrary(query)
    graph = _graph(
        (
            _transition("prim_unsafe", "entry", "forbidden"),
            _transition("prim_safe", "entry", "terminal"),
        ),
        safe_classes=("entry", "terminal"),
        recoverable_classes=("entry", "terminal"),
        forbidden_classes=("forbidden",),
    )
    governor = OnlineGovernor(library, graph)
    selector = UnfilteredPrimitiveSelector(library)

    first_decision = selector.decide(_state()).to_record()
    second_decision = selector.decide(_state()).to_record()
    first_comparison = compare_governor_to_unfiltered(_state(), governor, selector).to_record()
    second_comparison = compare_governor_to_unfiltered(_state(), governor, selector).to_record()

    assert first_decision == second_decision
    assert first_comparison == second_comparison
    json.dumps(first_decision, sort_keys=True)
    json.dumps(first_comparison, sort_keys=True)
