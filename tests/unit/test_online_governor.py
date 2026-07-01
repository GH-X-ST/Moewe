from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from moewe.governor import OnlineGovernor, OnlineGovernorConfig
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


def _candidate(
    primitive_id: str,
    *,
    entry_class: str = "entry",
    exit_class: str = "terminal",
    represented_primitive_ids: tuple[str, ...] | None = None,
) -> PrimitiveLibraryCandidate:
    return PrimitiveLibraryCandidate(
        primitive_id=primitive_id,
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class=entry_class,
        exit_class=exit_class,
        represented_primitive_ids=(primitive_id,) if represented_primitive_ids is None else represented_primitive_ids,
        feature_record={},
    )


def _transition(
    primitive_id: str,
    entry_class: str,
    exit_class: str,
    *,
    retained: bool = True,
    rollout_success: bool = True,
    safety_margin: float = 1.0,
    energy_change: float = 1.0,
    design_case_id: str = "design_still_air_trim_recovery",
) -> PrimitiveTransition:
    return PrimitiveTransition(
        primitive_id=primitive_id,
        design_case_id=design_case_id,
        case_set="library_design",
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class=entry_class,
        exit_class=exit_class,
        retained=retained,
        rollout_success=rollout_success,
        min_safety_margin_m=safety_margin,
        terminal_specific_energy_change_j_kg=energy_change,
        terminal_specific_energy_margin_j_kg=None,
        max_angle_of_attack_rad=0.1,
        max_command_abs_rad=0.1,
        failure_reason=None if rollout_success else "floor",
        retention_reason="retained" if retained else "rollout_failure:floor",
        scenario_id=design_case_id,
    )


def _graph(
    transitions: tuple[PrimitiveTransition, ...],
    *,
    safe_classes: tuple[str, ...] = ("entry", "terminal"),
    recoverable_classes: tuple[str, ...] = ("entry", "terminal"),
    terminal_success_classes: tuple[str, ...] = ("terminal",),
    forbidden_classes: tuple[str, ...] = (),
    dead_end_classes: tuple[str, ...] = (),
) -> ReturnabilityGraph:
    return ReturnabilityGraph(
        transitions=transitions,
        safe_classes=frozenset(safe_classes),
        forbidden_classes=frozenset(forbidden_classes),
        terminal_success_classes=frozenset(terminal_success_classes),
        recoverable_classes=frozenset(recoverable_classes),
        dead_end_classes=frozenset(dead_end_classes),
        entry_supported_classes=frozenset(transition.entry_class for transition in transitions),
        exit_observed_classes=frozenset(transition.exit_class for transition in transitions),
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


def test_governor_selects_recoverable_terminal_candidate() -> None:
    candidate = _candidate("prim_ok")
    graph = _graph((_transition("prim_ok", "entry", "terminal"),))
    governor = OnlineGovernor(_FixedLibrary(_query((candidate,))), graph)

    decision = governor.decide(_state())

    assert decision.selected_candidate_id == "prim_ok"
    assert decision.selected_primitive_id == "prim_ok"
    assert decision.selected_exit_class == "terminal"
    assert decision.selected_reason == "terminal_success_exit"
    assert decision.admissible_candidate_count == 1


def test_governor_rejects_forbidden_and_dead_end_exits() -> None:
    forbidden_candidate = _candidate("prim_forbidden", exit_class="forbidden")
    dead_end_candidate = _candidate("prim_dead", exit_class="dead")
    graph = _graph(
        (
            _transition("prim_forbidden", "entry", "forbidden"),
            _transition("prim_dead", "entry", "dead"),
        ),
        safe_classes=("entry", "dead"),
        recoverable_classes=("entry",),
        terminal_success_classes=(),
        forbidden_classes=("forbidden",),
        dead_end_classes=("dead",),
    )
    governor = OnlineGovernor(_FixedLibrary(_query((dead_end_candidate, forbidden_candidate))), graph)

    decision = governor.decide(_state())
    reasons_by_candidate = {
        evidence.candidate_id: set(evidence.rejection_reasons)
        for evidence in decision.candidate_evidence
    }

    assert decision.selected_candidate_id is None
    assert "no_admissible_candidate" in decision.rejection_reasons
    assert "forbidden_exit" in reasons_by_candidate["prim_forbidden"]
    assert "no_recoverable_exit" in reasons_by_candidate["prim_dead"]


def test_retrieval_fallback_is_rejected_by_default() -> None:
    candidate = _candidate("prim_available", entry_class="available", exit_class="terminal")
    graph = _graph(
        (_transition("prim_available", "available", "terminal"),),
        safe_classes=("requested", "available", "terminal"),
        recoverable_classes=("requested", "available", "terminal"),
    )
    query = _query(
        (candidate,),
        entry_class="requested",
        fallback_used=True,
        fallback_reason="no_exact_entry_class",
    )
    governor = OnlineGovernor(_FixedLibrary(query), graph)

    decision = governor.decide(_state())

    assert decision.fallback_used
    assert decision.selected_candidate_id is None
    assert "retrieval_fallback_not_allowed" in decision.rejection_reasons


def test_retrieval_fallback_can_use_supported_compatibility_class_when_enabled() -> None:
    candidate = _candidate("prim_available", entry_class="available", exit_class="terminal")
    graph = _graph(
        (_transition("prim_available", "available", "terminal"),),
        safe_classes=("available", "terminal"),
        recoverable_classes=("available", "terminal"),
    )
    query = _query(
        (candidate,),
        entry_class="requested",
        fallback_used=True,
        fallback_reason="no_exact_entry_class",
    )
    governor = OnlineGovernor(
        _FixedLibrary(query),
        graph,
        OnlineGovernorConfig(allow_retrieval_fallback=True),
    )

    decision = governor.decide(_state())

    assert decision.fallback_used
    assert decision.selected_candidate_id == "prim_available"
    assert "entry_not_safe" not in decision.rejection_reasons
    assert "entry_not_recoverable" not in decision.rejection_reasons


def test_candidate_evidence_is_deterministic_and_json_serialisable() -> None:
    candidate = _candidate("prim_ok", represented_primitive_ids=("prim_ok", "prim_rep"))
    graph = _graph(
        (
            _transition("prim_ok", "entry", "terminal", design_case_id="design_a"),
            _transition("prim_rep", "entry", "terminal", design_case_id="design_b"),
        )
    )
    governor = OnlineGovernor(_FixedLibrary(_query((candidate,))), graph)

    first = governor.candidate_evidence(candidate).to_record()
    second = governor.candidate_evidence(candidate).to_record()

    assert first == second
    assert first["matched_transition_ids"] == sorted(first["matched_transition_ids"])
    json.dumps(first, sort_keys=True)


def test_governor_tie_breaks_by_candidate_id() -> None:
    candidates = (_candidate("prim_b"), _candidate("prim_a"))
    graph = _graph(
        (
            _transition("prim_b", "entry", "terminal"),
            _transition("prim_a", "entry", "terminal"),
        )
    )
    governor = OnlineGovernor(
        _FixedLibrary(_query(candidates)),
        graph,
        OnlineGovernorConfig(min_safety_margin_m=0.0, min_terminal_specific_energy_change_j_kg=0.0),
    )

    decision = governor.decide(_state())

    assert decision.selected_candidate_id == "prim_a"
