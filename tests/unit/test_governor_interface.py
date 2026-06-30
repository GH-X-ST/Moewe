from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from moewe.governor import (
    DecisionType,
    ManoeuvrePrimitiveGovernor,
    PrimitiveRequest,
    RejectionReason,
)
from moewe.objectives import GateTraversalProposer, LiftExploitationProposer
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
        position_w_m=np.array([0.0, 0.0, 1.0]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([7.0, 0.0, 0.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def _candidate(
    primitive_id: str,
    *,
    exit_class: str = "terminal",
    aggressiveness: int = 1,
    feature: dict[str, object] | None = None,
) -> PrimitiveLibraryCandidate:
    features = {"aggressiveness_level": aggressiveness}
    if feature:
        features.update(feature)
    return PrimitiveLibraryCandidate(
        primitive_id=primitive_id,
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class="entry",
        exit_class=exit_class,
        represented_primitive_ids=(primitive_id,),
        feature_record=features,
    )


def _transition(
    primitive_id: str,
    exit_class: str,
    *,
    retained: bool = True,
    safety_margin: float = 1.0,
    failure_reason: str | None = None,
) -> PrimitiveTransition:
    return PrimitiveTransition(
        primitive_id=primitive_id,
        design_case_id="design",
        case_set="library_design",
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class="entry",
        exit_class=exit_class,
        retained=retained,
        rollout_success=failure_reason is None,
        min_safety_margin_m=safety_margin,
        terminal_specific_energy_change_j_kg=1.0,
        terminal_specific_energy_margin_j_kg=None,
        max_angle_of_attack_rad=0.1,
        max_command_abs_rad=0.1,
        failure_reason=failure_reason,
        retention_reason="retained" if retained else "rollout_failure",
        scenario_id="design",
    )


def _graph(transitions: tuple[PrimitiveTransition, ...]) -> ReturnabilityGraph:
    return ReturnabilityGraph(
        transitions=transitions,
        safe_classes=frozenset({"entry", "terminal", "dead"}),
        forbidden_classes=frozenset({"forbidden"}),
        terminal_success_classes=frozenset({"terminal"}),
        recoverable_classes=frozenset({"entry", "terminal"}),
        dead_end_classes=frozenset({"dead"}),
        entry_supported_classes=frozenset({"entry"}),
        exit_observed_classes=frozenset(transition.exit_class for transition in transitions),
    )


def _query(candidates: tuple[PrimitiveLibraryCandidate, ...]) -> PrimitiveLibraryQuery:
    return PrimitiveLibraryQuery(
        tier="balanced",
        entry_class="entry",
        candidates=candidates,
        fallback_used=False,
        fallback_reason=None,
        available_entry_classes=("entry",),
    )


def _request(requested_primitive_id: str | None = None) -> PrimitiveRequest:
    return PrimitiveRequest(
        request_id="req_gate",
        task_intent="gate_traversal",
        preferred_family="bank_pitch_dwell_recovery",
        requested_aggressiveness=2,
        requested_primitive_id=requested_primitive_id,
        objective_score_terms={"gate_alignment": 1.0},
    )


def test_request_and_decision_records_are_deterministic() -> None:
    request = _request("prim_ok")
    candidate = _candidate("prim_ok", aggressiveness=1)
    governor = ManoeuvrePrimitiveGovernor(_FixedLibrary(_query((candidate,))), _graph((_transition("prim_ok", "terminal"),)))

    decision = governor.decide(_state(), request)

    assert request.to_record() == request.to_record()
    assert decision.to_record() == decision.to_record()
    assert decision.to_record()["decision_type"] == "accept"
    assert {"request_id", "decision_type", "active_constraints", "runtime_ms"} <= set(decision.to_record())
    assert decision.runtime_ms >= 0.0
    json.dumps(decision.to_record(), sort_keys=True)


def test_manoeuvre_governor_accepts_requested_admissible_primitive() -> None:
    candidate = _candidate("prim_ok", aggressiveness=1)
    governor = ManoeuvrePrimitiveGovernor(_FixedLibrary(_query((candidate,))), _graph((_transition("prim_ok", "terminal"),)))

    decision = governor.decide(_state(), _request("prim_ok"))

    assert decision.decision_type == DecisionType.ACCEPT
    assert decision.selected_primitive_id == "prim_ok"
    assert decision.requested_primitive_id == "prim_ok"


def test_manoeuvre_governor_degrades_to_weaker_same_family() -> None:
    unsafe = _candidate("prim_hot", exit_class="forbidden", aggressiveness=2)
    safe = _candidate("prim_mild", exit_class="terminal", aggressiveness=0)
    graph = _graph(
        (
            _transition("prim_hot", "forbidden", failure_reason="wall"),
            _transition("prim_mild", "terminal"),
        )
    )
    governor = ManoeuvrePrimitiveGovernor(_FixedLibrary(_query((unsafe, safe))), graph)

    decision = governor.decide(_state(), _request("prim_hot"))

    assert decision.decision_type == DecisionType.DEGRADE
    assert decision.selected_primitive_id == "prim_mild"
    assert decision.degradation_stage is not None
    assert "nonreturnable_successor" in decision.rejection_reasons


def test_manoeuvre_governor_rejects_when_no_viable_primitive_exists() -> None:
    unsafe = _candidate("prim_hot", exit_class="forbidden", aggressiveness=2)
    graph = _graph((_transition("prim_hot", "forbidden", failure_reason="wall"),))
    governor = ManoeuvrePrimitiveGovernor(_FixedLibrary(_query((unsafe,))), graph)

    decision = governor.decide(_state(), _request("prim_hot"))

    assert decision.decision_type == DecisionType.REJECT
    assert RejectionReason.NO_VIABLE_ACTION.value in decision.rejection_reasons


def test_manoeuvre_governor_ranks_only_filtered_candidates() -> None:
    unsafe = _candidate("prim_unsafe", exit_class="forbidden", aggressiveness=0)
    safe = _candidate("prim_safe", exit_class="terminal", aggressiveness=1)
    graph = _graph(
        (
            _transition("prim_unsafe", "forbidden", failure_reason="wall"),
            _transition("prim_safe", "terminal"),
        )
    )
    governor = ManoeuvrePrimitiveGovernor(_FixedLibrary(_query((unsafe, safe))), graph)

    decision = governor.decide(_state(), _request(None))

    assert decision.decision_type == DecisionType.RANK
    assert decision.selected_primitive_id == "prim_safe"


def test_objective_proposer_change_does_not_bypass_governor_filters() -> None:
    unsafe = _candidate("prim_hot", exit_class="forbidden", aggressiveness=2)
    safe = _candidate("prim_mild", exit_class="terminal", aggressiveness=0)
    graph = _graph(
        (
            _transition("prim_hot", "forbidden", failure_reason="wall"),
            _transition("prim_mild", "terminal"),
        )
    )
    governor = ManoeuvrePrimitiveGovernor(_FixedLibrary(_query((unsafe, safe))), graph)
    gate_request = GateTraversalProposer(requested_aggressiveness=2).propose()
    lift_request = LiftExploitationProposer().propose()
    gate_request = PrimitiveRequest(**{**gate_request.to_record(), "requested_primitive_id": "prim_hot"})
    lift_request = PrimitiveRequest(**{**lift_request.to_record(), "requested_primitive_id": "prim_hot"})

    gate_decision = governor.decide(_state(), gate_request)
    lift_decision = governor.decide(_state(), lift_request)

    assert gate_decision.selected_primitive_id == "prim_mild"
    assert lift_decision.selected_primitive_id == "prim_mild"


def test_memory_enabled_request_remains_proposer_metadata_only() -> None:
    unsafe = _candidate("prim_hot", exit_class="forbidden", aggressiveness=2)
    safe = _candidate("prim_mild", exit_class="terminal", aggressiveness=0)
    graph = _graph(
        (
            _transition("prim_hot", "forbidden", failure_reason="wall"),
            _transition("prim_mild", "terminal"),
        )
    )
    governor = ManoeuvrePrimitiveGovernor(_FixedLibrary(_query((unsafe, safe))), graph)
    request = LiftExploitationProposer(memory_enabled=True).propose()
    request = PrimitiveRequest(**{**request.to_record(), "requested_primitive_id": "prim_hot"})

    decision = governor.decide(_state(), request)
    record = decision.to_record()

    assert record["request_summary"]["metadata"]["memory_evidence"] == "enabled"
    assert "successor_returnability" in record["active_constraints"]
    assert decision.selected_primitive_id == "prim_mild"
