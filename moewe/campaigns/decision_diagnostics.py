"""Decision-centric diagnostic records for governor comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from moewe.baselines import UnfilteredPrimitiveSelector
from moewe.governor import GovernorDecisionRecord, ManoeuvrePrimitiveGovernor, PrimitiveRequest
from moewe.sim.state import FlightState


@dataclass(frozen=True)
class DecisionDiagnosticRecord:
    """One in-memory request, decision, and counterfactual diagnostic."""

    request: PrimitiveRequest
    governor_decision: GovernorDecisionRecord
    ungoverned_selector_record: dict[str, object]
    governor_blocked_ungoverned_selection: bool
    reason: str | None
    predicted_successor_class: str | None
    actual_successor_class: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "objective_request": self.request.to_record(),
            "governor_decision": self.governor_decision.to_record(),
            "ungoverned_selector_counterfactual": dict(self.ungoverned_selector_record),
            "governor_blocked_ungoverned_selection": bool(self.governor_blocked_ungoverned_selection),
            "reason": self.reason,
            "predicted_successor_class": self.predicted_successor_class,
            "actual_successor_class": self.actual_successor_class,
        }


def build_decision_diagnostic_record(
    state: FlightState,
    request: PrimitiveRequest,
    governor: ManoeuvrePrimitiveGovernor,
    ungoverned_selector: UnfilteredPrimitiveSelector,
    *,
    tier: str | None = None,
    actual_successor_class: str | None = None,
) -> DecisionDiagnosticRecord:
    """Build an in-memory diagnostic record without writing plots or files."""

    governor_decision = governor.decide(state, request, tier=tier)
    ungoverned_record = ungoverned_selector.decide(state, tier=tier).to_record()
    ungoverned_selected = ungoverned_record.get("selected_candidate_id")
    governed_selected = governor_decision.selected_candidate_id
    blocked = (
        ungoverned_selected is not None
        and ungoverned_selected != governed_selected
        and (
            governor_decision.decision_type.value in {"degrade", "reject"}
            or bool(governor_decision.rejection_reasons)
        )
    )
    reason = governor_decision.rejection_reason
    if governor_decision.degradation_stage is not None:
        reason = governor_decision.degradation_stage.value
    return DecisionDiagnosticRecord(
        request=request,
        governor_decision=governor_decision,
        ungoverned_selector_record=ungoverned_record,
        governor_blocked_ungoverned_selection=blocked,
        reason=reason,
        predicted_successor_class=governor_decision.predicted_successor_class,
        actual_successor_class=actual_successor_class,
    )


__all__ = [
    "DecisionDiagnosticRecord",
    "build_decision_diagnostic_record",
]
