"""Manuscript-facing manoeuvre primitive governor interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from time import perf_counter
from typing import Protocol

from moewe.primitives import PrimitiveLibrary, PrimitiveLibraryCandidate
from moewe.returnability import (
    ReturnabilityCertificate,
    ReturnabilityGraph,
    ReturnabilityThresholds,
    compute_empirical_returnability_certificate,
)
from moewe.sim.state import FlightState

from .policy import CandidateReturnabilityEvidence, OnlineGovernor, OnlineGovernorConfig


class DecisionType(str, Enum):
    """First-class governor decision categories."""

    ACCEPT = "accept"
    DEGRADE = "degrade"
    REJECT = "reject"
    RANK = "rank"


class RejectionReason(str, Enum):
    """Public rejection reason codes used in decision records."""

    ENTRY_INCOMPATIBLE = "entry_incompatible"
    ACTUATOR_LIMIT = "actuator_limit"
    UNSAFE_PREDICTED_MARGIN = "unsafe_predicted_margin"
    NONRETURNABLE_SUCCESSOR = "nonreturnable_successor"
    EXCESSIVE_HARD_FAILURE_RISK = "excessive_hard_failure_risk"
    RUNTIME_BUDGET_EXCEEDED = "runtime_budget_exceeded"
    NO_VALID_SAME_FAMILY_DEGRADE = "no_valid_same_family_degrade"
    NO_VALID_TASK_PRESERVING_DEGRADE = "no_valid_task_preserving_degrade"
    NO_VALID_RECOVERY_DEGRADE = "no_valid_recovery_degrade"
    NO_VIABLE_ACTION = "no_viable_action"


class DegradationStage(str, Enum):
    """Structural degradation stages."""

    REQUESTED = "requested"
    WEAKER_SAME_FAMILY = "weaker_same_family"
    SAFER_TASK_PRESERVING = "safer_task_preserving"
    RECOVERY = "recovery"


class ActiveConstraint(str, Enum):
    """Constraints evaluated before ranking."""

    ENTRY_CLASS = "entry_class"
    ACTUATOR_FEASIBILITY = "actuator_feasibility"
    TIMING_COMPATIBILITY = "timing_compatibility"
    SAFETY_MARGIN = "safety_margin"
    SUCCESSOR_RETURNABILITY = "successor_returnability"
    HARD_FAILURE_RISK = "hard_failure_risk"


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple, frozenset, set)):
        return [_jsonable(child) for child in sorted(value, key=lambda item: str(item))]
    return value


def _unique_ordered(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


@dataclass(frozen=True)
class PrimitiveRequest:
    """Closed-loop primitive request produced by an objective proposer."""

    request_id: str
    task_intent: str
    preferred_family: str
    requested_aggressiveness: int
    target_region: dict[str, object] = field(default_factory=dict)
    objective_score_terms: dict[str, float] = field(default_factory=dict)
    requested_primitive_id: str | None = None
    candidate_ids: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty.")
        if not self.task_intent:
            raise ValueError("task_intent must be non-empty.")
        if not self.preferred_family:
            raise ValueError("preferred_family must be non-empty.")
        if int(self.requested_aggressiveness) < 0:
            raise ValueError("requested_aggressiveness must be non-negative.")
        for name, value in self.objective_score_terms.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"objective score term {name!r} must be finite.")
        object.__setattr__(self, "candidate_ids", tuple(str(item) for item in self.candidate_ids))
        object.__setattr__(self, "target_region", dict(self.target_region))
        object.__setattr__(
            self,
            "objective_score_terms",
            {str(key): float(value) for key, value in self.objective_score_terms.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_record(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "task_intent": self.task_intent,
            "preferred_family": self.preferred_family,
            "requested_aggressiveness": int(self.requested_aggressiveness),
            "target_region": _jsonable(self.target_region),
            "objective_score_terms": _jsonable(self.objective_score_terms),
            "requested_primitive_id": self.requested_primitive_id,
            "candidate_ids": list(self.candidate_ids),
            "metadata": _jsonable(self.metadata),
        }


class ObjectiveProposer(Protocol):
    """Protocol for modules that propose primitive requests without filtering."""

    def propose(self, state: FlightState | None = None) -> PrimitiveRequest:
        """Return an objective-driven primitive request."""


@dataclass(frozen=True)
class PrimitiveFamilyRelation:
    """Interpretable primitive-family relation used for degradation."""

    source_family: str
    weaker_same_family: tuple[str, ...] = ()
    task_preserving_families: tuple[str, ...] = ()
    recovery_families: tuple[str, ...] = ("recovery", "bank_pitch_dwell_recovery")

    def to_record(self) -> dict[str, object]:
        return {
            "source_family": self.source_family,
            "weaker_same_family": list(self.weaker_same_family),
            "task_preserving_families": list(self.task_preserving_families),
            "recovery_families": list(self.recovery_families),
        }


@dataclass(frozen=True)
class DegradationTrace:
    """One degradation stage attempt."""

    stage: DegradationStage
    candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    reason: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "candidate_ids": list(self.candidate_ids),
            "selected_candidate_id": self.selected_candidate_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DegradationPolicy:
    """Ordered structural degradation policy."""

    enabled: bool = True
    recovery_family_tokens: tuple[str, ...] = ("recovery", "safe_exit")

    def stage_candidates(
        self,
        request: PrimitiveRequest,
        requested_candidate: PrimitiveLibraryCandidate | None,
        candidates: tuple[PrimitiveLibraryCandidate, ...],
    ) -> tuple[tuple[DegradationStage, tuple[PrimitiveLibraryCandidate, ...]], ...]:
        """Return candidates grouped by degradation stage in evaluation order."""

        if not self.enabled:
            return ()
        requested_family = request.preferred_family if requested_candidate is None else requested_candidate.family
        requested_aggressiveness = _candidate_aggressiveness(requested_candidate, request.requested_aggressiveness)
        requested_id = None if requested_candidate is None else requested_candidate.primitive_id

        weaker = tuple(
            candidate
            for candidate in candidates
            if candidate.primitive_id != requested_id
            and candidate.family == requested_family
            and _candidate_aggressiveness(candidate, request.requested_aggressiveness) < requested_aggressiveness
        )
        task_preserving = tuple(
            candidate
            for candidate in candidates
            if candidate.primitive_id != requested_id
            and candidate not in weaker
            and _task_preserving_candidate(candidate, request)
        )
        recovery = tuple(
            candidate
            for candidate in candidates
            if candidate.primitive_id != requested_id
            and candidate not in weaker
            and candidate not in task_preserving
            and self.is_recovery_candidate(candidate)
        )
        return (
            (DegradationStage.WEAKER_SAME_FAMILY, _sort_candidates(weaker)),
            (DegradationStage.SAFER_TASK_PRESERVING, _sort_candidates(task_preserving)),
            (DegradationStage.RECOVERY, _sort_candidates(recovery)),
        )

    def is_recovery_candidate(self, candidate: PrimitiveLibraryCandidate) -> bool:
        label = f"{candidate.family} {candidate.exit_class} {_feature_text(candidate)}".lower()
        return any(token in label for token in self.recovery_family_tokens)


@dataclass(frozen=True)
class GovernorDecisionRecord:
    """Public decision record for one manoeuvre primitive governor call."""

    request_id: str
    decision_type: DecisionType
    selected_primitive_id: str | None
    requested_primitive_id: str | None
    selected_candidate_id: str | None
    requested_candidate_id: str | None
    degradation_level: int
    degradation_stage: DegradationStage | None
    rejection_reason: str | None
    rejection_reasons: tuple[str, ...]
    active_constraints: tuple[ActiveConstraint, ...]
    entry_class: str | None
    predicted_successor_class: str | None
    predicted_returnability: float | None
    predicted_safety_margin: float | None
    predicted_energy_delta: float | None
    predicted_lift_exposure: float | None
    candidate_count: int
    admissible_candidate_count: int
    filtered_count_by_stage: dict[str, int]
    fallback_used: bool
    fallback_reason: str | None
    runtime_ms: float
    evidence_record_id: str | None = None
    evidence_summary: dict[str, object] = field(default_factory=dict)
    request_summary: dict[str, object] = field(default_factory=dict)
    degradation_trace: tuple[DegradationTrace, ...] = ()

    def to_record(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "decision_type": self.decision_type.value,
            "selected_primitive_id": self.selected_primitive_id,
            "requested_primitive_id": self.requested_primitive_id,
            "selected_candidate_id": self.selected_candidate_id,
            "requested_candidate_id": self.requested_candidate_id,
            "degradation_level": int(self.degradation_level),
            "degradation_stage": None if self.degradation_stage is None else self.degradation_stage.value,
            "rejection_reason": self.rejection_reason,
            "rejection_reasons": list(self.rejection_reasons),
            "active_constraints": [constraint.value for constraint in self.active_constraints],
            "entry_class": self.entry_class,
            "predicted_successor_class": self.predicted_successor_class,
            "predicted_returnability": self.predicted_returnability,
            "predicted_safety_margin": self.predicted_safety_margin,
            "predicted_energy_delta": self.predicted_energy_delta,
            "predicted_lift_exposure": self.predicted_lift_exposure,
            "candidate_count": int(self.candidate_count),
            "admissible_candidate_count": int(self.admissible_candidate_count),
            "filtered_count_by_stage": _jsonable(self.filtered_count_by_stage),
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
            "runtime_ms": float(self.runtime_ms),
            "evidence_record_id": self.evidence_record_id,
            "evidence_summary": _jsonable(self.evidence_summary),
            "request_summary": _jsonable(self.request_summary),
            "degradation_trace": [trace.to_record() for trace in self.degradation_trace],
        }


class ManoeuvrePrimitiveGovernor:
    """Decision interface over the existing online governor."""

    def __init__(
        self,
        library: PrimitiveLibrary,
        graph: ReturnabilityGraph,
        config: OnlineGovernorConfig | None = None,
        *,
        certificate: ReturnabilityCertificate | None = None,
        degradation_policy: DegradationPolicy | None = None,
    ) -> None:
        self.library = library
        self.graph = graph
        self.config = OnlineGovernorConfig() if config is None else config
        self.online_governor = OnlineGovernor(library, graph, self.config)
        self.certificate = (
            compute_empirical_returnability_certificate(
                graph.transitions,
                terminal_success_classes=graph.terminal_success_classes,
                forbidden_classes=graph.forbidden_classes,
                thresholds=ReturnabilityThresholds(
                    p_return_min=1.0,
                    p_hard_max=0.0,
                    margin_min=float(self.config.min_safety_margin_m),
                ),
            )
            if certificate is None
            else certificate
        )
        self.degradation_policy = DegradationPolicy() if degradation_policy is None else degradation_policy

    def decide(
        self,
        state: FlightState,
        request: PrimitiveRequest,
        tier: str | None = None,
    ) -> GovernorDecisionRecord:
        """Supervise a closed-loop primitive request and return a public record."""

        start_s = perf_counter()
        query = self.library.query(state, tier=self.config.tier if tier is None else tier)
        online_decision = self.online_governor.decide(state, tier=tier)
        candidates = tuple(query.candidates)
        evidence_by_candidate = {
            evidence.candidate_id: evidence for evidence in online_decision.candidate_evidence
        }
        fatal_query_rejections = tuple(
            reason
            for reason in online_decision.rejection_reasons
            if reason not in {"no_admissible_candidate", "no_retrieval_candidate"}
        )
        requested_candidate = _find_requested_candidate(candidates, request.requested_primitive_id)
        trace: list[DegradationTrace] = []

        if request.requested_primitive_id is None:
            selected = _rank_admissible_candidate(
                candidates,
                evidence_by_candidate,
                request.candidate_ids,
                fatal_query_rejections,
            )
            if selected is None:
                return self._record(
                    request=request,
                    decision_type=DecisionType.REJECT,
                    online_decision=online_decision,
                    query_candidates=candidates,
                    requested_candidate=None,
                    selected_candidate=None,
                    selected_stage=None,
                    degradation_level=0,
                    rejection_reasons=_candidate_set_rejections(
                        online_decision,
                        evidence_by_candidate,
                        request.candidate_ids,
                    ),
                    runtime_ms=_elapsed_ms(start_s),
                    trace=tuple(trace),
                )
            return self._record(
                request=request,
                decision_type=DecisionType.RANK,
                online_decision=online_decision,
                query_candidates=candidates,
                requested_candidate=None,
                selected_candidate=selected,
                selected_stage=None,
                degradation_level=0,
                rejection_reasons=(),
                runtime_ms=_elapsed_ms(start_s),
                trace=tuple(trace),
            )

        if requested_candidate is not None and not fatal_query_rejections:
            evidence = evidence_by_candidate.get(requested_candidate.primitive_id)
            if evidence is not None and evidence.admissible:
                return self._record(
                    request=request,
                    decision_type=DecisionType.ACCEPT,
                    online_decision=online_decision,
                    query_candidates=candidates,
                    requested_candidate=requested_candidate,
                    selected_candidate=requested_candidate,
                    selected_stage=DegradationStage.REQUESTED,
                    degradation_level=0,
                    rejection_reasons=(),
                    runtime_ms=_elapsed_ms(start_s),
                    trace=tuple(trace),
                )

        requested_reasons = _requested_reasons(
            request,
            requested_candidate,
            evidence_by_candidate,
            fatal_query_rejections,
        )
        for level, (stage, stage_candidates) in enumerate(
            self.degradation_policy.stage_candidates(request, requested_candidate, candidates),
            start=1,
        ):
            selected = _rank_admissible_candidate(stage_candidates, evidence_by_candidate, (), fatal_query_rejections)
            trace.append(
                DegradationTrace(
                    stage=stage,
                    candidate_ids=tuple(candidate.primitive_id for candidate in stage_candidates),
                    selected_candidate_id=None if selected is None else selected.primitive_id,
                    reason=None if selected is not None else _no_degrade_reason(stage).value,
                )
            )
            if selected is not None:
                return self._record(
                    request=request,
                    decision_type=DecisionType.DEGRADE,
                    online_decision=online_decision,
                    query_candidates=candidates,
                    requested_candidate=requested_candidate,
                    selected_candidate=selected,
                    selected_stage=stage,
                    degradation_level=level,
                    rejection_reasons=requested_reasons,
                    runtime_ms=_elapsed_ms(start_s),
                    trace=tuple(trace),
                )

        rejection_reasons = [*requested_reasons, RejectionReason.NO_VIABLE_ACTION.value]
        if self.degradation_policy.enabled:
            rejection_reasons.extend(
                (
                    RejectionReason.NO_VALID_SAME_FAMILY_DEGRADE.value,
                    RejectionReason.NO_VALID_TASK_PRESERVING_DEGRADE.value,
                    RejectionReason.NO_VALID_RECOVERY_DEGRADE.value,
                )
            )
        return self._record(
            request=request,
            decision_type=DecisionType.REJECT,
            online_decision=online_decision,
            query_candidates=candidates,
            requested_candidate=requested_candidate,
            selected_candidate=None,
            selected_stage=None,
            degradation_level=0,
            rejection_reasons=_unique_ordered(rejection_reasons),
            runtime_ms=_elapsed_ms(start_s),
            trace=tuple(trace),
        )

    def _record(
        self,
        *,
        request: PrimitiveRequest,
        decision_type: DecisionType,
        online_decision: object,
        query_candidates: tuple[PrimitiveLibraryCandidate, ...],
        requested_candidate: PrimitiveLibraryCandidate | None,
        selected_candidate: PrimitiveLibraryCandidate | None,
        selected_stage: DegradationStage | None,
        degradation_level: int,
        rejection_reasons: tuple[str, ...] | list[str],
        runtime_ms: float,
        trace: tuple[DegradationTrace, ...],
    ) -> GovernorDecisionRecord:
        evidence = None if selected_candidate is None else self.online_governor.candidate_evidence(selected_candidate)
        selected_transition = (
            None
            if selected_candidate is None
            else self.online_governor._select_transition(selected_candidate, _candidate_entry_class(online_decision))
        )
        predicted_successor = _predicted_successor(selected_transition, evidence)
        returnability = _predicted_returnability(
            self.certificate,
            entry_class=_candidate_entry_class(online_decision),
            primitive_id=None if selected_candidate is None else selected_candidate.primitive_id,
            successor_class=predicted_successor,
        )
        reasons = _unique_ordered(list(rejection_reasons))
        summary = {} if evidence is None else evidence.to_record()
        return GovernorDecisionRecord(
            request_id=request.request_id,
            decision_type=decision_type,
            selected_primitive_id=None if selected_candidate is None else selected_candidate.primitive_id,
            requested_primitive_id=request.requested_primitive_id,
            selected_candidate_id=None if selected_candidate is None else selected_candidate.primitive_id,
            requested_candidate_id=None if requested_candidate is None else requested_candidate.primitive_id,
            degradation_level=int(degradation_level),
            degradation_stage=selected_stage if decision_type == DecisionType.DEGRADE else None,
            rejection_reason=None if not reasons else reasons[0],
            rejection_reasons=reasons,
            active_constraints=(
                ActiveConstraint.ENTRY_CLASS,
                ActiveConstraint.ACTUATOR_FEASIBILITY,
                ActiveConstraint.TIMING_COMPATIBILITY,
                ActiveConstraint.SAFETY_MARGIN,
                ActiveConstraint.SUCCESSOR_RETURNABILITY,
                ActiveConstraint.HARD_FAILURE_RISK,
            ),
            entry_class=getattr(online_decision, "entry_class", None),
            predicted_successor_class=predicted_successor,
            predicted_returnability=returnability,
            predicted_safety_margin=None if evidence is None else evidence.min_safety_margin_m,
            predicted_energy_delta=None if evidence is None else evidence.min_terminal_specific_energy_change_j_kg,
            predicted_lift_exposure=_candidate_lift_exposure(selected_candidate),
            candidate_count=len(query_candidates),
            admissible_candidate_count=sum(
                1
                for item in getattr(online_decision, "candidate_evidence", ())
                if item.admissible and not getattr(online_decision, "rejection_reasons", ())
            ),
            filtered_count_by_stage=_filtered_count_by_stage(getattr(online_decision, "candidate_evidence", ())),
            fallback_used=bool(getattr(online_decision, "fallback_used", False)),
            fallback_reason=getattr(online_decision, "fallback_reason", None),
            runtime_ms=runtime_ms,
            evidence_record_id=None if selected_candidate is None else f"{request.request_id}:{selected_candidate.primitive_id}",
            evidence_summary=summary,
            request_summary=request.to_record(),
            degradation_trace=trace,
        )


def _sort_candidates(candidates: tuple[PrimitiveLibraryCandidate, ...]) -> tuple[PrimitiveLibraryCandidate, ...]:
    return tuple(sorted(candidates, key=lambda candidate: candidate.primitive_id))


def _feature_text(candidate: PrimitiveLibraryCandidate) -> str:
    return " ".join(str(value) for value in candidate.feature_record.values())


def _candidate_aggressiveness(
    candidate: PrimitiveLibraryCandidate | None,
    default: int,
) -> int:
    if candidate is None:
        return int(default)
    for key in ("aggressiveness_level", "requested_aggressiveness"):
        value = candidate.feature_record.get(key)
        if isinstance(value, int):
            return max(0, int(value))
        if isinstance(value, float) and math.isfinite(value):
            return max(0, int(value))
    safety = candidate.feature_record.get("max_angle_of_attack_rad", 0.0)
    command = candidate.feature_record.get("max_command_abs_rad", 0.0)
    try:
        return int(max(0.0, round(10.0 * (abs(float(safety)) + abs(float(command))))))
    except (TypeError, ValueError):
        return int(default)


def _task_preserving_candidate(candidate: PrimitiveLibraryCandidate, request: PrimitiveRequest) -> bool:
    labels = f"{candidate.family} {candidate.exit_class} {_feature_text(candidate)}".lower()
    intent_tokens = [token for token in request.task_intent.lower().replace("_", " ").split() if token]
    return candidate.family == request.preferred_family or any(token in labels for token in intent_tokens)


def _find_requested_candidate(
    candidates: tuple[PrimitiveLibraryCandidate, ...],
    requested_primitive_id: str | None,
) -> PrimitiveLibraryCandidate | None:
    if requested_primitive_id is None:
        return None
    for candidate in candidates:
        represented = {candidate.primitive_id, *candidate.represented_primitive_ids}
        if requested_primitive_id in represented:
            return candidate
    return None


def _rank_admissible_candidate(
    candidates: tuple[PrimitiveLibraryCandidate, ...],
    evidence_by_candidate: dict[str, CandidateReturnabilityEvidence],
    candidate_ids: tuple[str, ...],
    fatal_query_rejections: tuple[str, ...],
) -> PrimitiveLibraryCandidate | None:
    if fatal_query_rejections:
        return None
    allowed = set(candidate_ids)
    admissible = [
        candidate
        for candidate in candidates
        if (not allowed or candidate.primitive_id in allowed or bool(allowed & set(candidate.represented_primitive_ids)))
        and (evidence_by_candidate.get(candidate.primitive_id) is not None)
        and evidence_by_candidate[candidate.primitive_id].admissible
    ]
    if not admissible:
        return None
    return min(
        admissible,
        key=lambda candidate: (
            -float(evidence_by_candidate[candidate.primitive_id].terminal_success_exit_count),
            -float(evidence_by_candidate[candidate.primitive_id].recoverable_exit_count),
            _candidate_aggressiveness(candidate, 0),
            candidate.primitive_id,
        ),
    )


def _candidate_set_rejections(
    online_decision: object,
    evidence_by_candidate: dict[str, CandidateReturnabilityEvidence],
    candidate_ids: tuple[str, ...],
) -> tuple[str, ...]:
    reasons = list(getattr(online_decision, "rejection_reasons", ()))
    allowed = set(candidate_ids)
    for candidate_id, evidence in evidence_by_candidate.items():
        if allowed and candidate_id not in allowed:
            continue
        reasons.extend(evidence.rejection_reasons)
    if not reasons:
        reasons.append(RejectionReason.NO_VIABLE_ACTION.value)
    return _unique_ordered(reasons)


def _requested_reasons(
    request: PrimitiveRequest,
    requested_candidate: PrimitiveLibraryCandidate | None,
    evidence_by_candidate: dict[str, CandidateReturnabilityEvidence],
    fatal_query_rejections: tuple[str, ...],
) -> tuple[str, ...]:
    reasons = list(fatal_query_rejections)
    if requested_candidate is None:
        reasons.append("requested_primitive_not_retrieved")
        return _unique_ordered(reasons)
    evidence = evidence_by_candidate.get(requested_candidate.primitive_id)
    if evidence is None:
        reasons.append("no_matching_transition")
    else:
        reasons.extend(_public_rejection_reason(reason) for reason in evidence.rejection_reasons)
    if requested_candidate.family != request.preferred_family:
        reasons.append("requested_family_mismatch")
    return _unique_ordered(reasons)


def _public_rejection_reason(reason: str) -> str:
    mapping = {
        "no_matching_transition_for_entry": RejectionReason.ENTRY_INCOMPATIBLE.value,
        "no_matching_transition": RejectionReason.ENTRY_INCOMPATIBLE.value,
        "forbidden_exit": RejectionReason.NONRETURNABLE_SUCCESSOR.value,
        "no_recoverable_exit": RejectionReason.NONRETURNABLE_SUCCESSOR.value,
        "safety_margin_below_minimum": RejectionReason.UNSAFE_PREDICTED_MARGIN.value,
        "energy_change_below_minimum": "insufficient_terminal_energy",
    }
    return mapping.get(reason, reason)


def _no_degrade_reason(stage: DegradationStage) -> RejectionReason:
    if stage == DegradationStage.WEAKER_SAME_FAMILY:
        return RejectionReason.NO_VALID_SAME_FAMILY_DEGRADE
    if stage == DegradationStage.SAFER_TASK_PRESERVING:
        return RejectionReason.NO_VALID_TASK_PRESERVING_DEGRADE
    return RejectionReason.NO_VALID_RECOVERY_DEGRADE


def _filtered_count_by_stage(evidence_items: tuple[CandidateReturnabilityEvidence, ...]) -> dict[str, int]:
    counts = {
        "entry": 0,
        "actuator_timing": 0,
        "safety": 0,
        "returnability": 0,
    }
    for evidence in evidence_items:
        reasons = set(evidence.rejection_reasons)
        if reasons & {"no_matching_transition", "no_matching_transition_for_entry"}:
            counts["entry"] += 1
        if reasons & {"forbidden_exit", "no_recoverable_exit", "no_retained_transition"}:
            counts["returnability"] += 1
        if reasons & {"safety_margin_below_minimum"}:
            counts["safety"] += 1
        if reasons & {"actuator_limit", "timing_incompatible"}:
            counts["actuator_timing"] += 1
    return counts


def _candidate_entry_class(online_decision: object) -> str:
    value = getattr(online_decision, "entry_class", None)
    return "" if value is None else str(value)


def _predicted_successor(
    transition: object | None,
    evidence: CandidateReturnabilityEvidence | None,
) -> str | None:
    if transition is not None:
        return getattr(transition, "exit_class", None)
    if evidence is None or not evidence.exit_classes:
        return None
    return sorted(evidence.exit_classes)[0]


def _predicted_returnability(
    certificate: ReturnabilityCertificate,
    *,
    entry_class: str,
    primitive_id: str | None,
    successor_class: str | None,
) -> float | None:
    if primitive_id is None:
        return None
    edge = certificate.edge_for(entry_class, primitive_id)
    if edge is not None:
        return edge.successor_return_probability
    if successor_class is None:
        return None
    return 1.0 if successor_class in certificate.returnable_classes else 0.0


def _candidate_lift_exposure(candidate: PrimitiveLibraryCandidate | None) -> float | None:
    if candidate is None:
        return None
    value = candidate.feature_record.get("mean_positive_vertical_wind_m_s")
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _elapsed_ms(start_s: float) -> float:
    return max(0.0, float((perf_counter() - start_s) * 1000.0))


__all__ = [
    "ActiveConstraint",
    "DecisionType",
    "DegradationPolicy",
    "DegradationStage",
    "DegradationTrace",
    "GovernorDecisionRecord",
    "ManoeuvrePrimitiveGovernor",
    "ObjectiveProposer",
    "PrimitiveFamilyRelation",
    "PrimitiveRequest",
    "RejectionReason",
]
