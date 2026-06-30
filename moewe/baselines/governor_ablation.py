"""Unfiltered primitive-selection baseline for governor ablations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from moewe.governor import GovernorDecision, OnlineGovernor
from moewe.primitives import PrimitiveLibrary, PrimitiveLibraryCandidate
from moewe.sim.state import FlightState


def _unique_ordered(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _finite_value(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        return default
    numeric = float(value)
    if not math.isfinite(numeric):
        return default
    return numeric


@dataclass(frozen=True)
class UnfilteredPrimitiveSelectorConfig:
    """Scoring settings for primitive selection without viability filtering."""

    tier: str = "balanced"
    allow_retrieval_fallback: bool = True
    energy_weight: float = 1.0
    safety_weight: float = 0.25
    lift_weight: float = 0.25
    gate_miss_weight: float = 0.25
    missing_feature_value: float = 0.0

    def __post_init__(self) -> None:
        if not self.tier:
            raise ValueError("Selector tier must be non-empty.")
        values = (
            self.energy_weight,
            self.safety_weight,
            self.lift_weight,
            self.gate_miss_weight,
            self.missing_feature_value,
        )
        try:
            finite = all(math.isfinite(float(value)) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError("Selector weights and defaults must be finite.") from exc
        if not finite:
            raise ValueError("Selector weights and defaults must be finite.")


@dataclass(frozen=True)
class UnfilteredCandidateScore:
    """Feature-score record for one retrieved primitive candidate."""

    candidate_id: str
    entry_class: str
    exit_class: str
    score: float
    terminal_specific_energy_change_j_kg: float
    min_safety_margin_m: float
    gate_miss_distance_m: float
    mean_positive_vertical_wind_m_s: float
    represented_primitive_ids: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "entry_class": self.entry_class,
            "exit_class": self.exit_class,
            "score": self.score,
            "terminal_specific_energy_change_j_kg": self.terminal_specific_energy_change_j_kg,
            "min_safety_margin_m": self.min_safety_margin_m,
            "gate_miss_distance_m": self.gate_miss_distance_m,
            "mean_positive_vertical_wind_m_s": self.mean_positive_vertical_wind_m_s,
            "represented_primitive_ids": list(self.represented_primitive_ids),
        }


@dataclass(frozen=True)
class UnfilteredSelectionDecision:
    """Selection result for feature-scored primitive candidates."""

    tier: str
    entry_class: str
    candidate_count: int
    selected_candidate_id: str | None
    selected_entry_class: str | None
    selected_exit_class: str | None
    selected_score: float | None
    fallback_used: bool
    fallback_reason: str | None
    candidate_scores: tuple[UnfilteredCandidateScore, ...]
    rejection_reasons: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "entry_class": self.entry_class,
            "candidate_count": self.candidate_count,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_entry_class": self.selected_entry_class,
            "selected_exit_class": self.selected_exit_class,
            "selected_score": self.selected_score,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "candidate_scores": [score.to_record() for score in self.candidate_scores],
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class GovernorAblationComparison:
    """Side-by-side governor and unfiltered selector records."""

    tier: str
    entry_class: str
    governor_selected_candidate_id: str | None
    unfiltered_selected_candidate_id: str | None
    same_selected_candidate: bool
    governor_blocked_unfiltered_selection: bool
    governor_rejection_reasons: tuple[str, ...]
    unfiltered_decision: dict[str, object]
    governor_decision: dict[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "entry_class": self.entry_class,
            "governor_selected_candidate_id": self.governor_selected_candidate_id,
            "unfiltered_selected_candidate_id": self.unfiltered_selected_candidate_id,
            "same_selected_candidate": self.same_selected_candidate,
            "governor_blocked_unfiltered_selection": self.governor_blocked_unfiltered_selection,
            "governor_rejection_reasons": list(self.governor_rejection_reasons),
            "unfiltered_decision": self.unfiltered_decision,
            "governor_decision": self.governor_decision,
        }


class UnfilteredPrimitiveSelector:
    """Feature-scored primitive selector without returnability admission filtering."""

    def __init__(
        self,
        library: PrimitiveLibrary,
        config: UnfilteredPrimitiveSelectorConfig | None = None,
    ) -> None:
        self.library = library
        self.config = UnfilteredPrimitiveSelectorConfig() if config is None else config

    def decide(self, state: FlightState, tier: str | None = None) -> UnfilteredSelectionDecision:
        """Score retrieved candidates and select the highest scoring one."""

        query_tier = self.config.tier if tier is None else tier
        query = self.library.query(state, tier=query_tier)
        scores = tuple(sorted((self._score_candidate(candidate) for candidate in query.candidates), key=_score_sort_key))
        rejection_reasons: list[str] = []
        if query.fallback_used and not self.config.allow_retrieval_fallback:
            rejection_reasons.append("retrieval_fallback_not_allowed")
        if not query.candidates:
            rejection_reasons.append("no_retrieval_candidate")
        rejection_reasons = list(_unique_ordered(rejection_reasons))

        selected = None if rejection_reasons or not scores else scores[0]
        return UnfilteredSelectionDecision(
            tier=query.tier,
            entry_class=query.entry_class,
            candidate_count=len(query.candidates),
            selected_candidate_id=None if selected is None else selected.candidate_id,
            selected_entry_class=None if selected is None else selected.entry_class,
            selected_exit_class=None if selected is None else selected.exit_class,
            selected_score=None if selected is None else selected.score,
            fallback_used=query.fallback_used,
            fallback_reason=query.fallback_reason,
            candidate_scores=scores,
            rejection_reasons=tuple(rejection_reasons),
        )

    def _score_candidate(self, candidate: PrimitiveLibraryCandidate) -> UnfilteredCandidateScore:
        feature = candidate.feature_record
        default = float(self.config.missing_feature_value)
        energy = _finite_value(feature.get("terminal_specific_energy_change_j_kg"), default)
        safety = _finite_value(feature.get("min_safety_margin_m"), default)
        gate_miss = _finite_value(feature.get("gate_miss_distance_m"), default)
        lift = _finite_value(feature.get("mean_positive_vertical_wind_m_s"), default)
        score = (
            self.config.energy_weight * energy
            + self.config.safety_weight * safety
            + self.config.lift_weight * lift
            - self.config.gate_miss_weight * gate_miss
        )
        return UnfilteredCandidateScore(
            candidate_id=candidate.primitive_id,
            entry_class=candidate.entry_class,
            exit_class=candidate.exit_class,
            score=float(score),
            terminal_specific_energy_change_j_kg=energy,
            min_safety_margin_m=safety,
            gate_miss_distance_m=gate_miss,
            mean_positive_vertical_wind_m_s=lift,
            represented_primitive_ids=tuple(sorted({candidate.primitive_id, *candidate.represented_primitive_ids})),
        )


def compare_governor_to_unfiltered(
    state: FlightState,
    governor: OnlineGovernor,
    selector: UnfilteredPrimitiveSelector,
    tier: str | None = None,
) -> GovernorAblationComparison:
    """Run governor and unfiltered selector paths on the same state."""

    governor_decision = governor.decide(state, tier=tier)
    unfiltered_decision = selector.decide(state, tier=tier)
    blocked = _governor_blocked_unfiltered(governor_decision, unfiltered_decision)
    same_selected = (
        governor_decision.selected_candidate_id is not None
        and governor_decision.selected_candidate_id == unfiltered_decision.selected_candidate_id
    )
    return GovernorAblationComparison(
        tier=governor_decision.tier,
        entry_class=governor_decision.entry_class,
        governor_selected_candidate_id=governor_decision.selected_candidate_id,
        unfiltered_selected_candidate_id=unfiltered_decision.selected_candidate_id,
        same_selected_candidate=same_selected,
        governor_blocked_unfiltered_selection=blocked,
        governor_rejection_reasons=_governor_rejection_reasons(governor_decision, unfiltered_decision),
        unfiltered_decision=unfiltered_decision.to_record(),
        governor_decision=governor_decision.to_record(),
    )


def _score_sort_key(score: UnfilteredCandidateScore) -> tuple[float, str]:
    return (-score.score, score.candidate_id)


def _governor_blocked_unfiltered(
    governor_decision: GovernorDecision,
    unfiltered_decision: UnfilteredSelectionDecision,
) -> bool:
    selected = unfiltered_decision.selected_candidate_id
    if selected is None or governor_decision.selected_candidate_id == selected:
        return False
    selected_evidence = [
        evidence
        for evidence in governor_decision.candidate_evidence
        if evidence.candidate_id == selected
    ]
    if selected_evidence and any(not evidence.admissible for evidence in selected_evidence):
        return True
    return bool(governor_decision.rejection_reasons)


def _governor_rejection_reasons(
    governor_decision: GovernorDecision,
    unfiltered_decision: UnfilteredSelectionDecision,
) -> tuple[str, ...]:
    reasons = list(governor_decision.rejection_reasons)
    selected = unfiltered_decision.selected_candidate_id
    if selected is not None:
        for evidence in governor_decision.candidate_evidence:
            if evidence.candidate_id == selected:
                reasons.extend(evidence.rejection_reasons)
    return _unique_ordered(reasons)
