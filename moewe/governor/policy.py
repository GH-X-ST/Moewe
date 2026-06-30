"""Deterministic online filtering for retrieved motion primitives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from moewe.primitives import PrimitiveLibrary, PrimitiveLibraryCandidate, PrimitiveLibraryQuery
from moewe.returnability import PrimitiveTransition, ReturnabilityGraph
from moewe.sim.state import FlightState


def _unique_ordered(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


@dataclass(frozen=True)
class OnlineGovernorConfig:
    """Policy thresholds for online primitive admission."""

    tier: str = "balanced"
    allow_retrieval_fallback: bool = False
    require_safe_entry: bool = True
    require_recoverable_entry: bool = True
    require_recoverable_exit: bool = True
    reject_forbidden_exit: bool = True
    min_safety_margin_m: float = -10.0
    min_terminal_specific_energy_change_j_kg: float = -100.0
    prefer_terminal_success: bool = True

    def __post_init__(self) -> None:
        if not self.tier:
            raise ValueError("Governor tier must be non-empty.")
        if not math.isfinite(float(self.min_safety_margin_m)):
            raise ValueError("Minimum safety margin must be finite.")
        if not math.isfinite(float(self.min_terminal_specific_energy_change_j_kg)):
            raise ValueError("Minimum terminal energy change must be finite.")


@dataclass(frozen=True)
class CandidateReturnabilityEvidence:
    """Returnability evidence attached to one retrieved primitive candidate."""

    candidate_id: str
    represented_primitive_ids: tuple[str, ...]
    matched_transition_ids: tuple[str, ...]
    entry_classes: tuple[str, ...]
    exit_classes: tuple[str, ...]
    retained_transition_count: int
    recoverable_exit_count: int
    terminal_success_exit_count: int
    forbidden_exit_count: int
    min_safety_margin_m: float | None
    min_terminal_specific_energy_change_j_kg: float | None
    admissible: bool
    rejection_reasons: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "represented_primitive_ids": list(self.represented_primitive_ids),
            "matched_transition_ids": list(self.matched_transition_ids),
            "entry_classes": list(self.entry_classes),
            "exit_classes": list(self.exit_classes),
            "retained_transition_count": self.retained_transition_count,
            "recoverable_exit_count": self.recoverable_exit_count,
            "terminal_success_exit_count": self.terminal_success_exit_count,
            "forbidden_exit_count": self.forbidden_exit_count,
            "min_safety_margin_m": self.min_safety_margin_m,
            "min_terminal_specific_energy_change_j_kg": self.min_terminal_specific_energy_change_j_kg,
            "admissible": self.admissible,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class GovernorDecision:
    """One online governor decision and the evidence used to produce it."""

    tier: str
    entry_class: str
    candidate_count: int
    admissible_candidate_count: int
    selected_candidate_id: str | None
    selected_primitive_id: str | None
    selected_entry_class: str | None
    selected_exit_class: str | None
    selected_transition_id: str | None
    selected_reason: str | None
    fallback_used: bool
    fallback_reason: str | None
    rejection_reasons: tuple[str, ...]
    candidate_evidence: tuple[CandidateReturnabilityEvidence, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "entry_class": self.entry_class,
            "candidate_count": self.candidate_count,
            "admissible_candidate_count": self.admissible_candidate_count,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_primitive_id": self.selected_primitive_id,
            "selected_entry_class": self.selected_entry_class,
            "selected_exit_class": self.selected_exit_class,
            "selected_transition_id": self.selected_transition_id,
            "selected_reason": self.selected_reason,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "rejection_reasons": list(self.rejection_reasons),
            "candidate_evidence": [evidence.to_record() for evidence in self.candidate_evidence],
        }


class OnlineGovernor:
    """Conservative governor over retrieved primitives and a returnability graph."""

    def __init__(
        self,
        library: PrimitiveLibrary,
        graph: ReturnabilityGraph,
        config: OnlineGovernorConfig | None = None,
    ) -> None:
        self.library = library
        self.graph = graph
        self.config = OnlineGovernorConfig() if config is None else config
        self._transitions_by_primitive = graph.transitions_by_primitive()

    def decide(self, state: FlightState, tier: str | None = None) -> GovernorDecision:
        """Return the best admissible primitive for the current state."""

        query_tier = self.config.tier if tier is None else tier
        query = self.library.query(state, tier=query_tier)
        global_rejections = self._query_rejections(query)
        compatibility_entry_class = self._candidate_entry_class(query)
        evidence = tuple(
            self._candidate_evidence(candidate, compatibility_entry_class)
            for candidate in query.candidates
        )
        admissible = tuple(item for item in evidence if item.admissible and not global_rejections)
        selected_evidence = self._select_evidence(admissible)
        selected_transition = None
        if selected_evidence is not None:
            selected_candidate = next(
                candidate
                for candidate in query.candidates
                if candidate.primitive_id == selected_evidence.candidate_id
            )
            selected_transition = self._select_transition(selected_candidate, compatibility_entry_class)
            if selected_transition is None:
                selected_evidence = None

        rejection_reasons = list(global_rejections)
        if selected_transition is None:
            if not query.candidates:
                rejection_reasons.append("no_retrieval_candidate")
            elif not global_rejections:
                rejection_reasons.append("no_admissible_candidate")
        rejection_reasons = list(_unique_ordered(rejection_reasons))

        selected_reason = None
        if selected_transition is not None:
            selected_reason = (
                "terminal_success_exit"
                if selected_transition.exit_class in self.graph.terminal_success_classes
                else "recoverable_exit"
            )

        return GovernorDecision(
            tier=query.tier,
            entry_class=query.entry_class,
            candidate_count=len(query.candidates),
            admissible_candidate_count=len(admissible),
            selected_candidate_id=None if selected_evidence is None else selected_evidence.candidate_id,
            selected_primitive_id=None if selected_transition is None else selected_transition.primitive_id,
            selected_entry_class=None if selected_transition is None else selected_transition.entry_class,
            selected_exit_class=None if selected_transition is None else selected_transition.exit_class,
            selected_transition_id=None if selected_transition is None else selected_transition.transition_id,
            selected_reason=selected_reason,
            fallback_used=query.fallback_used,
            fallback_reason=query.fallback_reason,
            rejection_reasons=tuple(rejection_reasons),
            candidate_evidence=evidence,
        )

    def candidate_evidence(self, candidate: PrimitiveLibraryCandidate) -> CandidateReturnabilityEvidence:
        """Return deterministic graph evidence for a single retrieved candidate."""

        return self._candidate_evidence(candidate, candidate.entry_class)

    def _query_rejections(self, query: PrimitiveLibraryQuery) -> tuple[str, ...]:
        reasons: list[str] = []
        if query.fallback_used and not self.config.allow_retrieval_fallback:
            reasons.append("retrieval_fallback_not_allowed")
        if query.entry_class in self.graph.forbidden_classes:
            reasons.append("entry_forbidden")
        if self.config.require_safe_entry and query.entry_class not in self.graph.safe_classes:
            reasons.append("entry_not_safe")
        if self.config.require_recoverable_entry and query.entry_class not in self.graph.recoverable_classes:
            reasons.append("entry_not_recoverable")
        return _unique_ordered(reasons)

    def _candidate_entry_class(self, query: PrimitiveLibraryQuery) -> str:
        if query.fallback_used and self.config.allow_retrieval_fallback and query.candidates:
            return query.candidates[0].entry_class
        return query.entry_class

    def _candidate_evidence(
        self,
        candidate: PrimitiveLibraryCandidate,
        entry_class: str,
    ) -> CandidateReturnabilityEvidence:
        primitive_ids = self._candidate_primitive_ids(candidate)
        all_matches = self._candidate_transitions(primitive_ids)
        compatible = tuple(transition for transition in all_matches if transition.entry_class == entry_class)
        retained = tuple(transition for transition in compatible if transition.retained)
        retained_non_forbidden = tuple(
            transition
            for transition in retained
            if transition.exit_class not in self.graph.forbidden_classes
        )
        viable = tuple(
            transition
            for transition in retained_non_forbidden
            if self._transition_meets_thresholds(transition)
            and (not self.config.require_recoverable_exit or transition.exit_class in self.graph.recoverable_classes)
        )
        forbidden_exit_count = len(
            tuple(transition for transition in compatible if transition.exit_class in self.graph.forbidden_classes)
        )
        recoverable_exit_count = len(
            {
                transition.exit_class
                for transition in retained_non_forbidden
                if transition.exit_class in self.graph.recoverable_classes
            }
        )
        terminal_success_exit_count = len(
            {
                transition.exit_class
                for transition in retained_non_forbidden
                if transition.exit_class in self.graph.terminal_success_classes
            }
        )
        min_safety_margin = self._min_or_none(
            transition.min_safety_margin_m for transition in retained_non_forbidden
        )
        min_energy_change = self._min_or_none(
            transition.terminal_specific_energy_change_j_kg
            for transition in retained_non_forbidden
        )
        rejection_reasons = self._candidate_rejections(
            all_matches=all_matches,
            compatible=compatible,
            retained=retained,
            retained_non_forbidden=retained_non_forbidden,
            forbidden_exit_count=forbidden_exit_count,
            recoverable_exit_count=recoverable_exit_count,
            min_safety_margin_m=min_safety_margin,
            min_terminal_specific_energy_change_j_kg=min_energy_change,
        )
        if self.config.reject_forbidden_exit and forbidden_exit_count:
            viable = tuple()
        return CandidateReturnabilityEvidence(
            candidate_id=candidate.primitive_id,
            represented_primitive_ids=primitive_ids,
            matched_transition_ids=tuple(transition.transition_id for transition in compatible),
            entry_classes=_unique_ordered([transition.entry_class for transition in compatible]),
            exit_classes=_unique_ordered([transition.exit_class for transition in compatible]),
            retained_transition_count=len(retained),
            recoverable_exit_count=recoverable_exit_count,
            terminal_success_exit_count=terminal_success_exit_count,
            forbidden_exit_count=forbidden_exit_count,
            min_safety_margin_m=min_safety_margin,
            min_terminal_specific_energy_change_j_kg=min_energy_change,
            admissible=bool(viable) and not rejection_reasons,
            rejection_reasons=rejection_reasons,
        )

    def _candidate_rejections(
        self,
        *,
        all_matches: tuple[PrimitiveTransition, ...],
        compatible: tuple[PrimitiveTransition, ...],
        retained: tuple[PrimitiveTransition, ...],
        retained_non_forbidden: tuple[PrimitiveTransition, ...],
        forbidden_exit_count: int,
        recoverable_exit_count: int,
        min_safety_margin_m: float | None,
        min_terminal_specific_energy_change_j_kg: float | None,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if not all_matches:
            reasons.append("no_matching_transition")
        elif not compatible:
            reasons.append("no_matching_transition_for_entry")
        if compatible and not retained:
            reasons.append("no_retained_transition")
        if self.config.reject_forbidden_exit and forbidden_exit_count:
            reasons.append("forbidden_exit")
        if retained and self.config.require_recoverable_exit and not recoverable_exit_count:
            reasons.append("no_recoverable_exit")
        if (
            retained_non_forbidden
            and min_safety_margin_m is not None
            and min_safety_margin_m < self.config.min_safety_margin_m
        ):
            reasons.append("safety_margin_below_minimum")
        if (
            retained_non_forbidden
            and min_terminal_specific_energy_change_j_kg is not None
            and min_terminal_specific_energy_change_j_kg
            < self.config.min_terminal_specific_energy_change_j_kg
        ):
            reasons.append("energy_change_below_minimum")
        return _unique_ordered(reasons)

    def _candidate_primitive_ids(self, candidate: PrimitiveLibraryCandidate) -> tuple[str, ...]:
        primitive_ids = [candidate.primitive_id, *candidate.represented_primitive_ids]
        return tuple(sorted(set(primitive_ids)))

    def _candidate_transitions(self, primitive_ids: tuple[str, ...]) -> tuple[PrimitiveTransition, ...]:
        transitions = [
            transition
            for primitive_id in primitive_ids
            for transition in self._transitions_by_primitive.get(primitive_id, ())
        ]
        return tuple(sorted(transitions, key=lambda transition: transition.transition_id))

    def _transition_meets_thresholds(self, transition: PrimitiveTransition) -> bool:
        return (
            transition.min_safety_margin_m >= self.config.min_safety_margin_m
            and transition.terminal_specific_energy_change_j_kg
            >= self.config.min_terminal_specific_energy_change_j_kg
        )

    def _select_evidence(
        self,
        candidates: tuple[CandidateReturnabilityEvidence, ...],
    ) -> CandidateReturnabilityEvidence | None:
        if not candidates:
            return None
        return min(candidates, key=self._evidence_sort_key)

    def _evidence_sort_key(self, evidence: CandidateReturnabilityEvidence) -> tuple[object, ...]:
        terminal_score = evidence.terminal_success_exit_count if self.config.prefer_terminal_success else 0
        safety = -math.inf if evidence.min_safety_margin_m is None else evidence.min_safety_margin_m
        energy = (
            -math.inf
            if evidence.min_terminal_specific_energy_change_j_kg is None
            else evidence.min_terminal_specific_energy_change_j_kg
        )
        return (
            -terminal_score,
            -evidence.recoverable_exit_count,
            -safety,
            -energy,
            evidence.candidate_id,
        )

    def _select_transition(
        self,
        candidate: PrimitiveLibraryCandidate,
        entry_class: str,
    ) -> PrimitiveTransition | None:
        transitions = tuple(
            transition
            for transition in self._candidate_transitions(self._candidate_primitive_ids(candidate))
            if transition.entry_class == entry_class
            and transition.retained
            and transition.exit_class not in self.graph.forbidden_classes
            and self._transition_meets_thresholds(transition)
            and (not self.config.require_recoverable_exit or transition.exit_class in self.graph.recoverable_classes)
        )
        if not transitions:
            return None
        return min(transitions, key=self._transition_sort_key)

    def _transition_sort_key(self, transition: PrimitiveTransition) -> tuple[object, ...]:
        terminal_score = 1 if transition.exit_class in self.graph.terminal_success_classes else 0
        recoverable_score = 1 if transition.exit_class in self.graph.recoverable_classes else 0
        return (
            -terminal_score if self.config.prefer_terminal_success else 0,
            -recoverable_score,
            -transition.min_safety_margin_m,
            -transition.terminal_specific_energy_change_j_kg,
            transition.transition_id,
        )

    @staticmethod
    def _min_or_none(values: Iterable[float]) -> float | None:
        materialised = [float(value) for value in values]
        if not materialised:
            return None
        return min(materialised)
