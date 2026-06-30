"""Decision-level smoke campaigns for selector comparison."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from moewe.baselines import (
    GovernorAblationComparison,
    UnfilteredPrimitiveSelector,
    compare_governor_to_unfiltered,
)
from moewe.governor import OnlineGovernor
from moewe.primitives import StructuredLibraryBuildReport
from moewe.sim.state import FlightState, STATE_SIZE


def _count(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class SelectorDecisionCampaignConfig:
    """Configuration for a finite selector-decision campaign."""

    tier: str = "balanced"
    max_cases: int | None = None
    retained_only: bool = True
    state_index: int = 0

    def __post_init__(self) -> None:
        if not self.tier:
            raise ValueError("Campaign tier must be non-empty.")
        if self.max_cases is not None and int(self.max_cases) <= 0:
            raise ValueError("max_cases must be positive when supplied.")
        if int(self.state_index) < 0:
            raise ValueError("state_index must be non-negative.")

    def to_record(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "max_cases": self.max_cases,
            "retained_only": self.retained_only,
            "state_index": int(self.state_index),
        }


@dataclass(frozen=True)
class SelectorCampaignCase:
    """One sampled decision state traced to structured primitive evidence."""

    case_id: str
    design_case_id: str
    case_set: str
    primitive_id: str
    entry_class: str
    exit_class: str
    state: FlightState
    seed: int | None = None
    source: str = "structured_library_report"

    def __post_init__(self) -> None:
        for name, value in (
            ("case_id", self.case_id),
            ("design_case_id", self.design_case_id),
            ("case_set", self.case_set),
            ("primitive_id", self.primitive_id),
            ("entry_class", self.entry_class),
            ("exit_class", self.exit_class),
            ("source", self.source),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty.")
        vector = np.asarray(self.state.as_vector(), dtype=float).reshape(STATE_SIZE)
        if not np.isfinite(vector).all():
            raise ValueError("Campaign case state must contain only finite values.")

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "design_case_id": self.design_case_id,
            "case_set": self.case_set,
            "primitive_id": self.primitive_id,
            "entry_class": self.entry_class,
            "exit_class": self.exit_class,
            "seed": self.seed,
            "source": self.source,
            "state_vector": [float(value) for value in self.state.as_vector()],
        }


@dataclass(frozen=True)
class SelectorDecisionRecord:
    """One campaign case evaluated by the governor and unfiltered selector."""

    case_id: str
    design_case_id: str
    case_set: str
    primitive_id: str
    expected_entry_class: str
    expected_exit_class: str
    tier: str
    governor_entry_class: str
    governor_selected_candidate_id: str | None
    unfiltered_selected_candidate_id: str | None
    same_selected_candidate: bool
    governor_blocked_unfiltered_selection: bool
    governor_rejection_reasons: tuple[str, ...]
    fallback_used: bool
    fallback_reason: str | None
    governor_candidate_count: int
    governor_admissible_candidate_count: int
    unfiltered_candidate_count: int
    comparison_record: dict[str, object]

    @classmethod
    def from_comparison(
        cls,
        case: SelectorCampaignCase,
        comparison: GovernorAblationComparison,
    ) -> "SelectorDecisionRecord":
        comparison_record = comparison.to_record()
        governor_decision = comparison_record["governor_decision"]
        unfiltered_decision = comparison_record["unfiltered_decision"]
        if not isinstance(governor_decision, dict) or not isinstance(unfiltered_decision, dict):
            raise ValueError("Comparison record must include decision dictionaries.")
        fallback_used = bool(governor_decision.get("fallback_used")) or bool(unfiltered_decision.get("fallback_used"))
        fallback_reason = governor_decision.get("fallback_reason") or unfiltered_decision.get("fallback_reason")
        return cls(
            case_id=case.case_id,
            design_case_id=case.design_case_id,
            case_set=case.case_set,
            primitive_id=case.primitive_id,
            expected_entry_class=case.entry_class,
            expected_exit_class=case.exit_class,
            tier=str(comparison_record["tier"]),
            governor_entry_class=str(governor_decision["entry_class"]),
            governor_selected_candidate_id=_optional_str(comparison_record["governor_selected_candidate_id"]),
            unfiltered_selected_candidate_id=_optional_str(comparison_record["unfiltered_selected_candidate_id"]),
            same_selected_candidate=bool(comparison_record["same_selected_candidate"]),
            governor_blocked_unfiltered_selection=bool(comparison_record["governor_blocked_unfiltered_selection"]),
            governor_rejection_reasons=tuple(str(reason) for reason in comparison_record["governor_rejection_reasons"]),
            fallback_used=fallback_used,
            fallback_reason=_optional_str(fallback_reason),
            governor_candidate_count=int(governor_decision["candidate_count"]),
            governor_admissible_candidate_count=int(governor_decision["admissible_candidate_count"]),
            unfiltered_candidate_count=int(unfiltered_decision["candidate_count"]),
            comparison_record=comparison_record,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "design_case_id": self.design_case_id,
            "case_set": self.case_set,
            "primitive_id": self.primitive_id,
            "expected_entry_class": self.expected_entry_class,
            "expected_exit_class": self.expected_exit_class,
            "tier": self.tier,
            "governor_entry_class": self.governor_entry_class,
            "governor_selected_candidate_id": self.governor_selected_candidate_id,
            "unfiltered_selected_candidate_id": self.unfiltered_selected_candidate_id,
            "same_selected_candidate": self.same_selected_candidate,
            "governor_blocked_unfiltered_selection": self.governor_blocked_unfiltered_selection,
            "governor_rejection_reasons": list(self.governor_rejection_reasons),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "governor_candidate_count": self.governor_candidate_count,
            "governor_admissible_candidate_count": self.governor_admissible_candidate_count,
            "unfiltered_candidate_count": self.unfiltered_candidate_count,
            "comparison_record": self.comparison_record,
        }


@dataclass(frozen=True)
class SelectorDecisionCampaignReport:
    """Compact JSON-serialisable decision campaign report."""

    tier: str
    records: tuple[SelectorDecisionRecord, ...]
    config: SelectorDecisionCampaignConfig

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("Selector decision campaign report requires at least one record.")
        if not self.tier:
            raise ValueError("Campaign report tier must be non-empty.")

    def to_summary(self) -> dict[str, object]:
        case_count = len(self.records)
        governor_reasons = [
            reason
            for record in self.records
            for reason in record.governor_rejection_reasons
        ]
        return {
            "case_count": case_count,
            "design_case_count": len({record.design_case_id for record in self.records}),
            "case_sets": sorted({record.case_set for record in self.records}),
            "governor_selection_count": sum(1 for record in self.records if record.governor_selected_candidate_id is not None),
            "unfiltered_selection_count": sum(1 for record in self.records if record.unfiltered_selected_candidate_id is not None),
            "same_selection_count": sum(1 for record in self.records if record.same_selected_candidate),
            "governor_blocked_unfiltered_count": sum(1 for record in self.records if record.governor_blocked_unfiltered_selection),
            "fallback_count": sum(1 for record in self.records if record.fallback_used),
            "governor_rejection_reason_counts": _count(governor_reasons),
            "mean_governor_candidate_count": _mean(record.governor_candidate_count for record in self.records),
            "mean_governor_admissible_candidate_count": _mean(
                record.governor_admissible_candidate_count for record in self.records
            ),
            "mean_unfiltered_candidate_count": _mean(record.unfiltered_candidate_count for record in self.records),
        }

    def to_record(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "config": self.config.to_record(),
            "summary": self.to_summary(),
            "records": [record.to_record() for record in self.records],
        }


def build_selector_campaign_cases_from_structured_report(
    report: StructuredLibraryBuildReport,
    config: SelectorDecisionCampaignConfig | None = None,
) -> tuple[SelectorCampaignCase, ...]:
    """Build deterministic selector decision cases from a structured report."""

    campaign_config = SelectorDecisionCampaignConfig() if config is None else config
    state_index = int(campaign_config.state_index)
    cases_by_id = {case.case_id: case for case in report.design_cases}
    if len(cases_by_id) != len(report.design_cases):
        raise ValueError("Structured report contains duplicate design case ids.")
    results = []
    for result in report.validation_report.results:
        if result.scenario_id not in cases_by_id:
            raise ValueError(f"Validation result is not traced to a design case: {result.scenario_id}")
        if campaign_config.retained_only and not result.retention.retained:
            continue
        if state_index >= len(result.rollout.states):
            raise ValueError("Campaign state_index is outside a rollout state history.")
        results.append(result)

    results = sorted(results, key=lambda result: (result.scenario_id, result.primitive_id, _seed_sort_key(result.seed)))
    built: list[SelectorCampaignCase] = []
    for result in results:
        design_case = cases_by_id[result.scenario_id]
        built.append(
            SelectorCampaignCase(
                case_id=f"{design_case.case_id}:{result.primitive_id}:state_{state_index}",
                design_case_id=design_case.case_id,
                case_set=design_case.case_set,
                primitive_id=result.primitive_id,
                entry_class=result.entry_class.label,
                exit_class=result.exit_class.label,
                state=result.rollout.states[state_index],
                seed=result.seed,
            )
        )
        if campaign_config.max_cases is not None and len(built) >= int(campaign_config.max_cases):
            break
    if not built:
        raise ValueError("No selector campaign cases could be built from the structured report.")
    return tuple(built)


def run_selector_decision_campaign(
    cases: tuple[SelectorCampaignCase, ...] | list[SelectorCampaignCase],
    governor: OnlineGovernor,
    selector: UnfilteredPrimitiveSelector,
    config: SelectorDecisionCampaignConfig | None = None,
) -> SelectorDecisionCampaignReport:
    """Run selector comparison over prebuilt decision cases without writing files."""

    campaign_config = SelectorDecisionCampaignConfig() if config is None else config
    case_tuple = tuple(cases)
    if not case_tuple:
        raise ValueError("At least one selector campaign case is required.")
    records = tuple(
        SelectorDecisionRecord.from_comparison(
            case,
            compare_governor_to_unfiltered(case.state, governor, selector, tier=campaign_config.tier),
        )
        for case in case_tuple
    )
    return SelectorDecisionCampaignReport(tier=campaign_config.tier, records=records, config=campaign_config)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _seed_sort_key(seed: int | None) -> tuple[int, int]:
    return (0, -1) if seed is None else (1, int(seed))


def _mean(values: Iterable[float]) -> float:
    materialised = [float(value) for value in values]
    return float(sum(materialised) / len(materialised))
