"""Selector rollout smoke campaigns for short segment evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from moewe.baselines import UnfilteredPrimitiveSelector, UnfilteredSelectionDecision
from moewe.governor import GovernorDecision, OnlineGovernor
from moewe.primitives import PrimitiveLibrary, PrimitiveRolloutConfig, rollout_primitive

from .selector_decision_campaign import SelectorCampaignCase

SELECTOR_GOVERNOR = "governor"
SELECTOR_UNFILTERED = "unfiltered"
SUPPORTED_SELECTORS = frozenset({SELECTOR_GOVERNOR, SELECTOR_UNFILTERED})


def _count(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _unique_ordered(values: Iterable[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


@dataclass(frozen=True)
class SelectorRolloutCampaignConfig:
    """Configuration for short selector rollout evidence."""

    tier: str = "balanced"
    selectors: tuple[str, ...] = (SELECTOR_GOVERNOR, SELECTOR_UNFILTERED)
    dt_s: float = 0.01
    max_duration_s: float = 0.10
    wind_mode: str = "panel"
    seed: int | None = 0

    def __post_init__(self) -> None:
        if not self.tier:
            raise ValueError("Rollout campaign tier must be non-empty.")
        selectors = tuple(self.selectors)
        if not selectors:
            raise ValueError("At least one selector is required.")
        unknown = sorted(set(selectors) - SUPPORTED_SELECTORS)
        if unknown:
            raise ValueError(f"Unsupported selector names: {unknown}")
        if not math.isfinite(float(self.dt_s)) or float(self.dt_s) <= 0.0:
            raise ValueError("dt_s must be positive and finite.")
        if not math.isfinite(float(self.max_duration_s)) or float(self.max_duration_s) <= 0.0:
            raise ValueError("max_duration_s must be positive and finite.")

    def to_record(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "selectors": list(self.selectors),
            "dt_s": float(self.dt_s),
            "max_duration_s": float(self.max_duration_s),
            "wind_mode": self.wind_mode,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SelectorRolloutRecord:
    """Segment evidence for one selected primitive rollout."""

    selector_name: str
    case_id: str
    design_case_id: str
    case_set: str
    expected_entry_class: str
    expected_exit_class: str
    selected_candidate_id: str | None
    selected_primitive_id: str | None
    fallback_used: bool
    fallback_reason: str | None
    rollout_success: bool
    failure_reason: str | None
    min_safety_margin_m: float | None
    terminal_specific_energy_change_j_kg: float | None
    max_angle_of_attack_rad: float | None
    max_command_abs_rad: float | None
    rollout_duration_s: float
    state_count: int
    decision_record: dict[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "selector_name": self.selector_name,
            "case_id": self.case_id,
            "design_case_id": self.design_case_id,
            "case_set": self.case_set,
            "expected_entry_class": self.expected_entry_class,
            "expected_exit_class": self.expected_exit_class,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_primitive_id": self.selected_primitive_id,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "rollout_success": self.rollout_success,
            "failure_reason": self.failure_reason,
            "min_safety_margin_m": self.min_safety_margin_m,
            "terminal_specific_energy_change_j_kg": self.terminal_specific_energy_change_j_kg,
            "max_angle_of_attack_rad": self.max_angle_of_attack_rad,
            "max_command_abs_rad": self.max_command_abs_rad,
            "rollout_duration_s": self.rollout_duration_s,
            "state_count": self.state_count,
            "decision_record": self.decision_record,
        }


@dataclass(frozen=True)
class SelectorRolloutCampaignReport:
    """Compact selector rollout evidence report."""

    records: tuple[SelectorRolloutRecord, ...]
    config: SelectorRolloutCampaignConfig

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("Selector rollout campaign report requires at least one record.")

    def to_summary(self) -> dict[str, object]:
        selectors = sorted({record.selector_name for record in self.records})
        by_selector = {
            selector: _selector_summary(
                tuple(record for record in self.records if record.selector_name == selector)
            )
            for selector in selectors
        }
        return {
            "record_count": len(self.records),
            "case_count": len({record.case_id for record in self.records}),
            "selectors": selectors,
            "by_selector": by_selector,
        }

    def to_record(self) -> dict[str, object]:
        return {
            "config": self.config.to_record(),
            "summary": self.to_summary(),
            "records": [record.to_record() for record in self.records],
        }


def run_selector_rollout_campaign(
    cases: tuple[SelectorCampaignCase, ...] | list[SelectorCampaignCase],
    governor: OnlineGovernor,
    selector: UnfilteredPrimitiveSelector,
    library: PrimitiveLibrary,
    config: SelectorRolloutCampaignConfig | None = None,
) -> SelectorRolloutCampaignReport:
    """Roll out primitives selected by existing selector paths."""

    campaign_config = SelectorRolloutCampaignConfig() if config is None else config
    case_tuple = tuple(cases)
    if not case_tuple:
        raise ValueError("At least one selector rollout campaign case is required.")

    records: list[SelectorRolloutRecord] = []
    for case in case_tuple:
        for selector_name in campaign_config.selectors:
            decision = _selector_decision(selector_name, case, governor, selector, campaign_config.tier)
            records.append(_rollout_selection(case, decision, library, campaign_config))
    return SelectorRolloutCampaignReport(records=tuple(records), config=campaign_config)


@dataclass(frozen=True)
class _SelectorDecision:
    selector_name: str
    selected_candidate_id: str | None
    selected_primitive_id: str | None
    fallback_used: bool
    fallback_reason: str | None
    decision_record: dict[str, object]


def _selector_decision(
    selector_name: str,
    case: SelectorCampaignCase,
    governor: OnlineGovernor,
    selector: UnfilteredPrimitiveSelector,
    tier: str,
) -> _SelectorDecision:
    if selector_name == SELECTOR_GOVERNOR:
        decision = governor.decide(case.state, tier=tier)
        return _governor_selection(decision)
    if selector_name == SELECTOR_UNFILTERED:
        decision = selector.decide(case.state, tier=tier)
        return _unfiltered_selection(decision)
    raise ValueError(f"Unsupported selector name: {selector_name}")


def _governor_selection(decision: GovernorDecision) -> _SelectorDecision:
    return _SelectorDecision(
        selector_name=SELECTOR_GOVERNOR,
        selected_candidate_id=decision.selected_candidate_id,
        selected_primitive_id=decision.selected_primitive_id or decision.selected_candidate_id,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        decision_record=decision.to_record(),
    )


def _unfiltered_selection(decision: UnfilteredSelectionDecision) -> _SelectorDecision:
    return _SelectorDecision(
        selector_name=SELECTOR_UNFILTERED,
        selected_candidate_id=decision.selected_candidate_id,
        selected_primitive_id=decision.selected_candidate_id,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        decision_record=decision.to_record(),
    )


def _rollout_selection(
    case: SelectorCampaignCase,
    decision: _SelectorDecision,
    library: PrimitiveLibrary,
    config: SelectorRolloutCampaignConfig,
) -> SelectorRolloutRecord:
    if decision.selected_primitive_id is None and decision.selected_candidate_id is None:
        return _non_rollout_record(case, decision, "no_selected_primitive")

    candidate_index = library.compressed.candidate_index
    primitive_id = None
    primitive = None
    for selected_id in _unique_ordered((decision.selected_primitive_id, decision.selected_candidate_id)):
        primitive = candidate_index.get(selected_id)
        if primitive is not None:
            primitive_id = selected_id
            break
    if primitive is None:
        return _non_rollout_record(case, decision, "missing_selected_primitive")

    rollout_config = PrimitiveRolloutConfig(
        dt_s=float(config.dt_s),
        wind_mode=config.wind_mode,
        max_duration_s=float(config.max_duration_s),
        scenario_id=f"{case.case_id}:{decision.selector_name}",
        seed=case.seed if case.seed is not None else config.seed,
    )
    result = rollout_primitive(
        primitive=primitive,
        initial_state=case.state,
        config=rollout_config,
    )
    evidence = result.evidence
    return SelectorRolloutRecord(
        selector_name=decision.selector_name,
        case_id=case.case_id,
        design_case_id=case.design_case_id,
        case_set=case.case_set,
        expected_entry_class=case.entry_class,
        expected_exit_class=case.exit_class,
        selected_candidate_id=decision.selected_candidate_id,
        selected_primitive_id=primitive_id,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        rollout_success=bool(evidence.rollout_success),
        failure_reason=evidence.failure_reason,
        min_safety_margin_m=float(evidence.min_safety_margin_m),
        terminal_specific_energy_change_j_kg=float(evidence.terminal_specific_energy_change_j_kg),
        max_angle_of_attack_rad=float(evidence.max_angle_of_attack_rad),
        max_command_abs_rad=float(evidence.max_command_abs_rad),
        rollout_duration_s=float(evidence.rollout_duration_s),
        state_count=len(result.states),
        decision_record=decision.decision_record,
    )


def _non_rollout_record(
    case: SelectorCampaignCase,
    decision: _SelectorDecision,
    reason: str,
) -> SelectorRolloutRecord:
    return SelectorRolloutRecord(
        selector_name=decision.selector_name,
        case_id=case.case_id,
        design_case_id=case.design_case_id,
        case_set=case.case_set,
        expected_entry_class=case.entry_class,
        expected_exit_class=case.exit_class,
        selected_candidate_id=decision.selected_candidate_id,
        selected_primitive_id=decision.selected_primitive_id,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        rollout_success=False,
        failure_reason=reason,
        min_safety_margin_m=None,
        terminal_specific_energy_change_j_kg=None,
        max_angle_of_attack_rad=None,
        max_command_abs_rad=None,
        rollout_duration_s=0.0,
        state_count=0,
        decision_record=decision.decision_record,
    )


def _selector_summary(records: tuple[SelectorRolloutRecord, ...]) -> dict[str, object]:
    selected = [record for record in records if record.selected_candidate_id is not None]
    successful = [record for record in records if record.rollout_success]
    failures = [record.failure_reason for record in records if record.failure_reason is not None]
    safety = [record.min_safety_margin_m for record in records if record.min_safety_margin_m is not None]
    energy = [
        record.terminal_specific_energy_change_j_kg
        for record in records
        if record.terminal_specific_energy_change_j_kg is not None
    ]
    return {
        "record_count": len(records),
        "selection_count": len(selected),
        "rollout_success_count": len(successful),
        "failure_reason_counts": _count(failures),
        "mean_min_safety_margin_m": _finite_mean(safety),
        "mean_terminal_specific_energy_change_j_kg": _finite_mean(energy),
    }


def _finite_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))
