from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

from moewe.baselines import UnfilteredSelectionDecision
from moewe.campaigns import SelectorCampaignCase
from moewe.campaigns.selector_rollout_campaign import (
    SelectorRolloutCampaignConfig,
    SelectorRolloutCampaignReport,
    SelectorRolloutRecord,
    run_selector_rollout_campaign,
)
from moewe.governor import GovernorDecision
from moewe.sim.state import FlightState


@dataclass(frozen=True)
class _CompressedStub:
    candidate_index: dict[str, object]


@dataclass(frozen=True)
class _LibraryStub:
    compressed: _CompressedStub


@dataclass(frozen=True)
class _GovernorStub:
    decision: GovernorDecision

    def decide(self, state: FlightState, tier: str = "balanced") -> GovernorDecision:
        assert state.finite()
        assert tier == self.decision.tier
        return self.decision


@dataclass(frozen=True)
class _SelectorStub:
    decision: UnfilteredSelectionDecision

    def decide(self, state: FlightState, tier: str = "balanced") -> UnfilteredSelectionDecision:
        assert state.finite()
        assert tier == self.decision.tier
        return self.decision


def _state() -> FlightState:
    return FlightState(
        position_w_m=np.array([0.0, 0.0, 2.0]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([7.0, 0.0, 0.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def _case() -> SelectorCampaignCase:
    return SelectorCampaignCase(
        case_id="design_a:prim_a:state_0",
        design_case_id="design_a",
        case_set="library_design",
        primitive_id="prim_a",
        entry_class="entry",
        exit_class="terminal",
        state=_state(),
    )


def _governor_decision(selected_candidate_id: str | None = None, selected_primitive_id: str | None = None) -> GovernorDecision:
    return GovernorDecision(
        tier="balanced",
        entry_class="entry",
        candidate_count=0 if selected_candidate_id is None else 1,
        admissible_candidate_count=0 if selected_candidate_id is None else 1,
        selected_candidate_id=selected_candidate_id,
        selected_primitive_id=selected_primitive_id,
        selected_entry_class=None,
        selected_exit_class=None,
        selected_transition_id=None,
        selected_reason=None,
        fallback_used=False,
        fallback_reason=None,
        rejection_reasons=() if selected_candidate_id is not None else ("no_admissible_candidate",),
        candidate_evidence=(),
    )


def _unfiltered_decision(selected_candidate_id: str | None = None) -> UnfilteredSelectionDecision:
    return UnfilteredSelectionDecision(
        tier="balanced",
        entry_class="entry",
        candidate_count=0 if selected_candidate_id is None else 1,
        selected_candidate_id=selected_candidate_id,
        selected_entry_class=None,
        selected_exit_class=None,
        selected_score=None,
        fallback_used=False,
        fallback_reason=None,
        candidate_scores=(),
        rejection_reasons=() if selected_candidate_id is not None else ("no_retrieval_candidate",),
    )


def _record(
    selector_name: str,
    *,
    selected: str | None = "prim_a",
    success: bool = True,
    failure_reason: str | None = None,
    safety: float | None = 1.0,
    energy: float | None = 2.0,
) -> SelectorRolloutRecord:
    return SelectorRolloutRecord(
        selector_name=selector_name,
        case_id="design_a:prim_a:state_0",
        design_case_id="design_a",
        case_set="library_design",
        expected_entry_class="entry",
        expected_exit_class="terminal",
        selected_candidate_id=selected,
        selected_primitive_id=selected,
        fallback_used=False,
        fallback_reason=None,
        rollout_success=success,
        failure_reason=failure_reason,
        min_safety_margin_m=safety,
        terminal_specific_energy_change_j_kg=energy,
        max_angle_of_attack_rad=0.1 if safety is not None else None,
        max_command_abs_rad=0.1 if safety is not None else None,
        rollout_duration_s=0.03 if safety is not None else 0.0,
        state_count=4 if safety is not None else 0,
        decision_record={"selector": selector_name},
    )


def test_rollout_campaign_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="At least one selector rollout campaign case"):
        run_selector_rollout_campaign(
            [],
            _GovernorStub(_governor_decision()),
            _SelectorStub(_unfiltered_decision()),
            _LibraryStub(_CompressedStub({})),
        )


def test_no_selected_primitive_records_explicit_reason() -> None:
    report = run_selector_rollout_campaign(
        [_case()],
        _GovernorStub(_governor_decision()),
        _SelectorStub(_unfiltered_decision()),
        _LibraryStub(_CompressedStub({})),
        SelectorRolloutCampaignConfig(selectors=("governor",)),
    )

    record = report.records[0]

    assert record.selector_name == "governor"
    assert record.rollout_success is False
    assert record.failure_reason == "no_selected_primitive"
    assert record.state_count == 0


def test_missing_selected_primitive_records_explicit_reason() -> None:
    report = run_selector_rollout_campaign(
        [_case()],
        _GovernorStub(_governor_decision()),
        _SelectorStub(_unfiltered_decision("missing_primitive")),
        _LibraryStub(_CompressedStub({})),
        SelectorRolloutCampaignConfig(selectors=("unfiltered",)),
    )

    record = report.records[0]

    assert record.selector_name == "unfiltered"
    assert record.selected_candidate_id == "missing_primitive"
    assert record.failure_reason == "missing_selected_primitive"
    assert record.rollout_success is False


def test_report_summary_counts_by_selector() -> None:
    report = SelectorRolloutCampaignReport(
        config=SelectorRolloutCampaignConfig(),
        records=(
            _record("governor", success=True),
            _record("governor", selected=None, success=False, failure_reason="no_selected_primitive", safety=None, energy=None),
            _record("unfiltered", success=False, failure_reason="floor", safety=-0.5, energy=-1.0),
        ),
    )

    summary = report.to_summary()

    assert summary["record_count"] == 3
    assert summary["case_count"] == 1
    assert summary["selectors"] == ["governor", "unfiltered"]
    assert summary["by_selector"]["governor"]["record_count"] == 2
    assert summary["by_selector"]["governor"]["selection_count"] == 1
    assert summary["by_selector"]["governor"]["rollout_success_count"] == 1
    assert summary["by_selector"]["governor"]["failure_reason_counts"] == {"no_selected_primitive": 1}
    assert summary["by_selector"]["governor"]["mean_min_safety_margin_m"] == 1.0
    assert summary["by_selector"]["unfiltered"]["failure_reason_counts"] == {"floor": 1}
    assert summary["by_selector"]["unfiltered"]["mean_terminal_specific_energy_change_j_kg"] == -1.0


def test_rollout_records_and_report_are_deterministic_and_json_serialisable() -> None:
    report = SelectorRolloutCampaignReport(
        config=SelectorRolloutCampaignConfig(selectors=("governor",)),
        records=(_record("governor"),),
    )

    first = report.to_record()
    second = report.to_record()

    assert first == second
    json.dumps(first, sort_keys=True)
