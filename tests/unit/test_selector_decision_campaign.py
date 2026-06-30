from __future__ import annotations

import json

import numpy as np
import pytest

from moewe.campaigns import (
    SelectorCampaignCase,
    SelectorDecisionCampaignConfig,
    SelectorDecisionCampaignReport,
    SelectorDecisionRecord,
    run_selector_decision_campaign,
)
from moewe.sim.state import FlightState


def _state() -> FlightState:
    return FlightState(
        position_w_m=np.array([1.0, 2.0, 3.0]),
        euler_rad=np.array([0.1, 0.0, 0.0]),
        velocity_b_m_s=np.array([7.0, 0.0, 0.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def _case(case_id: str = "design_a:prim_a:state_0") -> SelectorCampaignCase:
    return SelectorCampaignCase(
        case_id=case_id,
        design_case_id="design_a",
        case_set="library_design",
        primitive_id="prim_a",
        entry_class="entry",
        exit_class="terminal",
        state=_state(),
    )


def _comparison(
    *,
    governor_candidate_count: int = 2,
    governor_admissible_candidate_count: int = 1,
    unfiltered_candidate_count: int = 2,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> dict[str, object]:
    return {
        "tier": "balanced",
        "entry_class": "entry",
        "governor_selected_candidate_id": "prim_a",
        "unfiltered_selected_candidate_id": "prim_a",
        "same_selected_candidate": True,
        "governor_blocked_unfiltered_selection": False,
        "governor_rejection_reasons": [],
        "governor_decision": {
            "entry_class": "entry",
            "candidate_count": governor_candidate_count,
            "admissible_candidate_count": governor_admissible_candidate_count,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        },
        "unfiltered_decision": {
            "candidate_count": unfiltered_candidate_count,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        },
    }


def _record(
    case_id: str,
    *,
    governor_selected: str | None = "prim_a",
    unfiltered_selected: str | None = "prim_a",
    same: bool = True,
    blocked: bool = False,
    reasons: tuple[str, ...] = (),
    fallback_used: bool = False,
    governor_candidate_count: int = 2,
    governor_admissible_candidate_count: int = 1,
    unfiltered_candidate_count: int = 2,
) -> SelectorDecisionRecord:
    comparison = _comparison(
        governor_candidate_count=governor_candidate_count,
        governor_admissible_candidate_count=governor_admissible_candidate_count,
        unfiltered_candidate_count=unfiltered_candidate_count,
        fallback_used=fallback_used,
        fallback_reason="no_exact_entry_class" if fallback_used else None,
    )
    comparison.update(
        {
            "governor_selected_candidate_id": governor_selected,
            "unfiltered_selected_candidate_id": unfiltered_selected,
            "same_selected_candidate": same,
            "governor_blocked_unfiltered_selection": blocked,
            "governor_rejection_reasons": list(reasons),
        }
    )
    governor_decision = comparison["governor_decision"]
    unfiltered_decision = comparison["unfiltered_decision"]
    assert isinstance(governor_decision, dict)
    assert isinstance(unfiltered_decision, dict)
    return SelectorDecisionRecord(
        case_id=case_id,
        design_case_id="design_a" if case_id.endswith("a") else "design_b",
        case_set="library_design",
        primitive_id=case_id,
        expected_entry_class="entry",
        expected_exit_class="terminal",
        tier="balanced",
        governor_entry_class="entry",
        governor_selected_candidate_id=governor_selected,
        unfiltered_selected_candidate_id=unfiltered_selected,
        same_selected_candidate=same,
        governor_blocked_unfiltered_selection=blocked,
        governor_rejection_reasons=reasons,
        fallback_used=fallback_used,
        fallback_reason="no_exact_entry_class" if fallback_used else None,
        governor_candidate_count=governor_candidate_count,
        governor_admissible_candidate_count=governor_admissible_candidate_count,
        unfiltered_candidate_count=unfiltered_candidate_count,
        comparison_record=comparison,
    )


def test_campaign_config_validates_basic_limits() -> None:
    SelectorDecisionCampaignConfig(tier="balanced", max_cases=1, state_index=0)

    with pytest.raises(ValueError, match="tier"):
        SelectorDecisionCampaignConfig(tier="")
    with pytest.raises(ValueError, match="max_cases"):
        SelectorDecisionCampaignConfig(max_cases=0)
    with pytest.raises(ValueError, match="state_index"):
        SelectorDecisionCampaignConfig(state_index=-1)


def test_campaign_case_record_is_json_serialisable_with_state_vector() -> None:
    record = _case().to_record()

    assert record["case_id"] == "design_a:prim_a:state_0"
    assert len(record["state_vector"]) == 15
    json.dumps(record, sort_keys=True)


def test_campaign_summary_counts_selector_outcomes_deterministically() -> None:
    report = SelectorDecisionCampaignReport(
        tier="balanced",
        config=SelectorDecisionCampaignConfig(max_cases=3),
        records=(
            _record("case_a"),
            _record("case_b", governor_selected=None, unfiltered_selected="prim_b", same=False, blocked=True, reasons=("forbidden_exit",), fallback_used=True, governor_candidate_count=3, governor_admissible_candidate_count=0),
            _record("case_c", governor_selected="prim_c", unfiltered_selected="prim_d", same=False, blocked=True, reasons=("forbidden_exit", "entry_not_recoverable"), governor_candidate_count=1, governor_admissible_candidate_count=1, unfiltered_candidate_count=1),
        ),
    )

    summary = report.to_summary()

    assert summary["case_count"] == 3
    assert summary["design_case_count"] == 2
    assert summary["case_sets"] == ["library_design"]
    assert summary["governor_selection_count"] == 2
    assert summary["unfiltered_selection_count"] == 3
    assert summary["same_selection_count"] == 1
    assert summary["governor_blocked_unfiltered_count"] == 2
    assert summary["fallback_count"] == 1
    assert summary["governor_rejection_reason_counts"] == {
        "entry_not_recoverable": 1,
        "forbidden_exit": 2,
    }
    assert summary["mean_governor_candidate_count"] == 2.0
    assert summary["mean_governor_admissible_candidate_count"] == 2.0 / 3.0
    assert summary["mean_unfiltered_candidate_count"] == 5.0 / 3.0


def test_campaign_report_is_deterministic_and_json_serialisable() -> None:
    records = (_record("case_a"), _record("case_b", same=False, blocked=True, reasons=("forbidden_exit",)))
    report = SelectorDecisionCampaignReport(
        tier="balanced",
        records=records,
        config=SelectorDecisionCampaignConfig(max_cases=2),
    )

    first = report.to_record()
    second = report.to_record()

    assert first == second
    json.dumps(first, sort_keys=True)


def test_campaign_runner_rejects_empty_case_list() -> None:
    with pytest.raises(ValueError, match="At least one selector campaign case"):
        run_selector_decision_campaign([], governor=None, selector=None)
