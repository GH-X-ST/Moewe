from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from moewe.baselines import UnfilteredSelectionDecision
from moewe.campaigns import (
    RandomUpdraftChallengeConfig,
    RandomUpdraftChallengeMethodRecord,
    RandomUpdraftChallengeReport,
    build_random_updraft_challenge_cases,
    run_random_updraft_challenge_campaign,
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


def _governor_decision(selected_candidate_id: str | None = None) -> GovernorDecision:
    return GovernorDecision(
        tier="balanced",
        entry_class="entry",
        candidate_count=0 if selected_candidate_id is None else 1,
        admissible_candidate_count=0 if selected_candidate_id is None else 1,
        selected_candidate_id=selected_candidate_id,
        selected_primitive_id=selected_candidate_id,
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
    case_id: str = "challenge_case_a",
    family: str = "weak_random_single_source",
    selected: str | None = "prim_a",
    success: bool = True,
    crossed: bool | None = True,
    failure_reason: str | None = None,
    miss: float | None = 0.1,
    safety: float | None = 1.0,
    margin: float | None = 2.0,
) -> RandomUpdraftChallengeMethodRecord:
    return RandomUpdraftChallengeMethodRecord(
        case_id=case_id,
        case_set="random_challenge_after_freeze",
        environment_family=family,
        selector_name=selector_name,
        selected_candidate_id=selected,
        selected_primitive_id=selected,
        rollout_success=success,
        gate_crossed=crossed,
        gate_miss_distance_m=miss,
        min_safety_margin_m=safety,
        terminal_specific_energy_margin_j_kg=margin,
        terminal_specific_energy_change_j_kg=1.0 if margin is not None else None,
        max_angle_of_attack_rad=0.1 if safety is not None else None,
        max_command_abs_rad=0.2 if safety is not None else None,
        failure_reason=failure_reason,
        fallback_used=False,
        fallback_reason=None,
        decision_record={"selector": selector_name},
    )


def test_random_challenge_case_generation_is_deterministic() -> None:
    config = RandomUpdraftChallengeConfig(case_count=4, base_seed=91)

    first = [case.to_record() for case in build_random_updraft_challenge_cases(config)]
    second = [case.to_record() for case in build_random_updraft_challenge_cases(config)]

    assert first == second
    assert [case["environment_family"] for case in first] == [
        "weak_random_single_source",
        "hard_random_single_source",
        "random_two_source",
        "random_four_source",
    ]
    json.dumps(first, sort_keys=True)


def test_random_challenge_case_records_are_after_freeze_and_json_serialisable() -> None:
    cases = build_random_updraft_challenge_cases(RandomUpdraftChallengeConfig(case_count=2))

    for case in cases:
        record = case.to_record()
        assert record["case_set"] == "random_challenge_after_freeze"
        assert record["fan_count"] >= 1
        assert record["updraft_factor_record"]["used_for_library_construction"] is False
        assert record["updraft_factor_record"]["used_for_governor_tuning"] is False
        json.dumps(record, sort_keys=True)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"case_count": 0},
        {"tier": ""},
        {"selectors": ()},
        {"selectors": ("governor", "unknown")},
        {"selectors": ("governor", "governor")},
        {"dt_s": 0.0},
        {"max_duration_s": -0.1},
        {"reference_horizon_s": 0.0},
        {"wind_mode": "bad"},
        {"case_set": "library_design"},
    ),
)
def test_invalid_random_challenge_config_rejects_bad_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RandomUpdraftChallengeConfig(**kwargs)


def test_random_challenge_report_summary_counts_selectors_and_cases() -> None:
    report = RandomUpdraftChallengeReport(
        config=RandomUpdraftChallengeConfig(selectors=("governor", "unfiltered")),
        records=(
            _record("governor", case_id="case_a"),
            _record("governor", case_id="case_b", success=False, crossed=False, failure_reason="floor", miss=0.3),
            _record("unfiltered", case_id="case_a", success=False, crossed=False, failure_reason="timeout", margin=None),
        ),
    )

    summary = report.to_summary()

    assert summary["case_count"] == 2
    assert summary["record_count"] == 3
    assert summary["selectors"] == ["governor", "unfiltered"]
    assert summary["by_selector"]["governor"]["record_count"] == 2
    assert summary["by_selector"]["governor"]["selection_count"] == 2
    assert summary["by_selector"]["governor"]["rollout_success_count"] == 1
    assert summary["by_selector"]["governor"]["gate_crossed_count"] == 1
    assert summary["by_selector"]["governor"]["mean_gate_miss_distance_m"] == pytest.approx(0.2)
    assert summary["by_selector"]["governor"]["failure_reason_counts"] == {"floor": 1}
    assert summary["by_selector"]["unfiltered"]["failure_reason_counts"] == {"timeout": 1}
    json.dumps(report.to_record(), sort_keys=True)


def test_missing_selected_primitive_returns_failure_record() -> None:
    config = RandomUpdraftChallengeConfig(case_count=1, selectors=("governor",))
    case = build_random_updraft_challenge_cases(config)[0]

    report = run_random_updraft_challenge_campaign(
        [case],
        _GovernorStub(_governor_decision("missing_primitive")),
        _SelectorStub(_unfiltered_decision()),
        _LibraryStub(_CompressedStub({})),
        config,
    )

    record = report.records[0]

    assert record.selector_name == "governor"
    assert record.selected_candidate_id == "missing_primitive"
    assert record.rollout_success is False
    assert record.failure_reason == "missing_selected_primitive"
    json.dumps(report.to_record(), sort_keys=True)
