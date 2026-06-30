from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from moewe.baselines import BASELINE_COMMON_SCHEMA_FIELDS
from moewe.baselines import UnfilteredSelectionDecision
from moewe.benchmarks import (
    FinalBenchmarkConfig,
    build_final_benchmark_plan,
    require_full_benchmark_guards,
    run_final_benchmark_preflight,
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


def _governor_decision() -> GovernorDecision:
    return GovernorDecision(
        tier="balanced",
        entry_class="entry",
        candidate_count=0,
        admissible_candidate_count=0,
        selected_candidate_id=None,
        selected_primitive_id=None,
        selected_entry_class=None,
        selected_exit_class=None,
        selected_transition_id=None,
        selected_reason=None,
        fallback_used=False,
        fallback_reason=None,
        rejection_reasons=("no_admissible_candidate",),
        candidate_evidence=(),
    )


def _unfiltered_decision() -> UnfilteredSelectionDecision:
    return UnfilteredSelectionDecision(
        tier="balanced",
        entry_class="entry",
        candidate_count=0,
        selected_candidate_id=None,
        selected_entry_class=None,
        selected_exit_class=None,
        selected_score=None,
        fallback_used=False,
        fallback_reason=None,
        candidate_scores=(),
        rejection_reasons=("no_retrieval_candidate",),
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"full_case_count": 0},
        {"preflight_case_count": 0},
        {"preflight_case_count": 5, "full_case_count": 4},
        {"methods": ("governor", "governor")},
        {"methods": ("governor", "unknown")},
        {"case_families": ()},
        {"case_families": ("weak_random_single_source", "weak_random_single_source")},
        {"case_families": ("unknown_family",)},
        {"tier": ""},
        {"runs_full_simulation": True},
        {"writes_files_by_default": True},
    ),
)
def test_final_benchmark_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FinalBenchmarkConfig(**kwargs)


def test_final_benchmark_plan_counts_full_and_preflight_records() -> None:
    config = FinalBenchmarkConfig(full_case_count=12, preflight_case_count=4)

    plan = build_final_benchmark_plan(config)

    assert plan.expected_full_record_count == 12 * config.method_count
    assert plan.expected_preflight_record_count == 4 * config.method_count
    assert plan.to_record()["full_simulation_executed"] is False
    json.dumps(plan.to_record(), sort_keys=True)


def test_final_benchmark_preflight_report_is_deterministic_and_json_serialisable() -> None:
    config = FinalBenchmarkConfig(full_case_count=8, preflight_case_count=2)
    governor = _GovernorStub(_governor_decision())
    selector = _SelectorStub(_unfiltered_decision())
    library = _LibraryStub(_CompressedStub({}))

    first = run_final_benchmark_preflight(governor, selector, library, config).to_record()
    second = run_final_benchmark_preflight(governor, selector, library, config).to_record()

    assert first == second
    assert first["full_simulation_executed"] is False
    assert first["writes_files_by_default"] is False
    assert first["ready_for_full_simulation"] is True
    assert first["expected_full_record_count"] == 8 * config.method_count
    assert first["preflight_case_count"] == 2
    assert first["preflight_record_count"] == 2 * config.method_count
    json.dumps(first, sort_keys=True)


def test_final_benchmark_schema_contains_outcome_and_safety_metrics() -> None:
    report = run_final_benchmark_preflight(
        _GovernorStub(_governor_decision()),
        _SelectorStub(_unfiltered_decision()),
        _LibraryStub(_CompressedStub({})),
        FinalBenchmarkConfig(full_case_count=8, preflight_case_count=2),
    )
    schema = set(report.to_record()["schema_fields"])

    assert set(BASELINE_COMMON_SCHEMA_FIELDS) <= schema


def test_full_benchmark_guards_require_freeze_and_external_output() -> None:
    repo_root = Path.cwd()
    external_output = repo_root.parent / "moewe_full_benchmark_results"
    with pytest.raises(ValueError):
        require_full_benchmark_guards(controller_frozen=False, write_results=True, output_dir=external_output)
    with pytest.raises(ValueError):
        require_full_benchmark_guards(controller_frozen=True, write_results=False, output_dir=external_output)
    with pytest.raises(ValueError):
        require_full_benchmark_guards(
            controller_frozen=True,
            write_results=True,
            output_dir=repo_root / "results",
            public_repo_root=repo_root,
        )

    record = require_full_benchmark_guards(
        controller_frozen=True,
        write_results=True,
        output_dir=external_output,
        public_repo_root=repo_root,
    )

    assert record["guard_only"] is True
    assert record["runs_full_simulation"] is False
