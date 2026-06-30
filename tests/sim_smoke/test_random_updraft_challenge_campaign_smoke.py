from __future__ import annotations

import json
from pathlib import Path

from moewe.baselines import UnfilteredPrimitiveSelector
from moewe.campaigns import (
    RandomUpdraftChallengeConfig,
    build_random_updraft_challenge_cases,
    run_random_updraft_challenge_campaign,
)
from moewe.governor import OnlineGovernor
from moewe.primitives import (
    CompressionSpec,
    CompressionTierBudgets,
    PrimitiveLibrary,
    build_structured_library_design_cases,
    build_structured_primitive_library,
    compress_retained_primitives,
    dense_smoke_grammar,
)
from moewe.returnability import build_returnability_graph_from_report


def test_random_updraft_challenge_campaign_smoke_builds_report() -> None:
    config_path = Path("config/simulation/random_updraft_challenge_campaign_smoke.yaml")
    assert config_path.exists()
    config_text = config_path.read_text(encoding="utf-8")
    assert "writes_files_by_default: false" in config_text
    assert "not a final benchmark campaign" in config_text

    grammar = dense_smoke_grammar()
    structured_report = build_structured_primitive_library(
        grammar_spec=grammar,
        design_cases=build_structured_library_design_cases()[:2],
        max_primitives=6,
    )
    graph = build_returnability_graph_from_report(structured_report)
    compressed = compress_retained_primitives(
        structured_report.validation_report,
        spec=CompressionSpec(
            tier_budgets=CompressionTierBudgets(heavy=3, balanced=2, light=1, super_light=1, smoke=1)
        ),
        grammar_spec=grammar,
    )
    library = PrimitiveLibrary.from_compression(compressed)
    governor = OnlineGovernor(library, graph)
    selector = UnfilteredPrimitiveSelector(library)

    config = RandomUpdraftChallengeConfig(case_count=4, base_seed=91)
    cases = build_random_updraft_challenge_cases(config)
    report = run_random_updraft_challenge_campaign(cases, governor, selector, library, config)
    record = report.to_record()
    summary = record["summary"]

    assert summary["case_count"] == 4
    assert summary["record_count"] == 12
    assert set(summary["selectors"]) == {"governor", "unfiltered", "reference_tracking_pd"}
    assert {case.case_set for case in cases} == {"random_challenge_after_freeze"}
    assert {item["case_set"] for item in record["records"]} == {"random_challenge_after_freeze"}
    for selector_name in ("governor", "unfiltered", "reference_tracking_pd"):
        selector_summary = summary["by_selector"][selector_name]
        assert "failure_reason_counts" in selector_summary
        assert "rollout_success_count" in selector_summary
        assert "gate_crossed_count" in selector_summary
    json.dumps(record, sort_keys=True)
