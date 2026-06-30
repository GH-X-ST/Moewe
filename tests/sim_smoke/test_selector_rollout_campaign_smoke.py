from __future__ import annotations

import json
from pathlib import Path

from moewe.baselines import UnfilteredPrimitiveSelector
from moewe.campaigns import (
    SelectorDecisionCampaignConfig,
    build_selector_campaign_cases_from_structured_report,
)
from moewe.campaigns.selector_rollout_campaign import (
    SelectorRolloutCampaignConfig,
    run_selector_rollout_campaign,
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


def test_selector_rollout_campaign_smoke_builds_report() -> None:
    config_path = Path("config/simulation/selector_rollout_campaign_smoke.yaml")
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
    case_config = SelectorDecisionCampaignConfig(max_cases=4, retained_only=True, state_index=0)
    cases = build_selector_campaign_cases_from_structured_report(structured_report, case_config)
    rollout_config = SelectorRolloutCampaignConfig(dt_s=0.01, max_duration_s=0.10)

    report = run_selector_rollout_campaign(cases, governor, selector, library, rollout_config)
    record = report.to_record()
    summary = record["summary"]

    assert cases
    assert summary["case_count"] == len(cases)
    assert summary["selectors"] == ["governor", "unfiltered"]
    assert summary["record_count"] == len(cases) * 2
    assert record["records"]
    for selector_name in ("governor", "unfiltered"):
        selector_summary = summary["by_selector"][selector_name]
        assert "selection_count" in selector_summary
        assert "rollout_success_count" in selector_summary
        assert "failure_reason_counts" in selector_summary
    json.dumps(record, sort_keys=True)
