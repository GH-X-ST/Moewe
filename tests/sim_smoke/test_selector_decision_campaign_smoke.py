from __future__ import annotations

import json
from pathlib import Path

from moewe.baselines import UnfilteredPrimitiveSelector
from moewe.campaigns import (
    SelectorDecisionCampaignConfig,
    build_selector_campaign_cases_from_structured_report,
    run_selector_decision_campaign,
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


def test_selector_decision_campaign_smoke_builds_report() -> None:
    config_path = Path("config/simulation/selector_decision_campaign_smoke.yaml")
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
    campaign_config = SelectorDecisionCampaignConfig(max_cases=4, retained_only=True, state_index=0)
    cases = build_selector_campaign_cases_from_structured_report(structured_report, campaign_config)

    report = run_selector_decision_campaign(cases, governor, selector, campaign_config)
    record = report.to_record()
    summary = record["summary"]

    assert cases
    assert summary["case_count"] == len(cases)
    assert summary["case_count"] <= 4
    assert "governor_selection_count" in summary
    assert "unfiltered_selection_count" in summary
    assert "governor_rejection_reason_counts" in summary
    assert record["records"]
    json.dumps(record, sort_keys=True)
