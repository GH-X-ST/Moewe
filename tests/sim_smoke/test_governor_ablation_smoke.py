from __future__ import annotations

import json
from pathlib import Path

from moewe.baselines import UnfilteredPrimitiveSelector, compare_governor_to_unfiltered
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


def test_governor_ablation_smoke_builds_comparison_record() -> None:
    config_path = Path("config/baselines/governor_ablation_smoke.yaml")
    assert config_path.exists()
    config_text = config_path.read_text(encoding="utf-8")
    assert "no_returnability_filtering: true" in config_text
    assert "writes_files_by_default: false" in config_text

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
    retained_result = next(
        result
        for result in structured_report.validation_report.results
        if result.retention.retained
    )
    state = retained_result.rollout.states[0]

    comparison = compare_governor_to_unfiltered(state, governor, selector, tier="balanced")
    record = comparison.to_record()

    assert record["tier"] == "balanced"
    assert record["entry_class"]
    assert record["unfiltered_decision"]["candidate_count"] > 0
    assert "governor_decision" in record
    assert "unfiltered_decision" in record
    json.dumps(record, sort_keys=True)
