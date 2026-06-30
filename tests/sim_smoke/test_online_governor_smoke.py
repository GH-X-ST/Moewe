from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from moewe.governor import OnlineGovernor, check_governor_timing
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


def test_online_governor_smoke_builds_runtime_decision() -> None:
    config_path = Path("config/simulation/online_governor_smoke.yaml")
    assert config_path.exists()
    assert "allow_retrieval_fallback: false" in config_path.read_text(encoding="utf-8")

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
    retained_result = next(
        result
        for result in structured_report.validation_report.results
        if result.retention.retained
    )
    state = retained_result.rollout.states[0]

    decision = governor.decide(state, tier="balanced")
    decision_record = decision.to_record()
    timing = check_governor_timing(governor, [state], tier="balanced", repeat=3)
    timing_record = timing.to_record()

    assert decision.candidate_count > 0
    assert decision.fallback_used is False
    assert decision_record["candidate_evidence"]
    assert timing_record["decision_count"] == 3
    for key in ("mean_ms", "p95_ms", "p99_ms"):
        assert np.isfinite(timing_record[key])
        assert timing_record[key] >= 0.0
    json.dumps({"decision": decision_record, "timing": timing_record}, sort_keys=True)
