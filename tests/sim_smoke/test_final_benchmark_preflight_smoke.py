from __future__ import annotations

import json
from pathlib import Path

from moewe.baselines import BASELINE_METHOD_NAMES, UnfilteredPrimitiveSelector
from moewe.benchmarks import FinalBenchmarkConfig, run_final_benchmark_preflight
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


def test_final_benchmark_preflight_smoke_builds_deterministic_report() -> None:
    config_path = Path("config/simulation/final_benchmark_preflight.yaml")
    assert config_path.exists()
    config_text = config_path.read_text(encoding="utf-8")
    assert "runs_full_simulation: false" in config_text
    assert "writes_files_by_default: false" in config_text
    assert "stops before the full simulation run" in config_text

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
    config = FinalBenchmarkConfig(full_case_count=120, preflight_case_count=4)

    first = run_final_benchmark_preflight(governor, selector, library, config).to_record()
    second = run_final_benchmark_preflight(governor, selector, library, config).to_record()

    assert first == second
    assert first["full_simulation_executed"] is False
    assert first["writes_files_by_default"] is False
    assert first["preflight_case_count"] == 4
    expected_method_names = {"governor", *BASELINE_METHOD_NAMES}
    assert first["preflight_record_count"] == 4 * len(expected_method_names)
    assert first["expected_full_record_count"] == 120 * len(expected_method_names)
    assert set(first["method_names"]) == expected_method_names
    assert "rollout_success" in first["schema_fields"]
    json.dumps(first, sort_keys=True)
    assert not Path("results").exists()
    assert not Path("data").exists()
