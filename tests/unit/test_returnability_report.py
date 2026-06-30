from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from moewe.primitives import (
    CompressionSpec,
    CompressionTierBudgets,
    build_structured_library_design_cases,
    build_structured_primitive_library,
    compress_retained_primitives,
    dense_smoke_grammar,
)
from moewe.returnability import (
    ReturnabilityReport,
    build_returnability_graph_from_report,
    map_compressed_tier_to_returnability,
)


def test_returnability_report_serialises_summary_and_transitions() -> None:
    report = build_structured_primitive_library(
        design_cases=build_structured_library_design_cases()[:2],
        max_primitives=3,
    )
    graph = build_returnability_graph_from_report(report)
    returnability_report = ReturnabilityReport(graph)

    record = returnability_report.to_record()
    assert record["transition_count"] == report.validated_rollout_count
    assert record["primitive_count"] == report.primitive_candidate_count
    assert "design_case_coverage" in record
    assert "classes_without_recovery_successors" in record

    with TemporaryDirectory(prefix=".tmp_returnability_report_", dir=Path.cwd()) as directory:
        output = Path(directory) / "returnability_report.json"
        returnability_report.write_json(output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["summary"]["transition_count"] == report.validated_rollout_count
        assert "states" not in output.read_text(encoding="utf-8")


def test_compressed_tier_maps_back_to_returnability_transitions() -> None:
    grammar = dense_smoke_grammar()
    structured_report = build_structured_primitive_library(
        grammar_spec=grammar,
        design_cases=build_structured_library_design_cases()[:2],
        max_primitives=6,
    )
    graph = build_returnability_graph_from_report(structured_report)
    compression = CompressionSpec(tier_budgets=CompressionTierBudgets(heavy=3, balanced=2, light=1, super_light=1, smoke=1))
    compressed = compress_retained_primitives(
        structured_report.validation_report,
        spec=compression,
        grammar_spec=grammar,
    )

    trace = map_compressed_tier_to_returnability(compressed, graph, tier_name="balanced")

    assert trace
    assert all(record["represented_primitive_id"] for record in trace)
    assert all(record["returnability_transition_ids"] for record in trace)
    assert set().union(*(set(record["design_case_ids"]) for record in trace)) == {"design_still_air_trim_recovery", "design_still_air_gate_alignment"}
