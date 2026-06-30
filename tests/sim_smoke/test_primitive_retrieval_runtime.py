from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from moewe.primitives import (
    CompressionSpec,
    CompressionTierBudgets,
    PrimitiveLibrary,
    check_retrieval_timing,
    build_structured_library_design_cases,
    compress_retained_primitives,
    dense_smoke_grammar,
    run_validation_sweep,
)
from moewe.sim.state import FlightState


def _library_report():
    grammar = dense_smoke_grammar()
    scenarios = tuple(case.to_validation_scenario() for case in build_structured_library_design_cases()[:2])
    report = run_validation_sweep(grammar_spec=grammar, scenarios=scenarios, max_primitives=6)
    compressed = compress_retained_primitives(
        report,
        spec=CompressionSpec(
            tier_budgets=CompressionTierBudgets(heavy=3, balanced=2, light=1, super_light=1, smoke=1)
        ),
        grammar_spec=grammar,
    )
    return report, compressed, PrimitiveLibrary.from_compression(compressed)


def test_runtime_retrieval_returns_candidates_for_exact_entry_class() -> None:
    report, _, library = _library_report()
    state = report.results[0].rollout.states[0]

    query = library.query(state, tier="balanced")

    assert query.candidates
    assert not query.fallback_used
    assert query.check_record()["candidate_count"] == len(query.candidates)


def test_runtime_retrieval_fallback_is_explicit() -> None:
    _, _, library = _library_report()
    state = FlightState(
        position_w_m=np.array([0.0, 3.0, 4.0]),
        euler_rad=np.array([0.9, 0.6, 0.0]),
        velocity_b_m_s=np.array([13.0, 0.0, 0.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )

    query = library.query(state, tier="balanced")

    assert query.fallback_used
    assert query.fallback_reason == "no_exact_entry_class"
    assert query.available_entry_classes


def test_retrieval_timing_check_returns_finite_non_negative_values() -> None:
    report, _, library = _library_report()
    check = check_retrieval_timing(library, [report.results[0].rollout.states[0]], tier="balanced", repeat=3)
    record = check.to_record()

    assert record["query_count"] == 3
    for key in ("mean_ms", "p95_ms", "p99_ms"):
        assert np.isfinite(record[key])
        assert record[key] >= 0.0


def test_compressed_library_manifest_export_is_deterministic() -> None:
    _, compressed, _ = _library_report()

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
        first = Path(tmp_dir) / "primitive_library_tiers_first.json"
        second = Path(tmp_dir) / "primitive_library_tiers_second.json"
        compressed.write_json(first)
        compressed.write_json(second)
        first_payload = json.loads(first.read_text(encoding="utf-8"))
        second_payload = json.loads(second.read_text(encoding="utf-8"))

    assert first_payload == second_payload
    assert "balanced" in first_payload["tiers"]
    assert "raw_retained" in first_payload["tiers"]
