from __future__ import annotations

from moewe.primitives import (
    CompressionSpec,
    CompressionTierBudgets,
    build_structured_library_design_cases,
    compress_retained_primitives,
    dense_smoke_grammar,
    run_validation_sweep,
)


def _compressed():
    grammar = dense_smoke_grammar()
    scenarios = tuple(case.to_validation_scenario() for case in build_structured_library_design_cases()[:2])
    report = run_validation_sweep(grammar_spec=grammar, scenarios=scenarios, max_primitives=6)
    spec = CompressionSpec(tier_budgets=CompressionTierBudgets(heavy=3, balanced=2, light=1, super_light=1, smoke=1))
    return report, compress_retained_primitives(report, spec=spec, grammar_spec=grammar)


def test_compression_uses_existing_retained_primitives_only() -> None:
    report, compressed = _compressed()
    retained_ids = {result.primitive_id for result in report.results if result.retention.retained}

    assert retained_ids
    for tier in compressed.tiers.values():
        for entry in tier.entries:
            assert entry.primitive_id in retained_ids
            assert set(entry.represented_primitive_ids).issubset(retained_ids)


def test_compressed_tier_counts_do_not_exceed_raw_retained_count() -> None:
    _, compressed = _compressed()
    raw_count = compressed.tier("raw_retained").candidate_count

    assert compressed.tier("heavy").candidate_count <= raw_count
    assert compressed.tier("balanced").candidate_count <= raw_count
    assert compressed.tier("light").candidate_count <= raw_count
    assert compressed.tier("super_light").candidate_count <= raw_count
    assert compressed.tier("smoke").candidate_count <= raw_count


def test_group_representative_mapping_is_complete() -> None:
    _, compressed = _compressed()
    raw_ids = {
        primitive_id
        for entry in compressed.tier("raw_retained").entries
        for primitive_id in entry.represented_primitive_ids
    }
    represented_ids = {
        primitive_id
        for entry in compressed.tier("balanced").entries
        for primitive_id in entry.represented_primitive_ids
    }

    assert represented_ids == raw_ids
    assert compressed.tier("balanced").to_summary()["compression_method"] == "deterministic_grouped_medoids"
