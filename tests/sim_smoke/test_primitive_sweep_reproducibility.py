from __future__ import annotations

from moewe.primitives import (
    AcceptanceThresholds,
    EntryPerturbationSpec,
    PrimitiveRolloutConfig,
    ValidationScenario,
    run_validation_sweep,
)


def _randomized_scenario() -> ValidationScenario:
    half_width = [0.0] * 15
    half_width[0] = 0.02
    half_width[1] = 0.02
    half_width[2] = 0.02
    half_width[3] = 0.02
    half_width[6] = 0.1
    return ValidationScenario(
        scenario_id="randomized_reproducible_smoke",
        seed=13,
        entry_perturbation=EntryPerturbationSpec(half_width=tuple(half_width)),
        rollout_config=PrimitiveRolloutConfig(dt_s=0.01, max_duration_s=0.05),
        thresholds=AcceptanceThresholds(
            min_safety_margin_m=-10.0,
            max_angle_of_attack_rad=10.0,
            max_command_abs_rad=10.0,
            min_terminal_specific_energy_change_j_kg=-100.0,
        ),
    )


def test_validation_sweep_with_fixed_seed_is_reproducible() -> None:
    first = run_validation_sweep(scenarios=(_randomized_scenario(),), max_primitives=3)
    second = run_validation_sweep(scenarios=(_randomized_scenario(),), max_primitives=3)

    assert first.to_records() == second.to_records()
    assert first.to_summary() == second.to_summary()


def test_smoke_sweep_records_deterministic_primitive_ids() -> None:
    first = run_validation_sweep(scenarios=(_randomized_scenario(),), max_primitives=3)
    second = run_validation_sweep(scenarios=(_randomized_scenario(),), max_primitives=3)
    records = first.to_records()

    assert [record["primitive_id"] for record in records] == [record["primitive_id"] for record in second.to_records()]
    assert len({record["primitive_id"] for record in records}) == len(records)
    assert all(str(record["primitive_id"]).startswith("prim_") for record in records)
    assert all(record["seed"] is not None for record in records)
