from __future__ import annotations

from moewe.returnability import (
    PrimitiveTransition,
    ReturnabilityGraphConfig,
    compute_recoverable_classes,
    compute_returnability_class_sets,
)


def _transition(
    entry_class: str,
    exit_class: str,
    *,
    primitive_id: str = "prim_test",
    retained: bool = True,
    rollout_success: bool = True,
    energy_change: float = 0.0,
    failure_reason: str | None = None,
    retention_reason: str | None = "retained",
) -> PrimitiveTransition:
    return PrimitiveTransition(
        primitive_id=primitive_id,
        design_case_id="design_still_air_trim_recovery",
        case_set="library_design",
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class=entry_class,
        exit_class=exit_class,
        retained=retained,
        rollout_success=rollout_success,
        min_safety_margin_m=0.0,
        terminal_specific_energy_change_j_kg=energy_change,
        terminal_specific_energy_margin_j_kg=None,
        max_angle_of_attack_rad=0.1,
        max_command_abs_rad=0.1,
        failure_reason=failure_reason,
        retention_reason=retention_reason,
        scenario_id="design_still_air_trim_recovery",
    )


def test_recoverable_classes_use_fixed_point_backwards_from_terminal_success() -> None:
    transitions = (
        _transition("class_a", "class_b", primitive_id="prim_a"),
        _transition("class_b", "class_c", primitive_id="prim_b"),
    )

    recoverable = compute_recoverable_classes(transitions, terminal_success_classes={"class_c"})

    assert recoverable == frozenset({"class_a", "class_b", "class_c"})


def test_class_sets_identify_forbidden_and_dead_end_classes() -> None:
    transitions = (
        _transition("recover_entry", "terminal_class", primitive_id="prim_terminal", energy_change=20.0),
        _transition("dead_entry", "dead_exit", primitive_id="prim_dead", energy_change=0.0),
        _transition(
            "unsafe_entry",
            "unsafe_exit",
            primitive_id="prim_unsafe",
            retained=False,
            rollout_success=False,
            failure_reason="floor",
            retention_reason="rollout_failure:floor",
        ),
    )

    class_sets = compute_returnability_class_sets(
        transitions,
        config=ReturnabilityGraphConfig(min_terminal_specific_energy_change_j_kg=10.0),
    )

    assert "unsafe_exit" in class_sets.forbidden_classes
    assert "terminal_class" in class_sets.terminal_success_classes
    assert "recover_entry" in class_sets.recoverable_classes
    assert "dead_entry" in class_sets.dead_end_classes
    assert "dead_exit" in class_sets.dead_end_classes
