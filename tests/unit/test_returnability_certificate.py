from __future__ import annotations

import json

from moewe.returnability import (
    PrimitiveTransition,
    ReturnabilityThresholds,
    compute_empirical_returnability_certificate,
)


def _transition(
    primitive_id: str,
    entry_class: str,
    exit_class: str,
    *,
    retained: bool = True,
    failure_reason: str | None = None,
    margin: float = 1.0,
) -> PrimitiveTransition:
    return PrimitiveTransition(
        primitive_id=primitive_id,
        design_case_id="design",
        case_set="library_design",
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class=entry_class,
        exit_class=exit_class,
        retained=retained,
        rollout_success=failure_reason is None,
        min_safety_margin_m=margin,
        terminal_specific_energy_change_j_kg=1.0,
        terminal_specific_energy_margin_j_kg=None,
        max_angle_of_attack_rad=0.1,
        max_command_abs_rad=0.1,
        failure_reason=failure_reason,
        retention_reason="retained" if retained else "rollout_failure",
        scenario_id="design",
    )


def test_certificate_fixed_point_grows_to_predecessor_classes() -> None:
    certificate = compute_empirical_returnability_certificate(
        (
            _transition("prim_terminal", "entry", "terminal"),
            _transition("prim_pre", "pre_entry", "entry"),
        ),
        terminal_success_classes=frozenset({"terminal"}),
        thresholds=ReturnabilityThresholds(p_return_min=1.0, p_hard_max=0.0, margin_min=0.0),
    )

    assert {"terminal", "entry", "pre_entry"} <= set(certificate.returnable_classes)
    assert certificate.iterations >= 1
    json.dumps(certificate.to_record(), sort_keys=True)


def test_certificate_excludes_forbidden_successor_classes() -> None:
    certificate = compute_empirical_returnability_certificate(
        (_transition("prim_bad", "entry", "forbidden", failure_reason="wall"),),
        terminal_success_classes=frozenset({"terminal"}),
        forbidden_classes=frozenset({"forbidden"}),
    )

    assert "forbidden" in certificate.forbidden_classes
    assert "forbidden" not in certificate.returnable_classes
    assert certificate.edge_for("entry", "prim_bad") is not None
    assert certificate.edge_for("entry", "prim_bad").returnable_successor_mask["forbidden"] is False


def test_threshold_tightening_cannot_increase_returnable_set() -> None:
    transitions = (
        _transition("prim_a", "entry", "terminal"),
        _transition("prim_a", "entry", "dead", failure_reason="floor"),
    )

    loose = compute_empirical_returnability_certificate(
        transitions,
        terminal_success_classes=frozenset({"terminal"}),
        thresholds=ReturnabilityThresholds(p_return_min=0.5, p_hard_max=0.5, margin_min=-1.0),
    )
    tight = compute_empirical_returnability_certificate(
        transitions,
        terminal_success_classes=frozenset({"terminal"}),
        thresholds=ReturnabilityThresholds(p_return_min=1.0, p_hard_max=0.0, margin_min=0.0),
    )

    assert set(tight.returnable_classes) <= set(loose.returnable_classes)


def test_hard_failure_threshold_changes_admissibility() -> None:
    transitions = (
        _transition("prim_a", "entry", "terminal"),
        _transition("prim_a", "entry", "dead", failure_reason="floor"),
    )
    loose = compute_empirical_returnability_certificate(
        transitions,
        terminal_success_classes=frozenset({"terminal"}),
        thresholds=ReturnabilityThresholds(p_return_min=0.5, p_hard_max=0.5, margin_min=-1.0),
    )
    tight = compute_empirical_returnability_certificate(
        transitions,
        terminal_success_classes=frozenset({"terminal"}),
        thresholds=ReturnabilityThresholds(p_return_min=0.5, p_hard_max=0.0, margin_min=-1.0),
    )

    assert loose.admissible("entry", "prim_a") is True
    assert tight.admissible("entry", "prim_a") is False
