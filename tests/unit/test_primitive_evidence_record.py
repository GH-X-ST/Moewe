from __future__ import annotations

import json

from moewe.primitives import (
    PrimitiveGrammarSpec,
    PrimitiveLibraryCandidate,
    generate_primitives,
    primitive_evidence_record_from_candidate,
    primitive_evidence_record_from_primitive,
)
from moewe.returnability import PrimitiveTransition


def _transition(
    primitive_id: str,
    exit_class: str,
    *,
    failure_reason: str | None = None,
) -> PrimitiveTransition:
    return PrimitiveTransition(
        primitive_id=primitive_id,
        design_case_id="design",
        case_set="library_design",
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class="entry",
        exit_class=exit_class,
        retained=failure_reason is None,
        rollout_success=failure_reason is None,
        min_safety_margin_m=1.0 if failure_reason is None else -1.0,
        terminal_specific_energy_change_j_kg=2.0,
        terminal_specific_energy_margin_j_kg=None,
        max_angle_of_attack_rad=0.1,
        max_command_abs_rad=0.2,
        failure_reason=failure_reason,
        retention_reason="retained" if failure_reason is None else f"rollout_failure:{failure_reason}",
        scenario_id="design",
    )


def test_primitive_evidence_record_from_candidate_is_json_serialisable() -> None:
    candidate = PrimitiveLibraryCandidate(
        primitive_id="prim_a",
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class="entry",
        exit_class="terminal",
        represented_primitive_ids=("prim_a",),
        feature_record={"aggressiveness_level": 1, "mean_positive_vertical_wind_m_s": 0.3},
    )

    record = primitive_evidence_record_from_candidate(
        candidate,
        (_transition("prim_a", "terminal"),),
        recoverable_classes=frozenset({"terminal"}),
    )
    payload = record.to_record()

    assert payload["primitive_id"] == "prim_a"
    assert payload["successor_class_distribution"] == {"terminal": 1.0}
    assert payload["returnable_successor_mask"] == {"terminal": True}
    assert payload["validation_sample_count"] == 1
    assert record.to_json() == record.to_json()
    json.dumps(payload, sort_keys=True)


def test_successor_distribution_sums_to_one_when_nonempty() -> None:
    candidate = PrimitiveLibraryCandidate(
        primitive_id="prim_a",
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class="entry",
        exit_class="terminal",
        represented_primitive_ids=("prim_a",),
        feature_record={},
    )

    record = primitive_evidence_record_from_candidate(
        candidate,
        (_transition("prim_a", "terminal"), _transition("prim_a", "dead", failure_reason="floor")),
        recoverable_classes=frozenset({"terminal"}),
    )

    assert sum(record.successor_class_distribution.values()) == 1.0
    assert record.returnable_successor_mask["dead"] is False
    assert record.returnable_successor_mask["terminal"] is True
    assert record.hard_failure_probability == 0.5


def test_executable_primitive_maps_to_evidence_record_without_rollout() -> None:
    primitive = generate_primitives(PrimitiveGrammarSpec.smoke())[0]

    record = primitive_evidence_record_from_primitive(primitive)
    payload = record.to_record()

    assert payload["primitive_id"] == primitive.primitive_id
    assert payload["validation_sample_count"] == 0
    assert payload["validation_case_set_id"] == "unvalidated"
    assert payload["local_reference_state"]["state_vector"]
    assert payload["nominal_command"]["command_order"] == "[aileron, elevator, rudder]"
    assert record.to_json() == record.to_json()
