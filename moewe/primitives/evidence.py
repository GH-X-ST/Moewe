"""Lightweight evidence records for primitive smoke rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Iterable


def _sorted_dict(mapping: dict[str, object]) -> dict[str, object]:
    return {str(key): _jsonable(mapping[key]) for key in sorted(mapping)}


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return _sorted_dict(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _histogram(values: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        label = "none" if value is None else str(value)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _distribution(labels: Iterable[str]) -> dict[str, float]:
    materialised = [str(label) for label in labels]
    if not materialised:
        return {}
    counts = _histogram(materialised)
    total = float(len(materialised))
    return {key: count / total for key, count in counts.items()}


def _observed_distribution(values: Iterable[float | None]) -> dict[str, object]:
    materialised = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not materialised:
        return {"observed": [], "sample_count": 0}
    return {
        "observed": materialised,
        "sample_count": len(materialised),
        "minimum": min(materialised),
        "maximum": max(materialised),
    }


def _hard_failure(reason: str | None) -> bool:
    if reason is None:
        return False
    text = reason.lower()
    return any(token in text for token in ("floor", "wall", "ceiling", "stall", "bank", "non_finite"))


@dataclass(frozen=True)
class PrimitiveEvidence:
    """Public evidence summary for one primitive rollout."""

    primitive_id: str
    family: str
    controller_type: str
    rollout_success: bool
    min_safety_margin_m: float
    terminal_specific_energy_change_j_kg: float
    terminal_specific_energy_margin_j_kg: float | None
    max_angle_of_attack_rad: float
    max_command_abs_rad: float
    gate_miss_distance_m: float | None
    failure_reason: str | None
    scenario_id: str
    seed: int | None = None
    entry_class: str | None = None
    exit_class: str | None = None
    retained: bool | None = None
    retention_reason: str | None = None
    rollout_duration_s: float = 0.0

    def to_record(self) -> dict[str, object]:
        """Return a serialisable evidence record."""

        return {
            "primitive_id": self.primitive_id,
            "family": self.family,
            "controller_type": self.controller_type,
            "rollout_success": self.rollout_success,
            "min_safety_margin_m": self.min_safety_margin_m,
            "terminal_specific_energy_change_j_kg": self.terminal_specific_energy_change_j_kg,
            "terminal_specific_energy_margin_j_kg": self.terminal_specific_energy_margin_j_kg,
            "max_angle_of_attack_rad": self.max_angle_of_attack_rad,
            "max_command_abs_rad": self.max_command_abs_rad,
            "gate_miss_distance_m": self.gate_miss_distance_m,
            "failure_reason": self.failure_reason,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "entry_class": self.entry_class,
            "exit_class": self.exit_class,
            "retained": self.retained,
            "retention_reason": self.retention_reason,
            "rollout_duration_s": self.rollout_duration_s,
        }


@dataclass(frozen=True)
class PrimitiveEvidenceRecord:
    """Manuscript-facing evidence schema for one closed-loop primitive."""

    primitive_id: str
    family: str
    aggressiveness_level: int
    entry_class: str
    entry_set_or_entry_predicate: str
    local_reference_state: dict[str, object]
    nominal_command: dict[str, object]
    closed_loop_policy_id: str
    horizon: float
    actuator_feasibility_summary: dict[str, object]
    timing_compatibility_summary: dict[str, object]
    successor_class_distribution: dict[str, float]
    returnable_successor_mask: dict[str, bool]
    hard_failure_probability: float
    soft_failure_probability: float
    minimum_safety_margin_distribution: dict[str, object]
    terminal_energy_delta_distribution: dict[str, object]
    useful_lift_exposure_distribution: dict[str, object]
    task_progress_distribution: dict[str, object]
    validation_case_set_id: str
    validation_sample_count: int
    runtime_cost_summary: dict[str, object]
    failure_mode_histogram: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.primitive_id:
            raise ValueError("primitive_id must be non-empty.")
        if not self.family:
            raise ValueError("family must be non-empty.")
        if int(self.aggressiveness_level) < 0:
            raise ValueError("aggressiveness_level must be non-negative.")
        if not self.entry_class:
            raise ValueError("entry_class must be non-empty.")
        if not self.closed_loop_policy_id:
            raise ValueError("closed_loop_policy_id must be non-empty.")
        if not math.isfinite(float(self.horizon)) or float(self.horizon) < 0.0:
            raise ValueError("horizon must be finite and non-negative.")
        if int(self.validation_sample_count) < 0:
            raise ValueError("validation_sample_count must be non-negative.")
        for name, value in (
            ("hard_failure_probability", self.hard_failure_probability),
            ("soft_failure_probability", self.soft_failure_probability),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        total_probability = sum(float(value) for value in self.successor_class_distribution.values())
        if self.successor_class_distribution and not math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("successor_class_distribution must sum to one when nonempty.")
        missing_mask = set(self.successor_class_distribution) - set(self.returnable_successor_mask)
        if missing_mask:
            raise ValueError(f"returnable_successor_mask missing classes: {sorted(missing_mask)}")

    def to_record(self) -> dict[str, object]:
        return {
            "primitive_id": self.primitive_id,
            "family": self.family,
            "aggressiveness_level": int(self.aggressiveness_level),
            "entry_class": self.entry_class,
            "entry_set_or_entry_predicate": self.entry_set_or_entry_predicate,
            "local_reference_state": _jsonable(self.local_reference_state),
            "nominal_command": _jsonable(self.nominal_command),
            "closed_loop_policy_id": self.closed_loop_policy_id,
            "horizon": float(self.horizon),
            "actuator_feasibility_summary": _jsonable(self.actuator_feasibility_summary),
            "timing_compatibility_summary": _jsonable(self.timing_compatibility_summary),
            "successor_class_distribution": {
                key: float(self.successor_class_distribution[key])
                for key in sorted(self.successor_class_distribution)
            },
            "returnable_successor_mask": {
                key: bool(self.returnable_successor_mask[key])
                for key in sorted(self.returnable_successor_mask)
            },
            "hard_failure_probability": float(self.hard_failure_probability),
            "soft_failure_probability": float(self.soft_failure_probability),
            "minimum_safety_margin_distribution": _jsonable(self.minimum_safety_margin_distribution),
            "terminal_energy_delta_distribution": _jsonable(self.terminal_energy_delta_distribution),
            "useful_lift_exposure_distribution": _jsonable(self.useful_lift_exposure_distribution),
            "task_progress_distribution": _jsonable(self.task_progress_distribution),
            "validation_case_set_id": self.validation_case_set_id,
            "validation_sample_count": int(self.validation_sample_count),
            "runtime_cost_summary": _jsonable(self.runtime_cost_summary),
            "failure_mode_histogram": {
                key: int(self.failure_mode_histogram[key])
                for key in sorted(self.failure_mode_histogram)
            },
        }

    def to_json(self) -> str:
        """Return deterministic JSON for the public evidence schema."""

        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))


def primitive_evidence_record_from_transition(
    transition: object,
    *,
    recoverable_classes: frozenset[str] | set[str] = frozenset(),
) -> PrimitiveEvidenceRecord:
    """Build a degenerate evidence record from one graph transition."""

    return primitive_evidence_record_from_transitions(
        primitive_id=str(getattr(transition, "primitive_id")),
        family=str(getattr(transition, "family")),
        controller_type=str(getattr(transition, "controller_type")),
        entry_class=str(getattr(transition, "entry_class")),
        transitions=(transition,),
        recoverable_classes=recoverable_classes,
    )


def primitive_evidence_record_from_candidate(
    candidate: object,
    transitions: tuple[object, ...] | list[object],
    *,
    recoverable_classes: frozenset[str] | set[str] = frozenset(),
) -> PrimitiveEvidenceRecord:
    """Build a primitive evidence record for a runtime library candidate."""

    primitive_id = str(getattr(candidate, "primitive_id"))
    represented = set(getattr(candidate, "represented_primitive_ids", ()))
    represented.add(primitive_id)
    matched = tuple(
        transition
        for transition in transitions
        if str(getattr(transition, "primitive_id")) in represented
    )
    feature = dict(getattr(candidate, "feature_record", {}))
    if matched:
        return primitive_evidence_record_from_transitions(
            primitive_id=primitive_id,
            family=str(getattr(candidate, "family")),
            controller_type=str(getattr(candidate, "controller_type")),
            entry_class=str(getattr(candidate, "entry_class")),
            transitions=matched,
            recoverable_classes=recoverable_classes,
            feature_record=feature,
        )
    successor = str(getattr(candidate, "exit_class"))
    successor_distribution = {successor: 1.0}
    return PrimitiveEvidenceRecord(
        primitive_id=primitive_id,
        family=str(getattr(candidate, "family")),
        aggressiveness_level=_aggressiveness_from_feature(feature),
        entry_class=str(getattr(candidate, "entry_class")),
        entry_set_or_entry_predicate=str(getattr(candidate, "entry_class")),
        local_reference_state={},
        nominal_command={"command_order": "[aileron, elevator, rudder]"},
        closed_loop_policy_id=str(getattr(candidate, "controller_type")),
        horizon=float(_finite_or_none(feature.get("rollout_duration_s")) or 0.0),
        actuator_feasibility_summary={
            "actuator_feasible": feature.get("retention_reason") != "command_magnitude",
            "max_command_abs_rad": _finite_or_none(feature.get("max_command_abs_rad")),
        },
        timing_compatibility_summary={"timing_compatible": True},
        successor_class_distribution=successor_distribution,
        returnable_successor_mask={successor: successor in set(recoverable_classes)},
        hard_failure_probability=0.0,
        soft_failure_probability=0.0,
        minimum_safety_margin_distribution=_observed_distribution((_finite_or_none(feature.get("min_safety_margin_m")),)),
        terminal_energy_delta_distribution=_observed_distribution(
            (_finite_or_none(feature.get("terminal_specific_energy_change_j_kg")),)
        ),
        useful_lift_exposure_distribution=_observed_distribution(
            (_finite_or_none(feature.get("mean_positive_vertical_wind_m_s")),)
        ),
        task_progress_distribution=_observed_distribution((_finite_or_none(feature.get("gate_miss_distance_m")),)),
        validation_case_set_id=str(feature.get("scenario_id") or "runtime_candidate"),
        validation_sample_count=1,
        runtime_cost_summary={"source": "runtime_candidate_feature"},
        failure_mode_histogram={},
    )


def primitive_evidence_record_from_primitive(
    primitive: object,
    evidence: PrimitiveEvidence | None = None,
    *,
    entry_class: str | None = None,
    exit_class: str | None = None,
    recoverable_classes: frozenset[str] | set[str] = frozenset(),
) -> PrimitiveEvidenceRecord:
    """Build a schema record for an executable primitive object."""

    primitive_id = str(getattr(primitive, "primitive_id"))
    family = str(getattr(primitive, "family"))
    controller_type = str(getattr(primitive, "controller_type"))
    metadata = dict(getattr(primitive, "metadata", {}))
    reference = getattr(primitive, "reference")
    local_state = reference.state_at(0.0).as_vector().tolist()
    nominal_command = reference.command_at(0.0).tolist()
    successor = exit_class or (None if evidence is None else evidence.exit_class) or "unobserved"
    entry = entry_class or (None if evidence is None else evidence.entry_class) or "unobserved"
    rollout_duration = 0.0 if evidence is None else float(evidence.rollout_duration_s)
    horizon = max(float(getattr(reference, "total_duration_s", 0.0)), rollout_duration)
    failure_reason = None if evidence is None else evidence.failure_reason
    retained = None if evidence is None else evidence.retained
    hard_failure_probability = 0.0 if failure_reason is None else 1.0 if _hard_failure(failure_reason) else 0.0
    soft_failure_probability = 0.0 if failure_reason is None or hard_failure_probability else 1.0
    return PrimitiveEvidenceRecord(
        primitive_id=primitive_id,
        family=family,
        aggressiveness_level=_aggressiveness_from_feature(metadata),
        entry_class=entry,
        entry_set_or_entry_predicate=entry,
        local_reference_state={"state_vector": [float(value) for value in local_state]},
        nominal_command={
            "command_rad": [float(value) for value in nominal_command],
            "command_order": "[aileron, elevator, rudder]",
        },
        closed_loop_policy_id=controller_type,
        horizon=horizon,
        actuator_feasibility_summary={
            "actuator_feasible": failure_reason is None or "command" not in failure_reason,
            "retained": retained,
            "max_command_abs_rad": None if evidence is None else float(evidence.max_command_abs_rad),
        },
        timing_compatibility_summary={"timing_compatible": True, "horizon_s": horizon},
        successor_class_distribution={successor: 1.0},
        returnable_successor_mask={successor: successor in set(recoverable_classes)},
        hard_failure_probability=hard_failure_probability,
        soft_failure_probability=soft_failure_probability,
        minimum_safety_margin_distribution=_observed_distribution(
            () if evidence is None else (evidence.min_safety_margin_m,)
        ),
        terminal_energy_delta_distribution=_observed_distribution(
            () if evidence is None else (evidence.terminal_specific_energy_change_j_kg,)
        ),
        useful_lift_exposure_distribution=_observed_distribution(()),
        task_progress_distribution=_observed_distribution(
            () if evidence is None else (evidence.gate_miss_distance_m,)
        ),
        validation_case_set_id="unvalidated" if evidence is None else evidence.scenario_id,
        validation_sample_count=0 if evidence is None else 1,
        runtime_cost_summary={"source": "primitive_object", "rollout_duration_s": rollout_duration},
        failure_mode_histogram={} if failure_reason is None else {failure_reason: 1},
    )


def primitive_evidence_record_from_transitions(
    *,
    primitive_id: str,
    family: str,
    controller_type: str,
    entry_class: str,
    transitions: tuple[object, ...] | list[object],
    recoverable_classes: frozenset[str] | set[str] = frozenset(),
    feature_record: dict[str, object] | None = None,
) -> PrimitiveEvidenceRecord:
    """Build a public evidence record from graph transitions."""

    transition_tuple = tuple(transitions)
    if not transition_tuple:
        raise ValueError("At least one transition is required.")
    feature = {} if feature_record is None else dict(feature_record)
    successor_distribution = _distribution(str(getattr(transition, "exit_class")) for transition in transition_tuple)
    hard_count = sum(1 for transition in transition_tuple if _hard_failure(getattr(transition, "failure_reason", None)))
    failed_count = sum(1 for transition in transition_tuple if not bool(getattr(transition, "rollout_success", False)))
    sample_count = len(transition_tuple)
    case_sets = {
        str(getattr(transition, "case_set", "unknown"))
        for transition in transition_tuple
    }
    durations = [
        _finite_or_none(feature.get("rollout_duration_s")),
        *(_finite_or_none(getattr(transition, "rollout_duration_s", None)) for transition in transition_tuple),
    ]
    horizon = max([value for value in durations if value is not None] or [0.0])
    return PrimitiveEvidenceRecord(
        primitive_id=primitive_id,
        family=family,
        aggressiveness_level=_aggressiveness_from_feature(feature),
        entry_class=entry_class,
        entry_set_or_entry_predicate=entry_class,
        local_reference_state={},
        nominal_command={"command_order": "[aileron, elevator, rudder]"},
        closed_loop_policy_id=controller_type,
        horizon=float(horizon),
        actuator_feasibility_summary={
            "actuator_feasible": not any(
                "command" in str(getattr(transition, "failure_reason", "")).lower()
                or "command" in str(getattr(transition, "retention_reason", "")).lower()
                for transition in transition_tuple
            ),
            "max_command_abs_rad": max(
                float(getattr(transition, "max_command_abs_rad"))
                for transition in transition_tuple
                if _finite_or_none(getattr(transition, "max_command_abs_rad", None)) is not None
            ),
        },
        timing_compatibility_summary={
            "timing_compatible": True,
            "sample_count": sample_count,
        },
        successor_class_distribution=successor_distribution,
        returnable_successor_mask={
            successor: successor in set(recoverable_classes)
            for successor in successor_distribution
        },
        hard_failure_probability=hard_count / float(sample_count),
        soft_failure_probability=max(0, failed_count - hard_count) / float(sample_count),
        minimum_safety_margin_distribution=_observed_distribution(
            getattr(transition, "min_safety_margin_m", None) for transition in transition_tuple
        ),
        terminal_energy_delta_distribution=_observed_distribution(
            getattr(transition, "terminal_specific_energy_change_j_kg", None)
            for transition in transition_tuple
        ),
        useful_lift_exposure_distribution=_observed_distribution(
            (_finite_or_none(feature.get("mean_positive_vertical_wind_m_s")),)
        ),
        task_progress_distribution=_observed_distribution(
            (_finite_or_none(feature.get("gate_miss_distance_m")),)
        ),
        validation_case_set_id=";".join(sorted(case_sets)),
        validation_sample_count=sample_count,
        runtime_cost_summary={"source": "returnability_transition", "sample_count": sample_count},
        failure_mode_histogram=_histogram(getattr(transition, "failure_reason", None) for transition in transition_tuple),
    )


def _aggressiveness_from_feature(feature: dict[str, object]) -> int:
    value = feature.get("aggressiveness_level")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return max(0, int(value))
    command = _finite_or_none(feature.get("max_command_abs_rad")) or 0.0
    alpha = _finite_or_none(feature.get("max_angle_of_attack_rad")) or 0.0
    return max(0, int(round(10.0 * (abs(command) + abs(alpha)))))


__all__ = [
    "PrimitiveEvidence",
    "PrimitiveEvidenceRecord",
    "primitive_evidence_record_from_candidate",
    "primitive_evidence_record_from_primitive",
    "primitive_evidence_record_from_transition",
    "primitive_evidence_record_from_transitions",
]
