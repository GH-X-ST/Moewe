"""Returnability graph data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReturnabilityGraphConfig:
    """Transparent thresholds for smoke-scale returnability set construction."""

    min_terminal_safety_margin_m: float = -10.0
    min_terminal_specific_energy_change_j_kg: float = -100.0
    forbidden_reason_tokens: tuple[str, ...] = (
        "non_finite_state",
        "floor",
        "wall",
        "ceiling",
        "command_limit",
        "command_magnitude",
        "stall_limit",
        "angle_of_attack",
        "bank_limit",
        "min_safety_margin",
    )


@dataclass(frozen=True)
class PrimitiveTransition:
    """One primitive-labelled transition between classified states."""

    primitive_id: str
    design_case_id: str
    case_set: str
    family: str
    controller_type: str
    entry_class: str
    exit_class: str
    retained: bool
    rollout_success: bool
    min_safety_margin_m: float
    terminal_specific_energy_change_j_kg: float
    terminal_specific_energy_margin_j_kg: float | None
    max_angle_of_attack_rad: float
    max_command_abs_rad: float
    failure_reason: str | None
    retention_reason: str | None
    scenario_id: str
    seed: int | None = None

    @property
    def transition_id(self) -> str:
        return f"{self.design_case_id}:{self.primitive_id}:{self.entry_class}->{self.exit_class}"

    def to_record(self) -> dict[str, object]:
        return {
            "transition_id": self.transition_id,
            "primitive_id": self.primitive_id,
            "design_case_id": self.design_case_id,
            "case_set": self.case_set,
            "family": self.family,
            "controller_type": self.controller_type,
            "entry_class": self.entry_class,
            "exit_class": self.exit_class,
            "retained": self.retained,
            "rollout_success": self.rollout_success,
            "min_safety_margin_m": self.min_safety_margin_m,
            "terminal_specific_energy_change_j_kg": self.terminal_specific_energy_change_j_kg,
            "terminal_specific_energy_margin_j_kg": self.terminal_specific_energy_margin_j_kg,
            "max_angle_of_attack_rad": self.max_angle_of_attack_rad,
            "max_command_abs_rad": self.max_command_abs_rad,
            "failure_reason": self.failure_reason,
            "retention_reason": self.retention_reason,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ReturnabilityGraph:
    """Directed graph of primitive transitions and derived class sets."""

    transitions: tuple[PrimitiveTransition, ...]
    safe_classes: frozenset[str] = field(default_factory=frozenset)
    forbidden_classes: frozenset[str] = field(default_factory=frozenset)
    terminal_success_classes: frozenset[str] = field(default_factory=frozenset)
    recoverable_classes: frozenset[str] = field(default_factory=frozenset)
    dead_end_classes: frozenset[str] = field(default_factory=frozenset)
    entry_supported_classes: frozenset[str] = field(default_factory=frozenset)
    exit_observed_classes: frozenset[str] = field(default_factory=frozenset)

    @property
    def primitive_ids(self) -> frozenset[str]:
        return frozenset(transition.primitive_id for transition in self.transitions)

    @property
    def state_classes(self) -> frozenset[str]:
        return frozenset(self.entry_supported_classes | self.exit_observed_classes)

    def retained_transitions(self) -> tuple[PrimitiveTransition, ...]:
        return tuple(transition for transition in self.transitions if transition.retained)

    def transitions_by_primitive(self) -> dict[str, tuple[PrimitiveTransition, ...]]:
        grouped: dict[str, list[PrimitiveTransition]] = {}
        for transition in self.transitions:
            grouped.setdefault(transition.primitive_id, []).append(transition)
        return {
            primitive_id: tuple(sorted(items, key=lambda item: item.transition_id))
            for primitive_id, items in sorted(grouped.items())
        }

    def outgoing_retained(self, state_class: str) -> tuple[PrimitiveTransition, ...]:
        return tuple(
            transition
            for transition in self.transitions
            if transition.retained and transition.entry_class == state_class
        )

    def to_records(self) -> list[dict[str, object]]:
        return [transition.to_record() for transition in self.transitions]

    def to_summary(self) -> dict[str, object]:
        return {
            "node_count": len(self.state_classes) + len(self.primitive_ids),
            "state_class_count": len(self.state_classes),
            "transition_count": len(self.transitions),
            "retained_transition_count": len(self.retained_transitions()),
            "primitive_count": len(self.primitive_ids),
            "safe_class_count": len(self.safe_classes),
            "forbidden_class_count": len(self.forbidden_classes),
            "terminal_success_class_count": len(self.terminal_success_classes),
            "recoverable_class_count": len(self.recoverable_classes),
            "dead_end_class_count": len(self.dead_end_classes),
            "entry_supported_classes": sorted(self.entry_supported_classes),
            "exit_observed_classes": sorted(self.exit_observed_classes),
        }
