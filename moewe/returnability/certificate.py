"""Empirical returnability certificate scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .graph import PrimitiveTransition
from .sets import compute_returnability_class_sets


_HARD_FAILURE_TOKENS = (
    "floor",
    "wall",
    "ceiling",
    "stall",
    "angle_of_attack",
    "bank_limit",
    "non_finite",
)
_ACTUATOR_TOKENS = ("command", "actuator")


def _unique_sorted(values: set[str] | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def _probability_distribution(labels: tuple[str, ...]) -> dict[str, float]:
    if not labels:
        return {}
    total = float(len(labels))
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {label: counts[label] / total for label in sorted(counts)}


def _reason_text(transition: PrimitiveTransition) -> str:
    return " ".join(
        value
        for value in (transition.failure_reason, transition.retention_reason)
        if value is not None
    ).lower()


def _is_hard_failure(transition: PrimitiveTransition, forbidden_classes: frozenset[str]) -> bool:
    if transition.exit_class in forbidden_classes:
        return True
    text = _reason_text(transition)
    return any(token in text for token in _HARD_FAILURE_TOKENS)


def _actuator_feasible(transitions: tuple[PrimitiveTransition, ...]) -> bool:
    if not transitions:
        return False
    for transition in transitions:
        text = _reason_text(transition)
        if any(token in text for token in _ACTUATOR_TOKENS):
            return False
        if not math.isfinite(float(transition.max_command_abs_rad)):
            return False
    return True


@dataclass(frozen=True)
class ReturnabilityThresholds:
    """Empirical thresholds used by the class-based returnability certificate."""

    p_return_min: float = 1.0
    p_hard_max: float = 0.0
    margin_min: float = -10.0
    require_actuator_feasible: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.p_return_min) <= 1.0:
            raise ValueError("p_return_min must be in [0, 1].")
        if not 0.0 <= float(self.p_hard_max) <= 1.0:
            raise ValueError("p_hard_max must be in [0, 1].")
        if not math.isfinite(float(self.margin_min)):
            raise ValueError("margin_min must be finite.")

    def to_record(self) -> dict[str, object]:
        return {
            "p_return_min": float(self.p_return_min),
            "p_hard_max": float(self.p_hard_max),
            "margin_min": float(self.margin_min),
            "require_actuator_feasible": bool(self.require_actuator_feasible),
        }


@dataclass(frozen=True)
class ReturnabilityEdgeEvidence:
    """Evidence for one entry-class and primitive-labelled edge."""

    entry_class: str
    primitive_id: str
    successor_class_distribution: dict[str, float]
    returnable_successor_mask: dict[str, bool]
    hard_failure_probability: float
    minimum_safety_margin_m: float
    actuator_feasible: bool
    transition_count: int
    reasons: tuple[str, ...] = ()

    @property
    def successor_return_probability(self) -> float:
        return float(
            sum(
                probability
                for successor, probability in self.successor_class_distribution.items()
                if self.returnable_successor_mask.get(successor, False)
            )
        )

    def admissible(self, thresholds: ReturnabilityThresholds) -> bool:
        if self.successor_return_probability < float(thresholds.p_return_min):
            return False
        if self.hard_failure_probability > float(thresholds.p_hard_max):
            return False
        if self.minimum_safety_margin_m < float(thresholds.margin_min):
            return False
        if thresholds.require_actuator_feasible and not self.actuator_feasible:
            return False
        return True

    def to_record(self) -> dict[str, object]:
        return {
            "entry_class": self.entry_class,
            "primitive_id": self.primitive_id,
            "successor_class_distribution": {
                key: float(self.successor_class_distribution[key])
                for key in sorted(self.successor_class_distribution)
            },
            "returnable_successor_mask": {
                key: bool(self.returnable_successor_mask[key])
                for key in sorted(self.returnable_successor_mask)
            },
            "successor_return_probability": self.successor_return_probability,
            "hard_failure_probability": float(self.hard_failure_probability),
            "minimum_safety_margin_m": float(self.minimum_safety_margin_m),
            "actuator_feasible": bool(self.actuator_feasible),
            "transition_count": int(self.transition_count),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ReturnabilityCertificate:
    """Empirical, class-based returnability certificate."""

    thresholds: ReturnabilityThresholds
    iterations: int
    returnable_classes: frozenset[str]
    nonreturnable_classes: frozenset[str]
    forbidden_classes: frozenset[str]
    terminal_success_classes: frozenset[str]
    edges: tuple[ReturnabilityEdgeEvidence, ...]
    reasons_by_class: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def edge_for(self, entry_class: str, primitive_id: str) -> ReturnabilityEdgeEvidence | None:
        for edge in self.edges:
            if edge.entry_class == entry_class and edge.primitive_id == primitive_id:
                return edge
        return None

    def admissible(self, entry_class: str, primitive_id: str) -> bool:
        edge = self.edge_for(entry_class, primitive_id)
        return False if edge is None else edge.admissible(self.thresholds)

    def to_summary(self) -> dict[str, object]:
        return {
            "thresholds": self.thresholds.to_record(),
            "iterations": int(self.iterations),
            "returnable_classes": sorted(self.returnable_classes),
            "nonreturnable_classes": sorted(self.nonreturnable_classes),
            "forbidden_classes": sorted(self.forbidden_classes),
            "terminal_success_classes": sorted(self.terminal_success_classes),
            "edge_count": len(self.edges),
            "admissible_edge_count": sum(1 for edge in self.edges if edge.admissible(self.thresholds)),
            "reasons_by_class": {
                key: list(self.reasons_by_class[key])
                for key in sorted(self.reasons_by_class)
            },
        }

    def to_record(self) -> dict[str, object]:
        record = self.to_summary()
        record["edges"] = [edge.to_record() for edge in self.edges]
        return record


def compute_empirical_returnability_certificate(
    transitions: tuple[PrimitiveTransition, ...] | list[PrimitiveTransition],
    *,
    terminal_success_classes: frozenset[str] | set[str] | None = None,
    forbidden_classes: frozenset[str] | set[str] | None = None,
    thresholds: ReturnabilityThresholds | None = None,
) -> ReturnabilityCertificate:
    """Compute a conservative empirical returnability certificate."""

    transition_tuple = tuple(transitions)
    cfg = ReturnabilityThresholds() if thresholds is None else thresholds
    class_sets = compute_returnability_class_sets(transition_tuple)
    forbidden = (
        class_sets.forbidden_classes
        if forbidden_classes is None
        else frozenset(str(item) for item in forbidden_classes)
    )
    terminal = (
        class_sets.terminal_success_classes
        if terminal_success_classes is None
        else frozenset(str(item) for item in terminal_success_classes)
    )
    observed = frozenset(
        {transition.entry_class for transition in transition_tuple}
        | {transition.exit_class for transition in transition_tuple}
    )
    grouped = _group_transitions(transition_tuple)

    returnable = set(terminal) - set(forbidden)
    iterations = 0
    changed = True
    while changed:
        changed = False
        iterations += 1
        for (entry_class, _primitive_id), edge_transitions in grouped.items():
            if entry_class in forbidden or entry_class in returnable:
                continue
            edge = _edge_evidence(
                entry_class=entry_class,
                primitive_id=edge_transitions[0].primitive_id,
                transitions=edge_transitions,
                returnable_classes=frozenset(returnable),
                forbidden_classes=forbidden,
            )
            if edge.admissible(cfg):
                returnable.add(entry_class)
                changed = True
        if not grouped:
            break

    final_returnable = frozenset(sorted(returnable))
    edges = tuple(
        sorted(
            (
                _edge_evidence(
                    entry_class=entry_class,
                    primitive_id=primitive_id,
                    transitions=edge_transitions,
                    returnable_classes=final_returnable,
                    forbidden_classes=forbidden,
                )
                for (entry_class, primitive_id), edge_transitions in grouped.items()
            ),
            key=lambda edge: (edge.entry_class, edge.primitive_id),
        )
    )
    nonreturnable = frozenset(sorted(observed - final_returnable - forbidden))
    reasons_by_class = _class_reasons(nonreturnable, edges, cfg)
    return ReturnabilityCertificate(
        thresholds=cfg,
        iterations=max(0, iterations),
        returnable_classes=final_returnable,
        nonreturnable_classes=nonreturnable,
        forbidden_classes=frozenset(sorted(forbidden)),
        terminal_success_classes=frozenset(sorted(terminal)),
        edges=edges,
        reasons_by_class=reasons_by_class,
    )


def _group_transitions(
    transitions: tuple[PrimitiveTransition, ...],
) -> dict[tuple[str, str], tuple[PrimitiveTransition, ...]]:
    grouped: dict[tuple[str, str], list[PrimitiveTransition]] = {}
    for transition in transitions:
        grouped.setdefault((transition.entry_class, transition.primitive_id), []).append(transition)
    return {
        key: tuple(sorted(items, key=lambda item: item.transition_id))
        for key, items in sorted(grouped.items())
    }


def _edge_evidence(
    *,
    entry_class: str,
    primitive_id: str,
    transitions: tuple[PrimitiveTransition, ...],
    returnable_classes: frozenset[str],
    forbidden_classes: frozenset[str],
) -> ReturnabilityEdgeEvidence:
    successors = tuple(transition.exit_class for transition in transitions)
    distribution = _probability_distribution(successors)
    hard_count = sum(1 for transition in transitions if _is_hard_failure(transition, forbidden_classes))
    margins = [float(transition.min_safety_margin_m) for transition in transitions]
    reasons = _edge_reasons(
        distribution=distribution,
        returnable_classes=returnable_classes,
        hard_failure_probability=hard_count / float(len(transitions)) if transitions else 1.0,
        min_margin=min(margins) if margins else float("-inf"),
        actuator_feasible=_actuator_feasible(transitions),
    )
    return ReturnabilityEdgeEvidence(
        entry_class=entry_class,
        primitive_id=primitive_id,
        successor_class_distribution=distribution,
        returnable_successor_mask={
            successor: successor in returnable_classes
            for successor in sorted(distribution)
        },
        hard_failure_probability=hard_count / float(len(transitions)) if transitions else 1.0,
        minimum_safety_margin_m=min(margins) if margins else float("-inf"),
        actuator_feasible=_actuator_feasible(transitions),
        transition_count=len(transitions),
        reasons=reasons,
    )


def _edge_reasons(
    *,
    distribution: dict[str, float],
    returnable_classes: frozenset[str],
    hard_failure_probability: float,
    min_margin: float,
    actuator_feasible: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not distribution:
        reasons.append("no_successor_evidence")
    if not any(successor in returnable_classes for successor in distribution):
        reasons.append("nonreturnable_successor")
    if hard_failure_probability > 0.0:
        reasons.append("hard_failure_observed")
    if min_margin == float("-inf"):
        reasons.append("missing_safety_margin")
    if not actuator_feasible:
        reasons.append("actuator_infeasible")
    return _unique_sorted(reasons)


def _class_reasons(
    nonreturnable_classes: frozenset[str],
    edges: tuple[ReturnabilityEdgeEvidence, ...],
    thresholds: ReturnabilityThresholds,
) -> dict[str, tuple[str, ...]]:
    reasons: dict[str, tuple[str, ...]] = {}
    for state_class in sorted(nonreturnable_classes):
        class_edges = tuple(edge for edge in edges if edge.entry_class == state_class)
        if not class_edges:
            reasons[state_class] = ("no_outgoing_edge",)
            continue
        items: list[str] = []
        for edge in class_edges:
            if edge.successor_return_probability < thresholds.p_return_min:
                items.append("successor_return_probability_below_threshold")
            if edge.hard_failure_probability > thresholds.p_hard_max:
                items.append("hard_failure_probability_above_threshold")
            if edge.minimum_safety_margin_m < thresholds.margin_min:
                items.append("minimum_safety_margin_below_threshold")
            if thresholds.require_actuator_feasible and not edge.actuator_feasible:
                items.append("actuator_infeasible")
        reasons[state_class] = _unique_sorted(items)
    return reasons


__all__ = [
    "ReturnabilityCertificate",
    "ReturnabilityEdgeEvidence",
    "ReturnabilityThresholds",
    "compute_empirical_returnability_certificate",
]
