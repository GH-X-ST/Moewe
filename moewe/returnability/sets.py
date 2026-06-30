"""Class-set construction for returnability graphs."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import PrimitiveTransition, ReturnabilityGraphConfig


@dataclass(frozen=True)
class ReturnabilityClassSets:
    """Derived class sets used by the returnability graph."""

    safe_classes: frozenset[str]
    forbidden_classes: frozenset[str]
    terminal_success_classes: frozenset[str]
    recoverable_classes: frozenset[str]
    dead_end_classes: frozenset[str]
    entry_supported_classes: frozenset[str]
    exit_observed_classes: frozenset[str]


def _is_invalid_label(label: str) -> bool:
    return label.startswith("invalid:")


def _has_forbidden_reason(transition: PrimitiveTransition, config: ReturnabilityGraphConfig) -> bool:
    reasons = (transition.failure_reason, transition.retention_reason)
    for reason in reasons:
        if reason is None:
            continue
        for token in config.forbidden_reason_tokens:
            if token in reason:
                return True
    return False


def _transition_terminal_success(transition: PrimitiveTransition, config: ReturnabilityGraphConfig) -> bool:
    return (
        transition.retained
        and transition.rollout_success
        and transition.min_safety_margin_m >= config.min_terminal_safety_margin_m
        and transition.terminal_specific_energy_change_j_kg >= config.min_terminal_specific_energy_change_j_kg
    )


def compute_recoverable_classes(
    transitions: tuple[PrimitiveTransition, ...] | list[PrimitiveTransition],
    terminal_success_classes: frozenset[str] | set[str],
    forbidden_classes: frozenset[str] | set[str] = frozenset(),
) -> frozenset[str]:
    """Compute recoverable classes by fixed-point propagation over retained transitions."""

    forbidden = set(forbidden_classes)
    recoverable = set(terminal_success_classes) - forbidden
    changed = True
    while changed:
        changed = False
        for transition in transitions:
            if not transition.retained:
                continue
            if transition.entry_class in forbidden or transition.exit_class in forbidden:
                continue
            if transition.exit_class in recoverable and transition.entry_class not in recoverable:
                recoverable.add(transition.entry_class)
                changed = True
    return frozenset(sorted(recoverable))


def compute_returnability_class_sets(
    transitions: tuple[PrimitiveTransition, ...] | list[PrimitiveTransition],
    config: ReturnabilityGraphConfig | None = None,
) -> ReturnabilityClassSets:
    """Derive safe, forbidden, terminal, recoverable, and dead-end state classes."""

    cfg = ReturnabilityGraphConfig() if config is None else config
    transition_tuple = tuple(transitions)
    entry_supported = frozenset(transition.entry_class for transition in transition_tuple)
    exit_observed = frozenset(transition.exit_class for transition in transition_tuple)
    observed = set(entry_supported | exit_observed)

    forbidden = {label for label in observed if _is_invalid_label(label)}
    for transition in transition_tuple:
        if _has_forbidden_reason(transition, cfg):
            forbidden.add(transition.exit_class)

    safe = frozenset(sorted(observed - forbidden))
    terminal = frozenset(
        sorted(
            transition.exit_class
            for transition in transition_tuple
            if transition.exit_class not in forbidden and _transition_terminal_success(transition, cfg)
        )
    )
    recoverable = compute_recoverable_classes(transition_tuple, terminal, frozenset(forbidden))
    dead_end = frozenset(sorted(safe - recoverable))
    return ReturnabilityClassSets(
        safe_classes=safe,
        forbidden_classes=frozenset(sorted(forbidden)),
        terminal_success_classes=terminal,
        recoverable_classes=recoverable,
        dead_end_classes=dead_end,
        entry_supported_classes=entry_supported,
        exit_observed_classes=exit_observed,
    )
