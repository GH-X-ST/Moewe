"""Small reports derived from returnability graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moewe.primitives.compress import CompressedPrimitiveLibrary

from .graph import ReturnabilityGraph
from .manifest import write_csv_records, write_json_manifest


def _count(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class ReturnabilityReport:
    """Compact serialisable summary for returnability graph evidence."""

    graph: ReturnabilityGraph

    @property
    def classes_without_outgoing_transitions(self) -> tuple[str, ...]:
        outgoing = {transition.entry_class for transition in self.graph.retained_transitions()}
        return tuple(sorted(self.graph.safe_classes - outgoing))

    @property
    def classes_without_recovery_successors(self) -> tuple[str, ...]:
        missing: list[str] = []
        for state_class in sorted(self.graph.safe_classes):
            if state_class in self.graph.terminal_success_classes:
                continue
            outgoing = self.graph.outgoing_retained(state_class)
            if not any(transition.exit_class in self.graph.recoverable_classes for transition in outgoing):
                missing.append(state_class)
        return tuple(missing)

    @property
    def primitives_to_forbidden_classes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    transition.primitive_id
                    for transition in self.graph.transitions
                    if transition.exit_class in self.graph.forbidden_classes
                }
            )
        )

    @property
    def primitives_to_dead_end_classes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    transition.primitive_id
                    for transition in self.graph.transitions
                    if transition.exit_class in self.graph.dead_end_classes
                }
            )
        )

    def to_record(self) -> dict[str, object]:
        return {
            **self.graph.to_summary(),
            "classes_without_outgoing_transitions": list(self.classes_without_outgoing_transitions),
            "classes_without_recovery_successors": list(self.classes_without_recovery_successors),
            "primitives_to_forbidden_classes": list(self.primitives_to_forbidden_classes),
            "primitives_to_dead_end_classes": list(self.primitives_to_dead_end_classes),
            "design_case_coverage": _count([transition.design_case_id for transition in self.graph.transitions]),
        }

    def to_transition_records(self) -> list[dict[str, object]]:
        return self.graph.to_records()

    def write_json(self, path: str | Path) -> None:
        write_json_manifest(
            path,
            {
                "summary": self.to_record(),
                "transitions": self.to_transition_records(),
            },
        )

    def write_csv(self, path: str | Path) -> None:
        write_csv_records(path, self.to_transition_records())


def map_compressed_tier_to_returnability(
    compressed: CompressedPrimitiveLibrary,
    graph: ReturnabilityGraph,
    tier_name: str = "balanced",
) -> list[dict[str, object]]:
    """Map compressed tier entries back to represented primitive transition evidence."""

    transitions_by_primitive = graph.transitions_by_primitive()
    records: list[dict[str, object]] = []
    for entry in compressed.tier(tier_name).entries:
        for represented_id in entry.represented_primitive_ids:
            transitions = transitions_by_primitive.get(represented_id, ())
            records.append(
                {
                    "tier_name": tier_name,
                    "tier_primitive_id": entry.primitive_id,
                    "represented_primitive_id": represented_id,
                    "returnability_transition_ids": [transition.transition_id for transition in transitions],
                    "design_case_ids": sorted({transition.design_case_id for transition in transitions}),
                    "entry_classes": sorted({transition.entry_class for transition in transitions}),
                    "exit_classes": sorted({transition.exit_class for transition in transitions}),
                }
            )
    return records
