"""Runtime primitive-library retrieval without viability scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.state import FlightState

from .classify import PrimitiveStateClassifier
from .compress import CompressedPrimitiveEntry, CompressedPrimitiveLibrary


@dataclass(frozen=True)
class PrimitiveLibraryCandidate:
    """Runtime-visible primitive candidate record."""

    primitive_id: str
    family: str
    controller_type: str
    entry_class: str
    exit_class: str
    represented_primitive_ids: tuple[str, ...]
    feature_record: dict[str, object]

    @classmethod
    def from_entry(cls, entry: CompressedPrimitiveEntry) -> "PrimitiveLibraryCandidate":
        return cls(
            primitive_id=entry.primitive_id,
            family=entry.family,
            controller_type=entry.controller_type,
            entry_class=entry.entry_class,
            exit_class=entry.exit_class,
            represented_primitive_ids=entry.represented_primitive_ids,
            feature_record=entry.feature.to_record(),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "primitive_id": self.primitive_id,
            "family": self.family,
            "controller_type": self.controller_type,
            "entry_class": self.entry_class,
            "exit_class": self.exit_class,
            "represented_primitive_ids": list(self.represented_primitive_ids),
            "feature": self.feature_record,
        }


@dataclass(frozen=True)
class PrimitiveLibraryQuery:
    """Primitive retrieval result for a classified current state."""

    tier: str
    entry_class: str
    candidates: tuple[PrimitiveLibraryCandidate, ...]
    fallback_used: bool
    fallback_reason: str | None
    available_entry_classes: tuple[str, ...]

    def check_record(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "entry_class": self.entry_class,
            "candidate_count": len(self.candidates),
            "candidate_ids": [candidate.primitive_id for candidate in self.candidates],
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "available_entry_classes": list(self.available_entry_classes),
        }


@dataclass(frozen=True)
class PrimitiveLibrary:
    """Runtime library wrapper for compressed primitive tiers."""

    compressed: CompressedPrimitiveLibrary
    classifier: PrimitiveStateClassifier = PrimitiveStateClassifier()

    @classmethod
    def from_compression(
        cls,
        compressed: CompressedPrimitiveLibrary,
        classifier: PrimitiveStateClassifier | None = None,
    ) -> "PrimitiveLibrary":
        return cls(compressed=compressed, classifier=PrimitiveStateClassifier() if classifier is None else classifier)

    def query(self, state: FlightState, tier: str = "balanced") -> PrimitiveLibraryQuery:
        label = self.classifier.classify(state)
        compressed_tier = self.compressed.tier(tier)
        available = tuple(sorted({entry.entry_class for entry in compressed_tier.entries}))
        exact_entries = [entry for entry in compressed_tier.entries if entry.entry_class == label.label]
        fallback_used = False
        fallback_reason = None
        selected_entries = exact_entries
        if not selected_entries:
            fallback_used = True
            if not available:
                fallback_reason = "empty_tier"
                selected_entries = []
            else:
                fallback_reason = "no_exact_entry_class"
                selected_class = _nearest_entry_class(label.label, available)
                selected_entries = [entry for entry in compressed_tier.entries if entry.entry_class == selected_class]
        candidates = tuple(PrimitiveLibraryCandidate.from_entry(entry) for entry in selected_entries)
        return PrimitiveLibraryQuery(
            tier=tier,
            entry_class=label.label,
            candidates=candidates,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            available_entry_classes=available,
        )


def _nearest_entry_class(label: str, available: tuple[str, ...]) -> str:
    requested = set(label.split("|"))

    def distance(candidate: str) -> tuple[int, str]:
        candidate_parts = set(candidate.split("|"))
        return (len(requested.symmetric_difference(candidate_parts)), candidate)

    return min(available, key=distance)
